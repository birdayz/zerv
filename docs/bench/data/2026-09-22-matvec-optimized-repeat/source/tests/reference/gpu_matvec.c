/* External one-node ggml Vulkan oracle/competitor, never an engine dependency. */
#define _POSIX_C_SOURCE 200809L
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-vulkan.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *s) { fprintf(stderr, "%s\n", s); exit(2); }
static uint64_t now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock failed");
    return (uint64_t)t.tv_sec*1000000000 + t.tv_nsec;
}
static void read_exact(FILE *f, void *p, size_t n) {
    if (fread(p, 1, n, f) != n) fail("truncated case");
}
static void write_exact(FILE *f, const void *p, size_t n) {
    if (fwrite(p, 1, n, f) != n) fail("short output write");
}
int main(int argc, char **argv) {
    if (argc != 4) fail("usage: gpu-matvec CASE OUTPUT ITERATIONS (0 = correctness only)");
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1 || sizeof(float) != 4 || sizeof(double) != 8 || sizeof(long double) < 10)
        fail("unsupported host");
    char *end; errno = 0;
    unsigned long iterations = strtoul(argv[3], &end, 10);
    if (errno || !*argv[3] || *end || argv[3][0] == '-' || iterations > 100000) fail("invalid iterations");
    FILE *f = fopen(argv[1], "rb");
    if (!f) fail("open case failed");
    uint32_t h[8]; read_exact(f, h, sizeof(h));
    if (h[0] != 0x38564d5a || h[1] != 1 || h[7] || !h[3] || h[3] > 32768 || !h[4] || h[4] > 1048576)
        fail("invalid case header");
    enum ggml_type type = (enum ggml_type)h[2];
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1 && type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K)
        fail("unsupported type");
    const size_t k = h[3], m = h[4];
    const struct ggml_type_traits *traits = ggml_get_type_traits(type);
    if (!traits || k % traits->blck_size || k/traits->blck_size*traits->type_size*m != h[5] || h[6] != k*4)
        fail("invalid case layout");
    size_t row_bytes = h[5]/m;
    unsigned char *packed = malloc(h[5]);
    float *x = malloc(k*4), *row = malloc(k*4), *actual = malloc(m*4);
    double *ideal = malloc(m*8), *sumabs = malloc(m*8);
    if (!packed || !x || !row || !actual || !ideal || !sumabs) fail("host allocation failed");
    read_exact(f, packed, h[5]); read_exact(f, x, k*4);
    if (fgetc(f) != EOF || ferror(f) || fclose(f)) fail("trailing case bytes/read error");
    for (size_t c = 0; c < k; ++c) if (!isfinite(x[c])) fail("nonfinite input");
    for (size_t r = 0; r < m; ++r) {
        if (type == GGML_TYPE_F32) memcpy(row, packed+r*row_bytes, k*4);
        else {
            if (!traits->to_float) fail("missing independent decoder");
            traits->to_float(packed+r*row_bytes, row, (int64_t)k);
        }
        long double dot = 0, absolute = 0;
        for (size_t c = 0; c < k; ++c) {
            if (!isfinite(row[c])) fail("nonfinite decoded weight");
            long double p = (long double)row[c] * (long double)x[c];
            dot += p; absolute += fabsl(p);
        }
        ideal[r] = (double)dot; sumabs[r] = (double)absolute;
    }
    char description[512];
    if (ggml_backend_vk_get_device_count() < 1) fail("no reference Vulkan device");
    ggml_backend_vk_get_device_description(0, description, sizeof(description));
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend || !ggml_backend_is_vk(backend)) fail("reference Vulkan initialization failed");
    fprintf(stderr, "oracle=%s commit=%s device=%s disable_mmvq=%s\n", ggml_version(), ggml_commit(), description,
            getenv("GGML_VK_DISABLE_MMVQ") ? getenv("GGML_VK_DISABLE_MMVQ") : "unset");
    struct ggml_init_params params = { .mem_size = 4*ggml_tensor_overhead()+ggml_graph_overhead_custom(8, false), .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) fail("reference context failed");
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, type, (int64_t)k, (int64_t)m);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)k, 1);
    struct ggml_tensor *y = ggml_mul_mat(ctx, a, b);
    ggml_set_name(a, "weights"); ggml_set_name(b, "input"); ggml_set_name(y, "output");
    if (!ggml_backend_supports_op(backend, y)) fail("reference does not support this matvec");
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t allocation = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!allocation || ggml_nbytes(a) != h[5] || ggml_nbytes(b) != k*4 || ggml_nbytes(y) != m*4) fail("GPU tensor allocation/layout failed");
    fprintf(stderr, "backend_allocation_bytes=%zu\n", ggml_backend_buffer_get_size(allocation));
    ggml_backend_tensor_set(a, packed, 0, h[5]); ggml_backend_tensor_set(b, x, 0, k*4);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) fail("reference compute failed");
    if (iterations) {
        for (int i = 0; i < 3; ++i)
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) fail("warmup failed");
        for (int trial = 0; trial < 7; ++trial) {
            uint64_t start = now();
            for (unsigned long i = 0; i < iterations; ++i)
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) fail("timed compute failed");
            uint64_t elapsed = now()-start;
            printf("{\"trial\":%d,\"iterations\":%lu,\"elapsed_ns\":%llu}\n", trial, iterations, (unsigned long long)elapsed);
        }
    }
    ggml_backend_tensor_get(y, actual, 0, m*4);
    f = fopen(argv[2], "wbx");
    if (!f) fail("output exists or cannot be created");
    write_exact(f, ideal, m*8); write_exact(f, sumabs, m*8); write_exact(f, actual, m*4);
    if (fclose(f)) fail("close output failed");
    ggml_backend_buffer_free(allocation); ggml_free(ctx); ggml_backend_free(backend);
    free(sumabs); free(ideal); free(actual); free(row); free(x); free(packed);
    return fflush(stdout) ? 3 : 0;
}
