/* External ggml benchmark only; never linked into native zerv. */
#define _POSIX_C_SOURCE 200809L
#include <ggml.h>
#include <openssl/sha.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void fail(const char *message) { fprintf(stderr, "%s\n", message); exit(2); }
static uint64_t now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock failed");
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}
static void digest(const void *data, size_t n, char hex[65]) {
    unsigned char bytes[32];
    if (!SHA256(data, n, bytes)) fail("hash failed");
    for (int i = 0; i < 32; ++i) snprintf(hex + i * 2, 3, "%02x", bytes[i]);
}
int main(int argc, char **argv) {
    if (argc != 3) fail("usage: model-quant-bench MODEL ABSOLUTE_TENSOR_OFFSET");
    char *end;
    errno = 0;
    unsigned long long offset = strtoull(argv[2], &end, 10);
    if (errno || !*argv[2] || *end || argv[2][0] == '-') fail("invalid offset");
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1 || sizeof(float) != 4) fail("unsupported host");
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size < 0) fail("cannot open model");
    const size_t tensor_size = 55705600;
    if (offset > (uint64_t)st.st_size || tensor_size > (uint64_t)st.st_size - offset || offset % 2) fail("invalid tensor extent/alignment");
    void *mapping = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) fail("mmap failed");
    const unsigned char *input = (const unsigned char *)mapping + offset;
    const struct ggml_type_traits *traits = ggml_get_type_traits(GGML_TYPE_Q4_1);
    if (!traits || traits->blck_size != 32 || traits->type_size != 20 || !traits->to_float) fail("unexpected type traits");
    ggml_to_float_t decode = traits->to_float;
    const size_t rows_list[] = {1, 64, 5120}, iterations_list[] = {10000, 256, 4};
    for (int workload = 0; workload < 3; ++workload) {
        size_t rows = rows_list[workload], iterations = iterations_list[workload];
        size_t values = 17408 * rows, encoded_n = values / 32 * 20;
        float *output = malloc(values * sizeof(*output));
        if (!output) fail("allocation failed");
        char input_hash[65], output_hash[65];
        digest(input, encoded_n, input_hash);
        for (int i = 0; i < 3; ++i) decode(input, output, (int64_t)values);
        for (int trial = 0; trial < 7; ++trial) {
            uint64_t start = now();
            for (size_t i = 0; i < iterations; ++i) {
                decode(input, output, (int64_t)values);
                __asm__ __volatile__("" : : "r"(output) : "memory");
            }
            uint64_t elapsed = now() - start;
            if (!elapsed) fail("zero elapsed time");
            digest(output, values * sizeof(*output), output_hash);
            printf("{\"format\":\"q4_1\",\"rows\":%zu,\"values\":%zu,\"trial\":%d,\"iterations\":%zu,\"elapsed_ns\":%llu,\"input_sha256\":\"%s\",\"output_sha256\":\"%s\"}\n",
                   rows, values, trial, iterations, (unsigned long long)elapsed, input_hash, output_hash);
        }
        free(output);
    }
    munmap(mapping, (size_t)st.st_size); close(fd);
    return fflush(stdout) ? 3 : 0;
}
