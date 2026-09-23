/* External benchmark/oracle ONLY; never in native zerv builds or runtime. */
#define _POSIX_C_SOURCE 200809L
#include <llama.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) { fprintf(stderr, "%s\n", message); exit(2); }
static void *allocate(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) fail("allocation failed");
    return p;
}
static void read_exact(FILE *f, void *p, size_t n) {
    if (fread(p, 1, n, f) != n) fail("truncated corpus");
}
static uint32_t read_u32(FILE *f) {
    unsigned char b[4]; read_exact(f, b, 4);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
static char *read_bytes(FILE *f, uint32_t *n) {
    *n = read_u32(f);
    if (*n > 1048576) fail("field too large");
    char *p = allocate((size_t)*n + 1);
    read_exact(f, p, *n); p[*n] = 0;
    return p;
}
struct record { char *name, *text, *raw; uint32_t text_n, raw_n, ids_n; llama_token *ids; };
static struct record read_record(FILE *f) {
    struct record r; uint32_t n;
    r.name = read_bytes(f, &n);
    if (!n) fail("empty name");
    for (uint32_t i = 0; i < n; ++i) {
        char c = r.name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-' || c == '_')) fail("invalid name");
    }
    r.text = read_bytes(f, &r.text_n);
    r.ids_n = read_u32(f);
    if (r.ids_n > 1048576) fail("too many IDs");
    r.ids = allocate((size_t)r.ids_n * sizeof(*r.ids));
    for (uint32_t i = 0; i < r.ids_n; ++i) {
        uint32_t id = read_u32(f);
        if (id >= 248320) fail("invalid token ID");
        r.ids[i] = (llama_token)id;
    }
    r.raw = read_bytes(f, &r.raw_n);
    return r;
}
static void release(struct record *r) { free(r->name); free(r->text); free(r->raw); free(r->ids); }
static uint64_t now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock failed");
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}
static llama_token *encode(const struct llama_vocab *v, const struct record *r, int32_t *n) {
    /* Each UTF-8 byte can fall back to one ID. No capacity-probe tokenization. */
    int32_t capacity = r->text_n ? (int32_t)r->text_n : 1;
    llama_token *ids = allocate((size_t)capacity * sizeof(*ids));
    *n = llama_tokenize(v, r->text, (int32_t)r->text_n, ids, capacity, false, true);
    if (*n < 0 || *n > capacity) fail("llama tokenize capacity failure");
    return ids;
}
static void check_ids(const struct record *r, const llama_token *ids, int32_t n) {
    if (n != (int32_t)r->ids_n || memcmp(ids, r->ids, (size_t)n * sizeof(*ids))) {
        fprintf(stderr, "encode mismatch: %s\n", r->name); exit(3);
    }
}
static int32_t decode(const struct llama_vocab *v, const struct record *r, char *out) {
    int32_t n = llama_detokenize(v, r->ids, (int32_t)r->ids_n, out, (int32_t)r->raw_n, false, true);
    if (n < 0 || n > (int32_t)r->raw_n) fail("llama detokenize capacity failure");
    return n;
}
static void check_raw(const struct record *r, const char *out, int32_t n) {
    if (n != (int32_t)r->raw_n || memcmp(out, r->raw, (size_t)n)) {
        fprintf(stderr, "decode mismatch: %s\n", r->name); exit(3);
    }
}
static void hex(const void *data, size_t n) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; ++i) printf("%02x", p[i]);
}
static void validate(const struct llama_vocab *v, const struct record *r, int do_encode) {
    int32_t n;
    if (do_encode) { llama_token *ids = encode(v, r, &n); check_ids(r, ids, n); free(ids); }
    char *out = allocate(r->raw_n);
    n = decode(v, r, out); check_raw(r, out, n); free(out);
}
static void benchmark(const struct llama_vocab *v, const struct record *r) {
    char *out = allocate(r->raw_n);
    for (int op = 0; op < 2; ++op) {
        for (int trial = -3; trial < 7; ++trial) {
            int iterations = trial < 0 ? 1 : op == 0 ? 100 : 1000;
            int32_t n = 0;
            uint64_t start = now();
            for (int i = 0; i < iterations; ++i) {
                if (op == 0) {
                    llama_token *ids = encode(v, r, &n);
                    __asm__ __volatile__("" : : "r"(ids), "r"(n) : "memory");
                    free(ids);
                } else {
                    n = decode(v, r, out);
                    __asm__ __volatile__("" : : "r"(out), "r"(n) : "memory");
                }
            }
            uint64_t elapsed = now() - start;
            if (!elapsed) fail("zero elapsed time");
            if (op == 1) check_raw(r, out, n);
            llama_token *ids = encode(v, r, &n);
            check_ids(r, ids, n);
            validate(v, r, 0);
            if (trial >= 0) {
                printf("{\"workload\":\"%s\",\"operation\":\"%s\",\"trial\":%d,\"iterations\":%d,\"elapsed_ns\":%llu,\"output_hex\":\"",
                       r->name, op == 0 ? "encode" : "decode", trial, iterations, (unsigned long long)elapsed);
                if (op == 0) {
                    for (int32_t i = 0; i < n; ++i) {
                        uint32_t id = (uint32_t)ids[i];
                        unsigned char b[4] = {id, id >> 8, id >> 16, id >> 24}; hex(b, 4);
                    }
                } else hex(out, r->raw_n);
                puts("\"}");
            }
            free(ids);
        }
    }
    free(out);
}
int main(int argc, char **argv) {
    if (argc != 3 && (argc != 4 || strcmp(argv[3], "--bench"))) fail("usage: tokenizer-bench MODEL CORPUS [--bench]");
    FILE *f = fopen(argv[2], "rb");
    if (!f) fail("cannot open corpus");
    char magic[4]; read_exact(f, magic, 4);
    if (memcmp(magic, "ZTBC", 4) || read_u32(f) != 1) fail("invalid corpus header");
    uint32_t counts[3];
    for (int i = 0; i < 3; ++i) { counts[i] = read_u32(f); if (counts[i] > 100000) fail("too many records"); }
    llama_backend_init();
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = true; params.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], params);
    if (!model) fail("model load failed");
    const struct llama_vocab *v = llama_model_get_vocab(model);
    if (llama_vocab_n_tokens(v) != 248320) fail("wrong vocabulary");
    for (int group = 0; group < 3; ++group) {
        for (uint32_t i = 0; i < counts[group]; ++i) {
            struct record r = read_record(f);
            validate(v, &r, group != 1);
            if (group == 2 && argc == 4) benchmark(v, &r);
            release(&r);
        }
    }
    if (fgetc(f) != EOF || ferror(f)) fail("trailing data/read error");
    if (argc == 3) printf("{\"cases\":%u,\"decode_cases\":%u,\"workloads\":%u}\n", counts[0], counts[1], counts[2]);
    fclose(f); llama_model_free(model); llama_backend_free();
    return fflush(stdout) ? 4 : 0;
}
