// External reference capture of llama.cpp's *batched* prompt path (block 14 quality
// baseline). Test/oracle tool only: links the pinned installed libllama/ggml; never part
// of zerv. Feeds a fixed token sequence (teacher-forced, from tokens.json) with the given
// batch settings, requests logits for every position, and records the named F32 tensors
// (e.g. l_out-63) for every token. The precision path is whatever ggml selects under the
// caller's environment (default = llama-server's default).
// Usage: llama_batch_capture MODEL TOKENS.json NAMES.txt OUT_DIR N_CTX N_BATCH N_UBATCH FLASH(0|1) KV(f16|f32)
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#define MAX_NAMES 64
#define MAX_TOKENS 16384

struct capture {
    char names[MAX_NAMES][64];
    int seen[MAX_NAMES];   // tokens recorded so far per name (ubatches arrive in order)
    int n_names;
    FILE * blob;
    FILE * index;
    uint64_t offset;
    int failed;
    unsigned char * scratch;
    size_t scratch_size;
    float * gathered;
    size_t gathered_size;
};

static void die(const char * message) {
    fprintf(stderr, "llama_batch_capture: %s\n", message);
    exit(2);
}

static int slot(const struct capture * c, const char * name) {
    for (int i = 0; i < c->n_names; ++i) if (strcmp(c->names[i], name) == 0) return i;
    return -1;
}

static bool callback(struct ggml_tensor * t, bool ask, void * user) {
    struct capture * c = user;
    const int s = slot(c, t->name);
    if (ask) return s >= 0;
    if (s < 0) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1) { c->failed = 1; return false; }
    const size_t span = ggml_nbytes(t);
    if (span > c->scratch_size) {
        c->scratch = realloc(c->scratch, span);
        if (!c->scratch) die("out of memory");
        c->scratch_size = span;
    }
    ggml_backend_tensor_get(t, c->scratch, 0, span);
    const int64_t n = ggml_nelements(t);
    if ((size_t) n > c->gathered_size) {
        c->gathered = realloc(c->gathered, (size_t) n * sizeof(float));
        if (!c->gathered) die("out of memory");
        c->gathered_size = (size_t) n;
    }
    int64_t k = 0;
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            const size_t at = (size_t) (i0 * t->nb[0] + i1 * t->nb[1]);
            if (at + sizeof(float) > span) { c->failed = 1; return false; }
            memcpy(&c->gathered[k++], c->scratch + at, sizeof(float));
        }
    if (fwrite(c->gathered, sizeof(float), (size_t) n, c->blob) != (size_t) n) die("write failed");
    fprintf(c->index, "{\"name\":\"%s\",\"first_token\":%d,\"tokens\":%lld,\"width\":%lld,\"offset\":%llu}\n",
            t->name, c->seen[s], (long long) t->ne[1], (long long) t->ne[0], (unsigned long long) c->offset);
    c->seen[s] += (int) t->ne[1];
    c->offset += (uint64_t) n * sizeof(float);
    return true;
}

static char * read_file(const char * path, size_t * size) {
    FILE * f = fopen(path, "rb");
    if (!f) die("cannot open input");
    if (fseek(f, 0, SEEK_END) != 0) die("seek");
    long length = ftell(f);
    if (length < 0 || length > (1 << 24)) die("input too large");
    rewind(f);
    char * data = malloc((size_t) length + 1);
    if (!data || fread(data, 1, (size_t) length, f) != (size_t) length) die("read failed");
    data[length] = 0;
    fclose(f);
    *size = (size_t) length;
    return data;
}

// Minimal parser for {"tokens":[...]} (the oracle fixture's tokens.json format).
static int parse_tokens(const char * json, llama_token * out) {
    const char * p = strstr(json, "\"tokens\"");
    if (!p || !(p = strchr(p, '['))) die("tokens.json lacks tokens");
    int n = 0;
    ++p;
    while (*p && *p != ']') {
        char * end;
        long v = strtol(p, &end, 10);
        if (end == p) { ++p; continue; }
        if (n == MAX_TOKENS || v < 0) die("too many / invalid tokens");
        out[n++] = (llama_token) v;
        p = end;
    }
    return n;
}

int main(int argc, char ** argv) {
    if (argc != 10) die("usage: MODEL TOKENS.json NAMES.txt OUT_DIR N_CTX N_BATCH N_UBATCH FLASH KV");
    const int n_ctx = atoi(argv[5]), n_batch = atoi(argv[6]), n_ubatch = atoi(argv[7]), flash = atoi(argv[8]);
    const bool kv_f16 = strcmp(argv[9], "f16") == 0;
    if (!kv_f16 && strcmp(argv[9], "f32") != 0) die("KV must be f16 or f32");
    if (n_ctx < 16 || n_ctx > MAX_TOKENS || n_batch < 1 || n_ubatch < 1 || n_ubatch > n_batch) die("bad counts");
    static struct capture c;
    size_t tokens_size, names_size;
    char * tokens_json = read_file(argv[2], &tokens_size);
    char * names = read_file(argv[3], &names_size);
    for (char * line = strtok(names, "\n"); line; line = strtok(NULL, "\n")) {
        if (!*line) continue;
        if (c.n_names == MAX_NAMES || strlen(line) >= 64) die("too many/long names");
        snprintf(c.names[c.n_names++], 64, "%s", line);
    }
    static llama_token tokens[MAX_TOKENS];
    const int n = parse_tokens(tokens_json, tokens);
    if (n <= 0 || n > n_ctx) die("token count exceeds context");
    char path[4096];
    snprintf(path, sizeof path, "%s/tensors.bin", argv[4]);
    c.blob = fopen(path, "wbx");
    snprintf(path, sizeof path, "%s/index.jsonl", argv[4]);
    c.index = fopen(path, "wx");
    snprintf(path, sizeof path, "%s/logits.bin", argv[4]);
    FILE * logits_file = fopen(path, "wbx");
    if (!c.blob || !c.index || !logits_file) die("outputs must be fresh");

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) die("model load failed");
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) n_ubatch;
    cp.n_seq_max = 1;
    cp.type_k = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cp.type_v = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (c.n_names > 0) {
        cp.cb_eval = callback;
        cp.cb_eval_user_data = &c;
    }
    cp.no_perf = true;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    const int n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int first = 0; first < n; first += n_batch) {
        const int count = n - first < n_batch ? n - first : n_batch;
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[first + i];
            batch.pos[i] = first + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0 || c.failed) die("decode failed");
        for (int i = 0; i < count; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (!logits || fwrite(logits, sizeof(float), (size_t) n_vocab, logits_file) != (size_t) n_vocab) die("logits");
        }
    }
    llama_batch_free(batch);
    if (fclose(logits_file) || fclose(c.blob) || fclose(c.index)) die("close failed");
    printf("{\"tokens\":%d,\"n_vocab\":%d,\"n_batch\":%d,\"n_ubatch\":%d,\"flash\":%d,\"kv\":\"%s\"}\n", n, n_vocab, n_batch, n_ubatch, flash, argv[9]);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(tokens_json);
    free(names);
    free(c.scratch);
    free(c.gathered);
    return 0;
}
