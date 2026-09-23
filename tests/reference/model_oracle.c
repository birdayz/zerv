// External reference capture for Qwen3.8 execution. Test/oracle tool only: links the
// pinned installed libllama/ggml; never part of zerv. One token per llama_decode, so
// every projection is a matvec; the driver sets GGML_VK_DISABLE_MMVQ=1 (FP32 inputs).
// Usage: model_oracle MODEL PROMPT.txt NAMES.txt OUT_DIR N_GENERATE N_CTX
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#define MAX_NAMES 128
#define MAX_TOKENS 8192

struct capture {
    char names[MAX_NAMES][64];
    int n_names;
    int token;          // index of the token currently being decoded
    FILE * blob;
    FILE * index;
    uint64_t offset;
    char last_name[GGML_MAX_NAME];
    int occurrence;     // duplicate names inside one decode (e.g. Kcur before/after RoPE)
    int failed;
    unsigned char * scratch;
    size_t scratch_size;
    float * gathered;
    size_t gathered_size;
};

static void die(const char * message) {
    fprintf(stderr, "model_oracle: %s\n", message);
    exit(2);
}

static bool wanted(const struct capture * c, const char * name) {
    char base[GGML_MAX_NAME];
    snprintf(base, sizeof base, "%s", name);
    char * dash = strrchr(base, '-');
    if (dash && dash[1] >= '0' && dash[1] <= '9') *dash = 0;
    for (int i = 0; i < c->n_names; ++i) if (strcmp(c->names[i], base) == 0) return true;
    return false;
}

static bool callback(struct ggml_tensor * t, bool ask, void * user) {
    struct capture * c = user;
    if (ask) return wanted(c, t->name);
    if (!wanted(c, t->name)) return true;
    if (t->type != GGML_TYPE_F32) {
        fprintf(c->index, "{\"token\":%d,\"name\":\"%s\",\"skipped_type\":%d}\n", c->token, t->name, (int) t->type);
        return true;
    }
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
    // Gather a possibly strided view into contiguous ggml order (ne0 fastest).
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t at = (size_t) (i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                    if (at + sizeof(float) > span) { c->failed = 1; return false; }
                    memcpy(&c->gathered[k++], c->scratch + at, sizeof(float));
                }
    if (fwrite(c->gathered, sizeof(float), (size_t) n, c->blob) != (size_t) n) die("write failed");
    c->occurrence = strcmp(c->last_name, t->name) == 0 ? c->occurrence + 1 : 0;
    snprintf(c->last_name, sizeof c->last_name, "%s", t->name);
    fprintf(c->index, "{\"token\":%d,\"name\":\"%s\",\"op\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"offset\":%llu,\"count\":%lld}\n",
            c->token, t->name, ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (unsigned long long) c->offset, (long long) n);
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

int main(int argc, char ** argv) {
    if (argc != 7) die("usage: MODEL PROMPT NAMES OUT_DIR N_GENERATE N_CTX");
    const int n_generate = atoi(argv[5]);
    const int n_ctx = atoi(argv[6]);
    if (n_generate < 0 || n_ctx < 16 || n_ctx > MAX_TOKENS) die("bad counts");
    static struct capture c;
    size_t prompt_size, names_size;
    char * prompt = read_file(argv[2], &prompt_size);
    char * names = read_file(argv[3], &names_size);
    for (char * line = strtok(names, "\n"); line; line = strtok(NULL, "\n")) {
        if (!*line) continue;
        if (c.n_names == MAX_NAMES || strlen(line) >= 64) die("too many/long names");
        snprintf(c.names[c.n_names++], 64, "%s", line);
    }
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
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_seq_max = 1;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = callback;
    cp.cb_eval_user_data = &c;
    cp.no_perf = true;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");

    static llama_token tokens[MAX_TOKENS];
    const int n_prompt = llama_tokenize(vocab, prompt, (int32_t) prompt_size, tokens, MAX_TOKENS, false, true);
    if (n_prompt <= 0 || n_prompt + n_generate > n_ctx) die("tokenization failed or context too small");
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int total = n_prompt;
    for (int t = 0; t < total; ++t) {
        c.token = t;
        c.last_name[0] = 0;
        struct llama_batch batch = llama_batch_get_one(&tokens[t], 1);
        if (llama_decode(ctx, batch) != 0 || c.failed) die("decode failed");
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits || fwrite(logits, sizeof(float), (size_t) n_vocab, logits_file) != (size_t) n_vocab) die("logits");
        if (t == total - 1 && total < n_prompt + n_generate) {
            int best = 0;
            for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
            tokens[total++] = best;  // greedy, lowest index on ties
        }
    }
    snprintf(path, sizeof path, "%s/tokens.json", argv[4]);
    FILE * out = fopen(path, "wx");
    if (!out) die("tokens output");
    fprintf(out, "{\"n_prompt\":%d,\"n_vocab\":%d,\"tokens\":[", n_prompt, n_vocab);
    for (int i = 0; i < total; ++i) fprintf(out, "%s%d", i ? "," : "", tokens[i]);
    fprintf(out, "]}\n");
    if (fclose(out) || fclose(logits_file) || fclose(c.blob) || fclose(c.index)) die("close failed");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(prompt);
    free(names);
    free(c.scratch);
    free(c.gathered);
    return 0;
}
