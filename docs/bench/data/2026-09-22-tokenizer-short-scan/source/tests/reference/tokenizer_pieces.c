/* External oracle only. Never part of a native zerv build or runtime. */
#include <llama.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int write_u32(FILE *out, uint32_t value) {
    unsigned char b[4] = {value, value >> 8, value >> 16, value >> 24};
    return fwrite(b, 1, 4, out) == 4;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: pieces MODEL OUTPUT\n"); return 2; }
    FILE *out = fopen(argv[2], "wbx");
    if (!out) { perror("output"); return 2; }
    llama_backend_init();
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], params);
    if (!model) { fclose(out); llama_backend_free(); return 3; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t count = llama_vocab_n_tokens(vocab);
    int ok = count > 0 && write_u32(out, (uint32_t)count);
    char *buffer = malloc(65536);
    if (!buffer) ok = 0;
    for (int32_t id = 0; ok && id < count; ++id) {
        const int32_t n = llama_token_to_piece(vocab, id, buffer, 65536, 0, true);
        if (n < 0 || n > 65536) { fprintf(stderr, "invalid piece size %d for %d\n", n, id); ok = 0; break; }
        ok = write_u32(out, (uint32_t)n) && fwrite(buffer, 1, (size_t)n, out) == (size_t)n;
    }
    free(buffer);
    llama_model_free(model);
    llama_backend_free();
    if (fclose(out) != 0) ok = 0;
    return ok ? 0 : 4;
}
