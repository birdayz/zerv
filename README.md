# zerv

LLM inference server in Zig, written from scratch: GGUF loader, tokenizer, chat template,
Vulkan layer, GPU kernels (GLSL, plus hand-written RDNA3 machine code for the prefill GEMM),
KV and recurrent-state management, sampler, scheduler and HTTP. No C++ dependencies. It serves
OpenAI-compatible `POST /v1/chat/completions`, streaming or not.

Target: Qwen3.8-27B Q4_0 on one RX 7900 XTX (24 GB). The goal is the fastest correct
server for that pair. Nothing else is supported yet.

## Build and run

```sh
bazel build --config=release //src:zerv          # Bazel, see docs/development.md
bazel-bin/src/zerv --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf --prefill-precision f16
bazel-bin/src/zerv --model ... --prefill-precision f16 --parallel 8 --kv-type f16   # 8 users at once
```

`zerv` without arguments prints every option. Speed knobs that change kernels or scheduling
keep the previous behaviour selectable.

## Correctness

- Checked against an FP64 reference and llama.cpp captures: intermediate tensors, logits and
  greedy tokens ([verification](docs/specs/verification.md)).
- Speculative decoding (MTP) is lossless: the output is identical to plain decoding, greedy
  or sampled.
- With `--parallel N`, every response is byte-identical to serving it alone, whatever the
  load. llama-server `-np 8` gave 3–5 different greedy outputs per prompt depending on load.

## Numbers

Hardware: RX 7900 XTX with Mesa 26.2.3 (RADV). The competitor is llama-server build 10964
(Vulkan, `-fa on -b 2048 -ub 512`) on the same GGUF, same machine and same session. Every
number links to a report with commands, hashes and raw data.

### One user

| | zerv | llama-server |
| --- | --- | --- |
| TTFT, 23 / 836 / 3,223 prompt tokens | 75 ms / 0.71 s / 2.49 s | 166 ms / 1.21 s / 3.40 s |
| TTFT, 12,034 prompt tokens | 10.1 s | 12.3 s |
| decode, no speculation | 48 tok/s | 41 tok/s |
| decode with MTP, best config of each (code / json / think / prose) | 110 / 114 / 99 / 74 tok/s | 106 / 109 / 91 / 63 tok/s |

TTFT: [report](docs/bench/2026-09-24-gemm-f16x-isa.md) (zerv with `--prefill-precision f16`).
Decode: [report](docs/bench/2026-09-24-speculative.md).

### Several users, 8 slots

The competitors are vLLM 0.30.0 (official ROCm image, RedHatAI W4A16 checkpoint, FP8 KV cache)
and llama-server (same GGUF as zerv). All run cold (no prompt cache), interleaved in 2 rounds.
Competitor cells show each one's best configuration per metric (vLLM prefill chunk 2048 or 512,
llama `-b 2048` or `-b 512`), and vLLM's faster round at 1 user. Medians; full tables: [18c.2](docs/bench/2026-09-25-multiuser.md), [packed prefill](docs/bench/2026-09-26-packed-prefill.md).

| Closed loop, short prompts | zerv | vLLM | llama-server |
| --- | --- | --- | --- |
| aggregate tok/s, 1 / 2 / 4 / 8 users | **47 / 85 / 138 / 151** | 36 / 64 / 108 / 152–158 | 38 / 64 / 93 / 131 |
| TTFT p50 / p95, 8 users | **0.59 / 1.05 s** | 0.80 / 1.17 s | 3.3 / 3.6 s |
| token gap p50 / p99 / max, 8 users | 48 / 149 / **193** ms | 48 / **53** / 466 ms | 47 / 55 / 798 ms |

| A 4,936-token prompt arrives while 6 users stream | zerv | vLLM | llama-server |
| --- | --- | --- | --- |
| streaming users' tok/s during its prefill (all 6) | **40** | 13 | 11 |
| their worst token gap | **169 ms** | 597 ms | 692 ms |
| TTFT of a 280-token prompt sent 100 ms later | **1.4 s** | 5.4 s | 6.6 s |
| TTFT of the long prompt | 5.8 s | **5.3 s** | 6.7 s |

- The interference rows use each competitor's best setting for streaming users (vLLM chunk
  512, llama `-b 512`). With chunk 2048, vLLM's long TTFT is 4.9 s, at 5.7 tok/s and
  1.9 s gaps for the others.
- With `--parallel N`, zerv's responses are byte-identical to serving them alone. The
  other two are not batch-invariant.

### Where zerv is behind

- **Steady gap p99 at 8 users.** zerv spreads prefill into many ~140 ms stalls (3% of
  gaps); vLLM makes rare 0.5–0.9 s ones. 8-user throughput is a tie with vLLM.
- **Batched decode past 4 rows.** 8 rows cost 2.3× one row.
- **No prefix cache with `--parallel N` > 1.**
- vLLM's and SGLang's speculative decoding (MTP) have not been benchmarked yet.

## More

- [Work queue](TODO.md), one block at a time
- [Docs index](docs/README.md): research, specs, benchmark reports
- [Development and test commands](docs/development.md)
- [Agent instructions](AGENTS.md)
