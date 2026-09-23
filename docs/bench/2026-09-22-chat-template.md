# 2026-09-22 — official text chat-template component

Native allocation-free rendering passes **100 independent Jinja fixture cases**,
including 5 rejection cases and all supported options/history/Unicode stripping.
Both Debug and ReleaseFast passed. Three cases differ between official/publisher
sources (missing user, tool-response-only user, developer prefix); native explicitly
implements the official subset. This is not BPE, generation, or HTTP serving.

## Repeatable command and setup

```sh
.tools/tokenizer-oracle-venv/bin/python bench/run_chat.py \
  --config third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json \
  --output docs/bench/data/2026-09-22-chat-template --cpu 2
```

Use a new output directory. Isolated oracle environment: Python 3.14.7, Jinja2
3.1.6 (full versions in manifest); Zig 0.16.0 ReleaseFast; Ryzen 9 3900X, CPU 2.
No system packages/clocks changed. Reference template compilation and fixture JSON
parsing excluded. Both validate all 95 successful rendered outputs byte-for-byte
and identical concatenated output SHA before timing. Three alternating rounds,
3 warmups, 7 trials × 100 corpus traversals per worker; 9,500 renders per trial.
Native fixed output buffer is reused; Jinja produces Python strings. Both perform
UTF-8/content handling applicable to their input representations (native bytes vs
Python's already-decoded strings).

[Manifest](data/2026-09-22-chat-template/manifest.json) records compiler, binary,
source, config, corpus/output hashes and full package versions; all raw JSONL and
command logs are beside it. [Contract](../specs/chat-template.md).

| Engine | Median ns/render | Min–max | Sample stddev | Trials |
| --- | ---: | ---: | ---: | ---: |
| Native specialized text renderer | 467.96 | 378.41–494.05 | 41.38 | 21 |
| Precompiled Jinja template | 130,828.69 | 126,082.36–146,205.01 | 5,765.13 | 21 |

Native/Jinja median ratio 0.00358. This large gap compares a narrow compiled native
text-only renderer against a general sandboxed template interpreter; it is **not**
a claim to beat tuned native tokenizers or llama-server. Tools/multimodal/developer
extensions are rejected rather than implemented. Per-request formatting is usually
small beside model prefill/decode. Full serving evidence remains required.

## Independent repeat / retained source

Same command, new output `docs/bench/data/2026-09-22-chat-template-repeat`.
The harness now preserves a complete build/test/fixture/benchmark source snapshot
under `source/` alongside hashes; first-run results are retained unmodified.
[Repeat manifest](data/2026-09-22-chat-template-repeat/manifest.json).

Repeat medians: native **375.38 ns/render** (367.83–466.92, stddev 35.75 ns), Jinja
**124,825.39 ns/render** (122,438.37–135,328.82, stddev 2,679.40 ns), 21 trials each.
Ratio **0.00301**, with the same exact-output gates. Variation in these short native
renders reinforces why the result is a component observation, not serving speed.
