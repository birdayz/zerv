# MTP draft cost and a reduced draft vocabulary (block 17c, 2026-09-24)

Question: what does drafting cost per speculative cycle, and does a draft head limited
to the first N token ids (`--spec-draft-vocab N`) make speculative decoding faster
without changing outputs? [Spec](../specs/speculative.md) ("Draft vocabulary").

## Setup

- RX 7900 XTX, Mesa 26.2.3 RADV; Qwen3.8-27B Q4_0 (model sha256 prefix
  `ede16c7b36e578ca`); zerv built from `3c03b07` plus the uncommitted working tree
  (server sha256 prefix `40f08ad12cf4ec69`, `zig build server -Doptimize=ReleaseFast
  -Dcpu=native`).
- Draft cost: `zerv-mtp-check MODEL TOKENS OUTDIR [N]` (new timing section: wall-clock
  `draft(t, k)` with 1 pending row after a step and 3 after a 3-row commit, k = 1..4,
  median of 41 after 8 warm-up calls; the fixtures `third_party/mtp-check/tokens-{short-nothink,long-think}.json`).
  Raw data: [mtp-check/](data/2026-09-24-draft-vocab/mtp-check/).
- Serving: `bench/run_serving.py`, 3 drafts with the adaptive policy, 2 repeats,
  greedy, 512 tokens:
  - `--workload bench/workloads/decode-v1.json --output docs/bench/data/2026-09-24-draft-vocab/decode-v1`
  - `--workload bench/workloads/decode-multilingual-v1.json --output docs/bench/data/2026-09-24-draft-vocab/multilingual-v1`
    (new workload: five languages, long answers in the prompt's language)
  - engines `"zerv;zerv-spec3;zerv-spec3@spec-draft-vocab=131072;zerv-spec3@spec-draft-vocab=65536;zerv-spec3@spec-draft-vocab=32768"`.

## Draft cost

A draft pass reads the MTP layer (253 MiB) and the Q6_K output head (995 MiB):
1.31 GB, 1.42 ms at 920 GB/s. The head alone is 80% of the bytes and already runs at
916 GB/s ([decode baseline](2026-09-24-decode-baseline.md)).

| draft(t, k), ms (short-nothink, 1 pending row) | k = 1 | k = 2 | k = 3 | k = 4 |
| --- | ---: | ---: | ---: | ---: |
| full vocabulary (248,320) | 1.601 | 3.140 | 4.656 | 6.178 |
| N = 131,072 | 1.053 | 2.036 | 3.028 | 4.022 |
| N = 65,536 | 0.741 | 1.406 | 2.085 | 2.758 |
| N = 32,768 | 0.584 | 1.099 | 1.625 | 2.150 |

The long-think fixture and 3 pending rows are within 0.1 ms of these values (raw data).
With full drafts, 3 drafts cost 4.66 ms per cycle against 25.0 ms for the 4-row
verify + commit, so **drafting is 16% of a 3-draft cycle**.

## Correctness (gate 1)

- Full vocabulary with this build: all 18 dump files byte-identical to earlier runs on
  both fixtures (`timing-short`, `2026-09-24-fma-long-think`, `2026-09-24-final-long-think`).
- N = 131,072, 65,536 and 32,768 on both fixtures: every draft-logits dump equals the
  first N values of the full run bitwise (for each chain step whose inputs match the
  full run), and every draft is the first argmax of its dumped prefix. Scenario C
  (steps ≡ draft/verify/commit) 3/3 in every run. One divergence, as specified: at
  N = 32,768 the full head's draft 36093 is out of range and the reduced head drafts
  3299; the chain after it differs.

## Serving: decode-v1 (English, code, JSON, reasoning)

Decode tok/s, median of 2; accepted/verified drafts of repeat 0. All outputs are
byte-identical to plain decode (`zerv`, 50.7–50.8 tok/s).

| engine | code | json | think | prose |
| --- | --- | --- | --- | --- |
| 3 drafts, full (default) | 120.6 (369/423) | 124.9 (374/407) | 107.7 (351/463) | 75.6 (278/615) |
| N = 131,072 | 127.6 (369/423) | 132.3 (374/407) | 113.8 (350/461) | 80.2 (277/607) |
| N = 65,536 | **131.9** (369/425) | **134.1** (371/414) | 117.8 (350/461) | **82.6** (275/603) |
| N = 32,768 | 129.8 (364/431) | 125.9 (359/443) | **118.2** (347/450) | 81.4 (262/571) |

N = 65,536: +7 to +9% with acceptance nearly unchanged.

## Serving: multilingual (negative result)

| engine | chinese | japanese | russian | german | korean |
| --- | --- | --- | --- | --- | --- |
| plain decode | 50.6 | 50.6 | 50.6 | 50.6 | 50.6 |
| 3 drafts, full (default) | **74.3** (268/549) | **73.1** (264/553) | **75.2** (272/528) | **87.5** (310/515) | **69.8** (253/586) |
| N = 131,072 | 75.8 (258/537) | 68.0 (225/570) | 61.4 (187/525) | 85.3 (286/482) | 61.0 (191/531) |
| N = 65,536 | 49.6 (89/319) | 58.6 (160/397) | 60.4 (169/376) | 84.4 (273/401) | 60.9 (171/400) |
| N = 32,768 | 51.7 (88/338) | 59.6 (146/348) | 59.0 (145/364) | 80.6 (249/381) | 58.1 (142/392) |

All outputs byte-identical to plain decode. Share of generated tokens at ids ≥ N
(llama-tokenize on the plain-decode outputs):

| language | ≥ 32,768 | ≥ 65,536 | ≥ 131,072 | ≥ 151,643 |
| --- | ---: | ---: | ---: | ---: |
| chinese | 82.2% | 82.2% | 6.2% | 0.0% |
| japanese | 65.4% | 61.1% | 26.2% | 23.6% |
| russian | 64.8% | 56.1% | 52.0% | 37.1% |
| german | 29.2% | 17.4% | 12.9% | 12.9% |
| korean | 64.3% | 51.0% | 44.7% | 34.2% |

For English, code and JSON the same shares are 0.2–3.7% at ≥ 65,536 (decode-v1
outputs; the 75.5k-token long-v2 prompt: 2.2%).

## Interpretation

- Output is unaffected in every run, as designed: verification decides.
- Id order is a good frequency proxy only for English-like text. The Chinese pieces sit
  at ids 65,536–131,071 and much of the Cyrillic, Korean and Japanese vocabulary lies
  above 131,072. A prefix head cannot draft those tokens, and acceptance collapses:
  Chinese at N = 65,536 falls below plain decode.
- Decision: **the knob stays opt-in** (`--spec-draft-vocab full` by default, the
  previous behavior). N = 65,536 is the measured best for English and code deployments
  (+7–9% on decode-v1); for other languages it is slower.
- Follow-up: a frequency-ranked subset (row indirection in the head matvec plus an
  index → id map in the argmax) could keep the gain across languages. It needs a
  representative token-frequency source; the ranking must not be fit to the benchmark
  outputs.
