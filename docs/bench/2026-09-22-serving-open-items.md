# Serving open items: overload/disconnect tests, graceful shutdown, host memory, output parity (block 12b)

Date: 2026-09-22 (runs logged 2026-09-23 UTC). Question: close the block-12b serving
gaps listed in TODO:

- real-socket tests for overload and disconnect;
- a graceful drain on SIGINT/SIGTERM;
- keeping the 15.5 GB file mapping out of RSS while serving;
- `reasoning_content` formatting parity with llama-server.

Also confirm that none of this changes serving speed.

Setup:

- RX 7900 XTX (RADV), model `Qwen3.8-27B-Q4_0.gguf` (sha256 `ede16c7b…`).
- llama-server build 10964 (commit b29c606e28).
- zerv built by the gates below: ReleaseFast, `-Dcpu=native`, sha256
  `854f7b8f0db24642e60f41eb53673d5b807f7518d59baa528ce3c65e6a7516ad`. The session
  gate, shutdown checks, parity check and serving runs all used this same binary.
- GPU jobs ran one at a time.

## Changes

- **Host memory** (`src/main.zig`): after the weights are resident and the tokenizer
  and sampling defaults are copied out, the GGUF container is freed and the file is
  unmapped.
- **Graceful shutdown** ([serving spec](specs/serving.md#scheduling-and-lifecycle)).
  - `Server.run(listener, stop)` polls a flag that the SIGINT/SIGTERM handler sets.
    A second signal exits with 130.
  - On stop, `run` stops accepting and closes the listener. Chat requests on
    existing connections then get 503 `shutting_down`, `/ready` returns 503, and
    every response carries `connection: close`.
  - Admitted requests finish within `--drain-timeout` (30 s by default). The
    remaining connection tasks are then canceled. Generation observes cancellation
    between tokens (`io.checkCancel()`, with the device idle).
  - Transient accept errors (fd limits, memory, aborted connections) are logged and
    retried instead of ending the server.
  - A canceled generation is counted in `failed`, and is not logged as an error.
- **Output text parity** ([research](research/output-parsing.md),
  [session spec](specs/session.md#termination-and-text)). The llama.cpp source
  (Qwen3-Coder PEG parser, stop handling, token rendering) shows four differences
  from zerv. All four are fixed:
  1. **Trailing whitespace.** Reasoning keeps its trailing whitespace. zerv trimmed
     it.
  2. **Stop strings** now match the raw text, which includes `</think>`. zerv
     matched them per channel.
  3. **Splitting happens in the text.** `</think>` (consumed) and `<tool_call>`
     (starts content) are found in the text, not by token id. At the end of the
     stream, a held partial delimiter and an incomplete UTF-8 tail are dropped.
  4. **Control tokens** render as nothing (`Tokenizer.outputPiece`, new
     `Kind.control`).

  Also, `reasoning_content` is omitted when empty, as llama-server does.

## Verification

| Check | Result |
| --- | --- |
| `zig fmt --check`, `zig build test` (Debug, ReleaseFast): 68 tests | pass. New tests: overload 503, disconnect cancel, drain, drain deadline, splitter rules and every-chunking property, raw-text stops, spelled delimiters, control tokens, dropped tails. |
| Debug `zig build test` repeated after the last socket-test change (socket tests are timing-sensitive) | 13/13 pass. One earlier failure (ECONNRESET when a probe raced the listener close) was fixed in the test; the change is below. |
| `zig build gpu-test` (Debug, ReleaseFast): 14 | pass |
| Python unittest: 53 (new `test_output_reference.py`) | pass |
| [Session gate](bench/data/2026-09-22-serving-open-items/session-gate.json) (`tools/check_session.py`): libllama greedy oracle, with the expected split recomputed by an independent Python version of the parser rules | 4/4 pass |
| Shutdown on the real binary: [run 1](bench/data/2026-09-22-serving-open-items/shutdown.json), [run 2](bench/data/2026-09-22-serving-open-items/shutdown-run2.json) | pass, both runs (details below) |
| [Output parity vs llama-server](bench/data/2026-09-22-serving-open-items/parity-run1/report.json) (`tools/check_parity.py`, 6 greedy cases × JSON/SSE, workload `output-parity-v1.json`) | 12/12 equal |
| [Negative control](bench/data/2026-09-22-serving-open-items/parity-control-pre-change-oldlabels/report.json): same check, pre-change zerv binary from the 13f serving run | 2 cases fail (below), 4 equal |

The race fix, and its counting: the new "refused" probe in the drain test once got
ECONNRESET, because a connect raced the listener close. It now retries on reset.
The failing run came before the fix and is not counted in the 13/13.

The shutdown scenarios, identical in both runs:

- **SIGINT during a 96-token stream.** A new connection 0.2 s after the signal is
  refused. The stream completes (`finish_reason: length`, `[DONE]`). The process
  exits 0 about 2 s later and logs `zerv: stopped`.
- **SIGTERM with an idle keep-alive connection.** Exits 0 in 0.03–0.16 s.
- **A second SIGINT during a 2000-token generation.** Exits 130 in 0.06–0.11 s.

The parity cases cover:

- thinking on, with reasoning complete;
- `max_tokens` cutting inside reasoning;
- stop `</think>`;
- stop `\n\n` inside reasoning;
- thinking off, with stop `\n\n` and with a length cut.

The reference is `llama-fp32-full`. Case `think-low-complete` has reasoning ending
`391\n`, which zerv now reproduces.

The negative control shows that the check catches the old behavior:

- `think-low-complete`: reasoning lacked the trailing `\n`.
- `think-stop-think-end`: the old per-channel stop never saw `</think>`, so zerv
  kept generating content (63 vs 58 tokens).

  That run's report labels this `token-divergence`. The label was renamed to
  `COUNT-MISMATCH` afterwards, because a token-count difference can also be a
  termination-semantics difference. The comparison logic is unchanged.

## Serving (regression check and host memory)

`bench/run_serving.py --engines zerv,llama-fa-ub512`, workload v2, 3 repeats per run.
Raw data: [run1](bench/data/2026-09-22-serving-open-items-serving-run1/),
[repeat](bench/data/2026-09-22-serving-open-items-serving-repeat/) (pinned run1
binary). Each cell is the median TTFT in ms, then the median decode rate in tok/s:

| Prompt | zerv run1 | zerv repeat | llama run1 | llama repeat |
| --- | --- | --- | --- | --- |
| 23 tok | 100.8 / 54.1 | 100.8 / 54.1 | 161.1 / 44.1 | 161.7 / 44.0 |
| 81 tok (think) | 241.7 / 49.4 | 241.5 / 49.4 | 368.1 / 41.5 | 369.3 / 41.5 |
| 836 tok | 2016 / 49.2 | 2009 / 49.3 | 1200 / 41.4 | 1204 / 41.3 |
| 3223 tok | 7755 / — | 7729 / — | 3391 / — | 3402 / — |

This is unchanged from the 13f runs (101 / 242 / 2020 / 7760 ms; 54 / 49.4 / 49.2 tok/s).

Host memory while serving:

| | VmRSS | VmHWM |
| --- | --- | --- |
| zerv | 60–62 MB (was 15.5 GB) | 15.47 GB |
| llama-server | 1.21 GB | 15.84 GB |

The peak (VmHWM) is reached during load, while the mapping is still being read.
Lowering it would need a read-and-upload path that streams through a bounded buffer
instead of the mapping. This is not done; the mapped pages are clean and
reclaimable.

**Output equality.** zerv's four outputs now equal `llama-f32`, `llama-fa-ub256`
([serving-v2-run1](bench/data/2026-09-22-serving-v2-run1/)) and `llama-fp32-full`
([13f run](bench/data/2026-09-22-gemm-accuracy-serving-run1/)) byte for byte,
including `reasoning_content`. `llama-fa-ub512` and `llama-nofa-ub512` still differ
on `decode-think`, from about character 300 ("show equation chain" vs "mention last
nonzero remainder"). That is a greedy token divergence from their quantized
activation path, not formatting.

## Correction to an earlier report

[2026-09-22-serving.md](2026-09-22-serving.md) said that on `decode-think` "the tokens
are the same, but llama-server keeps a trailing `\n`". That holds only for the
FP32/ub256 llama configurations. The ub512 configurations produce a different greedy
continuation. A note was added there.

## Remaining / not done

- **Peak host RSS during load.** See above.
- **`reasoning_effort: "high"`.** zerv maps it to the template's `xhigh`. llama-server
  passes it to the template, which raises (HTTP 500). This is a documented, deliberate
  API difference.
- **Invalid UTF-8 inside the output.** Replacement characters in zerv versus a
  reference parser failure mode. Documented in the research note; not matched.

## Commands

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
zig fmt --check build.zig src bench/*.zig tools/*.zig tests/*.zig
zig build test; zig build test -Doptimize=ReleaseFast
zig build gpu-test; zig build gpu-test -Doptimize=ReleaseFast
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tools/check_session.py --output docs/bench/data/2026-09-22-serving-open-items/session-gate.json
python3 tools/check_shutdown.py --output docs/bench/data/2026-09-22-serving-open-items/shutdown.json   # and shutdown-run2.json
python3 tools/check_parity.py --output docs/bench/data/2026-09-22-serving-open-items/parity-run1 --zerv-binary zig-out/bin/zerv
python3 tools/check_parity.py --output docs/bench/data/2026-09-22-serving-open-items/parity-control-pre-change \
    --zerv-binary third_party/serving-bench/2026-09-22-gemm-accuracy-serving-run1/zerv   # renamed to …-oldlabels after the run
python3 bench/run_serving.py --output docs/bench/data/2026-09-22-serving-open-items-serving-run1 --engines zerv,llama-fa-ub512
python3 bench/run_serving.py --output docs/bench/data/2026-09-22-serving-open-items-serving-repeat --engines zerv,llama-fa-ub512 \
    --zerv-binary third_party/serving-bench/2026-09-22-serving-open-items-serving-run1/zerv
```

The first `check_shutdown.py` run used a version of the script that did not catch the
expected client-side `IncompleteRead` in scenario 3. The traceback went to stdout; the
verdicts are unaffected. Run 2 uses the final script.
