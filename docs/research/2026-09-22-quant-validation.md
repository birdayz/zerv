# 2026-09-22 — native quant decoding correctness

## Scope and sequencing

[Operation/ABI research](quant-blocks.md) and the
[functional specification](../specs/quantization.md) were written before native code.
The external golden generator ran successfully first, comparing installed ggml
outputs with independent Python scalar arithmetic. Native implementation followed;
[explicit SIMD research](quant-vector.md) preceded the subsequent CPU optimization.
No foreign source/header/library is imported into the native module.

## Executed verification

Pinned Zig 0.16.0:

```sh
zig fmt --check build.zig src bench/quant.zig tests/root.zig
zig build test --summary all
zig build test -Doptimize=ReleaseFast --summary all
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

- **5/5 native tests pass in both Debug and ReleaseFast.** The benchmark harness
  repeats these gates before building/timing each comparison.
- Every finite binary16 encoding (63,488), times every Q4/Q8 coefficient: 2,031,616
  Q4 and 16,252,928 Q8 output values match external golden fingerprints exactly.
- Explicit 11-block fixtures per format match full output bytes with deliberately
  byte-unaligned input. Includes positive/negative zero, subnormals, normal values,
  and maximal positive/negative finite scales.
- All 2,048 non-finite binary16 encodings per format reject, including invalid
  later blocks, with every output value untouched. Empty/length mismatch tests pass.
- Testing allocator reports no leaks. Implementation itself allocates no memory.
- **4/4 Python harness tests pass:** missing/duplicate/extra trials, altered
  input/output hashes, workload changes, invalid elapsed values and schema errors
  cannot be accepted as valid benchmarks.
- Native test/benchmark ELF files are static with no dynamic `NEEDED` dependencies;
  source/build scan finds no C/C++ header imports or reference-library links.

Golden regeneration was also executed into
`third_party/quantization-candidate.json`; `cmp` against the committed fixture
succeeded byte-for-byte. This verifies that the recorded generator reproduces its
data, not just that native code matches one opaque file.

Reference identity: ggml 0.24.0, reported commit `456172ec-dirty`; exact library
hash and generator hash are retained in `tests/fixtures/quantization.json`.
Ordinary native tests need no Python, reference library, or `third_party/` source.
See [dated component report](../bench/2026-09-22-quant-decode.md) for actual timing
results and raw build/test logs; test duration is not itself a performance result.

## What this does not establish

Finite-domain scalar-product coverage is not a formal proof of every possible row
sequence/layout. It does not validate other quant types, GGUF parsing, reductions,
matmul, GPU kernels, tokenization/templates, model logits, generation, concurrent
state, or Chat Completions. Those require separate complete research and independent
reference fixtures before implementation. The active model-serving goal remains open.
