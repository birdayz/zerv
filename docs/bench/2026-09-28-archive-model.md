# Model archive adapter — 2026-09-28

Observed first successful gate: 257 and 80,000 tokens, same Qwen3.8 Q4_0 weights,
f16 KV and f16 prefill, two model slots, shared pool. Capture logical recurrent/conv
and KV-page bytes into a RAM image; continue four teacher-forced tokens for reference
logits; release the source pages, poison **all** physical KV and state, map target
pages in reverse order in slot 1, restore, export and compare **all** bytes; compare
all four vocabulary rows bit-for-bit. No disk in this adapter gate.

[Successful initial evidence](data/2026-09-28-archive-model-verified/), including
source/model/binary hashes and commands. 257: 182,059,008 bytes; 80,000:
5,399,773,184 bytes. Both exact state and four exact vocabulary rows. Elapsed whole
gate 0.817 s / 128.718 s (not transfer-only timing; includes prefill, poison,
continuation and verification; not a repeated performance result).

CPU 81/81 passed. [GPU gate logs](data/2026-09-28-archive-model/): GPU Debug,
ReleaseFast and spill tests 3/3; host-driver 2/2. Debug actually ran on both drivers;
ReleaseFast/spills cached. After the teardown fix, the harness reran host-driver
Debug successfully (ReleaseFast cached).

Failures retained:
- First native build: invalid error-set use and use of a nonexistent fill API;
  corrected to existing errors and bounded staging copies for poisoning.
- [First model run](data/2026-09-28-archive-model-gate/): exact bytes/logits at 257,
  then SIGABRT during teardown. `seg_tail` initializes only `pack_seqs - 1`
  commands per plan; cleanup traversed the full seven-column array as if densely
  initialized. With two slots it deinitialized an uninitialized command. Cleanup
  now walks the actual initialized width, preserving partial-initialization cleanup.
  The two-slot model checker is its executable regression gate; this is separate
  from the retired hang investigation.
- [Next attempt](data/2026-09-28-archive-model-fixed/): format gate rejected the
  changed cleanup line. No model execution in that attempt; formatted before retry.

Model addressing is an opaque copy, not new inference arithmetic; independent
POSIX/hash archive fixtures and existing independent model oracles remain separate
required gates. Repeated adapter timings and the fresh oracle check are recorded
below; production disk owner and scheduler evidence is in the
[integration report](2026-09-28-disk-prefix-serving.md).

Completed follow-up: [five trials after one warmup + 80k repeat](data/2026-09-28-archive-model-trials/)
all exact. At 257 tokens capture durations: 70.637, 68.974, 69.749, 70.144,
70.951 ms; restores: 43.370, 42.968, 44.339, 43.659, 42.542 ms. Capture includes
first-touch/copy into the RAM golden image; restore includes its CPU copy into
1 MiB imported staging and per-quantum synchronous GPU fences. Thus not comparable
to the earlier large-buffer bandwidth probe. The 80k repeat captures in 4.505 s,
restores in 1.708 s (one sample, not a speedup claim). No equivalent llama-server
opaque-image adapter; actual serving comparison remains an integration gate.

Fresh independent FP64/libllama model gate passed decode (`0`) and prefill (`512`),
all intermediate/logits bounds and 337/337 greedy selections:
[data](data/2026-09-28-archive-model/oracle.json), [log](data/2026-09-28-archive-model/oracle.log).
Command:
```
tools/py tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-26-hermetic --work-dir third_party/archive-model-oracle-check --report docs/bench/data/2026-09-28-archive-model/oracle.json --modes 0,512 --runtime host
```
Full oracle capture provenance remains at `third_party/archive-model-oracle-check`;
reference source provenance is the pinned hermetic oracle ledger. These independent
arithmetic gates supplement—not replace—the exact model archive roundtrip.

The production-disk gate initially did **not execute**: its all-tests prerequisite
hit 300-second `kv_system_release_fast` timeouts, twice, in the distinct-long-
conversations test. Retained full logs/manifests:
[data](data/2026-09-28-archive-disk-gate/), [retry](data/2026-09-28-archive-disk-retry/).
The new delayed-I/O scheduler tests pass Debug and ReleaseFast. These timeouts
are not reclassified as passes, and the retired hang goal is not reopened.

Third disk-gate attempt passed CPU 81/81 and host GPU 2/2 but failed its 257-token
checker with `PageInUse` before restore. Harness error: destroying hot-cache metadata
does not unpin pages (normal shutdown destroys the pool afterwards). The checker now
releases the source sequence, calls the real cache eviction API to drop its pins,
then destroys the metadata before poisoning/remapping. The private-page guard
correctly refused the test's still-pinned pages; no weakened ownership check.
[Failed attempt](data/2026-09-28-archive-disk-third/).

The eviction-corrected disk checker passed 257 tokens, then failed at 80k with
`PagesMissing`. Again a checker contract error: a hot checkpoint rebinds the source
to its prefix pages, dropping the checker's extra reservation. At exactly 80,000
(page-aligned) tokens the next step needs a new page. The real scheduler calls
`grow` before each decode; the checker now does too, including after a disk begin.
The 257 case did not cross that page boundary. No production admission guard was
relaxed. [Attempt](data/2026-09-28-archive-disk-eviction/).
