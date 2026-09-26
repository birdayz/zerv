# KV admission and preemption in other engines (research, 2026-09-27)

Question: how do other serving engines admit requests into a shared KV pool without
reserving each request's worst-case output, and what do they do when the pool runs dry
during decode? This is the input for zerv's next step after the 18d.2 shared pool
(docs/specs/concurrent.md). Today zerv reserves prompt + output limit at admission, and a
request without `max_tokens` reserves up to the full per-request context.

## Sources (local, read 2026-09-27)

| Engine | Revision | Path | Files read |
| --- | --- | --- | --- |
| vLLM | git 98dff2a8 (2026-09-08), matches the benchmarked v0.30.0 image line | third_party/research-serving/vllm | vllm/v1/core/sched/scheduler.py (`schedule`, `_preempt_request`), vllm/v1/core/kv_cache_manager.py (`allocate_slots`), vllm/config/scheduler.py, docs/configuration/optimization.md |
| SGLang | 0.5.20 release tarball | third_party/sglang/sglang-0.5.20 | srt/managers/scheduler.py (`init_req_max_new_tokens`, retract path), srt/managers/schedule_batch.py (`check_decode_mem`, `retract_decode`, `release_req`), srt/managers/scheduler_components/new_token_ratio_tracker.py, srt/managers/schedule_policy.py, srt/environ.py |
| llama.cpp server | b29c606e (the hermetic in-graph build) | third_party/hermetic-src/llama.cpp-b29c606e… | tools/server/server-context.cpp (decode-failure path, `n_ctx_slot`) |
| TensorRT-LLM | **not read locally** | — | from its public docs, from memory; verify before relying on it |

## vLLM (V1): reserve the prompt, grow per step, preempt by recompute

- **Admission.** `allocate_slots` is called for every scheduled chunk.
  - With `scheduler_reserve_full_isl=True` (the default), a new request is admitted only if
    its *full prompt* fits in the free blocks: "prevents over-admission and KV cache
    thrashing with chunked prefill". Output tokens are **not** reserved.
  - Optional `watermark` (default 0): a fraction of blocks kept free when admitting
    waiting or preempted requests.
- **Growth.** Every step allocates the blocks the scheduled tokens need, plus speculative
  lookahead.
- **Pressure.** When `allocate_slots` fails for a running request, the scheduler preempts:
  - FCFS: `self.running.pop()`, the most recently admitted request;
  - priority policy: the highest (priority, arrival) value;
  - it repeats until the allocation fits or the victim is the request itself.
- **Preemption = recompute.** `_preempt_request` frees the request's blocks, sets
  `num_computed_tokens = 0` and puts it at the *front* of the waiting queue.
  - On resume it prefills prompt + generated tokens again. The prefix cache may still hold
    some of the freed blocks, which makes that cheaper.
  - The docs say V1 defaults to RECOMPUTE, not SWAP, "as recomputation has lower overhead
    in the V1 architecture". V0 had a swap mode; V1 keeps no swap path in the scheduler.
- **Output.** A recomputed request is not bitwise identical to an uninterrupted run: the
  re-prefill is different arithmetic, and vLLM does not claim batch invariance by default.
  Hybrid (mamba) state is rebuilt by the re-prefill, with block-aligned state caching in
  "align" mode.

## SGLang: reserve a *learned fraction* of the output, retract on pressure

- **Missing `max_tokens`.** `init_req_max_new_tokens` sets it to `1 << 30`, then caps it at
  `max_req_len - input - 1` and at the pool size. It is not a small default.
- **Admission** (`PrefillAdder`, schedule_policy.py):
  - a request is charged `prompt + (max_new_tokens × new_token_ratio − generated)`;
  - `new_token_ratio` starts at 0.7 (`SGLANG_INIT_NEW_TOKEN_RATIO` ×
    `schedule_conservativeness`) and decays linearly over 600 steps to 0.7 × 0.14 ≈ 0.1;
  - so the scheduler over-admits on purpose, betting that most requests stop early.
- **Pressure.** Before each decode step `check_decode_mem` asks the allocator whether the
  step fits, evicting radix-cache nodes first. If not, `retract_decode`:
  - retracts requests in order of **fewest generated tokens, then longest prompt** (or by
    priority when that policy is set), always keeping at least one;
  - resets `new_token_ratio` to `(decoded + 20·n) / Σ max_new_tokens`, so admission turns
    conservative after a retraction;
  - aborts the last remaining request with an error if even it cannot fit.
- **Retraction = recompute.**
  - `release_req` frees the KV **without inserting it into the radix tree** ("we need the
    space instantly"; a TODO says to insert it instead).
  - Only in PD-disaggregated *decode* mode is the KV backed up to a host
    "retraction pool" and restored without recompute. If that pool is full, the request
    is aborted.

## llama.cpp server (`-kvu`): no admission control

- Slots take cells from the unified cache as `llama_decode` runs; nothing is reserved.
  `--kv-unified-per-slot N` only caps each slot's context.
- **Pressure.** When `llama_decode` returns 1 (no KV slot), the server tries
  `try_clear_idle_slots()` (drops cached prompts of idle slots), then halves `n_batch` and
  retries.
- **At `n_batch == 1`:** it sends "Context size has been exceeded." to **every
  processing slot**, clears them, and **throws**. The source has TODOs to terminate only
  the largest sequence instead.
- **Observed** (third_party/multiuser/invalid-80k-v1): with the ~93k-token prompt,
  `-c 94208` and 6 streaming users, llama-server logged
  `decode() failed: Context size has been exceeded` and exited.

## TensorRT-LLM (from public docs, not verified locally)

The capacity scheduler has named policies:
- `GUARANTEED_NO_EVICT`: reserve worst-case KV at admission, like zerv's current 18d.2;
- `MAX_UTILIZATION`: admit optimistically and pause or evict requests under pressure;
- `STATIC_BATCH`.

Verify against its source before citing anything specific.

## What this means for zerv

| | zerv 18d.2 | vLLM | SGLang | llama.cpp |
| --- | --- | --- | --- | --- |
| Reserved at admission | prompt + output limit | full prompt | prompt + ratio × output (0.7 → 0.1) | nothing |
| No `max_tokens` | reserves up to the context | reserves the prompt only | ~pool-sized limit × ratio | — |
| Out of pages during decode | cannot happen | preempt the youngest | retract the fewest-generated | fail every slot, exit |
| Preemption mechanism | — | recompute | recompute (host backup only in PD decode) | — |
| Bitwise identical after preemption | n/a | no | no | n/a |

Observations:

1. **Everyone admits on the prompt, not the output.** Worst-case reservation (ours,
   TensorRT-LLM `GUARANTEED_NO_EVICT`) is the safe but underused choice. With clients that
   omit `max_tokens`, ours degrades to one request at a time when `--context max`.
2. **Everyone preempts by recompute.** It needs no host memory and resumes cheaply when
   prefix-cache blocks survive. Costs:
   - a full re-prefill of prompt + output, 46 s at 29k tokens for zerv today;
   - and it breaks bitwise determinism across preemption.
3. **Swap to host fits our constraints better.** Our invariant is bitwise
   identity: every serving gate compares against sequences run alone.
   - Copying the pages plus the 157 MB recurrent state back and forth is exact by
     construction.
   - Cost scales with context, not with compute: an estimated ~30 ms each way for an 8k
     sequence over PCIe.
   - For a hybrid model, recompute is even more expensive, because the recurrent state
     can only be rebuilt by a full re-prefill unless a snapshot exists.
   - It is also the host tier of the planned checkpoint store.
4. **Recompute exactness in zerv is an open question.** Is prefill of token t bitwise equal
   to decoding t? If not (GEMM vs matvec accumulation order), recompute would change later
   tokens. This must be measured with batch-check before recompute is even offered as an
   option.
5. **Victim choice:**
   - vLLM takes the youngest admitted request, SGLang the one with the fewest generated
     tokens.
   - With swap, the cost is bytes moved, so the cheapest victim is the shortest sequence,
     which also has made the least progress.
   - A fairness rule is needed so that one request is not swapped repeatedly (SGLang
     adjusts its ratio for this; vLLM puts the victim at the front of the queue).
6. **Admission estimate.**
   - SGLang's decaying ratio bets that requests stop early, and falls back to retraction
     when the bet fails.
   - With cheap swap, the bet can be aggressive. Alternatively reserve prompt + one growth
     chunk, as vLLM does, and let swap absorb the tail.

## Open questions before a spec

1. Pinned host memory: how much, and is it registered through our Vulkan path (host-visible
   memory import, transfer queue)? Measure the achievable host↔device bandwidth on this box.
2. Can swap-out and swap-in overlap with compute on a separate transfer queue without
   stalling decode steps?
3. Measure prefill-vs-decode bitwise equality (point 4) so the recompute option is decided
   on evidence.
4. What happens when host memory is also full: queue new requests, or abort (SGLang
   aborts)? Aborting a running stream should be the last resort.
5. The interaction with the prefix cache: pages kept by finished requests should be evicted
   before any running request is swapped.
