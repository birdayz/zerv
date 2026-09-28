# KV tiering papers: Mooncake, Pensieve, CachedAttention, LMCache

Research date: **2026-09-28**. User asks for papers, especially Moonshot AI's,
after rejecting the production archive's write-through policy. **Research only**;
no implementation or new performance measurement. The earlier write-through
integration remains the implemented baseline. The replacement in
[disk-prefix-cache.md](../specs/disk-prefix-cache.md#proposed-replacement-eviction-driven-spill-user-discussion-not-implemented)
is a proposal, not a shipped change or implementation-ready specification.

## Finding / correction

Our **unconditional full-image NVMe write that pauses its source request** is a poor
capacity-cache policy on the measured single-GPU system: [the serving experiment](../bench/2026-09-28-disk-prefix-serving.md)
wrote 8.78 GB and raised cold-turn TTFT p50 from 25.41 to 40.22 s. This is observed,
not something inferred from a paper.

But the stronger claim that write-through tiering is universally wrong is **not
supported by these sources**. Separate three questions:

1. **Admission/trigger:** every new object, selectively hot objects, or anticipated
   eviction? Are we preserving optional cached work or transferring required live
   state to a different inference node?
2. **Execution:** does copying block the request, or run asynchronously with bounded
   leases/headroom and real copy/compute overlap?
3. **Granularity:** newly produced/deduplicated blocks, or another complete prefix
   image including bytes already saved by earlier checkpoints?

Mooncake's published design eagerly moves **incremental layer-wise KV to DRAM** for
prefill/decode disaggregation. Its current SSD implementation separately supports
both eager persistence and offload-on-eviction. HiCache explicitly offers eager,
selective eager and eviction-driven policies. The evidence supports choosing an
appropriate policy per tier, not banning all eager copies.

## Primary sources and provenance

Full URL/revision/local path/size/SHA256 inventory:
[sources.json](2026-09-28-kv-tier-papers/sources.json). Third-party papers/code/tool
wheel are under gitignored `third_party/`, never production dependencies.

| Source | Pinned version | Sections inspected / purpose |
|---|---|---|
| [Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot](https://www.usenix.org/conference/fast25/presentation/qin) | FAST '25 proceedings, pp.155–170; publisher PDF hash in ledger | §3.1 workflow, §3.2 store/eviction/transfer; distinguish cluster architecture from SSD policy |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/html/2407.00079v4) | arXiv 2407.00079v4, 2025-09-03 | §3, §5–6; incremental DRAM writes, pipelining, cache/compute/transfer scheduling |
| [Mooncake repository](https://github.com/kvcache-ai/Mooncake/tree/f8b50d5f1a2fa4d33d029ed1e82188de70b137f0) | `f8b50d5f1a2fa4d33d029ed1e82188de70b137f0`, 2026-09-28 | SSD deployment/design docs, master implementation/configuration, policy tests, HiCache integration design |
| [Stateful Large Language Model Serving with Pensieve](https://arxiv.org/html/2312.05516v3) | arXiv 2312.05516v3 | §4.3, §5: GPU/CPU cache, cost-aware chunk eviction, ahead-of-time copies, retrieval priority |
| [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](https://arxiv.org/html/2403.19708v3) | arXiv 2403.19708v3 | §3.2–3.3: asynchronous GPU→RAM saving, scheduler-aware RAM/disk placement, reserved headroom |
| [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/html/2510.09665v2) | arXiv 2510.09665v2 | §4–6: chunk transfers, delayed decode stores, bounded duplication window, ownership/connector hooks |

These are focused reads of the relevant mechanisms, not reproduced evaluations.
The FAST paper and arXiv version are different documents; section numbers and
implementation facts are not interchangeable. Current repository SSD details must
not be retroactively attributed to the 2025 paper. HiCache integration documentation
is **not a separate peer-reviewed paper**. No third-party tests or engines were run
for this literature review.

## 1. What Mooncake actually does

### Published architecture

**Source-backed, FAST §3.1–3.2:** a global KV cache pool and separate prefill/decode
instances. Prefill stores **new incremental KV** into CPU memory. Layer-wise
transfer overlaps prefill and moves data toward the chosen decode node's CPU
memory. The cache stores paged blocks, generally 16–512 tokens, with hash identities
including prefix context. Full-pool eviction uses LRU and excludes blocks accessed
by ongoing requests. The implementation offers transfer submission and completion
status rather than treating submission as completion.

This is not a mandate to serialize every complete conversation to local SSD. The
FAST text inspected describes pool management but does not specify the modern
`offload_on_evict` SSD control plane. Also, much of Mooncake's benefit concerns
many GPU nodes and aggregated 100–400 Gbps NIC bandwidth; our one-card local disk
path cannot inherit its end-to-end speedup numbers.

**Source-backed, arXiv §6:** longest cached prefix is not automatically the best
execution choice. Scheduling considers load, transfer time, additional prefill and
TTFT constraints. Lesson for us: compare the cost of a disk restore plus remaining
prefill to recomputation (and shorter hot hits), rather than maximizing cached
bytes regardless of latency. Our current longer-disk-hit selection is simpler.

### Current SSD policy: both modes exist

At the pinned repository revision:

- `mooncake-store/include/master_config.h`: `offload_on_evict = false`;
  offload itself is also opt-in.
- `mooncake-store/src/master_service.cpp`, `PutEnd` near line 5330: if offload is
  enabled and `!offload_on_evict_`, enqueue offload work for completed memory
  replicas and pin a source replica. This is **eager background persistence**, not
  the application waiting for every physical disk completion.
- `BatchEvict`, near lines 11575–11650: in eviction mode, an existing disk replica
  allows memory eviction without another write. Otherwise enqueue/pin one source
  replica; other redundant unpinned replicas may be reclaimed immediately. Actual
  source bytes are not freed just because offload was queued.
- If the queue cannot accept the work, normal mode preserves the object and retries;
  `offload_force_evict` explicitly permits dropping it instead. Queue limits and
  force-eviction policy are distinct from asynchronous I/O. These are object-count
  limits; our variable-sized images also require byte accounting.
- `NotifyOffloadSuccess`, near line 8592: completed disk replicas are registered,
  source references/tasks resolved, and failed offloads do not publish successful
  disk replicas. Unmount and update races have explicit handling.
- The master also has other backend/HA branches (`enable_oplog_`, DFS, etc.). This
  inspection is of the ordinary local-SSD path, not a claim every flag combination
  uses exactly that path.

Independent source cross-checks (read, **not executed**):
`master_service_offload_scenario_test.cpp` contains
`PutEndQueuesOffloadByDefault` and `OffloadOnEvictSkipsQueueingAtPutEnd`.
`offload_on_evict_test.cpp` checks that a queued offload is not delivered after an
upsert preempts it. This is directly relevant to generation-qualified ownership;
we will use our own immutable IDs/oracle, not import their C++ implementation.

The repository's high-level SSD overview says offload is pressure-driven, whereas
the deployment flag table and code expose eager default behavior. **Use the
specific configuration and implementation, not that broad overview sentence.**
Likewise, the documented 0.90/0.80 watermark pair controls **disk eviction**, not
the RAM spill threshold. Do not copy those values into our RAM policy by mistake.

### HiCache integration: do not conflate levels

`docs/source/design/hicache-design.md` describes L1 GPU, L2 host, L3 shared storage;
Mooncake L3 may itself primarily be **distributed DRAM**, with its own SSD tier.
Its `write_through`, `write_through_selective`, and `write_back` policies therefore
do not all mean “write local SSD on every checkpoint.” It also describes node/page
keys, suppressing writes of already-present L3 data, asynchronous backup queues,
and prefetch policies (`best_effort`, `wait_complete`, timeout). Useful mechanisms,
but neither CUDA overlap nor a distributed backend is necessary for our initial
local implementation.

## 2. Pensieve: anticipate reclamation, do not wait for allocation failure

**Source-backed, §4.3:** retain completed conversations' GPU state rather than
immediately freeing it. When free GPU cache slots fall below a threshold (25% is
an example in the paper), start selected GPU→CPU copies. GPU storage is reclaimed
lazily when needed, allowing preparation before reuse. CPU pressure drops cached
chunks; this paper's evaluated hierarchy is GPU/CPU, **not an NVMe spill design**.

The 32-token experimental chunks are ranked by `Cost(chunk, context) / inactivity`
(the lower value is evicted first). Recompute cost is profiled, not assumed uniform.
Leading tokens are preferentially dropped because their attention recomputation is
cheaper. **Do not transplant that leading-token policy to our hybrid model**:
its recurrent/conv state requires checkpoint-aligned recovery. Likewise, retain
our radix sharing rather than independently duplicating shared conversation bytes.

§5 reports CPU↔GPU simultaneous copies slowing both directions on its hardware;
it prioritizes recovery over ahead-of-time eviction. This is evidence to benchmark
read/write interference on our card, not proof of the same 18–20% loss here.

**Applicable:** bounded proactive copying, lazy reclamation, cost-sensitive victims,
read priority. The important improvement over my previous explanation is to begin
work **before** the allocator is out of memory, not merely react at the hard limit.

## 3. CachedAttention: the most directly relevant RAM→SSD paper

**Source-backed, §3.2.2:** eager/asynchronous layer-wise GPU→host saving, different
prefill/decode schedules, and a reserved GPU write buffer so unfinished copies do
not block the next job. Again, eager host saving can be a purposeful architecture.

**Source-backed, §3.3:** host memory plus disk, a reserved host fetching buffer,
and eviction triggered at a free-memory threshold to keep that buffer available.
Queued requests provide look-ahead information: prefetch their disk-resident
state, avoid discarding entries needed soon, and prefer later-needed entries for
host eviction. Its eviction/fetch unit is a **whole conversation**, not radix pages.

**Applicable:** reserve transfer headroom, use known queued demand, start prefetch
before a request reaches the GPU, and separate RAM spill from disk eviction.
**Not directly applicable:** its positional-encoding/truncation changes and its
claim that a whole session is the natural unit. Those are different semantics from
our exact hybrid-model checkpoints and shared-prefix tree. No arithmetic or context
semantics changes are proposed here.

## 4. LMCache: bounded duplication is a deliberate latency tradeoff

**Source-backed, §5.1–5.3:** coalesce pages into configurable larger chunks (default
256 tokens), batch newly decoded KV before storing, pipeline layers, and share
buffers via reference counts until all transfers complete. Avoid thousands of tiny
operations and re-copying an entire growing prefix.

Its dynamic GPU→CPU offloading keeps only a **window** of free cached GPU pages
already duplicated or pending duplication. Too small a window stalls allocation
when it catches up to the copy cursor; a larger window buys headroom with memory
and bandwidth. The paper explicitly says extending this dynamic mechanism to other
tiers is possible but **not supported there**. Treat a RAM→NVMe application as our
proposal, not an existing measured LMCache result.

The connector's explicit load/store start/wait hooks reinforce that “asynchronous”
is a lifetime protocol; the engine cannot recycle pages still being read by DMA.
Its paper does not establish our GPU/CPU/device-error contract, which needs our
own tests.

## Proposed policy for zerv after this review

Not a completed spec and no implementation started:

1. **Checkpoint creation:** retain the exact recurrent snapshot and reference the
   immutable KV prefix. Do not enqueue an NVMe write. Snapshot-slot consumption is
   a capacity dimension separate from host KV bytes.
2. **GPU reclamation preparation:** keep a bounded reserve of cached pages safe to
   reclaim. Copy selected cold pages to host when projected demand consumes that
   reserve, not every time a token/checkpoint is produced. Actual transfer overlap
   on this Vulkan path must be measured; current fences are synchronous.
3. **Host reclamation preparation:** similarly prepare a bounded reserve of entries
   with complete disk backing. Candidates are cold, unleased cache state with no
   near-term queued reuse. Write only if valid disk backing is absent; count pending
   source bytes/slots as resident until completion. A full disk-backed image still
   requires every referenced segment, even if some live in GPU and some in RAM.
4. **Foreground reads over discretionary writes**, with explicit bandwidth/queue
   limits and anti-starvation/cancel/drop rules. Restore or admission must not wait
   indefinitely behind speculative spill work. Active request swap has separate
   ownership/priority and is not disposable cache.
5. **Best-effort persistence for cache entries:** if bounded headroom is exhausted,
   discard an eligible optional entry rather than force every request to wait for
   its disk preservation. Leased/in-flight buffers must drain before release;
   never discard live inference state. This differs from Mooncake's normal
   data-preserving queue-full behavior and must be an explicit product policy.
6. **Independent logical checkpoint, physical snapshot slot, and disk record IDs.**
   More disk capacity must not consume one RAM snapshot slot per retained prefix.
   Keep backing after a read to avoid writing unchanged data a second time.
7. **Eventually deduplicate immutable disk KV blocks**, with checkpoint manifests
   referring to them plus a recurrent snapshot. Mooncake's block identity is the
   useful lesson; our full-image archive duplicates overlapping prefixes. Keep
   policy integration and shared disk extents as separate runnable increments;
   do not introduce both ownership changes at once.

Watermark values and a cost model are **not tuned yet**. Headroom must account for
measured spill latency, allocation bursts, bytes, snapshot slots, and in-flight
reads/writes. A byte reserve cannot protect against snapshot-slot exhaustion, nor
can a queue of N objects bound memory without object-size accounting. A restore
cost comparison must include queueing, hashing, CPU copies, GPU upload and remaining
prefill—not raw NVMe bandwidth alone.

## Before coding / executable acceptance

Extend the independent residency oracle with separate entry/slot/backing identities,
pending write leases and pressure-driven transitions. Directed traces must cover:

- zero NVMe writes for insert/hit workloads fitting RAM;
- proactive spill under byte pressure and under snapshot-slot pressure alone;
- no duplicate write on re-eviction of a clean disk-backed checkpoint;
- new longer prefixes do not mutate old records;
- source generation cannot be reclaimed/reused before acknowledgment;
- mixed GPU/host radix ancestry, shared segments, partial pages, and concurrent hits;
- queue saturation, reads overtaking optional writes, bounded reserve consumption;
- disk failure/cancellation/drain and no lost live state;
- coherent recurrent/conv + KV state at one exact token position (not arbitrary
  attention-only chunk recovery).

Then exact poisoned/permuted 80k state + full-vocab checks through production owners,
followed by repeated cold/reuse/churn/pressure serving against eager baseline,
disk-off, host-only and tuned llama-server/compatible competitors. Report bytes
written per admitted/reused prefix, source retention time, avoided recompute,
TTFT/gaps, throughput, staging/metadata/host/VRAM peaks, losses and failed runs.
No published paper throughput multiplier is an acceptance result for our engine.

## Reproduction / limits of this research

Fetch primary URLs in `sources.json` with `curl -fLsS --max-time 60 URL -o PATH`,
creating parent directories first, and verify SHA256. GitHub paths use the pinned
commit, not `main`; arXiv paper paths use explicit revisions.

```sh
tools/py docs/research/2026-09-28-kv-tier-papers/extract_html.py \
 third_party/mooncake/arxiv-2407.00079v4/paper.html \
 third_party/kv-tier-papers/{2312.05516v3,2403.19708v3,2510.09665v2}/paper.html
tools/py docs/research/2026-09-28-kv-tier-papers/extract_pdf.py \
 third_party/mooncake/fast25/paper.pdf
tools/py docs/research/2026-09-28-kv-tier-papers/build_ledger.py
```

`pdftotext` was unavailable. Instead `extract_pdf.py` downloads and verifies the
PyPI **pypdf 6.1.1** pure-Python wheel in the gitignored research directory and
imports it only under the pinned `tools/py` interpreter. No system/package install,
repo runtime dependency, GPU job, driver or filesystem change. Its wheel SHA is
`7781f99493208a37a7d4275601d883e19af24e62a525c25844d22157c2e4cde7`.
Extraction logs are alongside this note. HTML math may contain duplicate accessibility
text; inspect the original equations when implementing a cost model. Search-engine
access hit a CAPTCHA and was not bypassed; primary arXiv APIs, publisher PDF and
pinned GitHub sources supplied the material directly.
