# Host overhead of decode and speculative decode (2026-09-24)

Question (user guidance: look for CPU-side optimizations, comptime where it pays off, but
measure first): how much of decode time is host work, and is any of it worth specializing?

## Instrumentation

`session.Result` now carries `backend_ns` (time inside backend `step`/`draft`/`verify`/
`commit` calls: submission, GPU work and fence wait) and `sample_ns` (time inside the
sampler) next to `decode_ns`. The server logs one line per request:
`decode time: T ms total, B ms in backend calls, S ms sampling, O ms other host work`
(O = token handling: detokenizing, UTF-8, stop strings, reasoning split, streaming).
Cost: three to six clock reads per token.

## Measurement

`python3 bench/run_serving.py --workload bench/workloads/decode-v1.json --output
docs/bench/data/2026-09-24-host-overhead/decode-v1 --engines "zerv,zerv-spec3" --repeats 1`
(greedy, 512 tokens; server logs in the data directory):

| engine | case | total ms | backend | sampling | other host |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 drafts | code | 4,180 | 4,144 (99.14%) | 30.0 (0.72%) | 5.8 (0.14%) |
| 3 drafts | json | 4,044 | 4,006 (99.06%) | 31.6 (0.78%) | 6.4 (0.16%) |
| 3 drafts | think | 4,715 | 4,677 (99.19%) | 31.6 (0.67%) | 6.5 (0.14%) |
| 3 drafts | prose | 6,688 | 6,650 (99.43%) | 31.6 (0.47%) | 6.8 (0.10%) |
| plain | 4 cases | 10,065–10,109 | 99.5–99.6% | 31.6–35.6 (0.31–0.35%) | 7.7–13.7 (0.08–0.14%) |

- Sampling is about 60 µs per token: the greedy scan of one 248,320-float logits row
  (1 MB of mapped host memory) at about 17 GB/s, close to one core's memory bandwidth.
  The component benchmark measured 49 µs for the same scan
  ([sampler](2026-09-24-sampler.md)).
- Cross-check with a cost model (per cycle: `draft(t, 3)` 4.67 ms from `zerv-mtp-check`,
  plus verify+commit at the average verified rows from `zerv-spec-check`; the earlier
  verify-fusion serving data): measured decode spans exceed the model by 0.54–0.93 ms per
  cycle (1.8–3.1%). The breakdown above puts 99% of the time inside backend calls, so
  most of that gap is GPU time and submission latency at serving positions and commit
  sizes, not session host work.
- Inside the backend calls, each synchronous submit + fence wait costs about 50 µs of
  GPU idle ([driver bench](2026-09-22-gpu-driver.md): tiny-dispatch round trip). A
  speculative cycle has three (commit, draft, verify): about 0.15 ms of about 29 ms.

## Conclusion

Host-caused GPU idle is about 1.4% of a speculative cycle (sampling about 0.2 ms, three
round trips about 0.15 ms, token handling about 0.05 ms) and about 0.6% of a plain step.
No comptime or CPU specialization is warranted now: the sampler dispatch is a few
branches per token, and the scan is bandwidth-bound.

Candidates if this share ever matters (each under 1%, recorded in TODO.md):
- greedy without penalties: argmax of the verify rows on the GPU, reading 4–5 ids
  instead of 4–5 MB of logits (about 0.2 ms per cycle);
- commit and the next draft in one submission (both are known once the host has
  sampled): one round trip fewer per cycle (about 0.05 ms);
- a single-pass vectorized greedy argmax (one read of the row instead of max, then find;
  about 20 µs per token).
