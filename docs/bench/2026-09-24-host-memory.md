# Host-resident embedding and prefix-cache snapshots (2026-09-24)

Question: VRAM bounds the context (FP32 KV, 128 KiB per token). Which large buffers can
live in system memory, what does that cost in TTFT and decode, and what should the
defaults be? Knobs: `--embedding-memory host|device` and `--prefix-cache-memory
device|host` ([model spec](../specs/model.md), "Host-resident data";
[prefix cache spec](../specs/prefix-cache.md)).

## Setup

- RX 7900 XTX, Mesa 26.2.3 RADV, PCIe 4.0 x16; Qwen3.8-27B-Q4_0 (sha256 `ede16c7b…`).
- zerv binary sha256 `cfbc6e0738292cda…` (both serving runs; the manifests hold the full
  hashes and commands).
- The token embedding (Q4_0, 682 MiB) is read one 2,880-byte row per token by `embed`
  and `embed_b`. A snapshot slot is 156,893,184 bytes (all DeltaNet and conv state); the
  server keeps 8 slots by default (1,197 MiB).

## Correctness (all run with the new code; raw files in [data](data/2026-09-24-host-memory/))

| Gate | Configuration | Result |
| --- | --- | --- |
| `tools/verify_model.py`, default oracle, modes 0/1/13/512/512:17 | host embedding | passes; all 28 capture and logit files byte-identical to the previous device-embedding captures (`third_party/model-native/2026-09-24-flash-final-default`) |
| `zerv-prefix-check` fp32 | host snapshots | gate passed: restore = split bitwise at 11 points, repeat and extend bitwise |
| `zerv-prefix-check` fp32 | device snapshots | gate passed |
| `zerv-prefix-check` f16 | host snapshots | gate passed |
| `zerv-spec-check` | host embedding | 11/11 cases: verify rows, committed state and follow-up decode bitwise equal to decode |
| `zerv-mtp-check` + FP64 reference | host embedding | both sequences pass (bound 1e-5) |
| serving-v2, 8 engine runs | all combinations | every case has one output hash across all engines |

The prefix check was changed in this step. It now compares restore with split directly
(`restore_vs_split`) and exits non-zero unless that gate holds. Earlier it only compared
both with the cold run, which differs off the chunk grid. It also takes
`[host|device]` and times the copies.

Process note: the first two prefix-check runs of this step used a stale
`zig-out/bin/zerv-prefix-check` built before the change: the default `zig build` step
runs tests and does not install tools. They are discarded. The tools were rebuilt with
their `*-build` steps (`prefix-check-build`, `spec-check-build`, `mtp-check-build`) before
the runs above.

## Snapshot copy cost (`zerv-prefix-check MODEL fp32 host|device`, 20 timed copies)

| Slot memory | save ms (median, min–max) | load ms |
| --- | --- | --- |
| device | 0.476 (0.464–0.530) | 0.452 (0.440–0.476) |
| host | 9.805 (9.771–9.841) | 15.285 (15.260–15.325) |

Host copies run at 16.0 GB/s (save, VRAM → RAM) and 10.3 GB/s (load).

## Serving (serving-v2, 2 repeats, greedy, default context 8192)

```
python3 bench/run_serving.py --workload bench/workloads/serving-v2.json \
  --output docs/bench/data/2026-09-24-host-memory/serving-v2-hostmem --repeats 2 \
  --engines "zerv-f16-spec3;zerv-f16-spec3@embedding-memory=device,prefix-cache-memory=device;zerv-f16;zerv-f16@embedding-memory=device,prefix-cache-memory=device"
python3 bench/run_serving.py --workload bench/workloads/serving-v2.json \
  --output docs/bench/data/2026-09-24-host-memory/serving-v2-embedding --repeats 2 \
  --engines "zerv-f16@prefix-cache-memory=device;zerv-f16@embedding-memory=device,prefix-cache-memory=device"
```

`ENGINE@flag=value,...` is a new harness feature: it appends `--flag value` to a zerv
engine's command, so any knob can be A/B-tested (`tests/test_serving_harness.py`).

TTFT ms (min/max of 2) and decode tok/s (median):

| Engine (embedding, snapshots) | short-nothink | decode-think | medium-prompt | long-prompt | VRAM loaded MiB |
| --- | --- | --- | --- | --- | --- |
| f16 spec3 (host, host) | 153/154, 127.6 | 325/327, 106.0 | 961/961, 97.5 | 2891/2902, 108.9 | 16,393 |
| f16 spec3 (device, device) | 143/144, 126.1 | 307/307, 105.4 | 940/946, 97.5 | 2905/2919, 108.1 | 18,273 |
| f16 (host, host) | 152/153, 54.9 | 327/328, 50.0 | 975/983, 49.8 | 2922/2941, 57.2 | 16,011 |
| f16 (device, device) | 142/142, 54.9 | 307/307, 49.6 | 947/952, 49.6 | 2917/2941, 57.3 | 17,890 |
| f16 (host, device) [run 2] | 142/142, 55.0 | 305/305, 49.6 | 929/938, 49.7 | 2878/2898, 57.3 | 17,208 |
| f16 (device, device) [run 2] | 142/142, 54.8 | 307/307, 49.5 | 940/948, 49.7 | 2901/2922, 57.1 | 17,890 |

## Interpretation and decisions

- **Host embedding: no measured cost** (TTFT within ±1% in either direction, decode
  equal), 682 MiB less VRAM. **Default: host.**
- **Host snapshots: +10 ms TTFT on short-nothink, +18–20 ms on decode-think and medium
  (one or two 9.8 ms saves on the prefill path), no decode cost**, 1,197 MiB less VRAM
  at 8 slots. That is 7% of short-request TTFT. **Default: device** (fastest); `--prefix-cache-memory
  host` is the knob for about 9.6k more tokens of FP32 KV context.
- Together they free 1.84 GiB (18,273 → 16,393 MiB with speculation).
- Possible next step (not done): save through a device staging slot (0.5 ms on the
  prefill path), then copy to the host on a transfer queue while compute continues.
  This would remove the host-snapshot TTFT cost for one slot's VRAM (150 MiB).
