#!/usr/bin/env python3
"""Interleaved ("race") comparison of f16 GEMM modules on the real model weights (block 16b).

The RX 7900 XTX sustains these GEMMs at its power cap and at the 110 C junction limit, so
clocks drift with thermal state; variants timed back to back in separate runs are not
comparable to a few percent. This runs every variant on every tensor in each round, rotating
the variant order per round, after one untimed heat soak, and reports per-round ratios
against the first variant (the reference) plus medians. Nothing else may use the GPU.

Variant spec: NAME=SPV_PATTERN@GRID[@wave=32][@xf16] where SPV_PATTERN contains {fmt}.
Usage: race_gemm_f16.py --output DIR --variant A=... --variant B=... [--only T1,T2]
       [--rounds 3] [--warmup-ms 3000] [--soak-ms 15000] [--reference-dump DIR]"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gemm_f16 import MODEL_SHA, ROOT, SHAPES, Sampler, device_dir, sha, zerv_build  # noqa: E402


def parse_variant(spec):
    name, _, rest = spec.partition("=")
    parts = rest.split("@")
    v = dict(name=name, pattern=parts[0], grid=parts[1], wave=None, x_f16=False)
    for opt in parts[2:]:
        if opt.startswith("wave="): v["wave"] = int(opt[5:])
        elif opt == "xf16": v["x_f16"] = True
        else: raise SystemExit(f"unknown variant option {opt!r}")
    return v


def command(tool, model, v, tensor, fmt, a, dump=None):
    cmd = [str(tool), str(model), "--coopmat", "1", "--variant", fmt, "--spv", v["pattern"].format(fmt=fmt), "--grid", v["grid"],
           "--k-chunk", "0", "--tensor", tensor, "--samples", str(a.samples), "--warmup-ms", str(a.warmup_ms)]
    if v["wave"]: cmd += ["--wave", str(v["wave"])]
    if v["x_f16"]: cmd += ["--x-f16", "1"]
    if dump: cmd += ["--dump", str(dump)]
    return cmd + [str(a.rows)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--only", help="comma list of tensor names")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup-ms", type=int, default=3000)
    p.add_argument("--soak-ms", type=int, default=15000)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--reference-dump", type=Path, help="bytewise gate: each variant's round-0 Y against this directory")
    a = p.parse_args()
    variants = [parse_variant(s) for s in a.variant]
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if sha(a.model) != MODEL_SHA: raise SystemExit("model mismatch")
    tool = zerv_build.binary("zerv-gemm-bench")
    dev = device_dir()
    only = set(a.only.split(",")) if a.only else None
    shapes = [(t, f) for t, f in SHAPES if not only or t in only]
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, model_sha256=MODEL_SHA, tool_sha256=sha(tool),
                    variants=[dict(v, spv={f: sha(Path(v["pattern"].format(fmt=f))) for _, f in shapes}) for v in variants],
                    power_cap_w=int(next((dev/"hwmon").glob("hwmon*/power1_cap")).read_text()) / 1e6,
                    performance_level=(dev/"power_dpm_force_performance_level").read_text().strip())
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    t, f = shapes[0]
    soak = command(tool, a.model, variants[0], t, f, a)
    soak[soak.index("--warmup-ms") + 1] = str(a.soak_ms)
    subprocess.run(soak, capture_output=True, check=True)
    rows = []
    with (out/"raw.jsonl").open("w") as raw:
        for rnd in range(a.rounds):
            order = variants[rnd % len(variants):] + variants[:rnd % len(variants)]
            for tensor, fmt in shapes:
                for v in order:
                    dump = None
                    if rnd == 0 and a.reference_dump:
                        dump = out/"dump"/v["name"]; dump.mkdir(parents=True, exist_ok=True)
                    sampler = Sampler(dev); sampler.start(); t0 = time.time()
                    r = subprocess.run(command(tool, a.model, v, tensor, fmt, a, dump), capture_output=True, text=True)
                    sampler.stop.set(); sampler.join()
                    if r.returncode: raise SystemExit(f"{v['name']} {tensor}: {r.stderr[-2000:]}")
                    timed = [s for s in sampler.samples if s[0] >= t0 + a.warmup_ms / 1000]
                    row = json.loads(r.stdout.splitlines()[-1])
                    row.update(variant=v["name"], round=rnd,
                               sclk_mhz_median=statistics.median(s[1] for s in timed) if timed else None,
                               power_w_median=statistics.median(s[2] for s in timed) if timed else None,
                               junction_c_median=statistics.median(s[3] for s in timed) if timed else None)
                    if dump is not None:
                        name = f"{tensor}-{a.rows}.bin"
                        row["bitwise_equal_reference"] = (dump/name).read_bytes() == (a.reference_dump/name).read_bytes()
                    raw.write(json.dumps(row) + "\n"); raw.flush(); rows.append(row)
                    print(f"r{rnd} {v['name']:10s} {tensor:26s} {row['tflops']:6.1f} TFLOP/s sclk={row['sclk_mhz_median']} "
                          f"P={row['power_w_median']} tj={row['junction_c_median']}" + (f" bitwise={row['bitwise_equal_reference']}" if dump is not None else ""), flush=True)
    ref = variants[0]["name"]
    summary = []
    for tensor, _ in shapes:
        for v in variants:
            mine = {r["round"]: r for r in rows if r["tensor"] == tensor and r["variant"] == v["name"]}
            base = {r["round"]: r for r in rows if r["tensor"] == tensor and r["variant"] == ref}
            ratios = [mine[k]["tflops"] / base[k]["tflops"] for k in mine]
            summary.append(dict(tensor=tensor, variant=v["name"], tflops_median=statistics.median(r["tflops"] for r in mine.values()),
                                tflops_min=min(r["tflops"] for r in mine.values()), tflops_max=max(r["tflops"] for r in mine.values()),
                                ratio_vs_ref_median=statistics.median(ratios), ratio_vs_ref_min=min(ratios), ratio_vs_ref_max=max(ratios),
                                sclk_mhz_median=statistics.median(r["sclk_mhz_median"] for r in mine.values()),
                                bitwise_equal_reference=mine[0].get("bitwise_equal_reference")))
    (out/"summary.json").write_text(json.dumps(dict(reference=ref, rows=summary), indent=1) + "\n")
    print(f"\nratio vs {ref} (median [min..max] over {a.rounds} rounds)")
    for s in summary:
        print(f"  {s['variant']:10s} {s['tensor']:26s} {s['tflops_median']:6.1f} TFLOP/s  x{s['ratio_vs_ref_median']:.3f} "
              f"[{s['ratio_vs_ref_min']:.3f}..{s['ratio_vs_ref_max']:.3f}]  sclk={s['sclk_mhz_median']}"
              + (f" bitwise={s['bitwise_equal_reference']}" if s["bitwise_equal_reference"] is not None else ""))


if __name__ == "__main__":
    main()
