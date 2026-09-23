#!/usr/bin/env python3
"""f16 WMMA prefill GEMM component benchmark on the real model weights (block 16b).

Runs zerv-gemm-bench on every f16-eligible shape with a sustained warm-up, while a
sampler thread records the GPU shader clock, power, junction temperature and vddgfx from
sysfs (medians over the timed phase). Writes raw JSON lines
plus a manifest. Nothing else may use the GPU meanwhile.
With --dump, every result Y is written and compared bytewise against --reference-dump
(the shipped kernel's dump), so variants that must be bit-identical are gated in the same
run that times them.
Usage: run_gemm_f16.py --output DIR [--spv-dir DIR --spv-pattern gemm_f16_{fmt}.spv]
       [--rows 512] [--warmup-ms 10000] [--dump] [--reference-dump DIR]"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
# f16-eligible projections (M >= 4096, M % 128 == 0) of the benchmark shape set.
SHAPES = [("blk.0.ffn_gate.weight", "q4_0"), ("blk.8.ffn_down.weight", "q4_0"), ("blk.0.ffn_down.weight", "q4_1"),
          ("blk.0.attn_qkv.weight", "q4_0"), ("blk.0.attn_gate.weight", "q4_0"), ("blk.0.ssm_out.weight", "q5_k"),
          ("blk.3.attn_q.weight", "q4_0"), ("blk.3.attn_output.weight", "q4_0")]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def device_dir():
    return next(p for p in sorted(Path("/sys/class/drm").glob("card*/device")) if (p/"pp_dpm_sclk").exists())


class Sampler(threading.Thread):
    def __init__(self, dev):
        super().__init__(daemon=True)
        self.hw = next((dev/"hwmon").glob("hwmon*"))
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self.samples.append((time.time(), int((self.hw/"freq1_input").read_text()) / 1e6, int((self.hw/"power1_average").read_text()) / 1e6,
                                     int((self.hw/"temp2_input").read_text()) / 1e3, int((self.hw/"in0_input").read_text())))
            except OSError:
                pass
            time.sleep(0.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spv-dir", type=Path, default=ROOT/"src/model/shaders", help="directory with gemm_f16_<format>.spv")
    p.add_argument("--rows", default="512")
    p.add_argument("--warmup-ms", type=int, default=10000)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--only", help="comma list of tensor names")
    p.add_argument("--spv-pattern", default="gemm_f16_{fmt}.spv")
    p.add_argument("--grid", default="128x128", help="workgroup tile MxROWS of the module")
    p.add_argument("--wave", type=int, help="required subgroup size (VK_EXT_subgroup_size_control)")
    p.add_argument("--x-f16", action="store_true", help="upload X pre-rounded to f16 (kernels reading f16 X)")
    p.add_argument("--dump", action="store_true", help="write each Y to OUTPUT/dump")
    p.add_argument("--reference-dump", type=Path, help="compare dumps bytewise against this directory")
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if sha(a.model) != MODEL_SHA: raise SystemExit("model mismatch")
    subprocess.run([str(ROOT/".tools/zig-x86_64-linux-0.16.0/zig"), "build", "gemm-bench-build", "-Doptimize=ReleaseFast", "-Dcpu=native"], cwd=ROOT, check=True)
    tool = ROOT/"zig-out/bin/zerv-gemm-bench"
    dev = device_dir()
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    env={k: v for k, v in __import__("os").environ.items() if k.startswith(("RADV_", "ACO_"))}, model_sha256=MODEL_SHA, tool_sha256=sha(tool),
                    spv={fmt: sha(a.spv_dir/a.spv_pattern.format(fmt=fmt)) for fmt in ("q4_0", "q4_1", "q5_k") if (a.spv_dir/a.spv_pattern.format(fmt=fmt)).exists()},
                    power_cap_w=int(next((dev/"hwmon").glob("hwmon*/power1_cap")).read_text()) / 1e6,
                    performance_level=(dev/"power_dpm_force_performance_level").read_text().strip(), runs=[])
    raw = (out/"raw.jsonl").open("w")
    only = set(a.only.split(",")) if a.only else None
    for tensor, fmt in SHAPES:
        if only and tensor not in only: continue
        cmd = [str(tool), str(a.model), "--coopmat", "1", "--variant", fmt, "--spv", str(a.spv_dir/a.spv_pattern.format(fmt=fmt)), "--grid", a.grid,
               "--k-chunk", "0", "--tensor", tensor, "--samples", str(a.samples), "--warmup-ms", str(a.warmup_ms)]
        if a.wave: cmd += ["--wave", str(a.wave)]
        if a.x_f16: cmd += ["--x-f16", "1"]
        if a.dump:
            (out/"dump").mkdir(exist_ok=True)
            cmd += ["--dump", str(out/"dump")]
        cmd += a.rows.split(",")
        sampler = Sampler(dev); sampler.start()
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        sampler.stop.set(); sampler.join()
        if r.returncode: raise SystemExit(f"{tensor}: {r.stderr[-2000:]}")
        # Clock/power over the sampling phase (after the warm-up) of each rows value.
        timed = [s for s in sampler.samples if s[0] >= t0 + a.warmup_ms / 1000]
        for line in r.stdout.splitlines():
            row = json.loads(line)
            row.update(sclk_mhz_median=statistics.median(s[1] for s in timed) if timed else None,
                       power_w_median=statistics.median(s[2] for s in timed) if timed else None,
                       junction_c_median=statistics.median(s[3] for s in timed) if timed else None,
                       vddgfx_mv_median=statistics.median(s[4] for s in timed) if timed else None)
            if a.reference_dump:
                name = f"{tensor}-{row['rows']}.bin"
                row["bitwise_equal_reference"] = (out/"dump"/name).read_bytes() == (a.reference_dump/name).read_bytes()
            raw.write(json.dumps(row) + "\n"); raw.flush()
            print(f"{tensor:26s} {fmt:5s} rows={row['rows']:5d} {row['tflops']:6.1f} TFLOP/s  sclk={row['sclk_mhz_median']} MHz power={row['power_w_median']} W tj={row['junction_c_median']} C vdd={row['vddgfx_mv_median']} mV"
                  + (f"  bitwise={row['bitwise_equal_reference']}" if a.reference_dump else ""), flush=True)
        manifest["runs"].append(dict(tensor=tensor, cmd=cmd))
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
