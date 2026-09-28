#!/usr/bin/env python3
"""Log GPU, CPU, board and NVMe temperatures (plus GPU power, clock, fan and throttle
reasons) from sysfs at a fixed rate, optionally around a load command. Read-only: it
changes no fan, clock or power setting.

Sources (all sysfs):
- every hwmon device: temp*/fan* inputs with their labels (amdgpu, k10temp, gigabyte_wmi,
  nvme, ...);
- amdgpu `gpu_metrics` format 1.3 (SMU 13.0.0 / Navi 31): edge, hotspot, memory and VR
  temperatures, socket power, gfx clock, fan RPM, gfx voltage and the ASIC-independent
  throttle status. Layout: Linux v7.2 include/kgd_pp_interface.h `gpu_metrics_v1_3`;
  throttle bits: pm/swsmu/inc/amdgpu_smu.h (copies under third_party/linux-amdgpu/).

Usage: thermal_log.py --output FILE.jsonl [--seconds 60] [--hz 2]
       [--load "CMD ..."] [--cooldown 60]
With --load, CMD starts after 5 s of baseline; logging continues until it exits, then for
--cooldown seconds. A summary (min / median / max per signal, throttle-bit shares while
loaded) is printed and written to FILE.summary.json."""
import sys
import argparse
import json
import shlex
import statistics
import struct
import subprocess
import time
from pathlib import Path

GPU_METRICS_V1_3 = struct.Struct("<HBB6H3HHQQ7H7HIHHHHII4HQ3HHQ")
assert GPU_METRICS_V1_3.size == 120
# SMU_THROTTLER_*_BIT (amdgpu_smu.h, v7.2)
THROTTLE_BITS = {0: "PPT0", 1: "PPT1", 2: "PPT2", 3: "PPT3", 4: "SPL", 5: "FPPT", 6: "SPPT", 7: "SPPT_APU",
                 16: "TDC_GFX", 17: "TDC_SOC", 18: "TDC_MEM", 19: "TDC_VDD", 20: "TDC_CVIP", 21: "EDC_CPU", 22: "EDC_GFX", 23: "APCC",
                 32: "TEMP_GPU", 33: "TEMP_CORE", 34: "TEMP_MEM", 35: "TEMP_EDGE", 36: "TEMP_HOTSPOT", 37: "TEMP_SOC",
                 38: "TEMP_VR_GFX", 39: "TEMP_VR_SOC", 40: "TEMP_VR_MEM0", 41: "TEMP_VR_MEM1", 42: "TEMP_LIQUID0", 43: "TEMP_LIQUID1",
                 44: "VRHOT0", 45: "VRHOT1", 46: "PROCHOT_CPU", 47: "PROCHOT_GFX", 56: "PPM", 57: "FIT"}


def hwmon_sources():
    """(key, path, scale) for every temperature and fan input."""
    out = []
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = (h/"name").read_text().strip()
        for f in sorted(h.glob("temp*_input")) + sorted(h.glob("fan*_input")):
            label_file = f.with_name(f.name.replace("_input", "_label"))
            label = label_file.read_text().strip() if label_file.exists() else f.name.replace("_input", "")
            out.append((f"{name}/{h.name}/{label}", f, 1000.0 if f.name.startswith("temp") else 1.0))
    return out


def gpu_metrics_path():
    for card in sorted(Path("/sys/class/drm").glob("card*/device/gpu_metrics")):
        head = card.read_bytes()[:4]
        if len(head) == 4 and struct.unpack("<HBB", head)[1:] == (1, 3):
            return card
    return None


def read_gpu_metrics(path):
    raw = path.read_bytes()
    if len(raw) < GPU_METRICS_V1_3.size: return None
    v = GPU_METRICS_V1_3.unpack_from(raw)
    (size, fmt, content, t_edge, t_hot, t_mem, t_vrgfx, t_vrsoc, t_vrmem, gfx_act, umc_act, mm_act, power, _energy, _clock,
     *rest) = v
    avg_clocks, cur_clocks = rest[0:7], rest[7:14]
    throttle, fan, _lw, _ls, _pad, _gacc, _macc = rest[14:21]
    _hbm = rest[21:25]
    _fwts, v_soc, v_gfx, v_mem, _pad1, indep = rest[25:31]
    return dict(edge_c=t_edge, hotspot_c=t_hot, mem_c=t_mem, vrgfx_c=t_vrgfx, vrsoc_c=t_vrsoc, vrmem_c=t_vrmem,
                gfx_activity=gfx_act, umc_activity=umc_act, socket_power_w=power, gfxclk_mhz=cur_clocks[0], uclk_mhz=cur_clocks[2],
                fan_rpm=fan, vgfx_mv=v_gfx, throttle=[n for b, n in THROTTLE_BITS.items() if indep >> b & 1], indep_throttle=indep)


def sample(sources, metrics):
    row = dict(t=time.time())
    for key, path, scale in sources:
        try:
            row[key] = int(path.read_text()) / scale
        except OSError:
            row[key] = None
    if metrics:
        row["gpu"] = read_gpu_metrics(metrics)
    return row


def summarize(rows, loaded):
    keys = [k for k in rows[0] if k not in ("t", "gpu", "phase")] + [f"gpu.{k}" for k in (rows[0].get("gpu") or {}) if k not in ("throttle", "indep_throttle")]

    def values(k, subset):
        out = []
        for r in subset:
            v = r["gpu"].get(k[4:]) if k.startswith("gpu.") and r.get("gpu") else r.get(k)
            if isinstance(v, (int, float)): out.append(v)
        return out
    summary = {}
    for phase in ("idle", "baseline", "load", "cooldown"):
        subset = [r for r in rows if r["phase"] == phase]
        if not subset: continue
        summary[phase] = {k: dict(min=min(v), median=statistics.median(v), max=max(v)) for k in keys if (v := values(k, subset))}
    load_rows = [r for r in rows if r["phase"] == loaded and r.get("gpu")]
    if load_rows:
        counts = {}
        for r in load_rows:
            for name in r["gpu"]["throttle"]: counts[name] = counts.get(name, 0) + 1
        summary["throttle_share_" + loaded] = {k: v / len(load_rows) for k, v in sorted(counts.items())}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60, help="duration without --load")
    p.add_argument("--hz", type=float, default=2)
    p.add_argument("--load", help="command to run as the load (shell-split)")
    p.add_argument("--cooldown", type=float, default=60)
    a = p.parse_args()
    if a.output.exists(): p.error("fresh output required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sources, metrics = hwmon_sources(), gpu_metrics_path()
    rows, period = [], 1.0 / a.hz
    proc, phase, t0 = None, "baseline" if a.load else "idle", time.time()
    load_end = None
    with a.output.open("w") as f:
        while True:
            now = time.time()
            if a.load:
                if proc is None and now - t0 >= 5:
                    proc = subprocess.Popen(shlex.split(a.load), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    phase = "load"
                if proc is not None and load_end is None and proc.poll() is not None:
                    load_end, phase = now, "cooldown"
                if load_end is not None and now - load_end >= a.cooldown: break
            elif now - t0 >= a.seconds:
                break
            row = sample(sources, metrics)
            row["phase"] = phase
            rows.append(row)
            f.write(json.dumps(row) + "\n"); f.flush()
            time.sleep(max(0.0, period - (time.time() - now)))
    summary = summarize(rows, "load" if a.load else "idle")
    summary["load"] = a.load
    summary["load_exit"] = proc.returncode if proc else None
    summary["samples"] = len(rows)
    a.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for phase, stats in summary.items():
        if not isinstance(stats, dict): continue
        print(f"== {phase}")
        for k, s in stats.items():
            print(f"  {k:45s} " + (f"{s['min']:8.1f} {s['median']:8.1f} {s['max']:8.1f}" if isinstance(s, dict) else f"{s:.2f}"))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
