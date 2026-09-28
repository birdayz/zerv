#!/usr/bin/env python3
"""Run pipeline_binary_lab race with a sysfs sampler (research tool, docs/research/native-isa-via-vulkan.md).

  race.py REF CAND [key=value ...] [--warm SECONDS] [--lab PATH]

REF is a SPIR-V module or a pipeline binary (.bin); CAND is a pipeline binary. Before the race,
an untimed warm-up race (--warm seconds, default 3) heats the card to its sustained state. Prints
the lab's JSON lines plus one line with the median shader clock (MHz), board power (W) and
junction temperature (C) sampled every 20 ms during the timed reps (samples with the shader
clock above 1000 MHz only; use batch=N so the GPU stays busy between fences). Refuses to run when another
GPU process of this project (zerv*, llama-server) is running.
"""
import json, os, pathlib, statistics, subprocess, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)
import host_info  # noqa: E402  (tools/host_info.py: the host, recorded without host tools)


def lab(args):
    """The lab executable: `--lab PATH` (removed from args), else built by Bazel."""
    if "--lab" in args:
        i = args.index("--lab"); path = args[i + 1]; del args[i:i + 2]
        return path
    return str(zerv_build.binary("pipeline_binary_lab", config=None))


def hwmon():
    for card in sorted(pathlib.Path("/sys/class/drm").glob("card[0-9]*")):
        hw = list((card / "device/hwmon").glob("hwmon*"))
        if hw and (hw[0] / "freq1_input").exists():
            return hw[0]
    raise SystemExit("no amdgpu hwmon")


class Sampler(threading.Thread):
    def __init__(self, hw):
        super().__init__(daemon=True)
        self.hw, self.stop, self.s = hw, threading.Event(), []

    def run(self):
        temp = next((p for p in self.hw.glob("temp*_label") if p.read_text().strip() == "junction"), None)
        while not self.stop.is_set():
            self.s.append((int((self.hw / "freq1_input").read_text()) / 1e6,
                           int((self.hw / "power1_average").read_text()) / 1e6,
                           int((temp.parent / temp.name.replace("_label", "_input")).read_text()) / 1e3 if temp else 0))
            time.sleep(0.02)


def busy():
    """Other GPU users of this project: the zerv server and tools, llama-server."""
    return host_info.processes(r"(^|/)(zerv(-[a-z0-9-]+)?|llama-server)( |$)")


def main():
    args = sys.argv[1:]
    exe = lab(args)
    warm = 3.0
    if "--warm" in args:
        i = args.index("--warm"); warm = float(args[i + 1]); del args[i:i + 2]
    ref, cand, kv = args[0], args[1], args[2:]
    b = busy()
    if b:
        raise SystemExit("GPU busy: " + "; ".join(b))
    if warm > 0:
        t0 = time.time()
        while time.time() - t0 < warm:
            subprocess.run([exe, "race", ref, cand, "reps=51"] + [x for x in kv if not x.startswith("reps=")],
                           check=True, capture_output=True)
    # The lab prints the bitwise line right before the timed reps: sample from then on.
    smp = Sampler(hwmon())
    pr = subprocess.Popen([exe, "race", ref, cand] + kv, stdout=subprocess.PIPE, text=True)
    out = []
    for line in pr.stdout:
        out.append(line)
        if '"check"' in line:
            smp.start()
    if pr.wait():
        raise SystemExit(f"lab failed: {pr.returncode}")
    if smp.is_alive():
        smp.stop.set(); smp.join()
    print("".join(out), end="")
    busy_s = [x for x in smp.s if x[0] > 1000]  # samples with the shader clock up
    if busy_s:
        f, p, t = zip(*busy_s)
        print(json.dumps({"sclk_mhz_median": statistics.median(f), "power_w_median": statistics.median(p),
                          "junction_c_median": statistics.median(t), "busy_samples": len(f), "samples": len(smp.s)}))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
