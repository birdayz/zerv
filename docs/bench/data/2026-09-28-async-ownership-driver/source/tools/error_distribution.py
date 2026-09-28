#!/usr/bin/env python3
"""Per-tensor error distributions of native captures against the FP64 oracle reference.

Complements the pass/fail gates of tests/reference/compare_model.py: for each capture
directory (zerv-model-capture output for one oracle case) and each tensor base name,
reports mean / median / p90 / max normalized L2 against FP64 over all (token, layer)
samples, plus the same for logits (decode-mode captures: every position). Use it to
judge whether a kernel change is systematically less accurate or only moves the worst
sample. Also reports the libllama-vs-FP64 distribution as a yardstick.

Usage: error_distribution.py --oracle-dir DIR --case NAME --run LABEL=CAPTURE_DIR [...]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests/reference"))
import model_capture  # noqa: E402
from compare_model import load_reference, nl2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(v.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--run", action="append", required=True, help="LABEL=CAPTURE_DIR")
    p.add_argument("--names", default="", help="comma list of tensor base names (default: all common)")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    work = a.oracle_dir/a.case
    ref = load_reference(work, "fp64-reference")
    ref_logits = np.fromfile(work/"fp64-logits.bin", dtype="<f4").astype(np.float64)
    runs = {label: model_capture.load(Path(d)) for label, d in (r.split("=", 1) for r in a.run)}
    runs["llama"] = model_capture.load(work)
    vocab = ref_logits.size // len(runs["llama"]["logits"])
    ref_logits = ref_logits.reshape(-1, vocab)
    base = lambda n: n.rsplit("-", 1)[0] if n[-1].isdigit() else n  # noqa: E731
    common = set(ref)
    for cap in runs.values(): common &= set(cap["tensors"])
    wanted = set(a.names.split(",")) if a.names else None
    report = {}
    for label, cap in runs.items():
        per = {}
        for name in sorted(common):
            if wanted and base(name) not in wanted: continue
            for t in cap["tensors"][name]:
                per.setdefault(base(name), []).append(nl2(model_capture.tensor(cap, name, t), ref[name][t]))
        logits = np.asarray(cap["logits"], dtype=np.float64)
        rows = [nl2(logits[t], ref_logits[t]) for t in range(len(logits)) if np.all(np.isfinite(logits[t]))]
        report[label] = dict(tensors={k: stats(v) for k, v in per.items()}, logits=stats(rows))
    labels = list(runs)
    print("tensor".ljust(24) + "".join(f"{lab+' mean':>14}{lab+' p90':>12}{lab+' max':>12}" for lab in labels))
    keys = sorted(report[labels[0]]["tensors"])
    for k in keys + ["logits"]:
        row = k.ljust(24)
        for lab in labels:
            s = report[lab]["logits"] if k == "logits" else report[lab]["tensors"][k]
            row += f"{s['mean']:14.3e}{s['p90']:12.3e}{s['max']:12.3e}"
        print(row)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(dict(case=a.case, runs={k: str(v) for k, v in (r.split("=", 1) for r in a.run)}, report=report), indent=1)+"\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
