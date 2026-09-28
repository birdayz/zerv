#!/usr/bin/env python3
"""Apply the declared block-10 gates (docs/specs/model.md) to a native capture.

Inputs: the committed oracle fixture, its work dir (FP64 reference + libllama capture
per case) and one native capture dir per case (zerv-model-capture output).
Writes a JSON report; exits nonzero if any gate fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_capture

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def load_reference(directory, prefix):
    index = json.loads((directory/(prefix+".json")).read_text())
    blob = np.memmap(directory/(prefix+".bin"), dtype="<f4", mode="r")
    return {e["name"]: np.asarray(blob[e["offset"]//4:e["offset"]//4+int(np.prod(e["shape"]))], dtype=np.float64).reshape(e["shape"]) for e in index}


def nl2(a, b):
    return float(np.linalg.norm(a-b))/max(float(np.linalg.norm(b)), 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=ROOT/"tests/fixtures/model/qwen38-oracle.json")
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--native", action="append", required=True, help="CASE=DIR")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fixture = json.loads(a.fixture.read_text())
    natives = dict(item.split("=", 1) for item in a.native)
    report = dict(fixture_sha256=sha(a.fixture), cases=[], passed=True)
    for case in fixture["cases"]:
        if case["name"] not in natives: raise SystemExit("missing native capture for "+case["name"])
        work = a.oracle_dir/case["name"]
        for name, record in case["files"].items():
            if sha(a.oracle_dir/name) != record["sha256"]: raise SystemExit("oracle file changed: "+name)
        ref = load_reference(work, "fp64-reference")
        ref_logits = np.fromfile(work/"fp64-logits.bin", dtype="<f4").astype(np.float64).reshape(len(case["tokens"]), -1)
        llama = model_capture.load(work)
        native_dir = Path(natives[case["name"]])
        native = model_capture.load(native_dir)
        if native["tokens"]["tokens"] != case["tokens"]: raise SystemExit("native ran different tokens")
        summary = json.loads((native_dir/"native-summary.json").read_text())
        failures = []
        if summary["plain_vs_capture_logit_mismatches"] or summary["reset_replay_logit_mismatches"]:
            failures.append("stateful determinism (gate 4)")
        names = sorted(set(native["tensors"]) & set(ref))
        requested = set((native_dir/"names.txt").read_text().split())
        missing = sorted(n for n in set(ref) - set(native["tensors"]) if (n.rsplit("-", 1)[0] if n[-1].isdigit() else n) in requested)
        if missing: failures.append("native lacks captured tensors: "+", ".join(missing[:8]))
        prefill = summary.get("mode") == "prefill"
        logit_positions = summary["logit_positions"] if prefill else list(range(len(case["tokens"])))
        for name in names:
            want = set(logit_positions) if (prefill and name == "result_norm") else set(range(len(case["tokens"])))
            if set(native["tensors"][name]) != want: failures.append(f"{name}: captured tokens differ from expected")
        per_name = {}
        for name in names:
            base = name.rsplit("-", 1)[0] if name[-1].isdigit() else name
            bound = max(4*case["llama_vs_fp64"][base]["normalized_l2"], 2e-6)
            for t in native["tensors"][name]:
                x = model_capture.tensor(native, name, t)
                if not np.all(np.isfinite(x)): failures.append(f"nonfinite {name} t={t}")
                e = nl2(x, ref[name][t])
                w = per_name.setdefault(base, dict(worst=0.0, bound=bound, llama=case["llama_vs_fp64"][base]["normalized_l2"], samples=0))
                w["worst"] = max(w["worst"], e); w["samples"] += 1
                if base == "model.input_embed":
                    if not np.array_equal(x, ref[name][t]): failures.append(f"embedding not exact t={t}")
                elif e > bound: failures.append(f"{name} t={t}: {e:.3e} > {bound:.3e}")
        nat_logits = np.asarray(native["logits"], dtype=np.float64)
        llama_logits = np.asarray(llama["logits"], dtype=np.float64)
        llama_logit_err = max(x["normalized_l2"] for x in case["positions"])
        logit_bound = max(4*llama_logit_err, 2e-6)
        margin_rule = 10*max(float(np.max(np.abs(llama_logits[t]-ref_logits[t]))) for t in range(len(case["tokens"])))
        positions, near_ties = [], []
        rows = [(t, nat_logits[t]) for t in logit_positions]
        if prefill:
            serving = np.fromfile(native_dir/"serving-logits.bin", dtype="<f4").astype(np.float64).reshape(-1, ref_logits.shape[1])
            first = summary["serving_first_position"]
            if len(serving) != len(case["tokens"]) - first: failures.append("serving logits count")
            rows += [(first+i, serving[i]) for i in range(len(serving))]
        for t, native_row in rows:
            nat_logits_t = native_row
            e = nl2(nat_logits_t, ref_logits[t])
            top = np.argsort(-ref_logits[t], kind="stable")[:2]
            margin = float(ref_logits[t][top[0]]-ref_logits[t][top[1]])
            agree = int(nat_logits_t.argmax()) == int(top[0])
            positions.append(dict(position=t, normalized_l2=e, max_abs=float(np.max(np.abs(nat_logits_t-ref_logits[t]))), margin=margin,
                                  native_argmax=int(nat_logits_t.argmax()), fp64_argmax=int(top[0]), llama_argmax=int(llama_logits[t].argmax())))
            if e > logit_bound: failures.append(f"logits t={t}: {e:.3e} > {logit_bound:.3e}")
            if margin > margin_rule and not agree: failures.append(f"greedy mismatch t={t} margin {margin:.4f}")
            if margin <= margin_rule: near_ties.append(t)
        result = dict(name=case["name"], mode=summary.get("mode", "decode"), chunk=summary.get("chunk", 0), tokens=len(case["tokens"]), checked_logit_rows=len(positions), per_name=per_name, logit_bound=logit_bound, margin_rule=margin_rule,
                      worst_logit_normalized_l2=max(x["normalized_l2"] for x in positions), near_ties=near_ties,
                      greedy_agreement=sum(x["native_argmax"] == x["fp64_argmax"] for x in positions), positions=positions,
                      native_summary={k: v for k, v in summary.items() if not k.endswith("step_ns")},
                      native_files={f.name: sha(f) for f in sorted(native_dir.iterdir()) if f.is_file()}, failures=failures)
        report["cases"].append(result)
        report["passed"] &= not failures
        worst = max(v["worst"]/v["bound"] for k, v in per_name.items() if k != "model.input_embed")
        print(case["name"], "tensors", len(names), "worst/bound", f"{worst:.3f}", "logits", f"{result['worst_logit_normalized_l2']:.3e}/{logit_bound:.3e}",
              "greedy", f"{result['greedy_agreement']}/{len(positions)}", "mode", result["mode"], result["chunk"], "near ties", near_ties, "failures", len(failures))
        for f in failures[:10]: print("  FAIL", f)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1)+"\n")
    if not report["passed"]: sys.exit(1)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
