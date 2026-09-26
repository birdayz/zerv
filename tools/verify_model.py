#!/usr/bin/env python3
"""Block-10 model gate: rebuild the native capture tool, run every oracle case through
the native forward pass, and apply tests/reference/compare_model.py."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True, help="fresh directory under third_party")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--kv-capacity-mib", type=int, default=0, help="cap each KV buffer (forces a KV split; 0 = default)")
    p.add_argument("--kv-type", choices=("f32", "f16"), default="f32", help="KV cache type (f16: block 17c; not gated by the FP32 bounds)")
    p.add_argument("--kv-page-tokens", default="128", help="KV page tokens: a multiple of 128 or 'context' (docs/specs/concurrent.md)")
    p.add_argument("--matvec-accumulation", choices=("fma", "separate"), default="fma", help="decode/verify projection accumulation (docs/specs/matvec-push.md)")
    p.add_argument("--modes", default="0,1,13,512", help="comma list: 0 = decode steps, N = prefill chunk N, N:S = chunk N with a boundary at token S (prefix-cache restore)")
    p.add_argument("--precision", choices=("fp32", "f16"), default="fp32",
                   help="prefill projection arithmetic passed to the capture tool (f16: block 14; not gated by the FP32 bounds)")
    p.add_argument("--gemm-code", choices=("spirv", "native"), default="spirv",
                   help="f16 gemm_f16x machine code (docs/specs/prefill.md, 'Native gemm_f16x machine code')")
    p.add_argument("--f16-small-tile", choices=("on", "off"), default="on",
                   help="f16 32-row tile for short plans (Options.f16_small_tile, block 18c.2; bitwise the 128-row tile)")
    p.add_argument("--tool", type=Path,
                   help="archived zerv-model-capture binary (e.g. a previous work dir's copy) run instead of rebuilding; "
                        "for baseline reruns (the manifest's source hashes then describe the tree, not the tool)")
    p.add_argument("--fixture", type=Path, default=ROOT/"tests/fixtures/model/qwen38-oracle.json",
                   help="oracle summary fixture (qwen38-oracle-long.json for the >512-token case)")
    a = p.parse_args()
    work = a.work_dir.resolve()
    if not work.is_relative_to(ROOT/"third_party") or work.exists() or a.report.exists(): p.error("fresh work dir under third_party and fresh report required")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    if sha(a.model) != MODEL_SHA: raise SystemExit("model mismatch")
    work.mkdir(parents=True)
    if a.tool is None:
        import zerv_build
        build = zerv_build.build_command("zerv-model-capture")
        source = zerv_build.binary("zerv-model-capture")
    else:
        build = ["archived-tool", str(a.tool)]
        source = a.tool
    tool = work/"zerv-model-capture"
    tool.write_bytes(source.read_bytes()); tool.chmod(0o755)
    fixture = json.loads(a.fixture.read_text())
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, model_sha256=MODEL_SHA, tool_sha256=sha(tool),
                    sources={str(f.relative_to(ROOT)): sha(f) for d in ("src/model", "src/matvec", "src/gpu") for f in sorted((ROOT/d).rglob("*")) if f.is_file()},
                    fixture_sha256=sha(a.fixture), commands=[build], cases={})
    passed = True
    outputs = []
    for mode_arg in a.modes.split(","):
        mode = int(mode_arg.split(":")[0])
        natives = []
        for case in fixture["cases"]:
            d = work/f"{case['name']}-{mode_arg.replace(':', 's')}"; d.mkdir()
            (d/"tokens.json").write_text(json.dumps(dict(n_prompt=case["n_prompt"], n_vocab=248320, tokens=case["tokens"]))+"\n")
            names = list(case["capture"]) + (["Kcur_roped"] if "Kcur" in case["capture"] else [])
            if mode >= 256: names = [n for n in names if n in ("l_out", "attn_output", "result_norm")]  # host capture budget
            (d/"names.txt").write_text("\n".join(names)+"\n")
            ctx = str(a.context) + (f":{a.kv_capacity_mib}" if a.kv_capacity_mib else "") + (f"@{a.kv_type}" if a.kv_type != "f32" else "") + (f"@{a.matvec_accumulation}" if a.matvec_accumulation != "fma" else "") + (f"@page={a.kv_page_tokens}" if a.kv_page_tokens != "128" else "")
            cmd = [str(tool), str(a.model), str(d/"tokens.json"), str(d/"names.txt"), str(d), ctx]
            # Capture MiB: all intermediates are ~34.5 MB per row (host-visible buffer).
            if mode > 0: cmd += [mode_arg, "3000" if mode >= 256 else str(max(1800, 36*mode))] + ([a.precision + (f"@{a.gemm_code}" if a.gemm_code != "spirv" else "") + ("@small=off" if a.f16_small_tile == "off" else "")] if a.precision != "fp32" else [])
            manifest["commands"].append(cmd)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            (d/"stderr.txt").write_text(r.stderr)
            if r.returncode: raise SystemExit(case["name"]+f" native capture (mode {mode_arg}) failed:\n"+r.stderr[-2000:])
            manifest["cases"][f"{case['name']}-{mode_arg}"] = r.stderr.strip().splitlines()[-1]
            natives += ["--native", f"{case['name']}={d}"]
        report = a.report.with_name(a.report.stem+f"-mode{mode_arg.replace(':', 's')}"+a.report.suffix)
        compare = [sys.executable, str(ROOT/"tests/reference/compare_model.py"), "--fixture", str(a.fixture), "--oracle-dir", str(a.oracle_dir), *natives, "--output", str(report)]
        manifest["commands"].append(compare)
        r = subprocess.run(compare, capture_output=True, text=True)
        print(f"mode {mode_arg}:", r.stdout, r.stderr, flush=True)
        outputs.append(r.stdout)
        passed &= r.returncode == 0
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), compare_stdout=outputs, passed=passed)
    (work/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    a.report.write_text(json.dumps(dict(passed=passed, modes=a.modes, outputs=outputs), indent=1)+"\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
