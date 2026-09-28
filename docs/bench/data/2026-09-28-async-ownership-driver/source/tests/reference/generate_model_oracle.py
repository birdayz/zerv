#!/usr/bin/env python3
"""Qwen3.8 execution oracle: libllama intermediate capture + independent FP64 NumPy forward on
identical token sequences. Large tensors go to a fresh --work-dir (under third_party); a small
hashed summary fixture goes to --output. Development tool only.

  tools/py tests/reference/generate_model_oracle.py --model GGUF --work-dir DIR --output FILE [--case-set long]

The capture program is //tests:oracle_model (llama.cpp and ggml at the pinned revisions, built
from source with the Vulkan backend); it runs on the test-only GPU runtime (the Vulkan loader
and Mesa RADV built from source, docs/specs/hermetic-build.md phase 4). Prompts are rendered by
render_chat.py under the same pinned interpreter.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

# One BLAS thread per process (set before numpy loads OpenBLAS): the FP64 reference runs its
# row work in a pool of worker processes (qwen35_reference.Projector); OpenBLAS would otherwise
# start one thread per CPU in each of them (22 workers x 24 threads measured on 2026-09-26).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_capture
import qwen35_reference as ref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import host_info  # noqa: E402  (tools/host_info.py)
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds, the GPU runtime)

TOKENIZER_CONFIG = ROOT/"third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json"
FULL = """model.input_embed attn_norm linear_attn_qkv_mixed z beta alpha beta_sigmoid a_softplus gate
conv_output_raw conv_output_silu q_conv_predelta k_conv_predelta attn_output final_output linear_attn_out
Qcur_full Kcur Vcur Qcur_normed Kcur_normed Qcur attn_pregate gate_sigmoid attn_gated attn_residual
attn_post_norm ffn_gate ffn_up ffn_swiglu ffn_out l_out result_norm""".split()
LONG = "attn_norm Kcur Vcur attn_output attn_gated final_output attn_residual ffn_out l_out result_norm".split()
CASES = [
    dict(name="short-nothink", generate=24, capture=FULL,
         messages=[{"role": "user", "content": "What is the capital of France? Answer with one word."}],
         options={"enable_thinking": False}),
    dict(name="long-think", generate=40, capture=LONG,
         messages=[{"role": "system", "content": "You are a careful assistant who explains mechanical systems clearly."},
                   {"role": "user", "content": (
                       "I ride a road bicycle with two chainrings in front and eleven cogs in the back. "
                       "When I climb steep hills I shift to the small chainring and the largest cog, but "
                       "on long descents I use the big chainring and the smallest cog. A friend told me "
                       "that crossing the chain, for example big chainring with biggest cog, wears the "
                       "drivetrain faster and can make noise. Can you explain why cross-chaining happens "
                       "to be harmful, how gear ratios are calculated from tooth counts, and give me a "
                       "simple rule of thumb for choosing gears on rolling terrain?")}],
         options={"enable_thinking": True}),
]

# A prompt longer than one 512-row prefill chunk (block 13h): the native gate then runs a
# full 512-row chunk plus a remainder. Only the tensors the native tool captures at
# chunk sizes >= 256 are recorded (host capture budget).
LONG_PROMPT = " ".join([
    "The town of Aldermoor sits in a narrow valley where two rivers meet, and for most of its history the town drew",
    "its drinking water from shallow wells dug beside the eastern river. In the spring floods the wells often turned",
    "cloudy, and the town council eventually decided to build a proper water system. The engineers they hired proposed",
    "a reservoir on the hillside north of the town, fed by a small dam across a mountain stream, with a slow sand",
    "filter and a covered storage tank. Water would flow downhill by gravity through cast iron mains, so no pumps would",
    "be needed except during the driest weeks of late summer. The council debated the plan for two years. Farmers",
    "upstream worried that the dam would take water they needed for their fields, so the engineers agreed to release",
    "a fixed minimum flow at all times and to measure it with a weir that anyone could inspect. Shopkeepers worried",
    "about the cost, so the council borrowed the money over thirty years and set a water rate based on the size of",
    "each connection rather than on metered use, because meters were expensive and unreliable at the time. Doctors in",
    "the town argued strongly for the filter, pointing out that outbreaks of fever had followed nearly every flood.",
    "Construction took four summers. The first summer was spent clearing the reservoir site and building a temporary",
    "road for the carts that carried stone and cement. In the second summer the dam was raised to half its final height,",
    "and an unexpected storm in August washed out part of the cofferdam, delaying the work by six weeks. The third summer",
    "saw the completion of the dam, the filter beds, and the storage tank, which was built partly underground to keep",
    "the water cool. In the final summer the mains were laid through the streets, trenches were dug by hand, and each",
    "house that wanted a connection paid a fee for a lead service pipe and a brass stop valve. When the system opened,",
    "the town held a celebration at the fountain in the market square, which ran continuously for a whole day. Over the",
    "following decade the number of fever cases fell sharply, the fire brigade began to rely on hydrants instead of",
    "bucket lines, and a brewery and a tannery moved to the town because of the reliable supply. Problems appeared",
    "later. The lead service pipes slowly corroded, the sand filter needed more frequent cleaning as the town grew,",
    "and in very dry years the minimum release to the farmers left the reservoir dangerously low. The council responded",
    "by adding a second, larger reservoir, replacing the lead pipes over twenty years, and eventually installing meters",
    "so that heavy users paid for what they consumed.",
])
LONG_CASES = [
    dict(name="long-prefill", generate=16, capture=["l_out", "attn_output", "result_norm"],
         messages=[{"role": "user", "content": LONG_PROMPT + "\n\nSummarize the history of the Aldermoor water system in three sentences."}],
         options={"enable_thinking": False}),
]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def metrics(a, b):
    diff = a - b
    nb = float(np.linalg.norm(b))
    return dict(normalized_l2=float(np.linalg.norm(diff))/max(nb, 1e-30), max_abs=float(np.max(np.abs(diff))),
                max_abs_reference=float(np.max(np.abs(b))), nonfinite=int(np.count_nonzero(~np.isfinite(a))))


def softmax(x):
    e = np.exp(x - x.max())
    return e/e.sum()


def write_tensors(directory, prefix, tensors):
    """Raw little-endian float32 blob + index (no pickle)."""
    index, offset = [], 0
    with (directory/(prefix+".bin")).open("xb") as f:
        for name, value in tensors.items():
            data = np.ascontiguousarray(value, dtype="<f4")
            f.write(data.tobytes())
            index.append(dict(name=name, shape=list(data.shape), offset=offset, sha256=hashlib.sha256(data.tobytes()).hexdigest()))
            offset += data.nbytes
    (directory/(prefix+".json")).write_text(json.dumps(index, indent=1)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-set", choices=("default", "long"), default="default",
                   help="default: short-nothink + long-think; long: one prompt longer than a 512-row chunk")
    a = p.parse_args()
    cases = CASES if a.case_set == "default" else LONG_CASES
    work = a.work_dir.resolve()
    if not work.is_relative_to(ROOT/"third_party"): p.error("work dir must be under third_party")
    if work.exists() or a.output.exists(): p.error("fresh work dir and output required")
    started = datetime.now(timezone.utc).isoformat()
    if sha(a.model) != ref.MODEL_SHA: raise ValueError("model mismatch")
    chat = json.loads((ROOT/"tests/fixtures/chat-template.json").read_text())
    if sha(TOKENIZER_CONFIG) != chat["config_sha256"]: raise ValueError("template config changed")
    renderer = [sys.executable, str(ROOT/"tests/reference/render_chat.py")]
    built, oracle = zerv_build.oracle("oracle_model")
    gpu_env, runtime = zerv_build.gpu_runtime()
    work.mkdir(parents=True)
    binary = work/"model_oracle"
    shutil.copy2(built, binary)
    env = {k: v for k, v in gpu_env.items() if not k.startswith(("GGML_", "LLAMA_"))}
    env["GGML_VK_DISABLE_MMVQ"] = "1"  # FP32 activations into every matvec
    deps = host_info.library_record(binary, env)
    fixture = dict(schema_version=2, model_sha256=ref.MODEL_SHA,
                   sources={name: sha(ROOT/"tests/reference"/name) for name in
                            ("generate_model_oracle.py", "qwen35_reference.py", "model_capture.py", "model_oracle.c", "generate_chat_goldens.py", "render_chat.py")},
                   oracle=dict(label=oracle["label"], sha256=oracle["sha256"]), gpu_runtime=runtime,
                   tokenizer_config_sha256=chat["config_sha256"], library_hashes={name: d["sha256"] for name, d in deps.items()},
                   numpy=np.__version__, python=sys.version.split()[0],
                   reference_environment={"GGML_VK_DISABLE_MMVQ": "1", "n_batch": 1, "n_ubatch": 1, "type_k": "F32", "type_v": "F32",
                                          "flash_attn": "disabled", "n_gpu_layers": 999, "greedy": "lowest index on ties"},
                   cases=[])
    sequences, captures, llama = [], [], []
    for case in cases:
        d = work/case["name"]; d.mkdir()
        prompt = subprocess.run(renderer, input=json.dumps(dict(messages=case["messages"], options=case["options"])),
                                text=True, capture_output=True, check=True).stdout
        (d/"prompt.txt").write_text(prompt)
        (d/"names.txt").write_text("\n".join(case["capture"])+"\n")
        cmd = [str(binary), str(a.model), str(d/"prompt.txt"), str(d/"names.txt"), str(d), str(case["generate"]), "1024"]
        r = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=3600)
        (d/"oracle.stderr").write_text(r.stderr)
        if r.returncode: raise RuntimeError("capture failed: "+r.stderr[-2000:])
        cap = model_capture.load(d)
        llama.append(cap)
        sequences.append(cap["tokens"]["tokens"])
        captures.append(set(case["capture"]) | ({"Kcur_roped"} if "Kcur" in case["capture"] else set()))
        print(case["name"], "tokens", len(cap["tokens"]["tokens"]), flush=True)
    results = ref.forward_many(ref.Weights(a.model, verify_hash=False), sequences, captures)
    for case, cap, (out, logits, state) in zip(cases, llama, results):
        d = work/case["name"]
        write_tensors(d, "fp64-reference", out)
        write_tensors(d, "fp64-state", state)
        np.ascontiguousarray(logits, dtype="<f4").tofile(d/"fp64-logits.bin")
        per_name, per_layer = {}, {}
        missing = sorted(set(cap["tensors"]) ^ set(out))
        if missing: raise ValueError("capture/reference name mismatch: "+", ".join(missing[:10]))
        for name, per in cap["tensors"].items():
            base = name.rsplit("-", 1)[0] if name[-1].isdigit() else name
            for t in per:
                m = metrics(model_capture.tensor(cap, name, t), out[name][t])
                if m["nonfinite"]: raise ValueError("nonfinite reference capture")
                if m["normalized_l2"] > 1e-3: raise ValueError(f"semantic mismatch {name} token {t}: {m}")
                w = per_name.setdefault(base, dict(normalized_l2=0.0, max_abs=0.0, samples=0))
                w["normalized_l2"] = max(w["normalized_l2"], m["normalized_l2"]); w["max_abs"] = max(w["max_abs"], m["max_abs"]); w["samples"] += 1
                if base == "l_out":
                    il = int(name.rsplit("-", 1)[1]); per_layer[il] = max(per_layer.get(il, 0.0), m["normalized_l2"])
        positions = []
        lg = np.asarray(cap["logits"], dtype=np.float64)
        tokens = cap["tokens"]["tokens"]
        for t in range(len(tokens)):
            ref_row = logits[t]; top = np.argsort(-ref_row, kind="stable")[:8]
            pr, pl = softmax(ref_row), softmax(lg[t])
            positions.append(dict(position=t, next_token=tokens[t+1] if t+1 < len(tokens) else None,
                                  reference_top=[[int(i), float(ref_row[i])] for i in top], margin=float(ref_row[top[0]]-ref_row[top[1]]),
                                  llama_argmax=int(lg[t].argmax()), **metrics(lg[t], ref_row),
                                  kl=float(np.sum(pr*(np.log(np.maximum(pr, 1e-300))-np.log(np.maximum(pl, 1e-300)))))))
        n_prompt = cap["tokens"]["n_prompt"]
        disagreements = [x["position"] for x in positions if x["llama_argmax"] != x["reference_top"][0][0]]
        files = {str(f.relative_to(work)): dict(sha256=sha(f), bytes=f.stat().st_size) for f in sorted(d.iterdir()) if f.is_file()}
        fixture["cases"].append(dict(name=case["name"], messages=case["messages"], options=case["options"], prompt=(d/"prompt.txt").read_text(),
                                     prompt_sha256=sha(d/"prompt.txt"), n_prompt=n_prompt, generate=case["generate"], tokens=tokens,
                                     capture=case["capture"], llama_vs_fp64=per_name, l_out_by_layer=[per_layer[i] for i in sorted(per_layer)],
                                     positions=positions, argmax_disagreements=disagreements, files=files))
        worst = max(v["normalized_l2"] for v in per_name.values())
        print(case["name"], "worst intermediate", f"{worst:.3e}", "logits", f"{max(x['normalized_l2'] for x in positions):.3e}",
              "argmax disagreements", disagreements, flush=True)
    (work/"run-manifest.json").write_text(json.dumps(dict(started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                                                          argv=sys.argv, library_dependencies=deps), indent=1)+"\n")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f: json.dump(fixture, f, indent=1); f.write("\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
