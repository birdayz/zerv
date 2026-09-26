#!/usr/bin/env python3
"""Market and serving summary per open-weight model, from saved snapshots.

    python3 tools/model_market_summary.py docs/research/2026-09-26-bigger-models

Reads openrouter-models.json, huggingface-models.json, inferencex-profit-estimator.json
and inferencex-deployments.json from the directory and prints the markdown tables of
docs/research/2026-09-26-bigger-models.md. Standard library only.
"""

import json
import statistics
import sys
from pathlib import Path

# OpenRouter id -> (Hugging Face id, InferenceX display model or None)
MODELS = {
    "qwen/qwen3.8-27b": ("Qwen/Qwen3.8-27B", None),
    "qwen/qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B", None),
    "qwen/qwen3.5-122b-a10b": ("Qwen/Qwen3.5-122B-A10B", None),
    "qwen/qwen3.5-397b-a17b": ("Qwen/Qwen3.5-397B-A17B", "Qwen-3.5-397B-A17B"),
    "qwen/qwen3.8-2.4t-a95b": ("Qwen/Qwen3.8-2.4T-A95B", None),
    "qwen/qwen3.8-flash": ("Qwen/Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next"),
    "z-ai/glm-5.3-flash": ("zai-org/GLM-5.3-Flash", None),
    "z-ai/glm-5.3": ("zai-org/GLM-5.3", "GLM-5.2"),
    "moonshotai/kimi-k3": ("moonshotai/Kimi-K3", "Kimi-K3"),
    "minimax/minimax-m3": ("MiniMaxAI/Minimax-M3", "MiniMax-M3"),
    "deepseek/deepseek-v4.1-flash": ("deepseek-ai/DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash"),
    "deepseek/deepseek-v4-flash-0731": ("deepseek-ai/DeepSeek-V4-Flash-0731", None),
    "deepseek/deepseek-v4-pro-0813": ("deepseek-ai/DeepSeek-V4-Pro-0813", "DeepSeek-V4-Pro"),
}
IX_KEYS = {"Kimi-K3": "kimik3", "GLM-5.2": "glm5.2", "MiniMax-M3": "minimaxm3",
           "DeepSeek-V4.1-Flash": "dsv41flash", "DeepSeek-V4-Pro": "dsv4",
           "Qwen-3.5-397B-A17B": "qwen3.5", "Qwen3.8-Flash-Next": "qwen3.8next"}


def load(directory, name):
    with open(Path(directory) / name, encoding="utf-8") as f:
        return json.load(f)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def market_rows(orm, hf):
    rows = []
    for oid, (hid, _) in MODELS.items():
        m = orm["models"][oid]
        full = [d for d in m["daily"] if d["date"] < "2026-09-26"]
        spend30 = sum(d["total_usage"] for d in full) * 30 / len(full)
        prompt = sum(d["total_prompt_tokens"] for d in full)
        completion = sum(d["total_completion_tokens"] for d in full)
        cached = sum(d["total_native_tokens_cached"] for d in full)
        outs = [float(e["output"]) * 1e6 for e in m["endpoints"]]
        ins = [float(e["input"]) * 1e6 for e in m["endpoints"]]
        h = hf["models"][hid]
        total = (h.get("safetensors") or {}).get("total")
        rows.append([oid, h.get("license"), f"{total / 1e9:,.0f}B" if total else "-",
                     len(m["endpoints"]), f"{statistics.median(ins):.2f}",
                     f"{min(outs):.2f} / {statistics.median(outs):.2f} / {max(outs):.2f}",
                     f"{spend30 / 1e3:,.0f}k", f"{sum(d['count'] for d in full) / len(full) / 1e6:.2f}M",
                     f"{prompt / completion:.0f}", f"{cached / prompt:.0%}"])
    return rows


def serving_rows(pe, deploy):
    rows = []
    for model, view in sorted(pe["views"].items()):
        p = view["params"]
        price = view["pricing"]
        best = {}
        for r in view["data"]["rows"]:
            hw = r["hwKey"].split("_")[0]
            if r["revenuePerGpuHour"] > best.get(hw, (0, ""))[0]:
                best[hw] = (r["revenuePerGpuHour"], r["hwKey"].split("_", 1)[1], r["tco"])
        amd = best.get("mi355x")
        nv8 = max((best[k] for k in ("b200", "b300") if k in best), default=None)
        rack = max((best[k] for k in ("gb200", "gb300") if k in best), default=None)
        sizes = sorted({d[1] for k, v in deploy["deployments"].items()
                        if k.startswith(IX_KEYS[model] + "|mi355x|") for d in v if d[1]})
        rows.append([model, f"{p['target']}", f"{price['inputPerMillion']:.3g} / {price['cachedInputPerMillion']:.2g} / {price['outputPerMillion']:.3g}",
                     f"{amd[0]:.2f} ({amd[1]})" if amd else "no row",
                     f"{amd[0] / amd[2]:.1f}x" if amd else "-",
                     f"{nv8[0]:.2f} ({nv8[1]})" if nv8 else "-",
                     f"{rack[0]:.2f} ({rack[1]})" if rack else "-",
                     ", ".join(str(s) for s in sizes) or "-"])
    return rows


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} DATA_DIR", file=sys.stderr)
        return 2
    orm, hf = load(argv[1], "openrouter-models.json"), load(argv[1], "huggingface-models.json")
    pe, deploy = load(argv[1], "inferencex-profit-estimator.json"), load(argv[1], "inferencex-deployments.json")
    print("### Market (OpenRouter, 30 days to 2026-09-25; price USD per 1M)\n")
    print(table(["model", "license", "params", "providers", "input median", "output min / median / max",
                 "spend per 30 d", "requests/day", "prompt:completion", "cached"], market_rows(orm, hf)))
    print("\n### Serving (InferenceX profit estimator: agentic traces, OpenRouter default price, "
          "revenue per GPU-hour at 100% busy, USD)\n")
    print(table(["InferenceX model", "target tok/s/user", "price in / cached / out",
                 "MI355X best", "MI355X revenue / InferenceX rent", "best 8-GPU NVIDIA (B200/B300)",
                 "best rack-scale NVIDIA (GB200/GB300)", "MI355X GPUs per decode replica"],
                serving_rows(pe, deploy)))
    return 0


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    sys.exit(main(sys.argv))
