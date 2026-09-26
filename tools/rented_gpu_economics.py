#!/usr/bin/env python3
"""Could a faster engine on rented GPUs undercut providers that own theirs?

Prints the tables of docs/research/2026-09-26-rented-gpu-economics.md from an
input file whose numbers carry their provenance. Standard library only.

    python3 tools/rented_gpu_economics.py docs/research/2026-09-26-rented-gpu-economics/inputs.json

Throughput model (one GPU, one model replica's share; an estimate, not a simulator):
- Prefill is compute bound: T_p = I * (linear + attention over C + I/2) / (peak * eta).
- A decode step of batch B at context L reads the weights once plus, per sequence,
  its KV (L tokens) and its recurrent state (read + written):
  t = (W + B * m) / (bandwidth * eta), bounded by compute and by memory capacity.
- Prefill steals time between decode steps, so a user sees t / (1 - f), where f is
  the prefill share of GPU time. The best t meets the interactivity target.
- eta is one efficiency for compute and bandwidth. For today's best engines it is
  fitted to InferenceX's measured Llama-3.3-70B throughput on the same GPU.
"""

import json
import math
import sys

GB = 1e9


class Workload:
    """Per-request token counts: uncached input I, cached prefix C, output O."""

    def __init__(self, uncached, cached, output):
        self.i, self.c, self.o = uncached, cached, output

    def tokens(self):
        return self.i + self.c + self.o


def replica_rate(model, gpu, precision, work, target, eta, cfg, tp):
    """Requests per second per GPU for a replica spread over tp GPUs (no
    communication cost), at user speed >= target; 0 if infeasible."""
    flops = gpu[f"{precision}_tflops"] * 1e12 * eta * tp
    bandwidth = gpu["bw_tbs"] * 1e12 * eta * tp
    weights = model["decode_weight_params"] * cfg["bytes_per_weight"][precision]
    kv = model["kv_elements_per_token"] * cfg["bytes_per_kv_element"]["value"]
    state = model["state_bytes_per_seq"]
    t_prefill = work.i * (model["prefill_linear_flops"]
                          + model["attn_flops_per_position"] * (work.c + work.i / 2)) / flops
    context = work.c + work.i + work.o / 2
    per_seq_bytes = context * kv + 2 * state
    per_seq_flops = model["decode_linear_flops"] + model["attn_flops_per_position"] * context
    room = tp * gpu["mem_gb"] * GB * cfg["memory_usable_fraction"]["value"] - weights
    capacity = math.floor(room / (work.tokens() * kv + state)) if room > 0 else 0
    best = 0.0
    t = weights / bandwidth
    while t <= 1.0 / target:
        batch = min((t * bandwidth - weights) / per_seq_bytes, t * flops / per_seq_flops, capacity)
        batch = math.floor(batch)
        if batch >= 1:
            t_decode = work.o * t / batch
            share = t_prefill / (t_prefill + t_decode)
            if t / (1 - share) <= 1.0 / target:
                best = max(best, 1.0 / (t_prefill + t_decode))
        t *= 1.01
    return best / tp


def best_rate(model, gpu, precision, work, target, eta, cfg, tps=(1, 2, 4, 8)):
    """Best requests per second per GPU over the tensor-parallel sizes tps."""
    return max(replica_rate(model, gpu, precision, work, target, eta, cfg, tp) for tp in tps)


def fit_eta(model, gpu, precision, work, target, measured_tokens_s, tp, cfg):
    """Efficiency at which the model, at the measured row's tensor-parallel size,
    reproduces its total tok/s/GPU; None if even eta = 1 falls short."""
    lo, hi = 0.01, 1.0
    rate = lambda e: replica_rate(model, gpu, precision, work, target, e, cfg, tp) * work.tokens()
    if rate(hi) < measured_tokens_s:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if rate(mid) < measured_tokens_s:
            lo = mid
        else:
            hi = mid
    return hi


def sota_measurements(rows, fit, target):
    """Best measured total tok/s/GPU per hardware at interactivity >= target."""
    best = {}
    for r in rows:
        if (r["isl"], r["osl"], r["precision"]) != (fit["isl"], fit["osl"], fit["precision"]):
            continue
        m = r["metrics"]
        if m["median_intvty"] < target:
            continue
        if m["tput_per_gpu"] > best.get(r["hardware"], (0, None, 0))[0]:
            best[r["hardware"]] = (m["tput_per_gpu"], r["framework"], r["decode_tp"])
    return best


def revenue_per_request(price, work):
    if "per_request_usd" in price:
        return price["per_request_usd"]
    return (work.i * price["input"] + work.c * price["cached"] + work.o * price["output"]) / 1e6


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fitted_etas(cfg, llama_rows, target):
    """Per GPU: (measured tok/s, engine, tp, fitted eta or None, Llama roofline at that tp)."""
    llama = cfg["models"]["llama-3.3-70b"]
    fit = cfg["sota_fit"]
    work = Workload(fit["isl"], 0, fit["osl"])
    out = {}
    for key, gpu in cfg["gpus"].items():
        meas = sota_measurements(llama_rows, fit, target).get(key)
        if not meas:
            out[key] = None
            continue
        tokens, engine, tp = meas
        roof = replica_rate(llama, gpu, "fp8", work, target, 1.0, cfg, tp) * work.tokens()
        out[key] = (tokens, engine, tp, fit_eta(llama, gpu, "fp8", work, target, tokens, tp, cfg), roof)
    return out


def report(cfg, llama_rows):
    qwen = cfg["models"]["qwen3.8-27b"]
    req = cfg["request"]
    work = Workload(req["uncached_input"], req["cached_input"], req["output"])
    gpus = cfg["gpus"]
    effs = cfg["our_engine_efficiency"]["values"]
    prices = cfg["prices"]
    out = []
    for target in cfg["interactivity_targets"]:
        fits = fitted_etas(cfg, llama_rows, target)
        out.append(f"\n### A{target}. Today's best engines against the roofline, >= {target} tok/s per user\n")
        rows = []
        rate = {}
        for key, gpu in gpus.items():
            f = fits[key]
            roof = best_rate(qwen, gpu, "fp8", work, target, 1.0, cfg)
            eta = f[3] if f else None
            sota = best_rate(qwen, gpu, "fp8", work, target, eta, cfg) if eta else None
            rate[key] = {"roof": roof, "today": sota, **{e: best_rate(qwen, gpu, "fp8", work, target, e, cfg) for e in effs}}
            rows.append([gpu["label"],
                         f"{f[0]:,.0f} ({f[1]}, TP{f[2]})" if f else "no row",
                         f"{f[4]:,.0f}" if f else "-",
                         f"{eta:.2f}" if eta else ("above roofline" if f else "-"),
                         f"{roof * 3600:,.0f}", f"{sota * 3600:,.0f}" if sota else "-"])
        out.append(table(["GPU", "Llama-70B measured tok/s/GPU (engine, TP)", "Llama-70B roofline at that TP",
                          "fitted efficiency", "Qwen3.8-27B roofline req/h/GPU", "Qwen3.8-27B today's engine req/h/GPU"], rows))

        out.append(f"\n### B{target}. Cost per request at 100% busy, milli-USD (per 1M output tokens: x1.025 USD)\n")
        rows, owner_cost = [], {}
        for key, gpu in gpus.items():
            r = rate[key]
            cell = lambda price, rr: f"{price / (rr * 3600) * 1e3:.2f}" if rr else "-"
            if r["today"]:
                owner_cost[key] = gpu["own"] / (r["today"] * 3600)
            rows.append([gpu["label"], cell(gpu["own"], r["today"]), cell(gpu["rent_ix"], r["today"]),
                         cell(gpu["rent_public"], r["today"])] + [cell(gpu["rent_public"], r[e]) for e in effs]
                        + [cell(gpu["rent_public"], r["roof"])])
        out.append(table(["GPU", "today's engine, owned", "today's, rented (InferenceX rate)", "today's, rented (public)"]
                         + [f"ours eta={e}, rented (public)" for e in effs] + ["roofline, rented (public)"], rows))
        setter = min(owner_cost, key=owner_cost.get)
        c = owner_cost[setter]
        out.append(f"\nLowest-cost producer with today's engines: owned **{gpus[setter]['label']}** at "
                   f"{c * 1e3:.3f} m$ per request. To match it, a renter needs:\n")
        rows = []
        for key, gpu in gpus.items():
            r = rate[key]
            need = lambda price: price / 3600 / c
            rows.append([gpu["label"],
                         f"{need(gpu['rent_public']) / r['roof']:.2f}" if r["roof"] else "-",
                         f"{need(gpu['rent_ix']) / r['roof']:.2f}" if r["roof"] else "-",
                         f"{need(gpu['rent_public']) / r['today']:.1f}x" if r["today"] else "-"])
        out.append(table(["GPU (rented)", "efficiency needed, public rent (>1 = impossible)",
                          "efficiency needed, InferenceX rent", "speedup over today's engine, public rent"], rows))

    target = cfg["interactivity_targets"][-1]
    fits = fitted_etas(cfg, llama_rows, target)
    out.append(f"\n### C. Rented GPU at public prices, >= {target} tok/s per user: break-even utilization and "
               "monthly margin at 60% (USD, 730 h)\n")
    rows = []
    for key, gpu in gpus.items():
        eta = fits[key][3] if fits[key] else None
        for name, e in [("today", eta)] + [(f"ours {e}", e) for e in effs]:
            if not e:
                continue
            r = best_rate(qwen, gpu, "fp8", work, target, e, cfg)
            cells = [gpu["label"], name]
            for p in prices:
                per_hour = r * 3600 * revenue_per_request(p, work)
                be = gpu["rent_public"] / per_hour
                cells.append(f"{be:.0%}" if be <= 1 else "never")
            for p in prices:
                cells.append(f"{(0.6 * r * 3600 * revenue_per_request(p, work) - gpu['rent_public']) * 730:,.0f}")
            rows.append(cells)
    out.append(table(["GPU", "engine"] + [f"break-even u, {p['key']}" for p in prices]
                     + [f"margin u=60%, {p['key']}" for p in prices], rows))
    out.append("\nRevenue per request: " + ", ".join(
        f"{p['key']} {revenue_per_request(p, work) * 1e3:.3f} m$" for p in prices) + ".")

    daily = cfg["market"]["requests_per_day"]
    out.append(f"\n### D. Whole OpenRouter Qwen3.8-27B demand ({daily:,} requests/day) in GPUs, 100% busy, "
               f">= {target} tok/s per user\n")
    rows = []
    for key, gpu in gpus.items():
        eta = fits[key][3] if fits[key] else None
        cells = [gpu["label"]]
        for e in (eta, effs[1]):
            r = best_rate(qwen, gpu, "fp8", work, target, e, cfg) if e else 0
            cells.append(f"{daily / 86400 / r:.1f}" if r else "-")
        rows.append(cells)
    out.append(table(["GPU", "today's engine", f"ours eta={effs[1]}"], rows))

    speeds = cfg["fast_tier_targets"]
    out.append(f"\n### E. Price of speed: cost per request (m$), our engine at eta={effs[1]}, rented (public), "
               "one GPU per replica (no tensor parallelism, no speculation)\n")
    rows = []
    for key, gpu in gpus.items():
        cells = [gpu["label"]]
        for sp in speeds:
            r = best_rate(qwen, gpu, "fp8", work, sp, effs[1], cfg, tps=(1,))
            cells.append(f"{gpu['rent_public'] / (r * 3600) * 1e3:.2f}" if r else "infeasible")
        rows.append(cells)
    out.append(table(["GPU"] + [f">= {sp} tok/s/user" for sp in speeds], rows))
    return "\n".join(out)


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} INPUTS.json INFERENCEX_LLAMA70B.json", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        cfg = json.load(f)
    with open(argv[2], encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    print(report(cfg, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
