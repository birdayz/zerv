#!/usr/bin/env python3
"""Unit economics of selling tokens from one GPU running zerv.

Reads an input file in which every number carries its provenance (measured,
sourced, derived, assumed) and prints the markdown tables of
docs/research/2026-09-26-token-economics.md. Standard library only.

    python3 tools/token_economics.py docs/research/2026-09-26-token-economics/inputs.json

Model (one GPU, time-shared between prefill and decode):
- A mode has a pure decode rate D (aggregate output tok/s with its users busy).
  Measured aggregates include the prefill of their short prompts, so
  O0/D = O0/A - I0/P for a measurement A with I0 prompt and O0 output tokens.
- For r uncached input tokens per output token, the full-load rates are
  R_out = 1 / (r/P + 1/D) and R_in = r * R_out.
- Revenue is usage-billed; utilization u scales it. Power is u * load +
  (1 - u) * idle (idle only when the machine would otherwise be off).
"""

import json
import sys


def value(x):
    """Returns the number of a labelled input ({"value": ...}) or a bare number."""
    return x["value"] if isinstance(x, dict) else x


def pure_decode_rate(aggregate, prompt_tokens, output_tokens, prefill):
    """Output tok/s without the prefill share that a measured aggregate contains."""
    if prompt_tokens == 0:
        return aggregate
    decode_seconds = output_tokens / aggregate - prompt_tokens / prefill
    if decode_seconds <= 0:
        raise ValueError("measurement inconsistent with the prefill rate")
    return output_tokens / decode_seconds


def full_load_rates(decode, prefill, input_per_output):
    """(output tok/s, input tok/s) with the GPU always busy."""
    out = 1.0 / (input_per_output / prefill + 1.0 / decode)
    return out, input_per_output * out


def revenue_usd_per_hour(rates, price):
    out, inp = rates
    return (inp * price["input"] + out * price["output"]) * 3600 / 1e6


def wall_watts(power):
    eff = value(power["psu_efficiency"])
    load = (value(power["gpu_load_w"]) + value(power["host_load_w"])) / eff
    idle = (value(power["gpu_idle_w"]) + value(power["host_idle_w"])) / eff
    return load, idle


def fixed_eur_month(cost):
    capex = cost.get("capex_eur_month")
    if capex is None:
        capex = (cost["gpu_eur"] + cost["host_eur"] - cost["resale_eur"]) / cost["life_months"]
    return capex + cost["internet_eur_month"]


class Scenario:
    """Monthly P&L of one mode x mix x price x electricity x cost basis."""

    def __init__(self, cfg, mode, mix, price, eur_kwh, cost):
        self.hours = cfg["hours_per_month"]
        prefill = value(cfg["prefill_tok_s"])
        decode = pure_decode_rate(mode["aggregate_tok_s"], mode["workload_prompt_tokens"],
                                  mode["workload_output_tokens"], prefill)
        self.rates = full_load_rates(decode, prefill, mix["input_per_output"])
        usd = value(cfg["usd_per_eur"])
        fee = value(cfg["platform_fee_fraction"])
        self.revenue_full = revenue_usd_per_hour(self.rates, price) / usd * self.hours * (1 - fee)
        load_w, idle_w = wall_watts(cfg["power"])
        if not cost["idle_power_counts"]:
            idle_w = 0.0
        self.energy_load = load_w / 1000 * self.hours * eur_kwh
        self.energy_idle = idle_w / 1000 * self.hours * eur_kwh
        self.fixed = fixed_eur_month(cost)

    def revenue(self, u):
        return u * self.revenue_full

    def cost(self, u):
        return self.fixed + u * self.energy_load + (1 - u) * self.energy_idle

    def margin(self, u):
        return self.revenue(u) - self.cost(u)

    def break_even(self):
        """Utilization where margin is zero, or None if even full load loses."""
        slope = self.revenue_full - (self.energy_load - self.energy_idle)
        if slope <= 0:
            return None
        u = (self.fixed + self.energy_idle) / slope
        return u if u <= 1 else None

    def cost_per_mtok_output(self):
        """EUR per 1M output tokens at full load, all cost charged to output."""
        tokens = self.rates[0] * 3600 * self.hours / 1e6
        return (self.fixed + self.energy_load) / tokens


def by_key(items, key):
    for item in items:
        if item["key"] == key:
            return item
    raise KeyError(key)


def pct(u):
    if u is None:
        return "never"
    return "any" if u == 0 else f"{u * 100:.0f}%"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def report(cfg):
    modes, mixes, prices = cfg["modes"], cfg["mixes"], cfg["prices_usd_per_mtok"]
    elec = cfg["electricity_eur_per_kwh"]
    base_kwh = value(by_key(elec, cfg["base_electricity"]))
    base_cost = by_key(cfg["cost_bases"], cfg["base_cost"])
    us = cfg["utilizations"]
    usd = value(cfg["usd_per_eur"])
    load_w, idle_w = wall_watts(cfg["power"])
    out = [f"Wall power (derived): {load_w:.0f} W under load, {idle_w:.0f} W idle. "
           f"Base: {cfg['base_electricity']} {base_kwh} EUR/kWh, cost basis '{base_cost['key']}' "
           f"({fixed_eur_month(base_cost):.1f} EUR/month fixed). 1 EUR = {usd} USD.", ""]

    out.append("### T1. Full-load capacity and revenue per hour (USD)\n")
    rows = []
    for mode in modes:
        for mix in mixes:
            s = {p["key"]: Scenario(cfg, mode, mix, p, base_kwh, base_cost) for p in prices}
            r_out, r_in = s["median"].rates
            rows.append([mode["key"], mix["input_per_output"], f"{r_out:.1f}", f"{r_in:.0f}",
                         f"{mode['per_user_tok_s']}", f"{mode['context_per_request']:,}"]
                        + [f"{s[p['key']].revenue_full * usd / cfg['hours_per_month']:.2f}" for p in prices])
    out.append(table(["mode", "in/out", "out tok/s", "in tok/s", "per-user tok/s", "context/request"]
                     + [f"$/h @{p['key']}" for p in prices], rows))

    for mode_key, mix_key in (("M8", "r10"), ("M1", "r10")):
        mode, mix = by_key(modes, mode_key), by_key(mixes, mix_key)
        out.append(f"\n### T2. {mode['name']}, input/output {mix['input_per_output']}: "
                   f"revenue / cost / margin, EUR per month ({cfg['base_electricity']}, '{base_cost['key']}')\n")
        rows = []
        for p in prices:
            s = Scenario(cfg, mode, mix, p, base_kwh, base_cost)
            with_ops = Scenario(cfg, mode, mix, p, base_kwh, base_cost)
            with_ops.fixed += value(cfg["ops_eur_month"])
            rows.append([p["key"]] + [f"{s.revenue(u):.0f} / {s.cost(u):.0f} / **{s.margin(u):.0f}**" for u in us]
                        + [pct(s.break_even()), pct(with_ops.break_even())])
        out.append(table(["price"] + [f"u={u:.0%}" for u in us] + ["break-even u", "incl. ops time"], rows))

    mode, mix = by_key(modes, "M8"), by_key(mixes, "r10")
    out.append(f"\n### T3. Margin sensitivity to electricity, EUR per month ({mode['key']}, "
               f"in/out {mix['input_per_output']}, '{base_cost['key']}')\n")
    rows = []
    for p in prices:
        for e in elec:
            s = Scenario(cfg, mode, mix, p, value(e), base_cost)
            rows.append([p["key"], e["key"]] + [f"{s.margin(u):.0f}" for u in us] + [pct(s.break_even())])
    out.append(table(["price", "electricity"] + [f"u={u:.0%}" for u in us] + ["break-even u"], rows))

    out.append(f"\n### T4. Break-even utilization ({cfg['base_electricity']}), per cost basis\n")
    rows = []
    for mode in modes:
        for mix in mixes:
            row = [mode["key"], mix["input_per_output"]]
            for p in prices:
                row.append(" / ".join(pct(Scenario(cfg, mode, mix, p, base_kwh, c).break_even())
                                      for c in cfg["cost_bases"]))
            rows.append(row)
    names = "/".join(c["key"] for c in cfg["cost_bases"])
    out.append(table(["mode", "in/out"] + [f"{p['key']} ({names})" for p in prices], rows))

    out.append(f"\n### T5. Cost per 1M output tokens at full load, EUR ('{base_cost['key']}', all cost on output)\n")
    rows = []
    for mode in modes:
        for mix in mixes:
            rows.append([mode["key"], mix["input_per_output"]]
                        + [f"{Scenario(cfg, mode, mix, prices[0], value(e), base_cost).cost_per_mtok_output():.2f}"
                           for e in elec])
    out.append(table(["mode", "in/out"] + [e["key"] for e in elec], rows))

    cloud = cfg["cloud"]
    out.append("\n### T6. Rented datacenter GPU: USD per 1M output tokens (all cost on output) "
               "and revenue per hour at the median price\n")
    median = by_key(prices, "median")
    rows = []
    for w in cloud["h100_workloads"]:
        rates = (w["output_tok_s"], w["input_tok_s"])
        rows.append([w["key"], f"{w['output_tok_s']:.0f}"]
                    + [f"{value(g) / (w['output_tok_s'] * 3600 / 1e6):.2f}" for g in cloud["gpus_usd_per_hour"]]
                    + [f"{revenue_usd_per_hour(rates, median):.2f}"])
    out.append(table(["H100 workload (Qwen3-32B)", "out tok/s"] + [g["key"] for g in cloud["gpus_usd_per_hour"]]
                     + ["revenue $/h @median"], rows))

    c3090 = cloud["rtx3090"]
    rent = value(c3090["usd_per_hour_vast_median"])
    out.append("\n### T7. Renting the card out instead (proxy: RTX 3090 on Vast.ai; no RX 7900 XTX market found)\n")
    rows = []
    for u in us:
        gross = u * rent / usd * cfg["hours_per_month"]
        power = (u * load_w + (1 - u) * idle_w) / 1000 * cfg["hours_per_month"] * base_kwh
        rows.append([f"{u:.0%}", f"{gross:.0f}", f"{power:.0f}", f"{gross - power - fixed_eur_month(base_cost):.0f}"])
    out.append(table(["u", "gross EUR/month", f"power EUR/month ({cfg['base_electricity']})",
                      f"margin EUR/month ('{base_cost['key']}')"], rows))
    for key in ("hyperqwen_c64_output_tok_s", "hyperqwen_c8_output_tok_s"):
        tok = value(c3090[key])
        out.append(f"\n- 3090 + HyperQwen, {key}: {tok} out tok/s -> "
                   f"{rent / (tok * 3600 / 1e6):.3f} USD/1M output at Vast median, "
                   f"{value(c3090['usd_per_hour_runpod']) / (tok * 3600 / 1e6):.3f} at RunPod list.")
    out.append("\n### T8. What a router would measure per request (unloaded server; estimate)\n")
    prefill = value(cfg["prefill_tok_s"])
    rows = []
    for prompt, completion in cfg["request_shapes"]:
        for mode in modes:
            if prompt + completion > mode["context_per_request"]:
                rows.append([f"{prompt:,} + {completion}", mode["key"], "does not fit", "-"])
                continue
            ttft = prompt / prefill
            total = ttft + completion / mode["per_user_tok_s"]
            rows.append([f"{prompt:,} + {completion}", mode["key"], f"{ttft:.1f}", f"{completion / total:.0f}"])
    out.append(table(["prompt + output tokens", "mode", "TTFT s", "output tok / total s"], rows))
    out.append(f"\nOps time, not included above: {value(cfg['ops_eur_month'])} EUR/month.")
    return "\n".join(out)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: {argv[0]} INPUTS.json", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        cfg = json.load(f)
    print(report(cfg))
    return 0


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    sys.exit(main(sys.argv))
