#!/usr/bin/env python3
"""Markdown tables from a run_multiuser.py summary.json: median [min-max] over rounds (steady)
and over rounds x reps (interference, queue).

  multiuser_table.py docs/bench/data/DATE-multiuser/final/summary.json [--names A=label,B=label]
"""
import sys
import argparse, json, statistics


def cell(xs, fmt="{:.0f}"):
    xs = [x for x in xs if x is not None]
    if not xs: return "-"
    m = fmt.format(statistics.median(xs))
    return m if len(xs) == 1 else f"{m} [{fmt.format(min(xs))}–{fmt.format(max(xs))}]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary")
    p.add_argument("--names", default="", help="comma list ENGINE=LABEL")
    a = p.parse_args()
    s = json.load(open(a.summary))
    label = dict(x.split("=", 1) for x in a.names.split(",") if "=" in x) if a.names else {}
    name = lambda n: label.get(n, n)

    print("### Steady (closed loop, short prompts, 256 tokens)\n")
    print("| Engine | Users | tok/s | TTFT p50 ms | TTFT p95 ms | gap p50 ms | gap p99 ms | gap max ms | stalled % | stalled mean ms | slowest user tok/s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n, r in s.items():
        for level in sorted({x["level"] for x in r["steady"]}):
            xs = [x for x in r["steady"] if x["level"] == level and not x["errors"]]
            errs = sum(len(x["errors"]) for x in r["steady"] if x["level"] == level)
            row = [cell([x["aggregate_tok_s"] for x in xs], "{:.1f}"), cell([x["ttft_p50_ms"] for x in xs]), cell([x["ttft_p95_ms"] for x in xs]),
                   cell([x.get("itl_p50_ms") for x in xs]), cell([x.get("itl_p99_ms") for x in xs]), cell([x.get("itl_max_ms") for x in xs]),
                   cell([x.get("itl_stalled_pct") for x in xs], "{:.1f}"), cell([x.get("itl_stalled_mean_ms") for x in xs]),
                   cell([x.get("user_tok_s_min") for x in xs], "{:.1f}")]
            print(f"| {name(n)} | {level}{' (' + str(errs) + ' errors)' if errs else ''} | " + " | ".join(row) + " |")

    print("\n### Interference (P-2 users streaming; a 4,936-token prompt, then a 280-token prompt 100 ms later)\n")
    print("| Engine | runs | long TTFT ms | short TTFT ms | running users' gap before, p50 ms | gap during prefill p50 / p99 / max ms | running tok/s during prefill |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for n, r in s.items():
        xs = [x for x in r["interference"] if not x["errors"]]
        errs = len(r["interference"]) - len(xs)
        print(f"| {name(n)} | {len(xs)}{' (' + str(errs) + ' failed)' if errs else ''} | {cell([x['long_ttft_ms'] for x in xs])} | {cell([x['short_ttft_ms'] for x in xs])} | "
              f"{cell([x['bg_itl_before_p50_ms'] for x in xs])} | {cell([x['bg_itl_during_p50_ms'] for x in xs])} / {cell([x['bg_itl_during_p99_ms'] for x in xs])} / "
              f"{cell([x['bg_itl_during_max_ms'] for x in xs])} | {cell([x['bg_tok_s_during'] for x in xs], '{:.1f}')} |")

    print("\n### Queue (P users streaming, 2 more arrive with no slot free)\n")
    print("| Engine | runs | queued TTFT ms | queued TTFT after the first slot frees, ms | running gap p50 / p99 / max ms | stalled % |")
    print("| --- | --- | --- | --- | --- | --- |")
    for n, r in s.items():
        xs = [x for x in r.get("queue", []) if not x["errors"]]
        errs = len(r.get("queue", [])) - len(xs)
        print(f"| {name(n)} | {len(xs)}{' (' + str(errs) + ' failed)' if errs else ''} | {cell([v for x in xs for v in x['queued_ttft_ms']])} | "
              f"{cell([v for x in xs for v in x['queued_ttft_after_first_free_ms']])} | {cell([x.get('running_itl_p50_ms') for x in xs])} / "
              f"{cell([x.get('running_itl_p99_ms') for x in xs])} / {cell([x.get('running_itl_max_ms') for x in xs])} | {cell([x.get('running_itl_stalled_pct') for x in xs], '{:.1f}')} |")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
