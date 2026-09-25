#!/usr/bin/env python3
"""For two ACO "After RA" IR dumps of the same shader (RADV_DEBUG=shaders), one with the
pre-RA scheduler (default) and one with ACO_DEBUG=nosched: for every LDS spill reload
(ds_read_addtid_b32, identified by its definition temp, which spilling assigns before
scheduling), count the same-slot spill stores (ds_write_addtid_b32) before it in its block.
A reload whose count differs was moved across an aliasing spill store.
Usage: reload_diff.py SCHED_IR NOSCHED_IR   (each: the After RA section of one module)"""
import re, sys
def reads(path):
    blocks, cur = {}, None
    for line in open(path).read().split("\n"):
        m = re.match(r"^BB(\d+)$", line.strip())
        if m: cur = int(m.group(1)); blocks[cur] = []; continue
        if cur is None or "addtid" not in line: continue
        off = int(re.search(r"offset0:(\d+)", line).group(1))
        if "ds_write_addtid_b32" in line: blocks[cur].append(("W", off, None))
        else: blocks[cur].append(("R", off, re.search(r"%(\d+):v", line).group(1)))
    out = {}
    for b, events in blocks.items():
        for i, (kind, off, temp) in enumerate(events):
            if kind == "R": out[temp] = (b, off, sum(1 for e in events[:i] if e[0] == "W" and e[1] == off))
    return out
s, n = reads(sys.argv[1]), reads(sys.argv[2])
diff = [t for t in s if t in n and s[t] != n[t]]
print(f"reloads {len(s)} / {len(n)}; moved across an aliasing store: {len(diff)}")
for t in diff: print(f"  %{t}: sched (block, offset, stores before) {s[t]} | nosched {n[t]}")
