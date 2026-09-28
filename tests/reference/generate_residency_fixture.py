#!/usr/bin/env python3
"""Independent finite ownership oracle; no zerv implementation is executed.

Regenerate: tools/py tests/reference/generate_residency_fixture.py
The reference derives free slots from sets of owned locations, not allocator arrays.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys


class Model:
    def __init__(self, count, hot, disk):
        self.count, self.hot, self.disk = count, hot, disk
        self.generations = [0] * count
        self.objects = {}

    def entry(self, h):
        e = self.objects.get(h["index"])
        if e is None or e["generation"] != h["generation"]:
            raise ValueError("InvalidHandle")
        return e

    def free(self, tier, capacity, error):
        available = set(range(capacity)) - {e[tier] for e in self.objects.values()}
        if not available:
            raise ValueError(error)
        return min(available)

    def apply(self, op, h, transfer, success):
        if op == "reserve":
            available = set(range(self.count)) - self.objects.keys()
            if not available:
                raise ValueError("NoEntry")
            index = min(available)
            hot = self.free("hot", self.hot, "NoHotSlot")
            self.generations[index] += 1
            self.objects[index] = dict(generation=self.generations[index], serial=0,
                                       phase="saving", hot=hot, disk=None, leases=0)
            return dict(index=index, generation=self.generations[index]), None, None
        if op == "complete":
            h = transfer["handle"]
        e = self.entry(h)
        if op == "saved":
            if e["phase"] != "saving":
                raise ValueError("InvalidState")
            if success:
                e["phase"] = "resident"
            else:
                del self.objects[h["index"]]
        elif op == "acquire":
            if e["phase"] != "resident":
                raise ValueError("Busy")
            e["leases"] += 1
            return None, None, e["hot"]
        elif op == "release":
            if e["phase"] != "resident" or e["leases"] == 0:
                raise ValueError("InvalidState")
            e["leases"] -= 1
        elif op in ("spill", "restore"):
            if op == "restore" and e["phase"] == "resident":
                return None, None, None
            if e["phase"] != ("resident" if op == "spill" else "disk") or e["leases"]:
                raise ValueError("Busy")
            if op == "spill" and e["disk"] is not None:
                e.update(phase="disk", hot=None)
                return None, None, None
            tier = "disk" if op == "spill" else "hot"
            slot = self.free(tier, self.disk if tier == "disk" else self.hot,
                             "NoDiskSlot" if tier == "disk" else "NoHotSlot")
            e[tier] = slot
            e["serial"] += 1
            e["phase"] = "writing" if op == "spill" else "reading"
            move = dict(handle=copy.deepcopy(h), serial=e["serial"], hot=e["hot"],
                        disk=e["disk"], direction="write" if op == "spill" else "read")
            return None, move, None
        elif op == "complete":
            write = transfer["direction"] == "write"
            if (e["phase"] != ("writing" if write else "reading") or
                    any(transfer[k] != e[k] for k in ("serial", "hot", "disk"))):
                raise ValueError("InvalidTransfer")
            if write:
                e.update(phase="disk" if success else "resident")
                e["hot" if success else "disk"] = None
            elif success:
                e["phase"] = "resident"
            else:
                e.update(phase="disk", hot=None)
        elif op == "discard":
            if e["phase"] != "resident":
                raise ValueError("Busy")
            e["disk"] = None
        elif op == "drop":
            if e["phase"] not in ("resident", "disk") or e["leases"]:
                raise ValueError("Busy")
            del self.objects[h["index"]]
        else:
            raise ValueError("UnknownOperation")
        return None, None, None

    def state(self):
        entries = [copy.deepcopy(self.objects.get(i, dict(generation=g, serial=0,
                   phase="free", hot=None, disk=None, leases=0)))
                   for i, g in enumerate(self.generations)]
        owners = lambda tier, n: [next((i for i, e in self.objects.items()
                                       if e[tier] == slot), None) for slot in range(n)]
        return dict(entries=entries, hot_owners=owners("hot", self.hot),
                    disk_owners=owners("disk", self.disk))


def generate(count, hot, disk, seed):
    model, rng, rows = Model(count, hot, disk), random.Random(seed), []
    handles, transfers = [], []

    def run(op, h=None, move=None, success=True):
        row = dict(op=op, handle=copy.deepcopy(h), transfer=copy.deepcopy(move), success=success)
        before = model.state()
        try:
            handle, transfer, slot = model.apply(op, h, move, success)
            row.update(error=None, returned_handle=handle, returned_transfer=transfer, slot=slot)
            if handle is not None:
                handles.append(handle)
            if transfer is not None:
                transfers.append(transfer)
        except ValueError as e:
            assert model.state() == before, "reference mutated on failure"
            row.update(error=str(e), returned_handle=None, returned_transfer=None, slot=None)
        row.update(model.state())
        rows.append(row)
        return row

    # Directed spill failure, same-generation retry, duplicate/out-of-order completion,
    # leased access, read failure, retained backing and stale handle after drop/reuse.
    h = run("reserve")["returned_handle"]
    run("saved", h)
    run("acquire", h)
    run("spill", h)
    run("drop", h)
    run("release", h)
    move = run("spill", h)["returned_transfer"]
    if move:
        run("drop", h)
        run("acquire", h)
        run("complete", move=move, success=False)
        new_move = run("spill", h)["returned_transfer"]
        run("complete", move=move)
        run("complete", move=new_move)
        read = run("restore", h)["returned_transfer"]
        run("complete", move=read, success=False)
        read2 = run("restore", h)["returned_transfer"]
        run("complete", move=read)
        run("complete", move=read2)
        run("spill", h)  # backed: no write
        read3 = run("restore", h)["returned_transfer"]
        run("complete", move=read3)
        run("discard", h)
    run("drop", h)
    run("reserve")
    run("saved", h)  # stale

    for _ in range(1600):
        live = [dict(index=i, generation=e["generation"]) for i, e in model.objects.items()]
        h = rng.choice(live if live and rng.random() < .85 else handles)
        e = model.objects.get(h["index"])
        choices = ["reserve", "drop", "spill", "restore", "acquire", "release", "discard", "saved"]
        if e is not None:
            # Favor progress while preserving invalid operations/stale completions.
            choices += {"saving": ["saved"] * 10, "resident": ["spill", "drop", "release"] * 2,
                        "disk": ["restore", "drop"] * 3, "writing": ["complete"] * 12,
                        "reading": ["complete"] * 12}[e["phase"]]
        op = rng.choice(choices)
        if op == "complete":
            candidates = [t for t in transfers if t["handle"] == h and e and t["serial"] == e["serial"]]
            move = rng.choice(candidates if candidates and rng.random() < .8 else transfers)
            if rng.random() < .1:
                move = dict(move, hot=hot + 1)  # forged destination
            run(op, move=move, success=rng.random() > .25)
        else:
            run(op, h, success=rng.random() > .2)
    return dict(count=count, hot=hot, disk=disk, seed=seed, rows=rows)


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "tests/fixtures/residency"
    out.mkdir(parents=True, exist_ok=True)
    cases = [generate(7, 2, 5, 186301), generate(3, 1, 0, 186302), generate(5, 3, 2, 186303)]
    payload = (json.dumps(cases, separators=(",", ":")) + "\n").encode()
    (out / "traces.json").write_bytes(payload)
    (out / "manifest.json").write_text(json.dumps(dict(schema=1,
        reference="Independent Python dictionary/set ownership model", python=sys.version,
        rows=sum(len(c["rows"]) for c in cases), sha256=hashlib.sha256(payload).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")
    print(f"generated {sum(len(c['rows']) for c in cases)} exact state transitions")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned interpreter")
    main()
