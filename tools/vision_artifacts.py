#!/usr/bin/env python3
"""Inventory the Qwen3.8 vision projector (mmproj GGUF) and check its tensors against the
BF16 vision tower of a safetensors checkpoint, bit for bit (docs/research/vision-qwen38.md).

  tools/vision_artifacts.py --mmproj models/qwen3.8-27b/mmproj-BF16.gguf \
      --safetensors models/RedHatAI/Qwen3.8-27B-INT4/<rev>/model.safetensors \
      --out docs/research/2026-09-26-vision/mmproj-inventory.json

Research tool only (reads headers and raw tensor bytes; no model code, no network).
Mapping: every GGUF tensor must equal exactly one checkpoint tensor's raw BF16 bytes; the
patch embedding's Conv3d kernel [1152, 3, 2, 16, 16] is compared as its two temporal
slices (the converter writes `v.patch_embd.weight` = [:, :, 0] and `.weight.1` = [:, :, 1]).
Exit status 1 if any tensor is unmatched, duplicated or of an unexpected type.
"""
import argparse, hashlib, json, struct, sys
from pathlib import Path

import numpy as np

GGML = {0: ("F32", 4), 1: ("F16", 2), 30: ("BF16", 2)}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    with open(path, "rb") as f:
        data = f.read(64 << 20)  # headers only; tensor data is read by offset below

    at = 0

    def take(fmt):
        nonlocal at
        v = struct.unpack_from(fmt, data, at)[0]
        at += struct.calcsize(fmt)
        return v

    def string():
        nonlocal at
        n = take("<Q")
        s = data[at:at + n].decode("utf-8")
        at += n
        return s

    def value(t):
        if t in SCALAR: return take(SCALAR[t])
        if t == 8: return string()
        if t == 9:
            et, n = take("<I"), take("<Q")
            return [value(et) for _ in range(n)]
        raise SystemExit(f"unknown GGUF value type {t}")

    if data[:4] != b"GGUF": raise SystemExit("not a GGUF file")
    at = 4
    version, n_tensors, n_kv = take("<I"), take("<Q"), take("<Q")
    meta = {}
    for _ in range(n_kv):
        k = string()
        meta[k] = value(take("<I"))
    tensors = []
    for _ in range(n_tensors):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        kind, offset = take("<I"), take("<Q")
        tensors.append(dict(name=name, ne=dims, type=kind, offset=offset))
    align = meta.get("general.alignment", 32)
    base = (at + align - 1) // align * align
    return version, meta, tensors, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    version, meta, tensors, base = read_gguf(a.mmproj)
    size = Path(a.mmproj).stat().st_size
    raw = np.memmap(a.mmproj, dtype=np.uint8, mode="r")

    # Checkpoint side: every model.visual.* tensor, hashed by raw bytes.
    with open(a.safetensors, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    sraw = np.memmap(a.safetensors, dtype=np.uint8, mode="r")
    by_hash = {}
    for name, t in header.items():
        if not name.startswith("model.visual."): continue
        if t["dtype"] != "BF16": raise SystemExit(f"{name}: {t['dtype']}, expected BF16")
        lo, hi = t["data_offsets"]
        b = np.asarray(sraw[8 + n + lo:8 + n + hi])
        if name == "model.visual.patch_embed.proj.weight":
            k = b.view("<u2").reshape(t["shape"])
            for s, suffix in ((0, "[:, :, 0]"), (1, "[:, :, 1]")):
                part = np.ascontiguousarray(k[:, :, s]).tobytes()
                by_hash.setdefault(hashlib.sha256(part).hexdigest(), []).append(name + suffix)
        else:
            by_hash.setdefault(hashlib.sha256(b.tobytes()).hexdigest(), []).append(name)

    rows, bad, used = [], [], set()
    for t in tensors:
        kind, width = GGML.get(t["type"], (f"type{t['type']}", 0))
        count = 1
        for d in t["ne"]: count *= d
        start = base + t["offset"]
        if width == 0 or start + count * width > size:
            bad.append(f"{t['name']}: type {kind} or range")
            continue
        h = hashlib.sha256(np.asarray(raw[start:start + count * width]).tobytes()).hexdigest()
        match = by_hash.get(h, [])
        if kind == "BF16" and len(match) != 1: bad.append(f"{t['name']}: {len(match)} checkpoint matches")
        for m in match:
            if m in used: bad.append(f"{t['name']}: {m} matched twice")
            used.add(m)
        rows.append(dict(name=t["name"], ne=t["ne"], type=kind, bytes=count * width, sha256=h, checkpoint=match))
    # F32 tensors (norms, biases in some converters) are converted, so they are compared by value.
    for r, t in zip(rows, tensors):
        if r["type"] != "F32" or r["checkpoint"]: continue
        start = base + t["offset"]
        v = np.asarray(raw[start:start + r["bytes"]]).view("<f4")
        found = []
        # The Conv3d kernel's temporal slices (the converter may widen them to F32).
        st = header["model.visual.patch_embed.proj.weight"]
        lo, hi = st["data_offsets"]
        k = (np.asarray(sraw[8 + n + lo:8 + n + hi]).view("<u2").astype(np.uint32) << 16).view("<f4").reshape(st["shape"])
        for s in (0, 1):
            if np.array_equal(np.ascontiguousarray(k[:, :, s]).reshape(-1), v):
                found.append(f"model.visual.patch_embed.proj.weight[:, :, {s}] (BF16 -> F32 exact)")
        for name, st in header.items():
            if not name.startswith("model.visual.") or list(st["shape"]) != list(reversed(r["ne"])) and [int(np.prod(st["shape"]))] != r["ne"]:
                continue
            lo, hi = st["data_offsets"]
            w = (np.asarray(sraw[8 + n + lo:8 + n + hi]).view("<u2").astype(np.uint32) << 16).view("<f4").reshape(-1)
            if w.shape == v.shape and np.array_equal(w, v): found.append(name + " (BF16 -> F32 exact)")
        r["checkpoint"] = found
        if len(found) != 1: bad.append(f"{r['name']}: F32 with {len(found)} exact checkpoint matches")
        for m in found: used.add(m.split(" (")[0])
    unused = sorted({m for ms in by_hash.values() for m in ms} - used)
    summary = dict(mmproj=a.mmproj, mmproj_bytes=size, gguf_version=version, data_offset=base,
                   tensors=len(tensors), checkpoint=a.safetensors, unmatched_checkpoint_tensors=unused, problems=bad,
                   metadata={k: v for k, v in meta.items()}, inventory=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    kinds = {}
    for r in rows: kinds[r["type"]] = kinds.get(r["type"], 0) + 1
    print(f"{len(tensors)} tensors {kinds}; problems {len(bad)}; checkpoint tensors not in the GGUF {len(unused)}")
    for p in bad[:20]: print("  ", p)
    for u in unused[:20]: print("   unused:", u)
    return 1 if bad or unused else 0


if __name__ == "__main__":
    sys.exit(main())
