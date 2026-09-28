"""Read model-oracle capture directories (index.jsonl + tensors.bin + logits.bin)."""
import json
from pathlib import Path

import numpy as np

# llama emits some names twice per token: the second occurrence is the post-RoPE value.
RENAMES = {("Kcur", "ROPE"): "Kcur_roped"}


def load(directory):
    d = Path(directory)
    rows = [json.loads(line) for line in (d/"index.jsonl").read_text().splitlines()]
    blob = np.memmap(d/"tensors.bin", dtype="<f4", mode="r")
    tokens = json.loads((d/"tokens.json").read_text())
    n_vocab = tokens["n_vocab"]
    logits = np.memmap(d/"logits.bin", dtype="<f4", mode="r").reshape(-1, n_vocab)
    tensors = {}
    for r in rows:
        if "skipped_type" in r: continue
        if " (" in r["name"]: continue  # ggml-derived views/reshapes of an already captured tensor
        base, _, il = r["name"].rpartition("-")
        if not base or not il.isdigit(): base, il = r["name"], None
        base = RENAMES.get((base, r["op"]), base)
        if r["op"] == "RESHAPE" and base == "Vcur": continue  # duplicate view of the projection
        name = base if il is None else f"{base}-{il}"
        start = r["offset"]//4
        tensors.setdefault(name, {})[r["token"]] = (start, r["count"], r["ne"])
    return dict(tokens=tokens, blob=blob, logits=logits, tensors=tensors)


def tensor(capture, name, token):
    start, count, _ = capture["tensors"][name][token]
    return np.asarray(capture["blob"][start:start+count], dtype=np.float64)
