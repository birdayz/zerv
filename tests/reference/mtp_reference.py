#!/usr/bin/env python3
"""Gate 2 of block 17b (docs/specs/speculative.md): FP64 reference of the Qwen3.8 MTP
(nextn) layer, llama.cpp `graph_mtp` semantics (docs/research/speculative-mtp.md),
recomputed from exactly the inputs zerv consumed (`zerv-mtp-check` dump).

MTP row q holds (h_{q-1}, x_q): e = rmsnorm(embed(x)) * enorm, g = rmsnorm(h) * hnorm,
u = eh_proj([e; g]); the trunk's gated attention layer (own KV, positions q, scale 1/16)
with residual u; post norm; SwiGLU FFN with residual; h' = rmsnorm(.) * shared_head_norm;
logits = output.weight h'. Chain rows are teacher-forced on zerv's drafts and zerv's h'
(so every chain step measures one step's arithmetic). Normalized L2 bound per the spec.
Development/test oracle only; never part of zerv.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_reference as ref

H = 5120
VOCAB = 248320
BOUND = 1e-5

# Q8_0 (the MTP's eh_proj; qwen35_reference.py is pinned by the trunk oracle fixtures, so
# the decoder is added here): f16 d, then 32 int8, value d * q (exact in FP64). Applied
# before the projector forks its workers, which inherit it.
ref.TYPES[8] = (32, 34)
_trunk_dequant = ref.dequant


def _dequant(kind, raw):
    if kind != 8: return _trunk_dequant(kind, raw)
    b = raw.reshape(-1, 34)
    q = np.ascontiguousarray(b[:, 2:34]).view(np.int8).astype(np.int64)
    return (q * ref.f16(b, 0)[:, None]).ravel()


ref.dequant = _dequant


def mtp_rows(W, projector, tokens, hs, eps, base):
    """tokens [T], hs [T, H] (row q's h = h_{q-1}); positions 0..T-1. Returns h' [T, H]."""
    T = len(tokens)
    L = ref.LayerWeights(W, 64, projector)
    emb = np.stack([W.matrix("token_embd.weight", t, 1)[0] for t in tokens])
    e = ref.rmsnorm(emb, L.vec("nextn.enorm.weight"), eps)
    g = ref.rmsnorm(hs, L.vec("nextn.hnorm.weight"), eps)
    u = L.mm(np.concatenate([e, g], axis=1), "nextn.eh_proj.weight")
    h = ref.rmsnorm(u, L.vec("attn_norm.weight"), eps)
    qf = L.mm(h, "attn_q.weight").reshape(T, 24, 512)
    k = L.mm(h, "attn_k.weight")
    v = L.mm(h, "attn_v.weight").reshape(T, 4, 256)
    q, gate = qf[:, :, :256], qf[:, :, 256:].reshape(T, 6144)
    q = ref.rmsnorm(q, L.vec("attn_q_norm.weight"), eps)
    k = ref.rmsnorm(k.reshape(T, 4, 256), L.vec("attn_k_norm.weight"), eps)
    ang = np.arange(T, dtype=np.float64)[:, None]*(base ** (-np.arange(32, dtype=np.float64)*2/64))[None, :]
    q, k = ref.rope(q, np.cos(ang), np.sin(ang)), ref.rope(k, np.cos(ang), np.sin(ang))
    att = np.empty((T, 24, 256))
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for hq in range(24):
        kv = hq // 6
        s = (q[:, hq, :] @ k[:, kv, :].T) / 16.0 + mask
        s = np.exp(s - s.max(axis=1, keepdims=True))
        s /= s.sum(axis=1, keepdims=True)
        att[:, hq, :] = s @ v[:, kv, :]
    att = att.reshape(T, 6144) * ref.sigmoid(gate)
    r = u + L.mm(att, "attn_output.weight")
    h2 = ref.rmsnorm(r, L.vec("post_attention_norm.weight"), eps)
    fg, fu = L.mm(h2, "ffn_gate.weight"), L.mm(h2, "ffn_up.weight")
    out = r + L.mm(fg*ref.sigmoid(fg)*fu, "ffn_down.weight")
    return ref.rmsnorm(out, L.vec("nextn.shared_head_norm.weight"), eps)


def nl2(a, b):
    return float(np.linalg.norm(a-b))/max(float(np.linalg.norm(b)), 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True, help="zerv-mtp-check output directory")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    W = ref.Weights(a.model)
    eps = float(np.float32(W.meta["qwen35.attention.layer_norm_rms_epsilon"]))
    base = float(W.meta["qwen35.rope.freq_base"])
    index = json.loads((a.dump/"index.json").read_text())
    load = lambda name, n: np.fromfile(a.dump/name, dtype="<f4").astype(np.float64).reshape(n, -1)
    prompt, nxt, N, M = index["prompt"], index["next"], index["drafts"], index["commit_rows"]
    P = len(prompt)
    hrows = load("hrows.bin", P)
    vhn = load("verify_hn.bin", N+1)
    zero = np.zeros((1, H))
    scen = {}
    # A: rows 0..P-1 (prompt catch-up), row P = (h_{P-1}, t0), chain rows P+1..P+N-1.
    mo_a = [load(f"A-{j}-mo.bin", 1) for j in range(1, N+1)]
    scen["A"] = dict(tokens=prompt+[nxt[0]]+index["drafts_a"][:N-1],
                     hs=np.concatenate([zero, hrows]+mo_a[:N-1]), first=P)
    # B: A's rows 0..P, then the committed rows P+1..P+M (h = verify rows 0..M-1), chain.
    mo_b = [load(f"B-{j}-mo.bin", 1) for j in range(1, N+1)]
    scen["B"] = dict(tokens=prompt+[nxt[0]]+nxt[1:M+1]+index["drafts_b"][:N-1],
                     hs=np.concatenate([zero, hrows, vhn[:M]]+mo_b[:N-1]), first=P+M)
    projector = ref.Projector(W.path, capacity=max(len(s["tokens"]) for s in scen.values())*17408)
    report = dict(bound=BOUND, dump=str(a.dump), scenarios={}, passed=True)
    try:
        heads = {}
        for name, s in scen.items():
            hp = mtp_rows(W, projector, s["tokens"], s["hs"], eps, base)
            heads[name] = hp[s["first"]:s["first"]+N]
        logits = projector(W, "output.weight", np.concatenate([heads["A"], heads["B"]]))
    finally:
        projector.close()
    for si, name in enumerate(("A", "B")):
        drafts = index["drafts_"+name.lower()]
        steps = []
        for j in range(1, N+1):
            ref_h = heads[name][j-1]
            ref_l = logits[si*N+j-1]
            z_h = load(f"{name}-{j}-mo.bin", 1)[0]
            z_l = load(f"{name}-{j}-logits.bin", 1)[0]
            top = np.argsort(ref_l)[-2:]
            margin = float((ref_l[top[1]]-ref_l[top[0]])/max(abs(ref_l[top[1]]), 1e-30))
            ref_arg = int(np.argmax(ref_l))
            # The drafter's probability output: softmax max of zerv's own logits, in FP64.
            want_p = 1.0/float(np.sum(np.exp(z_l-z_l.max())))
            got_p = index["probs_"+name.lower()][j-1]
            step = dict(j=j, h_nl2=nl2(z_h, ref_h), logits_nl2=nl2(z_l, ref_l), draft=drafts[j-1], ref_argmax=ref_arg,
                        zerv_argmax=int(np.argmax(z_l)), top2_margin=margin, prob=got_p, prob_rel_error=abs(got_p-want_p)/want_p)
            ok = step["h_nl2"] <= BOUND and step["logits_nl2"] <= BOUND and step["zerv_argmax"] == drafts[j-1] and (ref_arg == drafts[j-1] or margin < 1e-4) \
                and step["prob_rel_error"] <= 1e-4
            step["ok"] = ok
            report["passed"] &= ok
            steps.append(step)
        report["scenarios"][name] = steps
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
