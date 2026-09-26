#!/usr/bin/env python3
"""Hand-scheduled RDNA3 (gfx1100) machine code for gemm_f16x (Q4_0 weights x f16 X -> f32).

Research tool (docs/research/native-isa-via-vulkan.md, docs/bench/2026-09-24-gemm-f16x-isa.md).
Emits clang (-target amdgcn-mesa-mesa3d -mcpu=gfx1100) assembler source for the whole kernel:
prologue, software-pipelined main loop and epilogue. The result replaces the code of the RADV
pipeline binary of src/model/gemm_f16x.comp (bench/isa_lab/isa_tool.py splice); the interface
(bindings, push constants, workgroup, wave32, 256 VGPRs, 36,864 B LDS) is unchanged.

Arithmetic is identical to gemm_f16x.comp: every accumulator receives its 16x16x16 f16 WMMAs in
ascending k, and every weight is f16(d*(q-8)) computed as pk_add(0x6400|q, -1032) then pk_mul by d.

  gen_f16x.py OUT.s [--prio none|wmma] [--dq-part N] [--no-pipeline-b]

The RADV compute ABI of this pipeline (read from ACO's listing of the SPIR-V kernel, Mesa 26.2.3):
  s0 descriptor set address (low; high = 0xffff8000), s1..s7 push constants a_base, a_rs,
  x_base, x_rs, y_base, y_rs, k; s8/s9 workgroup id x/y; s10 bits 20..24 = subgroup id;
  v0 = local invocation id x. Descriptors (16 B): +0x00 A, +0x10 Y, +0x20 io, +0x30 X16.
"""
import argparse, sys

# ---------------------------------------------------------------- registers
ACC = 32            # acc[i][j] = v[ACC + 32 i + 8 j .. +7]
BB = (160, 192)     # B fragment double buffer: buffer b, fragment j at BB[b] + 8 j
AF = 224            # A fragments: fragment i at AF + 8 i
V_LDSR = 1          # LDS read base of this lane's A row (bytes): (wm*64 + lane%16) * 144
V_LDSW = 2          # LDS store base of this thread's block: r*144 + b*64
V_AADDR = 3         # dword-aligned byte address of this thread's Q4_0 block in stage 0
V_X = 4             # v4..v7: X byte address of token row j (16 * xrow[j])
V_SEL01, V_SEL23, V_DSH = 8, 9, 10   # per-lane perm selectors and scale shift (alignment)
RAW = 11            # v11..v15: the 5 dwords holding the thread's 18-byte block
V_D = 0             # scale d (low half)
LO, HI = 16, 24     # v16..v23 low-nibble words, v24..v31 high-nibble words (f16 pairs)

S_N1 = 47           # n - 1 (prologue only)
S_ADESC, S_XDESC, S_YDESC = 16, 20, 24
S_STEPS, S_STAGE, S_XOFF, S_AOFF = 28, 29, 30, 31
S_MASK, S_C64, S_NEG = 32, 33, 34
S_T = 35            # scratch SGPRs s35..s47

LDS_STAGE = 128 * 144   # bytes per LDS stage (128 rows x 72 halves)
FRAG_ROWS = 16 * 144    # bytes between A fragments (16 rows)


def vr(base, n):
    return f"v[{base}:{base + n - 1}]" if n > 1 else f"v{base}"


class Ins:
    """One instruction with what the wait-count pass needs."""
    __slots__ = ("text", "kind", "reads", "writes")

    def __init__(self, text, kind="salu", reads=(), writes=()):
        self.text, self.kind, self.reads, self.writes = text, kind, frozenset(reads), frozenset(writes)


def rng(base, n):
    return range(base, base + n)


class Gen:
    ORDERS = {
        "ij": [(i, j) for i in range(4) for j in range(4)],
        "ji": [(i, j) for j in range(4) for i in range(4)],
        "diag": [(i, (i + d) % 4) for d in range(4) for i in range(4)],
    }

    def __init__(self, prio, abl=(), order="ij", probe=False):
        self.prio, self.abl, self.probe = prio, set(abl), probe  # abl: timing ablations (wrong results), see --abl
        self.order = self.ORDERS[order]

    # ---- instruction constructors (VGPR read/write sets are exact for the pass below)
    def wmma(self, i, j, b):
        a, bb, c = AF + 8 * i, BB[b] + 8 * j, ACC + 32 * i + 8 * j
        ins = [Ins(f"v_wmma_f32_16x16x16_f16 {vr(c, 8)}, {vr(a, 8)}, {vr(bb, 8)}, {vr(c, 8)}", "wmma",
                   list(rng(a, 8)) + list(rng(bb, 8)) + list(rng(c, 8)), rng(c, 8))]
        if self.prio == "wmma":
            ins = [Ins("s_setprio 1")] + ins + [Ins("s_setprio 0")]
        return ins

    def ds_load_a(self, i, lds_off):
        a = AF + 8 * i
        return [Ins(f"ds_load_b128 {vr(a + 4 * h, 4)}, v{V_LDSR} offset:{lds_off + 16 * h}", "lgkm",
                    [V_LDSR], rng(a + 4 * h, 4)) for h in range(2)]

    def load_b(self, b, j, xoff):
        base = BB[b] + 8 * j
        return [Ins(f"buffer_load_b128 {vr(base + 4 * h, 4)}, v{V_X + j}, s[{S_XDESC}:{S_XDESC + 3}], "
                    f"s{S_XOFF} offen offset:{xoff + 16 * h}", "vm", [V_X + j], rng(base + 4 * h, 4))
                for h in range(2)]

    def load_raw(self, soff, imm):
        so = f"s{soff}" if isinstance(soff, int) else soff
        return [Ins(f"buffer_load_b128 {vr(RAW, 4)}, v{V_AADDR}, s[{S_ADESC}:{S_ADESC + 3}], {so} offen"
                    + (f" offset:{imm}" if imm else ""), "vm", [V_AADDR], rng(RAW, 4)),
                Ins(f"buffer_load_b32 v{RAW + 4}, v{V_AADDR}, s[{S_ADESC}:{S_ADESC + 3}], {so} offen offset:{imm + 16}",
                    "vm", [V_AADDR], [RAW + 4])]

    def dequant(self, store_off):
        """Dequantize the thread's block (RAW) into LO/HI and store 64 bytes at V_LDSW + store_off.
        Byte j of the payload holds k = j (low nibble) and k = j + 16 (high nibble). Word i covers
        payload bytes 4i..4i+3; L[2i] = f16 pair (k=4i, 4i+1), L[2i+1] = (4i+2, 4i+3); H likewise +16."""
        v = []
        for i in range(4):
            v.append(Ins(f"v_perm_b32 v{LO + 2 * i}, v{RAW + i + 1}, v{RAW + i}, v{V_SEL01}", "valu",
                         [RAW + i + 1, RAW + i, V_SEL01], [LO + 2 * i]))
            v.append(Ins(f"v_perm_b32 v{LO + 2 * i + 1}, v{RAW + i + 1}, v{RAW + i}, v{V_SEL23}", "valu",
                         [RAW + i + 1, RAW + i, V_SEL23], [LO + 2 * i + 1]))
        v.append(Ins(f"v_lshrrev_b32 v{V_D}, v{V_DSH}, v{RAW}", "valu", [V_DSH, RAW], [V_D]))
        for k in range(8):
            v.append(Ins(f"v_lshrrev_b32 v{HI + k}, 4, v{LO + k}", "valu", [LO + k], [HI + k]))
        for k in range(8):
            v.append(Ins(f"v_and_or_b32 v{LO + k}, v{LO + k}, s{S_MASK}, s{S_C64}", "valu", [LO + k], [LO + k]))
        for k in range(8):
            v.append(Ins(f"v_and_or_b32 v{HI + k}, v{HI + k}, s{S_MASK}, s{S_C64}", "valu", [HI + k], [HI + k]))
        for base in (LO, HI):
            for k in range(8):
                v.append(Ins(f"v_pk_add_f16 v{base + k}, v{base + k}, s{S_NEG} op_sel_hi:[1,0]", "valu",
                             [base + k], [base + k]))
        for base, off in ((LO, 0), (HI, 32)):
            for k in range(8):
                v.append(Ins(f"v_pk_mul_f16 v{base + k}, v{base + k}, v{V_D} op_sel_hi:[1,0]", "valu",
                             [base + k, V_D], [base + k]))
            for h in range(2):
                v.append(Ins(f"ds_store_b128 v{V_LDSW}, {vr(base + 4 * h, 4)} offset:{store_off + off + 16 * h}",
                             "lgkm_store", [V_LDSW] + list(rng(base + 4 * h, 4))))
        return v

    def barrier(self):
        # CU mode: LDS executes the workgroup's requests in order, so (as ACO) only wait until
        # this wave's DS/VMEM instructions have read their VGPR sources.
        if "nodep" in self.abl:
            return [Ins("s_barrier", "barrier")]
        return [Ins("s_waitcnt_depctr 0xffe3", "wait"), Ins("s_barrier", "barrier")]

    # ---- one K stage (64 k): 4 parts of 16 WMMAs, i-major
    def stage(self, c, has_next, dq_part):
        """Stage with LDS buffer c. Entry: B of parts 0/1 pending in BB[0]/BB[1], A of part 0 pending.
        has_next: prefetch/dequantize/store the next stage into buffer c^1 and issue the next
        stage's part-0 A and part-0/1 B loads (then the entry state holds again)."""
        out = []
        xs = 128 * c  # X byte offset of this stage relative to S_XOFF
        if "stamp" in self.abl or "padmsg" in self.abl:
            out += [Ins("s_sendmsg_rtn_b64 s[62:63], sendmsg(MSG_RTN_GET_REALTIME)"), Ins("s_waitcnt lgkmcnt(0)")]
        if "stamp" in self.abl or "padmov" in self.abl:
            out += [Ins("v_mov_b32 v8, s62"), Ins("v_mov_b32 v9, s63")]
        if "stamp" in self.abl or "padst" in self.abl:
            out += [Ins(f"buffer_store_b64 v[8:9], v16, s[{S_YDESC}:{S_YDESC + 3}], s61 offen"), Ins("s_add_u32 s61, s61, 8")]
        if "padwait" in self.abl:
            out += [Ins("s_waitcnt vmcnt(0)")]
        if has_next and "noraw" not in self.abl:
            out += self.load_raw(S_AOFF, 36 * c)
        for q in range(4):
            b = q % 2
            gap = [[] for _ in range(16)]  # instructions after WMMA number n of this part
            dq = self.dequant((c ^ 1) * LDS_STAGE) if (has_next and q == dq_part) else []
            if "nodq" in self.abl:
                dq = [x for x in dq if x.kind != "valu"]
            if "nost" in self.abl:
                dq = [x for x in dq if x.kind != "lgkm_store"]
            if dq:  # spread over the part: gaps 0..14 (leave the last gap for the loads)
                per = -(-len(dq) // 15)
                for n in range(15):
                    gap[n] += dq[n * per:(n + 1) * per]
            order = self.order
            lastA = {i: max(n for n, (a, _) in enumerate(order) if a == i) for i in range(4)}
            lastB = {j: max(n for n, (_, bj) in enumerate(order) if bj == j) for j in range(4)}
            first_next_a = min(lastA.values())
            for n, (i, j) in enumerate(order):
                if n == lastB[j] and "nob" not in self.abl:  # B_j of this part is dead: load B_j of part q + 2
                    nq = q + 2
                    if nq < 4:
                        gap[n] += self.load_b(b, j, xs + 32 * nq)
                    elif has_next:
                        gap[n] += self.load_b(b, j, xs + 128 + 32 * (nq - 4))
                if n == lastA[i]:  # A_i of this part is dead: load A_i of part q + 1
                    bar_here = q == 3 and has_next and n == first_next_a and "nobar" not in self.abl
                    if bar_here and not ("bar2" in self.abl and c == 0):
                        gap[n] += self.barrier()
                    if "barx2" in self.abl and q == 1 and has_next and n == first_next_a:
                        gap[n] += self.barrier()
                    if "noa" in self.abl:
                        pass
                    elif q < 3:
                        gap[n] += self.ds_load_a(i, c * LDS_STAGE + i * FRAG_ROWS + 32 * (q + 1))
                    elif has_next:
                        gap[n] += self.ds_load_a(i, (c ^ 1) * LDS_STAGE + i * FRAG_ROWS)
            for n, (i, j) in enumerate(order):
                out += self.wmma(i, j, b)
                out += gap[n]
        return out


def insert_waits(seq, vm, lgkm, name):
    """Insert s_waitcnt before every instruction that reads (or overwrites) VGPRs of a pending
    load. vm/lgkm: pending entries (oldest first) at entry; an entry is the set of VGPRs it
    writes (empty for stores). Both counters complete in order (no SMEM in these segments)."""
    out = []
    vm, lgkm = list(vm), list(lgkm)
    for ins in seq:
        need = set(ins.reads) | set(ins.writes)
        w = {}
        for cname, lst in (("vmcnt", vm), ("lgkmcnt", lgkm)):
            hit = max((idx for idx, regs in enumerate(lst) if regs & need), default=None)
            if hit is not None:
                w[cname] = len(lst) - hit - 1
                del lst[:hit + 1]
        if w:
            out.append(Ins("s_waitcnt " + " ".join(f"{k}({v})" for k, v in w.items()), "wait"))
        out.append(ins)
        if ins.kind == "vm":
            vm.append(ins.writes)
        elif ins.kind == "lgkm":
            lgkm.append(ins.writes)
        elif ins.kind == "lgkm_store":
            lgkm.append(frozenset())
        if len(vm) > 63 or len(lgkm) > 63:
            raise SystemExit(f"{name}: counter overflow")
    return out, vm, lgkm


def entry_state():
    vm = [frozenset(rng(BB[b] + 8 * j + 4 * h, 4)) for b in range(2) for j in range(4) for h in range(2)]
    lgkm = [frozenset(rng(AF + 8 * i + 4 * h, 4)) for i in range(4) for h in range(2)]
    return vm, lgkm


def check_exit(vm, lgkm, name):
    """The pending state at a stage exit must be the entry state (plus stores, which write
    no VGPRs); otherwise the next stage's computed wait counts would be wrong."""
    evm, elg = entry_state()
    ok = vm[len(vm) - len(evm):] == evm and all(not r for r in vm[:len(vm) - len(evm)]) and \
        lgkm[len(lgkm) - len(elg):] == elg and all(not r for r in lgkm[:len(lgkm) - len(elg)])
    if not ok:
        raise SystemExit(f"{name}: exit pending state differs from the entry state")


def text(seq):
    return [(i.text if i.kind == "label" else "\t" + i.text) for i in seq]


def prologue(g):
    """Setup, stage 0 in LDS buffer 0, and the loads of the first stage. Ends in the entry state.

    The row count n lives in the host-visible io buffer; at dispatch start all waves of the first
    round read it at once and it takes ~15 us (docs/bench/2026-09-24-gemm-f16x-isa.md). So it is
    loaded early and consumed late: the stage-0 A fetch, dequantization and barrier do not need it,
    and the first B loads use unclamped rows, reissued with rows >= n clamped to n - 1 only when the
    tile reaches past n (t0 + 256 > n)."""
    t = lambda k: S_T + k
    p = ["s_mov_b32 s11, s1", "s_movk_i32 s1, 0x8000",   # ABI, as ACO: s[0:1] = descriptor set
         "s_load_b128 s[12:15], s[0:1], 0x20",
         f"s_load_b128 s[{S_ADESC}:{S_ADESC + 3}], s[0:1], 0x0",
         f"s_load_b128 s[{S_XDESC}:{S_XDESC + 3}], s[0:1], 0x30",
         f"s_load_b128 s[{S_YDESC}:{S_YDESC + 3}], s[0:1], 0x10",
         "s_lshl_b32 s9, s9, 8",                        # t0
         "s_lshl_b32 s8, s8, 7",                        # m0
         f"s_lshr_b32 s{S_STEPS}, s7, 6",
         f"s_bfe_u32 s{t(0)}, s10, 0x10014",            # wm
         f"s_bfe_u32 s{t(1)}, s10, 0x40015",            # wn
         f"s_lshl_b32 s{t(1)}, s{t(1)}, 6",
         f"s_add_u32 s{t(1)}, s9, s{t(1)}",             # tr = t0 + wn*64 (kept to the epilogue)
         f"s_mul_i32 s{t(2)}, s{t(0)}, 0x2400",         # wm*64 rows * 144 B
         f"s_lshl_b32 s{t(0)}, s{t(0)}, 6",             # wm*64 (kept to the epilogue)
         f"s_mov_b32 s{S_MASK}, 0xf000f", f"s_mov_b32 s{S_C64}, 0x64006400", f"s_movk_i32 s{S_NEG}, 0xe408",
         f"s_mov_b32 s{S_STAGE}, 0", f"s_mov_b32 s{S_XOFF}, 0", f"s_mov_b32 s{S_AOFF}, 36"]
    lane, lr, r, b, a, o = 160, 161, 162, 163, 164, 165
    p += [f"v_mbcnt_lo_u32_b32 v{lane}, -1, 0", f"v_and_b32 v{lr}, 15, v{lane}",
          f"v_mad_u32_u24 v{V_LDSR}, 0x90, v{lr}, s{t(2)}",
          f"v_lshrrev_b32 v{r}, 1, v0", f"v_and_b32 v{b}, 1, v0",
          f"v_add_nc_u32 v{a}, s8, v{r}", f"v_mul_lo_u32 v{a}, v{a}, s2", f"v_add_nc_u32 v{a}, s11, v{a}",
          f"v_mad_u32_u24 v{a}, 18, v{b}, v{a}",        # byte address of the block in stage 0
          f"v_and_b32 v{V_AADDR}, -4, v{a}", f"v_and_b32 v{o}, 2, v{a}",
          f"v_lshlrev_b32 v{V_DSH}, 3, v{o}", f"v_mul_u32_u24 v{o}, 0x10001, v{o}",
          f"v_add_nc_u32 v{V_SEL01}, 0xc030c02, v{o}", f"v_add_nc_u32 v{V_SEL23}, 0xc050c04, v{o}",
          f"v_mul_u32_u24 v{V_LDSW}, 0x90, v{r}", f"v_lshl_add_u32 v{V_LDSW}, v{b}, 6, v{V_LDSW}",
          "s_waitcnt lgkmcnt(0)"]                        # descriptors
    if g.probe:
        p += ["s_sendmsg_rtn_b64 s[64:65], sendmsg(MSG_RTN_GET_REALTIME)", "s_waitcnt lgkmcnt(0)"]
    p += ["s_bitset0_b32 s13, 14", "s_buffer_load_b32 s12, s[12:15], 0x8",   # n: in flight until below
          f"s_lshr_b32 s{t(3)}, s{S_XDESC + 2}, 4", f"s_lshr_b32 s{t(4)}, s7, 3", f"s_sub_i32 s{t(3)}, s{t(3)}, s{t(4)}"]

    def xaddr(clamp, lr):
        q = []
        for j in range(4):
            q += [f"v_add3_u32 v{V_X + j}, s{t(1)}, {16 * j}, v{lr}"]
            if clamp:
                q += [f"v_min_u32 v{V_X + j}, s{S_N1}, v{V_X + j}"]
            q += [f"v_mul_lo_u32 v{V_X + j}, v{V_X + j}, s4", f"v_add_nc_u32 v{V_X + j}, s3, v{V_X + j}",
                  f"v_lshrrev_b32 v{V_X + j}, 3, v{V_X + j}", f"v_min_u32 v{V_X + j}, s{t(3)}, v{V_X + j}",
                  f"v_lshlrev_b32 v{V_X + j}, 4, v{V_X + j}"]
        return q

    def bloads():
        q = []
        for bq in range(2):
            for j in range(4):
                q += g.load_b(bq, j, 32 * bq)
        return q

    seq = [Ins(x) for x in p + xaddr(False, lr)]
    seq += g.load_raw("0", 0)
    seq += bloads()                                   # rows unclamped (fixed below if the tile passes n)
    for k in range(ACC, ACC + 128, 2):
        seq.append(Ins(f"v_dual_mov_b32 v{k}, 0 :: v_dual_mov_b32 v{k + 1}, 0", "valu", (), (k, k + 1)))
    seq += g.dequant(0)
    seq += g.barrier()
    seq += [Ins("s_waitcnt lgkmcnt(0)")]              # n (and this wave's LDS stores)
    if g.probe:
        seq += [Ins("s_sendmsg_rtn_b64 s[68:69], sendmsg(MSG_RTN_GET_REALTIME)"), Ins("s_waitcnt lgkmcnt(0)"),
                Ins("s_mov_b32 s66, 0"), Ins("s_mov_b32 s67, 0")]
    seq += [Ins("s_cmp_lt_u32 s9, s12"), Ins("s_cbranch_scc0 .Lend"),     # t0 >= n: nothing to do
            Ins(f"s_add_u32 s{S_N1}, s12, -1"),
            Ins(f"s_add_u32 s{t(5)}, s9, 0x100"), Ins(f"s_cmp_le_u32 s{t(5)}, s12"), Ins("s_cbranch_scc1 .Lxok")]
    # (the B loads have overwritten v160.., so the rare path recomputes lane % 16 in v16/v17,
    # free after the stage-0 dequantization)
    out, vm, lgkm = insert_waits(seq, [], [], "prologue")
    # rare path: the tile reaches past n. Wait for the unclamped loads, reload with clamped rows.
    fix = [Ins("s_waitcnt vmcnt(0)"), Ins("v_mbcnt_lo_u32_b32 v16, -1, 0"), Ins("v_and_b32 v17, 15, v16")] + \
        [Ins(x) for x in xaddr(True, 17)] + bloads()
    fout, fvm, _ = insert_waits(fix, [], [], "prologue fixup")
    if fvm != vm[len(vm) - len(fvm):] or any(vm[:len(vm) - len(fvm)]):
        raise SystemExit("prologue: fixup path pending state differs")
    tail = []
    for i in range(4):
        tail += g.ds_load_a(i, i * FRAG_ROWS)
    tout, vm, lgkm = insert_waits(tail, vm, lgkm, "prologue tail")
    check_exit(vm, lgkm, "prologue")
    return out + fout + [Ins(".Lxok:", "label")] + tout


def probe_epilogue():
    """Probe build: lane 0 of every wave stores 8 dwords at Y dword ((wg_y*gx + wg_x)*8 + wave)*8:
    HW_ID1, HW_ID2, start realtime (lo, hi), loop-end realtime (lo, hi), wave, 0."""
    e = ["s_sendmsg_rtn_b64 s[50:51], sendmsg(MSG_RTN_GET_REALTIME)", "s_waitcnt lgkmcnt(0)",
         "s_getreg_b32 s52, hwreg(HW_REG_HW_ID1)", "s_getreg_b32 s53, hwreg(HW_REG_HW_ID2)",
         "s_bfe_u32 s54, s10, 0x50014",                       # wave
         "s_lshr_b32 s55, s8, 7", "s_lshr_b32 s56, s9, 8",   # wg_x, wg_y
         f"s_lshr_b32 s57, s6, 7",                            # grid x = y_rs / 128 (y_rs = M in the lab)
         "s_mul_i32 s56, s56, s57", "s_add_u32 s55, s55, s56", "s_lshl_b32 s55, s55, 3", "s_add_u32 s55, s55, s54",
         "s_lshl_b32 s55, s55, 5", "s_lshl_b32 s58, s5, 2", "s_add_u32 s55, s55, s58",   # byte offset + y_base*4
         "v_mov_b32 v0, s52", "v_mov_b32 v1, s53", "v_mov_b32 v2, s48", "v_mov_b32 v3, s49",
         "v_mov_b32 v4, s50", "v_mov_b32 v5, s51", "v_mov_b32 v6, s54", "v_mov_b32 v7, 0",
         "v_mbcnt_lo_u32_b32 v8, -1, 0", "v_cmp_eq_u32 vcc_lo, 0, v8", "s_and_saveexec_b32 s59, vcc_lo",
         "v_mov_b32 v9, 0",
         f"buffer_store_b128 v[0:3], v9, s[{S_YDESC}:{S_YDESC + 3}], s55 offen",
         f"buffer_store_b128 v[4:7], v9, s[{S_YDESC}:{S_YDESC + 3}], s55 offen offset:16",
         # prologue stamps at byte 0x80000 + slot*32: after SMEM (n), raw A arrived, after the barrier
         "v_mov_b32 v0, s64", "v_mov_b32 v1, s65", "v_mov_b32 v2, s66", "v_mov_b32 v3, s67",
         "v_mov_b32 v4, s68", "v_mov_b32 v5, s69", "v_mov_b32 v6, 0", "v_mov_b32 v7, 0",
         "s_add_u32 s55, s55, 0x80000",
         f"buffer_store_b128 v[0:3], v9, s[{S_YDESC}:{S_YDESC + 3}], s55 offen",
         f"buffer_store_b128 v[4:7], v9, s[{S_YDESC}:{S_YDESC + 3}], s55 offen offset:16"]
    return ["\t" + x for x in e]


def epilogue(abl=()):
    t = lambda k: S_T + k
    e = [f"v_mbcnt_lo_u32_b32 v0, -1, 0", "v_and_b32 v1, 15, v0", "v_lshrrev_b32 v0, 4, v0",
         "v_mul_lo_u32 v1, v1, s6", "v_add_nc_u32 v1, v1, v0"]      # (lane%16)*y_rs + lane/16
    for j in range(4):  # y_base + (tr + 16 j)*y_rs + m0 + wm*64, then (+ lane part) * 4
        e += [f"s_add_u32 s{t(5)}, s{t(1)}, {16 * j}", f"s_mul_i32 s{t(5)}, s{t(5)}, s6",
              f"s_add_u32 s{t(5)}, s5, s{t(5)}", f"s_add_u32 s{t(5)}, s{t(5)}, s8",
              f"v_add3_u32 v{2 + j}, v1, s{t(5)}, s{t(0)}", f"v_lshlrev_b32 v{2 + j}, 2, v{2 + j}"]
    for i in range(4 if "noepi" not in abl else 0):
        for j in range(4):
            for r in range(8):
                e.append(f"buffer_store_b32 v{ACC + 32 * i + 8 * j + r}, v{2 + j}, s[{S_YDESC}:{S_YDESC + 3}], 0 offen"
                         + (f" offset:{64 * i + 8 * r}" if (i or r) else ""))
    return ["\t" + x for x in e]


def epilogue_lds(abl=()):
    """Stores through LDS: per chunk (token fragment j, M half ip) the wave writes its 32 (m) x 16 (t)
    f32 values into its own 2,304 B region of the LDS buffer the last stage did not read (s{S_T+7}
    = its byte base; rows of 144 B, conflict-free), reads them back as 16-byte row pieces and stores
    4 token rows x 128 B per buffer_store_b128. That buffer was last read before the previous
    stage's barrier, so no barrier is needed; LDS executes one wave's requests in order."""
    t = lambda k: S_T + k
    if "noepi" in abl:
        return []
    e = [f"s_bfe_u32 s{t(8)}, s10, 0x50014", f"s_mulk_i32 s{t(8)}, 0x900", f"s_add_u32 s{t(7)}, s{t(7)}, s{t(8)}",
         "v_mbcnt_lo_u32_b32 v0, -1, 0", "v_and_b32 v1, 15, v0", "v_lshrrev_b32 v2, 4, v0",
         f"v_mad_u32_u24 v3, 0x90, v1, s{t(7)}", "v_lshl_add_u32 v3, v2, 2, v3",          # write: t*144 + h*4
         "v_lshrrev_b32 v4, 3, v0", "v_and_b32 v5, 7, v0",
         f"v_mad_u32_u24 v6, 0x90, v4, s{t(7)}", "v_lshl_add_u32 v6, v5, 4, v6",          # read: (l/8)*144 + 16(l%8)
         "v_mul_lo_u32 v7, v4, s6", "v_lshl_add_u32 v7, v5, 2, v7", "v_lshlrev_b32 v7, 2, v7",  # ((l/8)*y_rs + 4(l%8))*4
         f"s_add_u32 s{t(9)}, s5, s8", f"s_add_u32 s{t(9)}, s{t(9)}, s{t(0)}"]          # y_base + m0 + wm*64
    seq = [Ins(x) for x in e]
    regs = list(range(160, 256)) + list(range(8, 32))  # 16 VGPRs per chunk for chunks 0..6
    chunk = 0
    for j in range(4):
        for ip in range(2):
            for ii in range(2):
                i = 2 * ip + ii
                for r in range(0, 8, 2):
                    a = ACC + 32 * i + 8 * j + r
                    seq.append(Ins(f"ds_store_2addr_b32 v3, v{a}, v{a + 1} offset0:{16 * ii + 2 * r} offset1:{16 * ii + 2 * r + 2}",
                                   "lgkm_store", [3, a, a + 1]))
            base = regs[16 * chunk:16 * chunk + 16] if chunk < 7 else None
            if base is None:  # chunk 7 reuses chunk 0's VGPRs: wait until those stores read them
                base = regs[0:16]
                seq.append(Ins("s_waitcnt_depctr 0xffe3", "wait"))
            for k in range(4):
                d = base[4 * k]
                assert base[4 * k:4 * k + 4] == list(range(d, d + 4))
                seq.append(Ins(f"ds_load_b128 v[{d}:{d + 3}], v6 offset:{576 * k}", "lgkm", [6], range(d, d + 4)))
            for k in range(4):
                d = base[4 * k]
                seq += [Ins(f"s_add_u32 s{t(10)}, s{t(1)}, {16 * j + 4 * k}"), Ins(f"s_mul_i32 s{t(10)}, s{t(10)}, s6"),
                        Ins(f"s_add_u32 s{t(10)}, s{t(10)}, s{t(9)}"), Ins(f"s_lshl_b32 s{t(10)}, s{t(10)}, 2")]
                if "noepi" not in abl:
                    seq.append(Ins(f"buffer_store_b128 v[{d}:{d + 3}], v7, s[{S_YDESC}:{S_YDESC + 3}], s{t(10)} offen"
                                   + (f" offset:{128 * ip}" if ip else ""), "vmstore", [7] + list(range(d, d + 4))))
            chunk += 1
    out, _, _ = insert_waits(seq, [], [], "epilogue")
    return text(out)


def generate(prio="wmma", dq_part=2, abl=(), order="ij", probe=False):
    g = Gen(prio, abl, order, probe)
    L = ["\t.text"]
    if probe:
        L += ["\ts_sendmsg_rtn_b64 s[48:49], sendmsg(MSG_RTN_GET_REALTIME)"]
    L += text(prologue(g))
    if g.abl & {"stamp", "padst"}:  # s61 = y_base*4 + 1 MiB + (wg linear * 8 + wave) * 1024; v16 = 0
        L += ["\ts_bfe_u32 s60, s10, 0x50014", "\ts_lshr_b32 s61, s8, 7", "\ts_lshr_b32 s59, s9, 8", "\ts_lshr_b32 s58, s6, 7",
              "\ts_mul_i32 s59, s59, s58", "\ts_add_u32 s61, s61, s59", "\ts_lshl_b32 s61, s61, 3", "\ts_add_u32 s61, s61, s60",
              "\ts_lshl_b32 s61, s61, 10", "\ts_add_u32 s61, s61, 0x100000", "\ts_lshl_b32 s59, s5, 2", "\ts_add_u32 s61, s61, s59",
              "\tv_mov_b32 v16, 0"]
    for x in g.abl:
        if x.startswith("active"):  # only the first N waves per SIMD run the loop
            L += ["\ts_bfe_u32 s60, s10, 0x40015", f"\ts_cmp_ge_u32 s60, {int(x[6:])}", "\ts_cbranch_scc1 .Lepi"]
    L += [f"\ts_cmp_eq_u32 s{S_STEPS}, 1", "\ts_cbranch_scc1 .Llast0"]
    bodies = {}
    for c in range(2):
        for nxt in (True, False):
            vm, lgkm = entry_state()
            seq, evm, elg = insert_waits(g.stage(c, nxt, dq_part), vm, lgkm, f"stage c={c} next={nxt}")
            if nxt and not g.abl:
                check_exit(evm, elg, f"stage c={c}")
            bodies[(c, nxt)] = seq
    # loop over stage pairs: s even at .Lloop; s + 1 == steps - 1 -> .Llast1; s + 2 == steps - 1 -> .Llast0
    if "align" in g.abl:  # timing experiment: hot loop start on a 64-byte boundary (s_nop padding)
        L += ["\t.p2align 6"]
    L += [".Lloop:"] + text(bodies[(0, True)])
    L += [f"\ts_add_u32 s{S_T + 6}, s{S_STAGE}, 2", f"\ts_cmp_eq_u32 s{S_T + 6}, s{S_STEPS}", "\ts_cbranch_scc1 .Llast1"]
    L += text(bodies[(1, True)])
    L += [f"\ts_add_u32 s{S_STAGE}, s{S_STAGE}, 2", f"\ts_addk_i32 s{S_XOFF}, 0x100", f"\ts_add_u32 s{S_AOFF}, s{S_AOFF}, 72",
          f"\ts_add_u32 s{S_T + 6}, s{S_STAGE}, 1", f"\ts_cmp_lg_u32 s{S_T + 6}, s{S_STEPS}", "\ts_cbranch_scc1 .Lloop"]
    lds_epi = "epi32" not in g.abl
    L += [".Llast0:"] + text(bodies[(0, False)]) + ([f"\ts_movk_i32 s{S_T + 7}, {LDS_STAGE}"] if lds_epi else []) + ["\ts_branch .Lepi"]
    L += [".Llast1:"] + text(bodies[(1, False)]) + ([f"\ts_mov_b32 s{S_T + 7}, 0"] if lds_epi else [])
    L += [".Lepi:"] + (probe_epilogue() if probe else epilogue_lds(g.abl) if lds_epi else epilogue(g.abl))
    L += [".Lend:", "\ts_nop 0", "\ts_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", "\ts_endpgm"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prio", default="wmma", choices=["none", "wmma"])
    ap.add_argument("--dq-part", type=int, default=2, choices=[1, 2])
    ap.add_argument("--abl", default="", help="comma list of timing ablations (results become wrong): "
                    "nodq (no dequant VALU), nobar (no barrier), noa (no A LDS loads), nob (no B loads), "
                    "noraw (no raw A loads)")
    ap.add_argument("--order", default="ij", choices=sorted(Gen.ORDERS))
    ap.add_argument("--probe", action="store_true", help="store per-wave hw ids and timestamps instead of Y")
    a = ap.parse_args()
    open(a.out, "w").write(generate(a.prio, a.dq_part, [x for x in a.abl.split(",") if x], a.order, a.probe))
