//! Resident Qwen3.8 forward pass on the native Vulkan driver. One sequence per Model,
//! externally serialized. One pre-recorded command per decode step; no per-step
//! allocation, descriptor update or re-recording.
const std = @import("std");
const gpu = @import("../gpu/root.zig");
const matvec = @import("../matvec/root.zig");
const gguf = @import("../artifact/gguf.zig");
const config = @import("config.zig");
const layout = @import("layout.zig");
const gemm = @import("gemm.zig");
const attention = @import("attention.zig");

pub const Options = struct {
    context: u32 = 8192,
    /// Weight bank, state arena and KV buffer upper bound (must stay below 4 GiB for uint32
    /// addressing).
    bank_capacity: u64 = 0xf000_0000,
    /// Most bytes of one KV buffer (0 = `bank_capacity`). Smaller values force the KV
    /// caches into more buffers (tests of the split; docs/specs/model.md, "KV buffers").
    kv_capacity: u64 = 0,
    staging_bytes: u64 = 64 * 1024 * 1024,
    timeout_ns: u64 = 60 * std.time.ns_per_s,
    /// Maximum prompt tokens per batched prefill chunk (0 = token-by-token only).
    prefill_rows: u32 = 512,
    /// Prefill projection arithmetic. `.f16` (explicit, lower precision; block 14) needs a
    /// device opened with cooperative matrices and subgroups of exactly 64.
    prefill_precision: gemm.Precision = .fp32,
    /// Device slots holding copies of the recurrent and convolution state
    /// (`snapshot_bytes` each) for prefix caching (docs/specs/prefix-cache.md).
    snapshots: u32 = 0,
    /// Where the snapshot slots live: `.device` (VRAM; save/load 0.5 ms) or `.host`
    /// (system RAM, no VRAM; save 9.8 ms and load 15.3 ms over PCIe, measured 2026-09-24).
    snapshot_memory: gpu.Location = .device,
    /// Where the token embedding (Q4_0, 682 MiB) lives: `.host` (system RAM; the embed
    /// kernels read one row per token over PCIe) or `.device` (weight bank 0).
    embedding_memory: gpu.Location = .host,
    /// With `context = context_max`: VRAM left free for other processes (desktop, other
    /// GPU users), beyond `vram_headroom`. Default as llama-server's `--fit-target`.
    context_reserve: u64 = 1024 * 1024 * 1024,
    /// Attention KV cache element type (docs/specs/model.md, "KV precision"): `.f32`
    /// (exact) or `.f16` (half the KV bytes; needs a device opened with `storage16`).
    kv_type: layout.KvType = .f32,
    /// Tokens per KV page (docs/specs/concurrent.md, "Addressing"; `--kv-page-tokens`): a
    /// multiple of `layout.kv_page_quantum`, or 0 for one page holding the whole context (the
    /// layout before paging). Values are identical for every page size.
    kv_page_tokens: u32 = layout.default_kv_page,
    /// Fuse the decode FFN input (gate, up and swiglu in one dispatch per layer where the
    /// two tensors share a bank; docs/specs/model.md). Values are identical either way;
    /// false records the separate projections and the swiglu kernel.
    decode_fusion: bool = true,
    /// The same fusion in the speculative verify pass (multi-row gate, up and swiglu in one
    /// dispatch per layer; `matvec.SwigluRowsPipeline`). Values are identical either way.
    verify_fusion: bool = true,
    /// Decode and verify projection accumulation (docs/specs/matvec-push.md): `.fma`
    /// (default) or `.separate` (the pre-FMA arithmetic and verify table).
    matvec_accumulation: matvec.Accumulation = .fma,
    /// Speculative verification rows (block 17b, docs/specs/speculative.md): 0 = off (no
    /// verify buffers or commands); 2..`matvec.max_rows` = drafts + 1. Needs
    /// `prefill_rows` >= this.
    verify_rows: u32 = 0,
    /// Load the MTP (nextn) layer and record the draft commands (block 17b,
    /// docs/specs/speculative.md): up to `verify_rows - 1` drafts per step. Needs
    /// `verify_rows` >= 2. Off: no MTP weights, regions or commands.
    mtp: bool = false,
    /// DeltaNet kernels with a separate state store offset (`delta`, `delta_b`; default),
    /// or the previous modules (`delta_legacy`, `delta_b_legacy`: 128 VGPRs spilled to
    /// scratch). Values are identical (docs/bench/2026-09-24-delta-spill.md).
    delta_state_out: bool = true,
    /// Machine code of the f16-mode gemm_f16x kernel (docs/specs/prefill.md, "Native
    /// gemm_f16x machine code"): `.spirv` (the previous behaviour) or `.native` (our RDNA3
    /// binary when the driver's global key matches, else SPIR-V; the device must be opened
    /// with `pipeline_binaries`; without it, SPIR-V). Values are identical either way.
    gemm_code: gemm.Code = .native,
    /// Draft head vocabulary (docs/specs/speculative.md, "Draft vocabulary"): the draft
    /// logits and argmax cover token ids 0..draft_vocab-1 only; 0 = the full vocabulary.
    /// Verification is unchanged, so outputs are too; acceptance and draft cost change.
    draft_vocab: u32 = 0,
    /// Sequence slots (docs/specs/concurrent.md, "18b.2 design"): each has its recurrent
    /// state and KV page table; `select` picks the one the single-sequence operations use.
    /// 1 = the single-sequence model, byte for byte. More than 1 needs the MTP off.
    slots: u32 = 1,
    /// Rows of the largest batched decode command (`decodeBatch`; 0 = none), at most
    /// `layout.io.batch_max` and the prefill rows.
    batch_rows: u32 = 0,
    /// KV pool pages shared by the slots (0: `slots` x the pages of `context`, each slot
    /// owning its share; otherwise slots beyond the pool start without pages).
    kv_pages: u32 = 0,
    /// Batched decode projection arithmetic (`--decode-precision`, docs/specs/concurrent.md
    /// "18e design"): `.f32` (exact, the single-sequence arithmetic) or `.f16` (opt-in: the
    /// f16 prefill arithmetic, f16 x f16 -> f32 WMMA, for the projections the prefill f16
    /// mode covers; batch-invariant within the mode). `.f16` needs `batch_rows` > 0 and a
    /// device with cooperative matrices and subgroups of 64. `step`/`verify` stay FP32.
    decode_precision: DecodePrecision = .f32,
};
pub const DecodePrecision = enum { f32, f16 };
pub const Error = gpu.Error || matvec.Error || config.Error || layout.Error || gemm.Error || error{ InvalidToken, ContextFull, TooManyPipelines, InvalidTensorBytes, CaptureFull, PrefillDisabled, ProbeFailed, InvalidPlan, UnsupportedDevice, InvalidContext, InvalidSnapshot, InsufficientVram, VerifyDisabled, VerifyPending, NoVerify, MtpDisabled, VramBudgetUnknown, InvalidDraftVocab, MtpNeedsOneSlot, InvalidBatch, InvalidSlot, PagesMissing, PageInUse };

/// One row of a batched decode (`Model.decodeBatch`): a sequence slot and its next token.
pub const BatchRow = struct { slot: u32, token: u32 };
/// KV pool pages at most (the host's page-owner table).
pub const max_pool_pages = 1 << 16;
const no_owner: u8 = 0xff;

/// Bytes of one recurrent-state snapshot: every DeltaNet state and convolution history.
/// The attention KV is not copied; it stays valid in the state arena below `position`.
pub const snapshot_bytes: u64 = (@as(u64, layout.State.ssm_words) + layout.State.conv_words) * 4;
pub const max_snapshots = 64;
/// Device-local memory beyond the model's buffers that `init` requires to be free:
/// driver objects (pipelines, descriptors, command buffers) and allocation padding.
pub const vram_headroom: u64 = 256 * 1024 * 1024;

/// `Options.context` value that asks `init` for the largest context that fits: free VRAM
/// (less `vram_headroom` and `Options.context_reserve`) and the device's allocation cap,
/// after every other buffer the options ask for, at most the trained context. Needs the
/// driver's memory budget.
pub const context_max: u32 = 0;
/// Allocation-cap slack for driver size alignment of the model's buffers.
const cap_margin: u64 = 16 * 1024 * 1024;

/// The context-independent inputs of the context-dependent buffers.
pub const ContextShape = struct { kv_capacity: u64, rows: u32, part_words: u64, x16: bool, decode_rows: u32, mtp: bool, kv: layout.KvType = .f32, kv_page: u32 = layout.default_kv_page, verify: bool = false, slots: u32 = 1, kv_pages: u32 = 0 };

/// Balanced groups of at most `cap` rows over `n` rows (batched decode projections):
/// ceil(n / cap) groups whose sizes differ by at most one.
const RowGroups = struct {
    n: u32,
    groups: u32,
    index: u32 = 0,
    first: u32 = 0,
    fn init(n: u32, cap: u32) RowGroups {
        return .{ .n = n, .groups = (n + cap - 1) / cap };
    }
    fn next(self: *RowGroups) ?struct { first: u32, count: u32 } {
        if (self.index == self.groups) return null;
        const count = self.n / self.groups + @intFromBool(self.index < self.n % self.groups);
        const first = self.first;
        self.first += count;
        self.index += 1;
        return .{ .first = first, .count = count };
    }
};

/// Multi-row input rows the projections are validated for: verify rows or batched decode
/// rows, whichever is larger (0: none); `rowCount` of them per dispatch at most.
fn rowSpan(options: Options) u32 {
    return @max(options.verify_rows, options.batch_rows);
}
fn rowCount(options: Options) u32 {
    return @min(matvec.max_rows, rowSpan(options));
}
/// Rows of the decode attention scratch.
fn decodeRows(options: Options) u32 {
    return @max(rowSpan(options), 1);
}

/// Bytes of the buffers that grow with the context: the KV caches, the state arena and
/// the activation arena. Errors when a layout cannot hold that context.
pub fn contextBytes(context: u32, shape: ContextShape) layout.Error!u64 {
    const s = try layout.stateWith(context, shape.kv_capacity, shape.mtp, shape.kv, shape.kv_page, shape.slots, shape.kv_pages);
    const a = try layout.actWith(context, shape.rows, shape.part_words, shape.x16, shape.decode_rows, shape.mtp, shape.verify, shape.slots);
    return a.words * 4 + s.words * 4 + s.kvBytes();
}

/// The largest multiple of 32 up to `limit` whose context-dependent buffers fit `room`
/// bytes (0: not even 32). Bytes grow with the context and layouts fail only above a
/// threshold, so a binary search over multiples of 32 is exact.
pub fn fitContext(room: u64, limit: u32, shape: ContextShape) u32 {
    var lo: u32 = 0; // fits (in units of 32; 0 = none)
    var hi: u32 = limit / 32 + 1; // does not fit
    while (hi - lo > 1) {
        const mid = lo + (hi - lo) / 2;
        const fits = if (contextBytes(mid * 32, shape)) |bytes| bytes <= room else |_| false;
        if (fits) lo = mid else hi = mid;
    }
    return lo * 32;
}

/// Recorded work groups, in execution order, reported to a `Probe`.
pub const Phase = enum { embed, norm, attn_in, qkprep, scores, pv, attention, gate, attn_out, lin_in, conv, delta, lin_out, ffn_in, swiglu, ffn_down, output };
/// Instrumentation hook called while recording, after each phase's dispatches and before
/// the following barrier (e.g. to write a GPU timestamp). Not called during execution.
pub const Probe = struct {
    context: *anyopaque,
    mark: *const fn (context: *anyopaque, commands: *gpu.Commands, phase: Phase, layer: i32) error{ProbeFailed}!void,
};
/// Optional recording extras; the production commands use none.
pub const Hooks = struct { capture: ?*Capture = null, probe: ?Probe = null };

const KernelId = enum { embed, norm, qkprep, conv, delta, swiglu, zero, reduce, embed_b, qk_b, gate, conv_b, delta_b, attn_scores, attn_pv, attn_combine, gnorm_b, rowcopy, argmax_a, argmax_b, copy2d, flash, attn_gmax, attn_cblock, delta_legacy, delta_b_legacy };
/// Kernels that read or write the KV cache: bound to a KV buffer (binding 2) instead of
/// the state arena; `kernels[id]` binds KV buffer 0, `kv_kernels[g - 1]` buffer g.
const kv_kernel_ids = [_]KernelId{ .qkprep, .qk_b, .attn_scores, .attn_pv, .flash };
fn kvSlot(id: KernelId) ?usize {
    for (kv_kernel_ids, 0..) |k, i| if (k == id) return i;
    return null;
}
const kernel_count = @typeInfo(KernelId).@"enum".fields.len;
const max_pipelines = 24;
const max_gemm_pipelines = 48; // banks x formats x tiles/kernels; the device caps all kernels at gpu.max_kernels

/// A recorded prefill command: the most chunk rows it handles. Plans differ only in
/// their split-K choices (sized for their row tiles), so a chunk runs on the smallest
/// plan covering it (docs/specs/prefill.md, block 13e).
pub const Plan = struct { rows: u32 };
pub const max_plans = plan_table.len;
/// Ascending; the last takes the remaining rows up to `Options.prefill_rows`.
const plan_table = [_]Plan{ .{ .rows = 32 }, .{ .rows = 64 }, .{ .rows = 128 }, .{ .rows = 256 }, .{ .rows = std.math.maxInt(u32) } };
pub const Chunk = struct { rows: u32, plan: usize };

/// Plans for a chunk capacity of `rows` (> 0): table entries clipped to `rows`, up to the
/// first entry that covers it. Returns the count written to `out`.
pub fn makePlans(rows: u32, out: *[max_plans]Plan) u8 {
    std.debug.assert(rows > 0);
    var count: u8 = 0;
    for (plan_table) |entry| {
        out[count] = .{ .rows = @min(entry.rows, rows) };
        count += 1;
        if (entry.rows >= rows) break;
    }
    return count;
}
/// First plan covering `n` rows (`n` <= the last plan's rows).
pub fn planIndex(plans: []const Plan, n: u32) usize {
    for (plans, 0..) |plan, i| if (n <= plan.rows) return i;
    unreachable;
}
/// Next chunk for `remaining` (>= 1) prompt tokens with capacity `rows`: full chunks on
/// the last plan, then the remainder on the first plan covering it.
pub fn chunkFor(plans: []const Plan, rows: u32, remaining: u32) Chunk {
    std.debug.assert(remaining > 0 and plans.len > 0);
    if (remaining >= rows) return .{ .rows = rows, .plan = plans.len - 1 };
    return .{ .rows = remaining, .plan = planIndex(plans, remaining) };
}
const split_slots = 4; // concurrently issued split-K projections per phase

/// Rows are config.hidden wide (HIDDEN in model.comp); `width` must equal it (it is the
/// run-time divisor of the mean, which keeps the pre-13c rounding).
const NormPush = extern struct { x: u32, a: u32, sum: u32, y: u32, w: u32, width: u32 = config.hidden, stride: u32, flags: u32, eps: f32 };
const EmbedPush = extern struct { tensor: u32, row_bytes: u32, out: u32, columns: u32 };
/// Row-capable decode kernels (docs/specs/speculative.md): `qkprep` reads RoPE row r at
/// `rope + 64 r` (decode: `layout.io.cos`); `conv` and `delta` process `rows` rows in order
/// and store their state only when `commit` is 1 (decode: 1 row, commit).
/// Decode-family pushes end in `slots, slot_rs` (docs/specs/concurrent.md, "18b.2 design"):
/// 0, 0 = no slot entry (single-slot model); prefill-family pushes end in `slots`.
pub const QkPush = extern struct { qf: u32, kc: u32, vc: u32, qn: u32, qr: u32, kn: u32, kr: u32, qw: u32, kw: u32, kcache: u32, vcache: u32, ctx: u32, eps: f32, rope: u32, pos: u32, ptab: u32, pstride: u32, slots: u32 = 0, slot_rs: u32 = 0 };
const ConvDPush = extern struct { mixed: u32, raw: u32, silu: u32, out: u32, w: u32, conv: u32, rows: u32 = 1, commit: u32 = 1, slots: u32 = 0, slot_rs: u32 = 0 };
/// `state_out`: where the kernel stores the state (== `ssm`; a separate value keeps the
/// compiler from holding the load addresses live across the row loop, block 17c).
const DeltaDPush = extern struct { qk: u32, v: u32, z: u32, beta_raw: u32, alpha: u32, beta_out: u32, softplus_out: u32, g_out: u32, o: u32, y: u32, ssm: u32, a_w: u32, dt_w: u32, norm_w: u32, eps: f32, rows: u32 = 1, commit: u32 = 1, state_out: u32, slots: u32 = 0, slot_rs: u32 = 0 };
const ConvPush = extern struct { mixed: u32, raw: u32, silu: u32, out: u32, w: u32, conv: u32, slots: u32 = 0 };
const DeltaPush = extern struct { qk: u32, v: u32, z: u32, beta_raw: u32, alpha: u32, beta_out: u32, softplus_out: u32, g_out: u32, o: u32, y: u32, ssm: u32, a_w: u32, dt_w: u32, norm_w: u32, eps: f32, state_out: u32, slots: u32 = 0 };
const SwigluPush = extern struct { g: u32, u: u32, y: u32, n: u32, rows_io: u32 = 0 };
const ZeroPush = extern struct { first: u32, count: u32 };
const EmbedBPush = extern struct { tensor: u32, row_bytes: u32, out: u32, columns: u32, tokens: u32, out_rs: u32 };
pub const QkBPush = extern struct { qf: u32, kc: u32, vc: u32, qn: u32, qr: u32, kn: u32, kr: u32, qw: u32, kw: u32, kcache: u32, vcache: u32, ctx: u32, rope: u32, eps: f32, ptab: u32, pstride: u32, slots: u32 = 0 };
const GatePush = extern struct { pregate: u32, qf: u32, gates: u32, gated: u32 };
/// Gated RMS norm of the batched DeltaNet output (model.comp K_GNORMB).
const GNormPush = extern struct { o: u32, y: u32, z: u32, norm_w: u32, eps: f32 };
/// MTP helpers (block 17b): row copy from an io-given source (bit 31: state arena) to
/// `dst` (flags bit 0: state arena); two-phase argmax into an io word.
const RowCopyPush = extern struct { src: u32, dst: u32, words: u32, flags: u32 };
const ArgmaxAPush = extern struct { x: u32, n: u32, part: u32 };
const ArgmaxBPush = extern struct { part: u32, out: u32, prob: u32 = layout.io.spec_drafts };
const Copy2dPush = extern struct { src: u32, dst: u32, width: u32, src_stride: u32, dst_stride: u32 };
/// Fused prefill attention (flash.comp): 8 query rows x the 6 query heads of one KV head
/// per workgroup.
const FlashPush = extern struct { qr: u32, kcache: u32, vcache: u32, out: u32, ctx: u32, scale: f32, ptab: u32, pstride: u32, slots: u32 = 0 };
pub const flash_rows = 8;
/// Query heads per fused-attention workgroup (GROUPS in flash.comp).
pub const flash_groups = 6;
const state_source: u32 = 0x8000_0000;
/// f16-mode producers (model.comp with F16OUT, block 16b): the FP32 kernel's push
/// followed by the f16 copy's offset (and row stride), in halves of the arena.
pub const HKernel = enum { norm, swiglu, gate };
const h_kernel_count = @typeInfo(HKernel).@"enum".fields.len;
const NormHPush = extern struct { base: NormPush, y16: u32, stride16: u32 };
const SwigluHPush = extern struct { base: SwigluPush, y16: u32 };
const GateHPush = extern struct { base: GatePush, gated16: u32 };
const h_push_sizes = [h_kernel_count]u32{ @sizeOf(NormHPush), @sizeOf(SwigluHPush), @sizeOf(GateHPush) };
pub fn hModule(id: HKernel) []align(4) const u8 {
    const M = struct {
        const norm align(4) = @embedFile("shaders/norm_h.spv").*;
        const swiglu align(4) = @embedFile("shaders/swiglu_h.spv").*;
        const gate align(4) = @embedFile("shaders/gate_h.spv").*;
    };
    return switch (id) {
        .norm => &M.norm,
        .swiglu => &M.swiglu,
        .gate => &M.gate,
    };
}
comptime {
    std.debug.assert(@sizeOf(NormHPush) == 44 and @sizeOf(SwigluHPush) == 24 and @sizeOf(GateHPush) == 20);
}
const push_sizes = [kernel_count]u32{ @sizeOf(EmbedPush), @sizeOf(NormPush), @sizeOf(QkPush), @sizeOf(ConvDPush), @sizeOf(DeltaDPush), @sizeOf(SwigluPush), @sizeOf(ZeroPush), @sizeOf(gemm.ReducePush), @sizeOf(EmbedBPush), @sizeOf(QkBPush), @sizeOf(GatePush), @sizeOf(ConvPush), @sizeOf(DeltaPush), attention.pushBytes(.scores), attention.pushBytes(.pv), attention.pushBytes(.combine), @sizeOf(GNormPush), @sizeOf(RowCopyPush), @sizeOf(ArgmaxAPush), @sizeOf(ArgmaxBPush), @sizeOf(Copy2dPush), @sizeOf(FlashPush), attention.pushBytes(.gmax), attention.pushBytes(.block), @sizeOf(DeltaDPush), @sizeOf(DeltaPush) };
const split_target: u32 = 384; // ~4 workgroups per compute unit on the target card
fn splitSlot(name: []const u8) usize {
    const suffixes = [_]struct { []const u8, usize }{
        .{ "attn_k.weight", 1 },    .{ "attn_v.weight", 2 },   .{ "attn_gate.weight", 1 },
        .{ "ssm_alpha.weight", 2 }, .{ "ssm_beta.weight", 3 }, .{ "ffn_up.weight", 1 },
    };
    for (suffixes) |entry| if (std.mem.endsWith(u8, name, entry[0])) return entry[1];
    return 0;
}
fn reduceGroups(rp: gemm.ReducePush, rows: u32) u32 {
    return @intCast(@min(std.math.divCeil(u64, @as(u64, rows) * rp.m, 256) catch unreachable, 4096));
}
comptime {
    std.debug.assert(config.hidden == 5120 and config.hidden % 256 == 0); // HIDDEN in model.comp; norm: 20 values per thread
}
// Norm flags; bit 31 must stay clear (the kernel derives its +0 accumulator seed from it).
const norm_add: u32 = 1;
const norm_rows_io: u32 = 2;
const norm_last_only: u32 = 4;

pub fn flashModule(kv: layout.KvType) []align(4) const u8 {
    return module(.flash, kv);
}
/// The KV writers' modules (exported for the f16 conversion test).
pub fn qkprepModule(kv: layout.KvType) []align(4) const u8 {
    return module(.qkprep, kv);
}
pub fn qkBModule(kv: layout.KvType) []align(4) const u8 {
    return module(.qk_b, kv);
}
/// Kernel `id`'s module; the KV kernels (`kv_kernel_ids`) have one per KV element type.
fn module(id: KernelId, kv: layout.KvType) []align(4) const u8 {
    const M = struct {
        const embed align(4) = @embedFile("shaders/embed.spv").*;
        const norm align(4) = @embedFile("shaders/norm.spv").*;
        const qkprep align(4) = @embedFile("shaders/qkprep.spv").*;
        const conv align(4) = @embedFile("shaders/conv.spv").*;
        const delta align(4) = @embedFile("shaders/delta.spv").*;
        const delta_legacy align(4) = @embedFile("shaders/delta_legacy.spv").*;
        const delta_b_legacy align(4) = @embedFile("shaders/delta_b_legacy.spv").*;
        const swiglu align(4) = @embedFile("shaders/swiglu.spv").*;
        const zero align(4) = @embedFile("shaders/zero.spv").*;
        const reduce align(4) = @embedFile("shaders/reduce.spv").*;
        const embed_b align(4) = @embedFile("shaders/embed_b.spv").*;
        const qk_b align(4) = @embedFile("shaders/qk_b.spv").*;
        const gate align(4) = @embedFile("shaders/gate.spv").*;
        const conv_b align(4) = @embedFile("shaders/conv_b.spv").*;
        const delta_b align(4) = @embedFile("shaders/delta_b.spv").*;
        const gnorm_b align(4) = @embedFile("shaders/gnorm_b.spv").*;
        const rowcopy align(4) = @embedFile("shaders/rowcopy.spv").*;
        const argmax_a align(4) = @embedFile("shaders/argmax_a.spv").*;
        const argmax_b align(4) = @embedFile("shaders/argmax_b.spv").*;
        const copy2d align(4) = @embedFile("shaders/copy2d.spv").*;
        const flash align(4) = @embedFile("shaders/attn_flash.spv").*;
        const qkprep16 align(4) = @embedFile("shaders/qkprep_kv16.spv").*;
        const qk_b16 align(4) = @embedFile("shaders/qk_b_kv16.spv").*;
        const flash16 align(4) = @embedFile("shaders/attn_flash_kv16.spv").*;
    };
    return switch (id) {
        .embed => &M.embed,
        .norm => &M.norm,
        .qkprep => if (kv == .f16) &M.qkprep16 else &M.qkprep,
        .conv => &M.conv,
        .delta => &M.delta,
        .delta_legacy => &M.delta_legacy,
        .delta_b_legacy => &M.delta_b_legacy,
        .swiglu => &M.swiglu,
        .zero => &M.zero,
        .reduce => &M.reduce,
        .embed_b => &M.embed_b,
        .qk_b => if (kv == .f16) &M.qk_b16 else &M.qk_b,
        .gate => &M.gate,
        .conv_b => &M.conv_b,
        .delta_b => &M.delta_b,
        .attn_scores => attention.module(.scores, kv),
        .attn_pv => attention.module(.pv, kv),
        .attn_combine => attention.module(.combine, kv),
        .attn_gmax => attention.module(.gmax, kv),
        .attn_cblock => attention.module(.block, kv),
        .gnorm_b => &M.gnorm_b,
        .rowcopy => &M.rowcopy,
        .argmax_a => &M.argmax_a,
        .argmax_b => &M.argmax_b,
        .copy2d => &M.copy2d,
        .flash => if (kv == .f16) &M.flash16 else &M.flash,
    };
}

const Proj = struct { pipe: u8 = 0, geo: matvec.Geometry = undefined };
/// Fused decode FFN input: gate and up of one layer plus the swiglu (block 17c).
const SwProj = struct { pipe: u8 = 0, geo: matvec.SwigluGeometry = undefined };
/// Fused verify FFN input (multi-row gate, up and swiglu; block 17c).
const SwRProj = struct { pipe: u8 = 0, geo: matvec.SwigluRowsGeometry = undefined };
const max_swiglu_pipelines = 8;
/// Multi-row projection of the speculative verify pass (validated for `verify_rows` rows).
const RProj = struct { pipe: u8 = 0, geo: matvec.RowsGeometry = undefined };
/// Batched projection: GEMM (optionally split-K into a partial slot + reduction). `x16`:
/// the kernel reads the f16 copy of its input (gemm_f16x), which its producer must write.
const GProj = struct { pipe: u8 = 0, push: gemm.Push = undefined, groups: [3]u32 = undefined, reduce: ?gemm.ReducePush = null, x16: bool = false };
fn any16(projs: []const *const GProj) bool {
    for (projs) |g| if (g.x16) return true;
    return false;
}
const Layer = struct {
    attn_norm: u32 = 0,
    post_norm: u32 = 0,
    gate: Proj = .{},
    up: Proj = .{},
    /// Set when gate and up share a bank, format and shape (then decode runs them and the
    /// swiglu as one dispatch); null: separate projections and the swiglu kernel.
    gate_up: ?SwProj = null,
    down: Proj = .{},
    q: Proj = .{},
    k: Proj = .{},
    v: Proj = .{},
    q_norm: u32 = 0,
    k_norm: u32 = 0,
    out: Proj = .{},
    qkv: Proj = .{},
    z: Proj = .{},
    alpha: Proj = .{},
    beta: Proj = .{},
    ssm_a: u32 = 0,
    dt: u32 = 0,
    conv_w: u32 = 0,
    ssm_norm: u32 = 0,
    ssm_out: Proj = .{},
    // Speculative verify counterparts (block 17b; unset without speculation).
    r_gate: RProj = .{},
    r_up: RProj = .{},
    /// Set when `Options.verify_fusion` and gate/up qualify as for `gate_up`.
    r_gate_up: ?SwRProj = null,
    r_down: RProj = .{},
    r_q: RProj = .{},
    r_k: RProj = .{},
    r_v: RProj = .{},
    r_out: RProj = .{},
    r_qkv: RProj = .{},
    r_z: RProj = .{},
    r_alpha: RProj = .{},
    r_beta: RProj = .{},
    r_ssm_out: RProj = .{},
    // Batched decode (docs/specs/concurrent.md): linear layers' lin_in into the plain regions.
    b_qkv: RProj = .{},
    b_z: RProj = .{},
    b_alpha: RProj = .{},
    b_beta: RProj = .{},
    // Batched decode, `decode_precision = .f16`: the WMMA projections (null: FP32 path).
    d_q: ?GProj = null,
    d_out: ?GProj = null,
    d_qkv: ?GProj = null,
    d_z: ?GProj = null,
    d_gate: ?GProj = null,
    d_up: ?GProj = null,
    d_down: ?GProj = null,
    d_ssm_out: ?GProj = null,
    // Batched (prefill) counterparts.
    g_gate: [max_plans]GProj = @splat(.{}),
    g_up: [max_plans]GProj = @splat(.{}),
    g_down: [max_plans]GProj = @splat(.{}),
    g_q: [max_plans]GProj = @splat(.{}),
    g_k: [max_plans]GProj = @splat(.{}),
    g_v: [max_plans]GProj = @splat(.{}),
    g_out: [max_plans]GProj = @splat(.{}),
    g_qkv: [max_plans]GProj = @splat(.{}),
    g_z: [max_plans]GProj = @splat(.{}),
    g_alpha: [max_plans]GProj = @splat(.{}),
    g_beta: [max_plans]GProj = @splat(.{}),
    g_ssm_out: [max_plans]GProj = @splat(.{}),
};

/// The MTP (nextn) layer (block 17b): param words and multi-row projections.
const MtpLayer = struct {
    attn_norm: u32 = 0,
    post_norm: u32 = 0,
    q_norm: u32 = 0,
    k_norm: u32 = 0,
    enorm: u32 = 0,
    hnorm: u32 = 0,
    head_norm: u32 = 0,
    eh: RProj = .{},
    q: RProj = .{},
    k: RProj = .{},
    v: RProj = .{},
    out: RProj = .{},
    gate: RProj = .{},
    up: RProj = .{},
    down: RProj = .{},
    /// Batched prompt catch-up (per prefill plan): only what the MTP KV depends on.
    g_eh: [max_plans]GProj = @splat(.{}),
    g_k: [max_plans]GProj = @splat(.{}),
    g_v: [max_plans]GProj = @splat(.{}),
};
/// MTP rows not yet run: the next pass's first `rows - 1` rows are `tokens` (accepted
/// drafts) and its h rows start at the word `src` (bit 31: state arena).
const MtpPending = struct { rows: u32 = 1, src: u32 = 0, tokens: [matvec.max_rows]u32 = @splat(0) };

/// Optional per-step intermediate capture into a host-visible buffer (verification only).
pub const Capture = struct {
    /// `words` per row; `rows` rows captured (prefill: the chunk capacity); `last_only`:
    /// a single row valid only for the chunk's last token.
    pub const Entry = struct { name: [40]u8 = undefined, name_len: u8 = 0, layer: i32, offset_words: u64, words: u32, rows: u32 = 1, stride: u32 = 0, last_only: bool = false };
    buffer: *gpu.Buffer,
    filter: []const []const u8,
    entries: []Entry,
    count: usize = 0,
    used_words: u64 = 0,

    pub fn entryName(entry: *const Entry) []const u8 {
        return entry.name[0..entry.name_len];
    }
    fn wants(self: *const Capture, name: []const u8) bool {
        for (self.filter) |item| if (std.mem.eql(u8, item, name)) return true;
        return false;
    }
};

pub const Model = struct {
    device: *gpu.Device,
    options: Options,
    hyper: config.Hyper,
    act_layout: layout.Act,
    state_layout: layout.State,
    bank_count: u8 = 0,
    banks: [layout.max_banks]gpu.Buffer = undefined,
    act: gpu.Buffer = undefined,
    state: gpu.Buffer = undefined,
    io: gpu.Buffer = undefined,
    /// KV buffers (`state_layout.kv_buffers`) and their extra attention kernels (buffers
    /// 1..).
    kv: [layout.max_kv_buffers]gpu.Buffer = undefined,
    kv_live: u8 = 0,
    kv_kernels: [layout.max_kv_buffers - 1][kv_kernel_ids.len]gpu.Kernel = undefined,
    kv_kernels_live: u8 = 0,
    kernels: [kernel_count]gpu.Kernel = undefined,
    // Speculative verification (block 17b, docs/specs/speculative.md); all empty when
    // `options.verify_rows` is 0.
    rpipes: [max_pipelines]matvec.RowsPipeline = undefined,
    rpipe_keys: [max_pipelines]u32 = undefined,
    rpipe_count: u8 = 0,
    output_rows: RProj = .{},
    /// Host-visible logits of the verify rows (`verify_rows` x vocab).
    verify_logits: gpu.Buffer = undefined,
    verify_logits_live: bool = false,
    /// Index n (1..verify_rows): verify of n rows / commit of n rows.
    verify_commands: [matvec.max_rows + 1]gpu.Commands = undefined,
    commit_commands: [matvec.max_rows + 1]gpu.Commands = undefined,
    live_verify: u8 = 0,
    live_commit: u8 = 0,
    /// Rows of the last `verify` not yet committed (0 = none).
    pending_verify: u32 = 0,
    /// Tokens of the last `verify`.
    verify_tokens: [matvec.max_rows]u32 = @splat(0),
    // MTP drafting (block 17b; empty unless `options.mtp`).
    mtp_layer: MtpLayer = .{},
    /// Draft head of a pass's last row r (input `mo` row r, output the MTP logits).
    mtp_head: [matvec.max_rows]Proj = @splat(.{}),
    /// [m][k]: a first pass of m rows, then k drafts in total. [n]: catch-up of n rows.
    draft_commands: [matvec.max_rows + 1][matvec.max_rows]gpu.Commands = undefined,
    catchup_commands: [matvec.max_rows + 1]gpu.Commands = undefined,
    save_commands: gpu.Commands = undefined,
    live_draft: u8 = 0,
    live_catchup: u8 = 0,
    save_live: bool = false,
    mtp_pending: MtpPending = .{},
    drafts: [matvec.max_rows]u32 = @splat(0),
    draft_probs: [matvec.max_rows]f32 = @splat(0),
    /// f16-mode producers; created only when the arena has the `x16` region.
    h_kernels: [h_kernel_count]gpu.Kernel = undefined,
    live_h_kernels: u8 = 0,
    pipelines: [max_pipelines]matvec.Pipeline = undefined,
    pipe_keys: [max_pipelines]u32 = undefined,
    pipeline_count: u8 = 0,
    swiglu_pipes: [max_swiglu_pipelines]matvec.SwigluPipeline = undefined,
    swiglu_keys: [max_swiglu_pipelines]u32 = undefined,
    swiglu_count: u8 = 0,
    swiglu_rpipes: [max_swiglu_pipelines]matvec.SwigluRowsPipeline = undefined,
    swiglu_rkeys: [max_swiglu_pipelines]u32 = undefined,
    swiglu_rcount: u8 = 0,
    /// Layers whose decode FFN input runs fused (gate, up and swiglu in one dispatch).
    fused_ffn_layers: u32 = 0,
    /// Layers whose verify FFN input runs fused.
    fused_verify_layers: u32 = 0,
    layers: [config.layers]Layer = @splat(.{}),
    output: Proj = .{},
    output_norm: u32 = 0,
    embed_offset: u32 = 0,
    /// The token embedding when `options.embedding_memory` is `.host` (mapped, filled at
    /// init); the embed kernels bind it instead of bank 0.
    embed_host: gpu.Buffer = undefined,
    embed_host_live: bool = false,
    step_commands: gpu.Commands = undefined,
    reset_commands: gpu.Commands = undefined,
    prefill_commands: [max_plans]gpu.Commands = undefined,
    plans: [max_plans]Plan = undefined,
    plan_count: u8 = 0,
    gemm_pipes: [max_gemm_pipelines]gpu.Kernel = undefined,
    gemm_keys: [max_gemm_pipelines]u32 = undefined,
    gemm_count: u8 = 0,
    part_slots: [split_slots]u32 = @splat(0),
    rows: u32 = 0, // prefill chunk capacity (0 = disabled)
    position: u32 = 0,
    /// Device-local bytes `init` needed (buffers + `vram_headroom`) and found free; kept
    /// after an `InsufficientVram` failure for the error message. `free` is null when the
    /// driver cannot report its budget (then no check is made).
    /// With `Options.context = context_max` and no context fitting at all, `needed` is the
    /// context-independent part (weights, snapshots, headroom, `context_reserve`) and
    /// `before_context` is set; `cap_limited` says the allocation cap
    /// (`Device.budget`), not free VRAM, left no room.
    vram: struct { needed: u64 = 0, free: ?u64 = null, before_context: bool = false, cap_limited: bool = false } = .{},
    snapshot_store: gpu.Buffer = undefined,
    snapshot_slots: u32 = 0,
    copy_commands: gpu.Commands = undefined,
    copy_live: bool = false,
    // Sequence slots (docs/specs/concurrent.md, "18b.2 design").
    /// io word of the slot entry the single-slot commands read (0 with one slot: none).
    slot_io: u32 = 0,
    /// The current slot (`select`); `position` is its position, `slot_positions` the others'.
    slot: u32 = 0,
    slot_positions: [layout.max_slots]u32 = @splat(0),
    /// Logical KV pages mapped per slot, and the owner slot of every pool page.
    mapped: [layout.max_slots]u32 = @splat(0),
    page_owner: [max_pool_pages]u8 = @splat(no_owner),
    /// Reset commands of slots 1.. (slot 0: `reset_commands`).
    slot_resets: [layout.max_slots]gpu.Commands = undefined,
    live_slot_resets: u8 = 0,
    /// Index B (1..`options.batch_rows`): the batched decode of B rows.
    batch_commands: [layout.io.batch_max + 1]gpu.Commands = undefined,
    live_batch: u8 = 0,
    /// Page-table writes: one slot's table staged on the host, then copied.
    ptab_staging: gpu.Buffer = undefined,
    ptab_commands: gpu.Commands = undefined,
    ptab_live: bool = false,
    // Cleanup progress for partial initialization.
    live_buffers: u8 = 0,
    live_kernels: u8 = 0,
    live_commands: u8 = 0,
    live_prefill: u8 = 0,

    /// In-place: pipelines and commands retain addresses of fields. Container bytes
    /// are read only during init (weights are uploaded and validated first).
    pub fn init(self: *Model, device: *gpu.Device, container: *const gguf.Container, options: Options) Error!void {
        self.* = .{ .device = device, .options = options, .hyper = try config.hyper(container), .act_layout = undefined, .state_layout = undefined };
        errdefer self.cleanup();
        const limits = device.properties.limits;
        const capacity = std.mem.alignBackward(u64, @min(options.bank_capacity, @as(u64, limits.maxStorageBufferRange), 0xffff_ffe0), 32);
        self.rows = options.prefill_rows;
        if (self.rows > 0) {
            // Scalar-X GEMM: 64-invocation groups must be subgroup-uniform (subgroups of at
            // most 64 with ballot), and score rows must be whole 32-key blocks.
            if (!device.subgroup.computeBallotWithin(64)) return error.UnsupportedDevice;
            // Fused attention: 64-thread groups of one or two subgroups with arithmetic.
            if (!device.subgroup.computeArithmeticFrom(32)) return error.UnsupportedDevice;
            // f16: gemm_f16 needs subgroups of 64 (the default); gemm_f16x requires size 32
            // with full subgroups.
            if (options.prefill_precision == .f16 and (!device.cooperative_matrix or device.subgroup.size != 64 or !device.full_subgroups or
                device.subgroup_sizes.min > gemm.f16x_subgroup or device.subgroup_sizes.max < gemm.f16x_subgroup)) return error.UnsupportedDevice;
            self.plan_count = makePlans(self.rows, &self.plans);
        }
        if (options.mtp and options.verify_rows < 2) return error.VerifyDisabled;
        if (options.slots == 0 or options.slots > layout.max_slots) return error.InvalidSlots;
        if (options.slots > 1 and options.mtp) return error.MtpNeedsOneSlot;
        if (options.batch_rows > layout.io.batch_max or options.batch_rows > @max(self.rows, 1)) return error.InvalidBatch;
        if (options.kv_pages > max_pool_pages) return error.InvalidSlots;
        if (options.decode_precision == .f16) {
            if (options.batch_rows == 0 or std.mem.alignForward(u32, options.batch_rows, gemm.f16n_tile_n) > @max(self.rows, 1)) return error.InvalidBatch;
            if (!device.cooperative_matrix or device.subgroup.size != 64) return error.UnsupportedDevice;
        }
        if (options.draft_vocab > config.vocab) return error.InvalidDraftVocab;
        const kv_capacity = if (options.kv_capacity != 0) std.mem.alignBackward(u64, @min(options.kv_capacity, capacity), 32) else capacity;
        // Split-K partial slots, sized from the fixed tensor inventory before any allocation.
        var slot_words: [split_slots]u64 = @splat(0);
        var sized: [config.tensor_count + config.mtp_tensor_count]config.Spec = undefined;
        sized[0..config.tensor_count].* = config.tensors();
        sized[config.tensor_count..].* = config.mtpTensors();
        const n_sized = if (options.mtp) sized.len else config.tensor_count;
        for (self.plans[0..self.plan_count]) |plan| for (sized[0..n_sized]) |*s| {
            if (s.role != .matrix or std.mem.eql(u8, config.specName(s), "output.weight")) continue;
            const chunk = gemm.splitChunk(@intCast(s.rows), plan.rows, @intCast(s.k), split_target, gemm.tileFor(@intCast(s.rows), plan.rows));
            // (f16-mode projections never split; sizing their FP32 split slot is harmless.)
            if (chunk == 0) continue;
            const words = @as(u64, gemm.splitCount(@intCast(s.k), chunk)) * plan.rows * s.rows;
            const slot = splitSlot(config.specName(s));
            slot_words[slot] = @max(slot_words[slot], std.mem.alignForward(u64, words, 64));
        };
        // f16-mode batched decode: split parts of the 16-row WMMA projections.
        if (options.decode_precision == .f16) for (sized[0..n_sized]) |*s| {
            if (s.role != .matrix or std.mem.eql(u8, config.specName(s), "output.weight")) continue;
            const chunk = gemm.f16nChunk(@intCast(s.rows), @intCast(s.k), split_target);
            if (chunk == 0) continue;
            const span = std.mem.alignForward(u64, options.batch_rows, gemm.f16n_tile_n);
            const words = @as(u64, gemm.splitCount(@intCast(s.k), chunk)) * span * s.rows;
            const slot = splitSlot(config.specName(s));
            slot_words[slot] = @max(slot_words[slot], std.mem.alignForward(u64, words, 64));
        };
        var part_words: u64 = 0;
        for (slot_words) |w| part_words += w;
        var x16 = false;
        if (options.prefill_precision == .f16) for (self.plans[0..self.plan_count]) |plan| {
            x16 = x16 or plan.rows % gemm.f16x_tile_n == 0;
        };
        if (options.verify_rows != 0 and (options.verify_rows < 2 or options.verify_rows > matvec.max_rows or self.rows < options.verify_rows)) return error.VerifyDisabled;
        if (options.snapshots > max_snapshots) return error.InvalidSnapshot;

        // The trunk's inventory, then the MTP layer's when requested.
        const max_tensors = config.tensor_count + config.mtp_tensor_count;
        var all_specs: [max_tensors]config.Spec = undefined;
        all_specs[0..config.tensor_count].* = config.tensors();
        all_specs[config.tensor_count..].* = config.mtpTensors();
        const n_tensors: usize = if (options.mtp) max_tensors else config.tensor_count;
        const specs = all_specs[0..n_tensors];
        var all_tensors: [max_tensors]*const gguf.Tensor = undefined;
        var all_items: [max_tensors]layout.Item = undefined;
        const tensors = all_tensors[0..n_tensors];
        const items = all_items[0..n_tensors];
        for (specs, tensors, items) |*s, *t, *item| {
            t.* = container.findTensor(config.specName(s)) orelse return error.MissingTensor;
            try config.check(s, t.*);
            const format: matvec.Format = if (s.role == .embedding) .q4_0 else try config.matrixFormat(t.*.kind);
            const shape: matvec.Shape = .{ .format = format, .columns = @intCast(s.k), .rows = @intCast(s.rows) };
            try matvec.validateWeights(shape, t.*.data);
            if (t.*.data.len % 4 != 0) return error.InvalidTensorBytes;
            // A host-resident embedding takes no bank space.
            const in_bank = s.role != .embedding or options.embedding_memory == .device;
            item.* = .{ .role = s.role, .bytes = if (in_bank) t.*.data.len else 0 };
        }
        var all_placements: [max_tensors]layout.Placement = undefined;
        const placements = all_placements[0..n_tensors];
        const banks = try layout.place(items, capacity, placements);

        // Device bytes that do not depend on the context, and the host buffers init
        // allocates (they count against the device's allocation cap, not VRAM).
        const snapshot_vram: u64 = if (options.snapshot_memory == .device) self.snapshotBytes() * options.snapshots else 0;
        var fixed: u64 = snapshot_vram;
        for (banks.bytes[0..banks.count]) |bytes| fixed += bytes;
        var host: u64 = layout.io.words(@max(self.rows, 1)) * 4 + options.staging_bytes + @as(u64, rowSpan(options)) * config.vocab * 4;
        if (options.snapshot_memory == .host) host += self.snapshotBytes() * options.snapshots;
        if (options.embedding_memory == .host) host += tensors[0].data.len;
        const budget = try device.memoryBudget();
        if (budget) |b| self.vram.free = b.free();

        // The context: given, or the largest that fits (`context_max`).
        const trained = self.hyper.context_length;
        const context = if (options.context != context_max) options.context else fit: {
            const free = self.vram.free orelse return error.VramBudgetUnknown;
            const cap = device.budget - device.allocated_bytes;
            const by_free = std.math.sub(u64, free, fixed + vram_headroom + options.context_reserve) catch 0;
            const by_cap = std.math.sub(u64, cap, fixed + host + cap_margin) catch 0;
            const shape: ContextShape = .{ .kv_capacity = kv_capacity, .rows = @max(self.rows, 1), .part_words = part_words, .x16 = x16, .decode_rows = decodeRows(options), .mtp = options.mtp, .kv = options.kv_type, .kv_page = options.kv_page_tokens, .verify = options.verify_rows > 1, .slots = options.slots, .kv_pages = options.kv_pages };
            const fitted = fitContext(@min(by_free, by_cap), @min(trained, layout.maxContext(kv_capacity, options.kv_type, options.kv_page_tokens)), shape);
            if (fitted == 0) {
                // Not even the smallest context fits: report what is needed before any KV
                // cache or activations, and which limit ran out.
                self.vram.needed = fixed + vram_headroom + options.context_reserve;
                self.vram.before_context = true;
                self.vram.cap_limited = by_cap < by_free;
                return error.InsufficientVram;
            }
            break :fit fitted;
        };
        if (context > trained) return error.ContextTooLarge;
        if (self.rows > 0 and context % 32 != 0) return error.InvalidContext;
        self.options.context = context;
        if (options.kv_type == .f16 and !device.storage16) return error.UnsupportedDevice;
        self.state_layout = try layout.stateWith(context, kv_capacity, options.mtp, options.kv_type, options.kv_page_tokens, options.slots, options.kv_pages);
        if (self.state_layout.pages > max_pool_pages or self.state_layout.words * 4 > capacity) return error.InvalidSlots;
        self.act_layout = try layout.actWith(context, @max(self.rows, 1), part_words, x16, decodeRows(options), options.mtp, options.verify_rows > 1, options.slots);
        self.slot_io = if (options.slots > 1) layout.io.slot else 0;
        // Decode (and verify) attention writes 24 x rows x context scores at `scores`.
        if (@as(u64, self.act_layout.part - self.act_layout.scores) < @as(u64, config.heads) * self.act_layout.decode_rows * context) return error.InvalidPlan;
        var at: u64 = self.act_layout.part;
        for (&self.part_slots, slot_words) |*slot, w| {
            slot.* = @intCast(at);
            at += w;
        }

        // Refuse to load when the device-local heap cannot hold every buffer, e.g. because
        // another process uses the GPU: oversubscribed VRAM is evicted to host memory by
        // the kernel driver and has ended in compute timeouts and a lost device.
        const needed: u64 = self.act_layout.words * 4 + self.state_layout.words * 4 + self.state_layout.kvBytes() + fixed + vram_headroom;
        self.vram.needed = needed;
        if (budget != null and needed > self.vram.free.?) return error.InsufficientVram;

        // Device arenas and weight banks.
        for (0..banks.count) |b| {
            self.banks[b] = try gpu.Buffer.init(device, banks.bytes[b], .device);
            self.bank_count += 1;
            self.live_buffers += 1;
        }
        self.act = try gpu.Buffer.init(device, self.act_layout.words * 4, .device);
        self.live_buffers += 1;
        self.state = try gpu.Buffer.init(device, self.state_layout.words * 4, .device);
        self.live_buffers += 1;
        self.io = try gpu.Buffer.init(device, layout.io.words(@max(self.rows, 1)) * 4, .host);
        self.live_buffers += 1;
        for (0..self.state_layout.kv_buffers) |g| {
            self.kv[g] = try gpu.Buffer.init(device, self.state_layout.kvBufferBytes(@intCast(g)), .device);
            self.kv_live += 1;
        }
        if (options.snapshots > 0) {
            self.copy_commands = try gpu.Commands.init(device);
            self.copy_live = true;
            self.snapshot_store = try gpu.Buffer.init(device, self.snapshotBytes() * options.snapshots, options.snapshot_memory);
            self.snapshot_slots = options.snapshots;
        }

        try self.upload(tensors, items, placements);
        try self.writePageTable();
        if (options.embedding_memory == .host) {
            const embedding = container.findTensor(config.specName(&specs[0])).?;
            std.debug.assert(specs[0].role == .embedding);
            self.embed_host = try gpu.Buffer.init(device, embedding.data.len, .host);
            self.embed_host_live = true;
            @memcpy((try self.embed_host.mapped())[0..embedding.data.len], embedding.data);
        }

        // The KV kernels' specialization constant 0: `KV_PAGE`, the tokens per page.
        const kv_constants = [_]u32{self.state_layout.page};
        for (0..kernel_count) |i| {
            const id: KernelId = @enumFromInt(i);
            const binding2 = if (kvSlot(id) != null) &self.kv[0] else &self.state;
            const binding0 = if (self.embed_host_live and (id == .embed or id == .embed_b)) &self.embed_host else &self.banks[0];
            self.kernels[i] = try gpu.Kernel.initWith(device, module(id, options.kv_type), &.{ binding0, &self.act, binding2, &self.io }, push_sizes[i], .{ .constants = if (kvSlot(id) != null) &kv_constants else &.{} });
            self.live_kernels += 1;
        }
        for (1..self.state_layout.kv_buffers) |g| {
            for (kv_kernel_ids, 0..) |id, j| self.kv_kernels[g - 1][j] = try gpu.Kernel.initWith(device, module(id, options.kv_type), &.{ &self.banks[0], &self.act, &self.kv[g], &self.io }, push_sizes[@intFromEnum(id)], .{ .constants = &kv_constants });
            self.kv_kernels_live += 1;
        }
        if (self.act_layout.x16 != null) for (0..h_kernel_count) |i| {
            self.h_kernels[i] = try gpu.Kernel.init(device, hModule(@enumFromInt(i)), &.{ &self.banks[0], &self.act, &self.state, &self.io }, h_push_sizes[i]);
            self.live_h_kernels += 1;
        };
        if (rowSpan(options) > 0) {
            self.verify_logits = try gpu.Buffer.init(device, @as(u64, rowSpan(options)) * config.vocab * 4, .host);
            self.verify_logits_live = true;
        }

        // Resolve parameter offsets and projections.
        for (specs, tensors, placements) |*s, t, place| {
            const name = config.specName(s);
            const word: u32 = @intCast(place.offset / 4);
            if (std.mem.startsWith(u8, name, std.fmt.comptimePrint("blk.{d}.", .{config.mtp_layer}))) {
                try self.resolveMtp(s, t, place, name[std.fmt.comptimePrint("blk.{d}.", .{config.mtp_layer}).len..]);
                continue;
            }
            if (s.role == .embedding) {
                self.embed_offset = if (self.embed_host_live) 0 else @intCast(place.offset);
                continue;
            }
            if (std.mem.eql(u8, name, "output_norm.weight")) {
                self.output_norm = word;
                continue;
            }
            if (std.mem.eql(u8, name, "output.weight")) {
                self.output = try self.projection(s, t, place, self.act_layout.hn, null);
                if (rowSpan(options) > 0) self.output_rows = try self.rprojection(s, t, place, self.act_layout.hn, null);
                if (self.act_layout.mtp) |M| {
                    for (self.mtp_head[0..options.verify_rows], 0..) |*head, row| head.* = try self.projectionRows(s, t, place, M.mo + @as(u32, @intCast(row)) * config.hidden, M.logits, self.draftVocab());
                }
                continue;
            }
            const rest = name["blk.".len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.').?;
            const il = std.fmt.parseInt(u32, rest[0..dot], 10) catch unreachable;
            const field = rest[dot + 1 ..];
            const L = &self.layers[il];
            const A = self.act_layout;
            if (std.mem.eql(u8, field, "attn_norm.weight")) L.attn_norm = word //
            else if (std.mem.eql(u8, field, "post_attention_norm.weight")) L.post_norm = word //
            else if (std.mem.eql(u8, field, "attn_q_norm.weight")) L.q_norm = word //
            else if (std.mem.eql(u8, field, "attn_k_norm.weight")) L.k_norm = word //
            else if (std.mem.eql(u8, field, "ssm_a")) L.ssm_a = word //
            else if (std.mem.eql(u8, field, "ssm_dt.bias")) L.dt = word //
            else if (std.mem.eql(u8, field, "ssm_conv1d.weight")) L.conv_w = word //
            else if (std.mem.eql(u8, field, "ssm_norm.weight")) L.ssm_norm = word //
            else if (std.mem.eql(u8, field, "ffn_gate.weight")) L.gate = try self.both(s, t, place, A.h, A.fg, &L.g_gate) //
            else if (std.mem.eql(u8, field, "ffn_up.weight")) L.up = try self.both(s, t, place, A.h, A.fu, &L.g_up) //
            else if (std.mem.eql(u8, field, "ffn_down.weight")) L.down = try self.both(s, t, place, A.sw, A.f, &L.g_down) //
            else if (std.mem.eql(u8, field, "attn_q.weight")) L.q = try self.both(s, t, place, A.h, A.qf, &L.g_q) //
            else if (std.mem.eql(u8, field, "attn_k.weight")) L.k = try self.both(s, t, place, A.h, A.kc, &L.g_k) //
            else if (std.mem.eql(u8, field, "attn_v.weight")) L.v = try self.both(s, t, place, A.h, A.vc, &L.g_v) //
            else if (std.mem.eql(u8, field, "attn_output.weight")) L.out = try self.both(s, t, place, A.gated, A.a, &L.g_out) //
            else if (std.mem.eql(u8, field, "attn_qkv.weight")) L.qkv = try self.both(s, t, place, A.h, A.mixed, &L.g_qkv) //
            else if (std.mem.eql(u8, field, "attn_gate.weight")) L.z = try self.both(s, t, place, A.h, A.z, &L.g_z) //
            else if (std.mem.eql(u8, field, "ssm_alpha.weight")) L.alpha = try self.both(s, t, place, A.h, A.alpha, &L.g_alpha) //
            else if (std.mem.eql(u8, field, "ssm_beta.weight")) L.beta = try self.both(s, t, place, A.h, A.beta_raw, &L.g_beta) //
            else if (std.mem.eql(u8, field, "ssm_out.weight")) L.ssm_out = try self.both(s, t, place, A.fo, A.a, &L.g_ssm_out) //
            else unreachable;
            if (options.batch_rows > 0 and options.decode_precision == .f16) {
                if (std.mem.eql(u8, field, "attn_q.weight")) L.d_q = try self.dprojection(s, t, place, A.h, A.qf) //
                else if (std.mem.eql(u8, field, "attn_output.weight")) L.d_out = try self.dprojection(s, t, place, A.gated, A.a) //
                else if (std.mem.eql(u8, field, "attn_qkv.weight")) L.d_qkv = try self.dprojection(s, t, place, A.h, A.mixed) //
                else if (std.mem.eql(u8, field, "attn_gate.weight")) L.d_z = try self.dprojection(s, t, place, A.h, A.z) //
                else if (std.mem.eql(u8, field, "ffn_gate.weight")) L.d_gate = try self.dprojection(s, t, place, A.h, A.fg) //
                else if (std.mem.eql(u8, field, "ffn_up.weight")) L.d_up = try self.dprojection(s, t, place, A.h, A.fu) //
                else if (std.mem.eql(u8, field, "ffn_down.weight")) L.d_down = try self.dprojection(s, t, place, A.sw, A.f) //
                else if (std.mem.eql(u8, field, "ssm_out.weight")) L.d_ssm_out = try self.dprojection(s, t, place, A.fo, A.a);
            }
            if (options.batch_rows > 0) {
                // Batched decode: linear layers' lin_in into the plain regions (rows are
                // committed at once); the other projections share the verify ones.
                if (std.mem.eql(u8, field, "attn_qkv.weight")) L.b_qkv = try self.rprojection(s, t, place, A.h, A.mixed) //
                else if (std.mem.eql(u8, field, "attn_gate.weight")) L.b_z = try self.rprojection(s, t, place, A.h, A.z) //
                else if (std.mem.eql(u8, field, "ssm_alpha.weight")) L.b_alpha = try self.rprojection(s, t, place, A.h, A.alpha) //
                else if (std.mem.eql(u8, field, "ssm_beta.weight")) L.b_beta = try self.rprojection(s, t, place, A.h, A.beta_raw);
            }
            if (rowSpan(options) > 0) {
                // Verify: linear layers write lin_in into their slot (kept for the commit).
                const slot: ?layout.SpecSlot = if (config.isAttention(il) or options.verify_rows == 0) null else A.specSlot(config.linearIndex(il));
                if (std.mem.eql(u8, field, "ffn_gate.weight")) L.r_gate = try self.rprojection(s, t, place, A.h, A.fg) //
                else if (std.mem.eql(u8, field, "ffn_up.weight")) L.r_up = try self.rprojection(s, t, place, A.h, A.fu) //
                else if (std.mem.eql(u8, field, "ffn_down.weight")) L.r_down = try self.rprojection(s, t, place, A.sw, A.f) //
                else if (std.mem.eql(u8, field, "attn_q.weight")) L.r_q = try self.rprojection(s, t, place, A.h, A.qf) //
                else if (std.mem.eql(u8, field, "attn_k.weight")) L.r_k = try self.rprojection(s, t, place, A.h, A.kc) //
                else if (std.mem.eql(u8, field, "attn_v.weight")) L.r_v = try self.rprojection(s, t, place, A.h, A.vc) //
                else if (std.mem.eql(u8, field, "attn_output.weight")) L.r_out = try self.rprojection(s, t, place, A.gated, A.a) //
                else if (slot != null and std.mem.eql(u8, field, "attn_qkv.weight")) L.r_qkv = try self.rprojection(s, t, place, A.h, slot.?.mixed) //
                else if (slot != null and std.mem.eql(u8, field, "attn_gate.weight")) L.r_z = try self.rprojection(s, t, place, A.h, slot.?.z) //
                else if (slot != null and std.mem.eql(u8, field, "ssm_alpha.weight")) L.r_alpha = try self.rprojection(s, t, place, A.h, slot.?.alpha) //
                else if (slot != null and std.mem.eql(u8, field, "ssm_beta.weight")) L.r_beta = try self.rprojection(s, t, place, A.h, slot.?.beta) //
                else if (std.mem.eql(u8, field, "ssm_out.weight")) L.r_ssm_out = try self.rprojection(s, t, place, A.fo, A.a);
            }
        }

        // Fused decode FFN input where gate and up of a layer share a bank, format and shape.
        var gate_index: [config.layers]?usize = @splat(null);
        var up_index: [config.layers]?usize = @splat(null);
        for (specs, 0..) |*s, i| {
            const name = config.specName(s);
            if (!std.mem.startsWith(u8, name, "blk.")) continue;
            const rest = name["blk.".len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
            const il = std.fmt.parseInt(u32, rest[0..dot], 10) catch continue;
            if (il >= config.layers) continue;
            if (std.mem.eql(u8, rest[dot + 1 ..], "ffn_gate.weight")) gate_index[il] = i;
            if (std.mem.eql(u8, rest[dot + 1 ..], "ffn_up.weight")) up_index[il] = i;
        }
        if (options.decode_fusion) for (&self.layers, gate_index, up_index) |*L, gi, ui| {
            const g = gi orelse continue;
            const u = ui orelse continue;
            L.gate_up = try self.swigluProjection(&specs[g], tensors[g], placements[g], &specs[u], tensors[u], placements[u]);
            if (L.gate_up != null) self.fused_ffn_layers += 1;
        };
        if (rowSpan(options) > 0 and options.verify_fusion) for (&self.layers, gate_index, up_index) |*L, gi, ui| {
            const g = gi orelse continue;
            const u = ui orelse continue;
            L.r_gate_up = try self.swigluRowsProjection(&specs[g], tensors[g], placements[g], &specs[u], tensors[u], placements[u]);
            if (L.r_gate_up != null) self.fused_verify_layers += 1;
        };

        self.step_commands = try gpu.Commands.init(device);
        self.live_commands += 1;
        self.reset_commands = try gpu.Commands.init(device);
        self.live_commands += 1;
        try self.step_commands.begin();
        try self.record(&self.step_commands, .{});
        try self.step_commands.end();
        try self.reset_commands.begin();
        try self.reset_commands.barrier(.compute, .compute);
        try self.reset_commands.barrier(.transfer, .compute);
        // With the MTP layer, the pending h follows the conv state and is zeroed too.
        const zero_words = layout.State.ssm_words + layout.State.conv_words + @as(u32, if (options.mtp) config.hidden else 0);
        try self.reset_commands.dispatch(&self.kernels[@intFromEnum(KernelId.zero)], std.mem.asBytes(&ZeroPush{ .first = self.state_layout.ssm, .count = zero_words }), .{ 4096, 1, 1 });
        try self.reset_commands.barrier(.compute, .compute);
        try self.reset_commands.end();
        for (1..self.state_layout.slots) |slot| {
            const c = &self.slot_resets[slot - 1];
            c.* = try gpu.Commands.init(device);
            self.live_slot_resets += 1;
            try c.begin();
            try c.barrier(.compute, .compute);
            try c.barrier(.transfer, .compute);
            try c.dispatch(&self.kernels[@intFromEnum(KernelId.zero)], std.mem.asBytes(&ZeroPush{ .first = @intCast(slot * self.state_layout.slot_words + self.state_layout.ssm), .count = zero_words }), .{ 4096, 1, 1 });
            try c.barrier(.compute, .compute);
            try c.end();
        }
        for (1..options.batch_rows + 1) |n| {
            self.batch_commands[n] = try gpu.Commands.init(device);
            self.live_batch += 1;
            try self.batch_commands[n].begin();
            try self.recordBatch(&self.batch_commands[n], @intCast(n));
            try self.batch_commands[n].end();
        }
        for (0..self.plan_count) |i| {
            self.prefill_commands[i] = try gpu.Commands.init(device);
            self.live_prefill += 1;
            try self.prefill_commands[i].begin();
            try self.recordPrefill(&self.prefill_commands[i], i, .{});
            try self.prefill_commands[i].end();
        }
        for (1..options.verify_rows + 1) |n| {
            self.verify_commands[n] = try gpu.Commands.init(device);
            self.live_verify += 1;
            try self.verify_commands[n].begin();
            try self.recordVerify(&self.verify_commands[n], @intCast(n));
            try self.verify_commands[n].end();
            self.commit_commands[n] = try gpu.Commands.init(device);
            self.live_commit += 1;
            try self.commit_commands[n].begin();
            try self.recordCommit(&self.commit_commands[n], @intCast(n));
            try self.commit_commands[n].end();
        }
        if (options.mtp) {
            const drafts = options.verify_rows - 1;
            for (1..options.verify_rows + 1) |m| {
                for (1..drafts + 1) |k| {
                    const c = &self.draft_commands[m][k - 1];
                    c.* = try gpu.Commands.init(device);
                    self.live_draft += 1;
                    try c.begin();
                    try self.recordDraft(c, @intCast(m), @intCast(k));
                    try c.end();
                }
                self.catchup_commands[m] = try gpu.Commands.init(device);
                self.live_catchup += 1;
                try self.catchup_commands[m].begin();
                try self.recordCatchup(&self.catchup_commands[m], @intCast(m));
                try self.catchup_commands[m].end();
            }
            self.save_commands = try gpu.Commands.init(device);
            self.save_live = true;
            try self.save_commands.begin();
            try self.save_commands.barrier(.host, .compute);
            try self.save_commands.barrier(.compute, .compute);
            try self.save_commands.dispatch(&self.kernels[@intFromEnum(KernelId.rowcopy)], std.mem.asBytes(&RowCopyPush{ .src = layout.io.spec(@max(self.rows, 1)).src + 2, .dst = self.state_layout.hp.?, .words = config.hidden, .flags = 1 }), .{ config.hidden / 256, 1, 1 });
            try self.save_commands.barrier(.compute, .compute);
            try self.save_commands.end();
        }
        try self.writeSlotEntry();
        try self.reset();
    }

    fn kvKernel(self: *Model, id: KernelId, cache: u32) *gpu.Kernel {
        const g = self.state_layout.kvBuffer(cache);
        if (g == 0) return &self.kernels[@intFromEnum(id)];
        return &self.kv_kernels[g - 1][kvSlot(id).?];
    }

    /// Bytes of one snapshot: the recurrent and conv state, plus the MTP's pending h.
    pub fn snapshotBytes(self: *const Model) u64 {
        return snapshot_bytes + @as(u64, if (self.options.mtp) config.hidden * 4 else 0);
    }

    fn resolveMtp(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, field: []const u8) Error!void {
        const word: u32 = @intCast(place.offset / 4);
        const A = self.act_layout;
        const L = &self.mtp_layer;
        if (std.mem.eql(u8, field, "attn_norm.weight")) L.attn_norm = word //
        else if (std.mem.eql(u8, field, "post_attention_norm.weight")) L.post_norm = word //
        else if (std.mem.eql(u8, field, "attn_q_norm.weight")) L.q_norm = word //
        else if (std.mem.eql(u8, field, "attn_k_norm.weight")) L.k_norm = word //
        else if (std.mem.eql(u8, field, "nextn.enorm.weight")) L.enorm = word //
        else if (std.mem.eql(u8, field, "nextn.hnorm.weight")) L.hnorm = word //
        else if (std.mem.eql(u8, field, "nextn.shared_head_norm.weight")) L.head_norm = word //
        else if (std.mem.eql(u8, field, "nextn.eh_proj.weight")) {
            L.eh = try self.rprojection(s, t, place, A.mtp.?.cat, A.x);
            for (self.plans[0..self.plan_count], L.g_eh[0..self.plan_count]) |plan, *g| g.* = try self.gprojection(s, t, place, A.mtp.?.cat, A.x, plan);
        } else if (std.mem.eql(u8, field, "attn_q.weight")) L.q = try self.rprojection(s, t, place, A.h, A.qf) //
        else if (std.mem.eql(u8, field, "attn_k.weight")) {
            L.k = try self.rprojection(s, t, place, A.h, A.kc);
            for (self.plans[0..self.plan_count], L.g_k[0..self.plan_count]) |plan, *g| g.* = try self.gprojection(s, t, place, A.h, A.kc, plan);
        } else if (std.mem.eql(u8, field, "attn_v.weight")) {
            L.v = try self.rprojection(s, t, place, A.h, A.vc);
            for (self.plans[0..self.plan_count], L.g_v[0..self.plan_count]) |plan, *g| g.* = try self.gprojection(s, t, place, A.h, A.vc, plan);
        } //
        else if (std.mem.eql(u8, field, "attn_output.weight")) L.out = try self.rprojection(s, t, place, A.gated, A.a) //
        else if (std.mem.eql(u8, field, "ffn_gate.weight")) L.gate = try self.rprojection(s, t, place, A.h, A.fg) //
        else if (std.mem.eql(u8, field, "ffn_up.weight")) L.up = try self.rprojection(s, t, place, A.h, A.fu) //
        else if (std.mem.eql(u8, field, "ffn_down.weight")) L.down = try self.rprojection(s, t, place, A.sw, A.f) //
        else unreachable;
    }

    pub fn deinit(self: *Model) void {
        self.cleanup();
    }

    fn cleanup(self: *Model) void {
        // The copy command retains the snapshot store and the state arena: release it first.
        if (self.copy_live) self.copy_commands.deinit() catch @panic("snapshot copy pending");
        self.copy_live = false;
        if (self.snapshot_slots > 0) self.snapshot_store.deinit() catch @panic("snapshot store in use");
        self.snapshot_slots = 0;
        const cmds = [_]*gpu.Commands{ &self.step_commands, &self.reset_commands };
        for (cmds[0..self.live_commands]) |c| c.deinit() catch @panic("model command still pending");
        self.live_commands = 0;
        for (self.prefill_commands[0..self.live_prefill]) |*c| c.deinit() catch @panic("prefill command still pending");
        self.live_prefill = 0;
        for (self.verify_commands[1 .. self.live_verify + 1]) |*c| c.deinit() catch @panic("verify command still pending");
        self.live_verify = 0;
        for (self.commit_commands[1 .. self.live_commit + 1]) |*c| c.deinit() catch @panic("commit command still pending");
        self.live_commit = 0;
        {
            var left = self.live_draft;
            outer: for (1..matvec.max_rows + 1) |m| for (0..matvec.max_rows) |k| {
                if (left == 0) break :outer;
                if (k >= self.options.verify_rows - 1) continue;
                self.draft_commands[m][k].deinit() catch @panic("draft command still pending");
                left -= 1;
            };
            self.live_draft = 0;
        }
        for (self.slot_resets[0..self.live_slot_resets]) |*c| c.deinit() catch @panic("reset command still pending");
        self.live_slot_resets = 0;
        for (self.batch_commands[1 .. self.live_batch + 1]) |*c| c.deinit() catch @panic("batch command still pending");
        self.live_batch = 0;
        if (self.ptab_live) {
            self.ptab_commands.deinit() catch @panic("page-table command pending");
            self.ptab_staging.deinit() catch @panic("page-table staging in use");
        }
        self.ptab_live = false;
        for (self.catchup_commands[1 .. self.live_catchup + 1]) |*c| c.deinit() catch @panic("catch-up command still pending");
        self.live_catchup = 0;
        if (self.save_live) self.save_commands.deinit() catch @panic("save command still pending");
        self.save_live = false;
        for (self.rpipes[0..self.rpipe_count]) |*p| p.deinit() catch @panic("model pipeline in use");
        self.rpipe_count = 0;
        for (self.pipelines[0..self.pipeline_count]) |*p| p.deinit() catch @panic("model pipeline in use");
        self.pipeline_count = 0;
        for (self.swiglu_pipes[0..self.swiglu_count]) |*p| p.deinit() catch @panic("model pipeline in use");
        self.swiglu_count = 0;
        for (self.swiglu_rpipes[0..self.swiglu_rcount]) |*p| p.deinit() catch @panic("model pipeline in use");
        self.swiglu_rcount = 0;
        for (self.gemm_pipes[0..self.gemm_count]) |*k| k.deinit() catch @panic("model gemm in use");
        self.gemm_count = 0;
        for (self.kv_kernels[0..self.kv_kernels_live]) |*set| for (set) |*k| k.deinit() catch @panic("model kernel in use");
        self.kv_kernels_live = 0;
        for (self.kernels[0..self.live_kernels]) |*k| k.deinit() catch @panic("model kernel in use");
        self.live_kernels = 0;
        for (self.h_kernels[0..self.live_h_kernels]) |*k| k.deinit() catch @panic("model kernel in use");
        self.live_h_kernels = 0;
        const n_banks = self.bank_count;
        var remaining = self.live_buffers;
        for (self.banks[0..n_banks]) |*b| {
            if (remaining == 0) break;
            b.deinit() catch @panic("model bank in use");
            remaining -= 1;
        }
        for (self.kv[0..self.kv_live]) |*b| b.deinit() catch @panic("KV buffer in use");
        self.kv_live = 0;
        if (self.verify_logits_live) self.verify_logits.deinit() catch @panic("verify logits in use");
        self.verify_logits_live = false;
        if (self.embed_host_live) self.embed_host.deinit() catch @panic("embedding in use");
        self.embed_host_live = false;
        const arenas = [_]*gpu.Buffer{ &self.act, &self.state, &self.io };
        for (arenas[0..remaining]) |b| b.deinit() catch @panic("model arena in use");
        self.live_buffers = 0;
        self.bank_count = 0;
    }

    /// Copies every tensor with bank space (`items[i].bytes` > 0) to its placement.
    fn upload(self: *Model, tensors: []const *const gguf.Tensor, items: []const layout.Item, placements: []const layout.Placement) Error!void {
        var staging = try gpu.Buffer.init(self.device, self.options.staging_bytes, .host);
        defer staging.deinit() catch @panic("staging in use");
        var cmd = try gpu.Commands.init(self.device);
        defer cmd.deinit() catch @panic("upload command pending");
        const chunk = std.mem.alignBackward(u64, self.options.staging_bytes, 4);
        for (tensors, items, placements) |t, item, place| {
            if (item.bytes == 0) continue; // not bank-resident (host embedding)
            var done: u64 = 0;
            while (done < t.data.len) {
                const n = @min(chunk, t.data.len - done);
                const mapped = try staging.mapped();
                @memcpy(mapped[0..n], t.data[done..][0..n]);
                try cmd.reset();
                try cmd.begin();
                try cmd.barrier(.transfer, .transfer);
                try cmd.copy(&staging, 0, &self.banks[place.bank], place.offset + done, n);
                try cmd.barrier(.transfer, .compute);
                try cmd.end();
                try cmd.run(self.options.timeout_ns);
                done += n;
            }
        }
    }

    /// Default KV page tables (docs/specs/concurrent.md, "18b.2 design"): slot s owns pool
    /// pages s * seq_pages .. while the pool holds them (one slot: the identity table); the
    /// other slots start without pages (`mapPages`). Also creates the page-table staging.
    fn writePageTable(self: *Model) Error!void {
        const S = self.state_layout;
        self.ptab_staging = try gpu.Buffer.init(self.device, @as(u64, self.act_layout.ptab_words) * 4, .host);
        self.ptab_commands = gpu.Commands.init(self.device) catch |e| {
            self.ptab_staging.deinit() catch @panic("page-table staging in use");
            return e;
        };
        self.ptab_live = true;
        for (0..S.slots) |slot| {
            if ((slot + 1) * S.seq_pages > S.pages) break;
            const words = std.mem.bytesAsSlice(u32, try self.ptab_staging.mapped());
            for (words[0..S.seq_pages], 0..) |*w, i| w.* = @intCast(slot * S.seq_pages + i);
            try self.commitPages(@intCast(slot), S.seq_pages);
        }
    }

    /// Append `pages` (pool page ids, each free) to `slot`'s page table: they hold the
    /// slot's next logical pages. The model enforces that a pool page belongs to one slot.
    pub fn mapPages(self: *Model, slot: u32, pages: []const u32) Error!void {
        const S = self.state_layout;
        if (slot >= S.slots) return error.InvalidSlot;
        if (pages.len > S.seq_pages - self.mapped[slot]) return error.InvalidToken;
        for (pages, 0..) |page, i| {
            if (page >= S.pages or self.page_owner[page] != no_owner) return error.PageInUse;
            for (pages[0..i]) |other| if (other == page) return error.PageInUse;
        }
        @memcpy(std.mem.bytesAsSlice(u32, try self.ptab_staging.mapped())[0..pages.len], pages);
        try self.commitPages(slot, @intCast(pages.len));
    }

    /// Release every page of `slot` to the pool; its table becomes empty (the slot needs
    /// `mapPages` before it runs again). Its state and position are kept.
    pub fn releasePages(self: *Model, slot: u32) Error!void {
        if (slot >= self.state_layout.slots) return error.InvalidSlot;
        for (self.page_owner[0..self.state_layout.pages]) |*owner| {
            if (owner.* == slot) owner.* = no_owner;
        }
        self.mapped[slot] = 0;
    }

    /// Pages mapped for `slot` (logical pages 0..).
    pub fn mappedPages(self: *const Model, slot: u32) u32 {
        return self.mapped[slot];
    }

    /// Copy staged entries 0..n-1 to `slot`'s table after its mapped pages; take ownership.
    fn commitPages(self: *Model, slot: u32, n: u32) Error!void {
        if (n == 0) return;
        const words = std.mem.bytesAsSlice(u32, try self.ptab_staging.mapped());
        const A = self.act_layout;
        const c = &self.ptab_commands;
        try c.reset();
        try c.begin();
        try c.barrier(.compute, .transfer);
        try c.copy(&self.ptab_staging, 0, &self.act, (@as(u64, A.ptab) + @as(u64, slot) * A.ptab_words + self.mapped[slot]) * 4, @as(u64, n) * 4);
        try c.barrier(.transfer, .compute);
        try c.end();
        try c.run(self.options.timeout_ns);
        for (words[0..n]) |page| self.page_owner[page] = @intCast(slot);
        self.mapped[slot] += n;
    }

    /// Positions below `end` of `slot` have mapped pages.
    fn ensureMapped(self: *const Model, slot: u32, end: u64) Error!void {
        if (end > @as(u64, self.mapped[slot]) * self.state_layout.page) return error.PagesMissing;
    }

    /// Make `slot` current: `reset`, `prefill`, `step`, `verify`/`commit` and the snapshots act
    /// on it; every slot keeps its own position.
    pub fn select(self: *Model, slot: u32) Error!void {
        if (slot >= self.state_layout.slots) return error.InvalidSlot;
        if (self.pending_verify != 0) return error.VerifyPending;
        if (slot == self.slot) return;
        self.slot_positions[self.slot] = self.position;
        self.slot = slot;
        self.position = self.slot_positions[slot];
        try self.writeSlotEntry();
    }

    /// The current slot and a slot's position.
    pub fn slotPosition(self: *const Model, slot: u32) u32 {
        return if (slot == self.slot) self.position else self.slot_positions[slot];
    }

    /// The single-slot entry of the current slot (read by the commands only when
    /// `slot_io` is set, i.e. with more than one slot).
    fn writeSlotEntry(self: *Model) Error!void {
        const words = std.mem.bytesAsSlice(u32, try self.io.mapped());
        const e = words[layout.io.slot..][0..layout.io.slot_entry];
        e.* = .{ self.position, self.slot * self.act_layout.ptab_words, self.slot * self.state_layout.slot_words, 0 };
    }

    /// State arena byte offset of the current slot.
    fn slotStateBytes(self: *const Model) u64 {
        return @as(u64, self.slot) * self.state_layout.slot_words * 4;
    }

    /// Batched decode (docs/specs/concurrent.md, "18b.2 design"): one token for each row's
    /// slot (distinct slots) at that slot's position, with the single-sequence arithmetic per
    /// row. Returns rows.len logits rows (row r at [r * vocab ..], borrowed until the next
    /// call) and advances each slot's position by one.
    pub fn decodeBatch(self: *Model, rows: []const BatchRow) Error![]const f32 {
        if (rows.len == 0 or rows.len > self.options.batch_rows) return error.InvalidBatch;
        if (self.pending_verify != 0) return error.VerifyPending;
        var seen: u64 = 0;
        for (rows) |row| {
            if (row.slot >= self.state_layout.slots) return error.InvalidSlot;
            const bit = @as(u64, 1) << @intCast(row.slot);
            if (seen & bit != 0) return error.InvalidBatch;
            seen |= bit;
            if (row.token >= config.vocab) return error.InvalidToken;
            const pos = self.slotPosition(row.slot);
            if (pos >= self.state_layout.context) return error.ContextFull;
            try self.ensureMapped(row.slot, @as(u64, pos) + 1);
        }
        const words = std.mem.bytesAsSlice(u32, try self.io.mapped());
        const n: u32 = @intCast(rows.len);
        words[layout.io.count] = n;
        const table = layout.io.batch(@max(self.rows, 1));
        const rope = layout.io.rope(@max(self.rows, 1));
        for (rows, 0..) |row, r| {
            const pos = self.slotPosition(row.slot);
            words[layout.io.tokens + r] = row.token;
            self.writeRope(words, @intCast(rope + 64 * r), pos, 1);
            words[table + 4 * r ..][0..4].* = .{ pos, row.slot * self.act_layout.ptab_words, row.slot * self.state_layout.slot_words, 0 };
        }
        try self.batch_commands[n].run(self.options.timeout_ns);
        for (rows) |row| {
            if (row.slot == self.slot) self.position += 1 else self.slot_positions[row.slot] += 1;
        }
        const bytes = try self.verify_logits.mapped();
        const floats: []align(1) const f32 = std.mem.bytesAsSlice(f32, bytes[0 .. rows.len * config.vocab * 4]);
        return @alignCast(floats);
    }

    fn projection(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: ?u32) Error!Proj {
        return self.projectionRows(s, t, place, input_word, output_word, @intCast(s.rows));
    }
    /// `projection` over the first `rows` (1..s.rows) weight rows only.
    fn projectionRows(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: ?u32, rows: u32) Error!Proj {
        std.debug.assert(rows >= 1 and rows <= s.rows);
        const format = try config.matrixFormat(t.kind);
        const aligned = (format == .q4_1 or format == .q5_k) and place.offset % 4 == 0;
        const to_io = output_word == null;
        const key: u32 = @as(u32, place.bank) | (@as(u32, @intFromEnum(format)) << 8) | (@as(u32, @intFromBool(aligned)) << 16) | (@as(u32, @intFromBool(to_io)) << 17);
        var index: ?u8 = null;
        for (self.pipe_keys[0..self.pipeline_count], 0..) |k, i| if (k == key) {
            index = @intCast(i);
        };
        if (index == null) {
            if (self.pipeline_count == max_pipelines) return error.TooManyPipelines;
            const i = self.pipeline_count;
            self.pipelines[i] = try matvec.Pipeline.initWith(format, aligned, self.options.matvec_accumulation, &self.banks[place.bank], &self.act, if (to_io) &self.io else &self.act);
            self.pipe_keys[i] = key;
            self.pipeline_count += 1;
            index = i;
        }
        const shape: matvec.Shape = .{ .format = format, .columns = @intCast(s.k), .rows = rows };
        const out_bytes: u64 = if (output_word) |w| @as(u64, w) * 4 else layout.io.logits * 4;
        return .{ .pipe = index.?, .geo = try self.pipelines[index.?].projection(shape, place.offset, @as(u64, input_word) * 4, out_bytes) };
    }

    /// The fused gate/up/swiglu decode projection of a layer, or null when the two tensors
    /// differ in bank, format or shape, or the format has no fused module.
    fn swigluProjection(self: *Model, sg: *const config.Spec, tg: *const gguf.Tensor, pg: layout.Placement, su: *const config.Spec, tu: *const gguf.Tensor, pu: layout.Placement) Error!?SwProj {
        if (tg.kind != tu.kind or pg.bank != pu.bank or sg.k != su.k or sg.rows != su.rows) return null;
        const format = try config.matrixFormat(tg.kind);
        const aligned = (format == .q4_1 or format == .q5_k) and pg.offset % 4 == 0 and pu.offset % 4 == 0;
        const key: u32 = @as(u32, pg.bank) | (@as(u32, @intFromEnum(format)) << 8) | (@as(u32, @intFromBool(aligned)) << 16);
        var index: ?u8 = null;
        for (self.swiglu_keys[0..self.swiglu_count], 0..) |k, i| if (k == key) {
            index = @intCast(i);
        };
        if (index == null) {
            if (self.swiglu_count == max_swiglu_pipelines) return error.TooManyPipelines;
            const i = self.swiglu_count;
            self.swiglu_pipes[i] = matvec.SwigluPipeline.initWith(format, aligned, self.options.matvec_accumulation, &self.banks[pg.bank], &self.act, &self.act) catch |e| switch (e) {
                error.InvalidShape => return null,
                else => return e,
            };
            self.swiglu_keys[i] = key;
            self.swiglu_count += 1;
            index = i;
        }
        const A = self.act_layout;
        const shape: matvec.Shape = .{ .format = format, .columns = @intCast(sg.k), .rows = @intCast(sg.rows) };
        return .{ .pipe = index.?, .geo = try self.swiglu_pipes[index.?].projection(shape, pg.offset, pu.offset, @as(u64, A.h) * 4, @as(u64, A.fg) * 4, @as(u64, A.fu) * 4, @as(u64, A.sw) * 4) };
    }

    /// The fused gate/up/swiglu verify projection of a layer (`verify_rows` rows), or null
    /// under the same conditions as `swigluProjection`.
    fn swigluRowsProjection(self: *Model, sg: *const config.Spec, tg: *const gguf.Tensor, pg: layout.Placement, su: *const config.Spec, tu: *const gguf.Tensor, pu: layout.Placement) Error!?SwRProj {
        if (tg.kind != tu.kind or pg.bank != pu.bank or sg.k != su.k or sg.rows != su.rows) return null;
        const format = try config.matrixFormat(tg.kind);
        if (format == .f32 or format == .q8_0) return null;
        const aligned = (format == .q4_1 or format == .q5_k) and pg.offset % 4 == 0 and pu.offset % 4 == 0;
        const key: u32 = @as(u32, pg.bank) | (@as(u32, @intFromEnum(format)) << 8) | (@as(u32, @intFromBool(aligned)) << 16);
        var index: ?u8 = null;
        for (self.swiglu_rkeys[0..self.swiglu_rcount], 0..) |k, i| if (k == key) {
            index = @intCast(i);
        };
        if (index == null) {
            if (self.swiglu_rcount == max_swiglu_pipelines) return error.TooManyPipelines;
            const i = self.swiglu_rcount;
            self.swiglu_rpipes[i] = try matvec.SwigluRowsPipeline.init(format, aligned, self.options.matvec_accumulation, rowCount(self.options), &self.banks[pg.bank], &self.act, &self.act);
            self.swiglu_rkeys[i] = key;
            self.swiglu_rcount += 1;
            index = i;
        }
        const A = self.act_layout;
        const shape: matvec.Shape = .{ .format = format, .columns = @intCast(sg.k), .rows = @intCast(sg.rows) };
        return .{ .pipe = index.?, .geo = try self.swiglu_rpipes[index.?].projectionSpan(shape, pg.offset, pu.offset, @as(u64, A.h) * 4, @as(u64, A.fg) * 4, @as(u64, A.fu) * 4, @as(u64, A.sw) * 4, @intCast(sg.k), @intCast(sg.rows), rowCount(self.options), rowSpan(self.options)) };
    }

    fn both(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: u32, g: *[max_plans]GProj) Error!Proj {
        for (self.plans[0..self.plan_count], g[0..self.plan_count]) |plan, *slot| slot.* = try self.gprojection(s, t, place, input_word, output_word, plan);
        return self.projection(s, t, place, input_word, output_word);
    }

    /// Multi-row verify projection (input rows stride K, output rows stride M, or the
    /// verify logits rows when `output_word` is null), validated for `verify_rows` rows.
    fn rprojection(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: ?u32) Error!RProj {
        const format = try config.matrixFormat(t.kind);
        const aligned = (format == .q4_1 or format == .q5_k) and place.offset % 4 == 0;
        const to_logits = output_word == null;
        const key: u32 = @as(u32, place.bank) | (@as(u32, @intFromEnum(format)) << 8) | (@as(u32, @intFromBool(aligned)) << 16) | (@as(u32, @intFromBool(to_logits)) << 17);
        var index: ?u8 = null;
        for (self.rpipe_keys[0..self.rpipe_count], 0..) |k, i| if (k == key) {
            index = @intCast(i);
        };
        if (index == null) {
            if (self.rpipe_count == max_pipelines) return error.TooManyPipelines;
            const i = self.rpipe_count;
            self.rpipes[i] = try matvec.RowsPipeline.initWith(format, aligned, self.options.matvec_accumulation, rowCount(self.options), &self.banks[place.bank], &self.act, if (to_logits) &self.verify_logits else &self.act);
            self.rpipe_keys[i] = key;
            self.rpipe_count += 1;
            index = i;
        }
        const shape: matvec.Shape = .{ .format = format, .columns = @intCast(s.k), .rows = @intCast(s.rows) };
        const out_bytes: u64 = if (output_word) |w| @as(u64, w) * 4 else 0;
        return .{ .pipe = index.?, .geo = try self.rpipes[index.?].projectionSpan(shape, place.offset, @as(u64, input_word) * 4, out_bytes, @intCast(s.k), @intCast(s.rows), rowCount(self.options), rowSpan(self.options)) };
    }

    /// The f16-mode batched decode projection (gemm_f16n over the batch span), or null when
    /// the projection stays FP32 (`gemm.f16DecodeEligible`).
    fn dprojection(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: u32) Error!?GProj {
        const format = try config.matrixFormat(t.kind);
        const v = gemm.variant(format) catch return null;
        const M: u32 = @intCast(s.rows);
        const K: u32 = @intCast(s.k);
        if (!gemm.f16DecodeEligible(v, M)) return null;
        const key: u32 = @as(u32, place.bank) | (@as(u32, @intFromEnum(v)) << 8) | (@as(u32, 3) << 24);
        var index: ?u8 = null;
        for (self.gemm_keys[0..self.gemm_count], 0..) |k, i| if (k == key) {
            index = @intCast(i);
        };
        if (index == null) {
            if (self.gemm_count == max_gemm_pipelines) return error.TooManyPipelines;
            const i = self.gemm_count;
            const buffers = [_]*gpu.Buffer{ &self.banks[place.bank], &self.act, &self.io, &self.act };
            self.gemm_pipes[i] = try gpu.Kernel.init(self.device, try gemm.moduleF16n(v), &buffers, @sizeOf(gemm.Push));
            self.gemm_keys[i] = key;
            self.gemm_count += 1;
            index = i;
        }
        var push: gemm.Push = .{ .a_base = @intCast(place.offset), .a_rs = @intCast(t.data.len / M), .x_base = input_word, .x_rs = K, .y_base = output_word, .y_rs = M, .m = M, .k = K };
        const span = std.mem.alignForward(u32, self.options.batch_rows, gemm.f16n_tile_n);
        push.k_chunk = gemm.f16nChunk(M, K, split_target);
        var reduce: ?gemm.ReducePush = null;
        var parts: u32 = 1;
        if (push.k_chunk != 0) {
            parts = gemm.splitCount(K, push.k_chunk);
            const part = self.part_slots[splitSlot(config.specName(s))];
            push.y_base = part;
            push.y_bs = span * M;
            reduce = .{ .part = part, .splits = parts, .part_bs = span * M, .y = output_word, .y_rs = M, .m = M };
        }
        const groups = try gemm.validateF16n(v, push, .{ .rows = span, .batches = parts }, self.banks[place.bank].size, self.act.size);
        return .{ .pipe = index.?, .push = push, .groups = groups, .reduce = reduce };
    }

    fn gprojection(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: u32, plan: Plan) Error!GProj {
        const format = try config.matrixFormat(t.kind);
        const v = try gemm.variant(format);
        const M: u32 = @intCast(s.rows);
        const K: u32 = @intCast(s.k);
        const tile = gemm.tileFor(M, plan.rows);
        const f16k: ?gemm.F16Kernel = if (self.options.prefill_precision == .f16) gemm.f16Kernel(v, M, K, plan.rows) else null;
        const kind: u32 = if (f16k) |k| 1 + @as(u32, @intFromEnum(k)) else 0;
        const key: u32 = @as(u32, place.bank) | (@as(u32, @intFromEnum(v)) << 8) | (@as(u32, @intFromEnum(tile)) << 16) | (kind << 24);
        var index: ?u8 = null;
        for (self.gemm_keys[0..self.gemm_count], 0..) |k, i| if (k == key) {
            index = @intCast(i);
        };
        if (index == null) {
            if (self.gemm_count == max_gemm_pipelines) return error.TooManyPipelines;
            const i = self.gemm_count;
            const buffers = [_]*gpu.Buffer{ &self.banks[place.bank], &self.act, &self.io, &self.act };
            self.gemm_pipes[i] = if (f16k) |k| switch (k) {
                .wave64 => try gpu.Kernel.init(self.device, try gemm.moduleF16(v), &buffers, @sizeOf(gemm.Push)),
                .x16 => try gpu.Kernel.initWith(self.device, try gemm.moduleF16x(v), &buffers, @sizeOf(gemm.Push), .{ .subgroup_size = gemm.f16x_subgroup, .full_subgroups = true, .binary = self.f16xBinary(v) }),
            } else try gpu.Kernel.init(self.device, try gemm.module(v, tile), &buffers, @sizeOf(gemm.Push));
            self.gemm_keys[i] = key;
            self.gemm_count += 1;
            index = i;
        }
        var push: gemm.Push = if (format == .f32)
            .{ .a_base = @intCast(place.offset / 4), .a_rs = K, .a_cs = 1, .x_base = input_word, .x_rs = K, .y_base = output_word, .y_rs = M, .m = M, .k = K }
        else
            .{ .a_base = @intCast(place.offset), .a_rs = @intCast(t.data.len / M), .x_base = input_word, .x_rs = K, .y_base = output_word, .y_rs = M, .m = M, .k = K };
        if (f16k) |k| switch (k) {
            .wave64 => {
                const groups = try gemm.validateF16(v, push, .{ .rows = plan.rows }, self.banks[place.bank].size, self.act.size);
                return .{ .pipe = index.?, .push = push, .groups = groups, .reduce = null };
            },
            .x16 => {
                // The producer of `input_word` writes its f16 copy to the x16 region with
                // row stride K (every consumer's K equals its input's width).
                push.x_base = 2 * (self.act_layout.x16 orelse return error.InvalidShape);
                push.x_rs = K;
                const groups = try gemm.validateF16x(v, push, .{ .rows = plan.rows }, self.banks[place.bank].size, self.act.size);
                return .{ .pipe = index.?, .push = push, .groups = groups, .reduce = null, .x16 = true };
            },
        };
        push.k_chunk = gemm.splitChunk(M, plan.rows, K, split_target, tile);
        const splits = gemm.splitCount(K, push.k_chunk);
        var reduce: ?gemm.ReducePush = null;
        if (splits > 1) {
            const part = self.part_slots[splitSlot(config.specName(s))];
            push.y_base = part;
            push.y_bs = plan.rows * M;
            reduce = .{ .part = part, .splits = splits, .part_bs = plan.rows * M, .y = output_word, .y_rs = M, .m = M };
        }
        const groups = try gemm.validate(v, tile, push, .{ .rows = plan.rows, .batches = splits }, self.banks[place.bank].size, self.act.size);
        return .{ .pipe = index.?, .push = push, .groups = groups, .reduce = reduce };
    }

    /// Clear recurrent and convolution state; the next step is position 0.
    pub fn reset(self: *Model) Error!void {
        self.pending_verify = 0; // an uncommitted verify is abandoned with the sequence
        try (if (self.slot == 0) &self.reset_commands else &self.slot_resets[self.slot - 1]).run(self.options.timeout_ns);
        self.position = 0;
        self.mtp_pending = self.hpPending();
    }

    /// MTP state after a reset, restore or prefill: the next pass is one row whose h is the
    /// pending h in the state arena.
    fn hpPending(self: *const Model) MtpPending {
        return .{ .rows = 1, .src = if (self.state_layout.hp) |hp| hp | state_source else 0 };
    }

    /// Copy the recurrent and convolution state (as of `position`) into snapshot `slot`.
    pub fn saveSnapshot(self: *Model, slot: u32) Error!void {
        if (self.pending_verify != 0) return error.VerifyPending;
        if (slot >= self.snapshot_slots) return error.InvalidSnapshot;
        try self.flushMtp();
        try self.snapshotCopy(&self.state, self.slotStateBytes() + @as(u64, self.state_layout.ssm) * 4, &self.snapshot_store, slot * self.snapshotBytes());
    }

    /// Restore snapshot `slot`, saved when `position` tokens had been processed; the next
    /// token is at `position`. The caller guarantees the attention KV below `position` is
    /// unchanged since the save (only positions at or above it were written).
    pub fn loadSnapshot(self: *Model, slot: u32, position: u32) Error!void {
        if (self.pending_verify != 0) return error.VerifyPending;
        if (slot >= self.snapshot_slots) return error.InvalidSnapshot;
        if (position > self.state_layout.context) return error.ContextFull;
        try self.snapshotCopy(&self.snapshot_store, slot * self.snapshotBytes(), &self.state, self.slotStateBytes() + @as(u64, self.state_layout.ssm) * 4);
        self.position = position;
        self.mtp_pending = self.hpPending();
    }

    fn snapshotCopy(self: *Model, source: *gpu.Buffer, source_offset: u64, destination: *gpu.Buffer, destination_offset: u64) Error!void {
        const c = &self.copy_commands;
        try c.reset();
        try c.begin();
        try c.barrier(.compute, .transfer);
        try c.copy(source, source_offset, destination, destination_offset, self.snapshotBytes());
        try c.barrier(.transfer, .compute);
        try c.end();
        try c.run(self.options.timeout_ns);
    }

    /// Advance one token at `self.position`; logits are borrowed until the next call.
    /// With the MTP layer, the pending MTP rows and this token's row run first (a catch-up
    /// pass: MTP KV only, the rows a draft's first pass would run), and the step's h
    /// becomes the pending h, so drafts after steps equal drafts after verify/commit.
    pub fn step(self: *Model, token: u32) Error![]const f32 {
        if (self.options.mtp) {
            if (self.pending_verify != 0) return error.VerifyPending;
            if (token >= config.vocab) return error.InvalidToken;
            if (self.position >= self.state_layout.context) return error.ContextFull;
            try self.catchupThrough(token);
        }
        const logits = try self.run(&self.step_commands, token);
        if (self.options.mtp) self.mtp_pending = .{ .rows = 1, .src = self.act_layout.hn };
        return logits;
    }

    /// MTP catch-up of the pending rows plus `token` at `position` (KV only, no head).
    fn catchupThrough(self: *Model, token: u32) Error!void {
        const pending = self.mtp_pending;
        const m = pending.rows;
        if (m == 0) return; // no known h (cannot happen after reset/prefill/commit)
        const words = std.mem.bytesAsSlice(u32, try self.io.mapped());
        const sp = layout.io.spec(self.rows);
        words[layout.io.count] = matvec.max_rows; // embed_b rows bound
        @memcpy(words[sp.tok..][0 .. m - 1], pending.tokens[0 .. m - 1]);
        words[sp.tok + m - 1] = token;
        const first = self.position + 1 - m;
        for (0..m) |i| words[sp.pos + i] = first + @as(u32, @intCast(i));
        self.writeRope(words, sp.rope, first, m);
        words[sp.src] = pending.src;
        words[sp.src + 1] = pending.src + config.hidden;
        try self.catchup_commands[m].run(self.options.timeout_ns);
    }

    /// Speculative verification (docs/specs/speculative.md): process `tokens` (1 ..
    /// `verify_rows`) at positions `position ..` with decode arithmetic per row and return
    /// their logits rows (row r at [r * vocab ..], borrowed until the next call). Writes
    /// the attention K/V of these positions but neither advances `position` nor changes
    /// the recurrent state: `commit` must follow before any other operation.
    pub fn verify(self: *Model, tokens: []const u32) Error![]const f32 {
        if (self.options.verify_rows == 0) return error.VerifyDisabled;
        if (self.pending_verify != 0) return error.VerifyPending;
        if (tokens.len == 0 or tokens.len > self.options.verify_rows) return error.InvalidToken;
        for (tokens) |token| if (token >= config.vocab) return error.InvalidToken;
        if (tokens.len > self.state_layout.context - self.position) return error.ContextFull;
        try self.ensureMapped(self.slot, @as(u64, self.position) + tokens.len);
        const words = std.mem.bytesAsSlice(u32, try self.io.mapped());
        words[layout.io.position] = self.position;
        words[layout.io.count] = @intCast(tokens.len);
        @memcpy(words[layout.io.tokens..][0..tokens.len], tokens);
        self.writeRope(words, layout.io.rope(self.rows), self.position, tokens.len);
        try self.verify_commands[tokens.len].run(self.options.timeout_ns);
        self.pending_verify = @intCast(tokens.len);
        @memcpy(self.verify_tokens[0..tokens.len], tokens);
        const bytes = try self.verify_logits.mapped();
        const floats: []align(1) const f32 = std.mem.bytesAsSlice(f32, bytes[0 .. tokens.len * config.vocab * 4]);
        return @alignCast(floats);
    }

    /// Keep the first `m` (1 .. verified rows) rows of the last `verify`: store the
    /// recurrent and convolution state after row m - 1 and advance `position` by m. With
    /// the MTP layer, rows p+1..p+m (the accepted drafts, then the next token) become the
    /// next pass's first rows, with h = the verify's final-norm rows 0..m-1.
    pub fn commit(self: *Model, m: u32) Error!void {
        if (self.pending_verify == 0) return error.NoVerify;
        if (m == 0 or m > self.pending_verify) return error.InvalidToken;
        try self.commit_commands[m].run(self.options.timeout_ns);
        self.pending_verify = 0;
        self.position += m;
        if (self.options.mtp) {
            self.mtp_pending = .{ .rows = m, .src = self.act_layout.hn };
            @memcpy(self.mtp_pending.tokens[0 .. m - 1], self.verify_tokens[1..m]);
        }
    }

    /// Draft `k` (1 .. verify_rows - 1) tokens after `token` (the next token, at
    /// `position`) with the MTP layer: one pass over the pending rows plus `token`, then
    /// k - 1 chained passes, each drafting the argmax of its logits. The main model is
    /// unchanged; the MTP KV of positions up to position + k - 1 is written (idempotent:
    /// repeating a draft rewrites the same values). Returns the drafts (borrowed until the
    /// next call).
    pub fn draft(self: *Model, token: u32, k: u32) Error![]const u32 {
        if (!self.options.mtp) return error.MtpDisabled;
        if (self.pending_verify != 0) return error.VerifyPending;
        if (token >= config.vocab) return error.InvalidToken;
        if (k == 0 or k >= self.options.verify_rows) return error.InvalidToken;
        if (self.position + k > self.state_layout.context) return error.ContextFull;
        try self.ensureMapped(self.slot, @as(u64, self.position) + k);
        const pending = self.mtp_pending;
        const m = pending.rows;
        const words = std.mem.bytesAsSlice(u32, try self.io.mapped());
        const sp = layout.io.spec(self.rows);
        words[layout.io.count] = matvec.max_rows; // embed_b rows bound
        @memcpy(words[sp.tok..][0 .. m - 1], pending.tokens[0 .. m - 1]);
        words[sp.tok + m - 1] = token;
        const first = self.position + 1 - m;
        for (0..m + k - 1) |i| words[sp.pos + i] = first + @as(u32, @intCast(i));
        self.writeRope(words, sp.rope, first, m + k - 1);
        words[sp.src] = pending.src;
        words[sp.src + 1] = pending.src + config.hidden;
        try self.draft_commands[m][k - 1].run(self.options.timeout_ns);
        // The pending rows stay pending: a repeated draft recomputes them identically, and
        // the commit after the verify replaces them.
        for (self.drafts[0..k], words[sp.draft..][0..k]) |*d, w| d.* = w;
        for (self.draft_probs[0..k], words[sp.prob..][0..k]) |*p, w| p.* = @bitCast(w);
        return self.drafts[0..k];
    }

    /// The gemm_f16x pipeline binary to offer `gpu.Kernel` (`Options.gemm_code`).
    fn f16xBinary(self: *const Model, v: gemm.Variant) ?gpu.Kernel.Binary {
        if (self.options.gemm_code != .native) return null;
        const n = gemm.nativeF16x(v) orelse return null;
        return .{ .data = n.data, .key = n.key, .global_key = n.global_key };
    }

    /// What `Options.gemm_code` resolves to for gemm_f16x on this device (for logs).
    pub fn gemmCodeStatus(self: *const Model) []const u8 {
        if (self.options.prefill_precision != .f16) return "unused (fp32 prefill)";
        if (self.options.gemm_code == .spirv) return "spirv (--gemm-code spirv)";
        const key = self.device.pipeline_key orelse return "spirv (fallback: pipeline binaries not enabled on this device, or the driver lacks VK_KHR_pipeline_binary)";
        const n = gemm.nativeF16x(.q4_0).?;
        if (!std.mem.eql(u8, &key, n.global_key)) return "spirv (fallback: the driver's pipeline key differs from the binary's; other Mesa build, GPU or driver options)";
        return "native";
    }

    /// The DeltaNet kernels (`Options.delta_state_out`).
    fn deltaKernel(self: *const Model) KernelId {
        return if (self.options.delta_state_out) .delta else .delta_legacy;
    }
    fn deltaBKernel(self: *const Model) KernelId {
        return if (self.options.delta_state_out) .delta_b else .delta_b_legacy;
    }

    /// Token ids the draft head covers (`Options.draft_vocab`, 0 = all).
    pub fn draftVocab(self: *const Model) u32 {
        return if (self.options.draft_vocab == 0) config.vocab else self.options.draft_vocab;
    }

    /// The MTP's softmax probability of each draft of the last `draft` (borrowed).
    pub fn draftProbs(self: *const Model, k: u32) []const f32 {
        return self.draft_probs[0..k];
    }

    /// Run the pending MTP rows that precede the next token (after a commit: the accepted
    /// drafts) and store the h of the last processed row as the pending h, so a prefill
    /// or snapshot can follow.
    fn flushMtp(self: *Model) Error!void {
        if (!self.options.mtp) return;
        const pending = self.mtp_pending;
        if (pending.rows == 0 or pending.src & state_source != 0) {
            // Nothing pending (a draft ran the rows; its h is not kept) or already in hp.
            self.mtp_pending = self.hpPending();
            return;
        }
        const words = std.mem.bytesAsSlice(u32, try self.io.mapped());
        const sp = layout.io.spec(self.rows);
        const n = pending.rows - 1;
        if (n > 0) {
            words[layout.io.count] = matvec.max_rows;
            @memcpy(words[sp.tok..][0..n], pending.tokens[0..n]);
            const first = self.position - n;
            for (0..n) |i| words[sp.pos + i] = first + @as(u32, @intCast(i));
            self.writeRope(words, sp.rope, first, n);
            words[sp.src] = pending.src;
            words[sp.src + 1] = pending.src + config.hidden;
            try self.catchup_commands[n].run(self.options.timeout_ns);
        }
        words[sp.src + 2] = pending.src + n * config.hidden;
        try self.save_commands.run(self.options.timeout_ns);
        self.mtp_pending = self.hpPending();
    }

    /// Host RoPE rows `0..n` for positions `first + r` (FP64 angle and trig, rounded
    /// once to FP32: the decode step's values).
    fn writeRope(self: *const Model, words: []align(1) u32, rope: u32, first: u32, n: usize) void {
        for (0..n) |r| {
            const pos: f64 = @floatFromInt(first + r);
            for (0..config.rope_dims / 2) |i| {
                const inv = std.math.pow(f64, self.hyper.rope_base, -@as(f64, @floatFromInt(2 * i)) / config.rope_dims);
                words[rope + r * 64 + i] = @bitCast(@as(f32, @floatCast(@cos(pos * inv))));
                words[rope + r * 64 + 32 + i] = @bitCast(@as(f32, @floatCast(@sin(pos * inv))));
            }
        }
    }

    /// Process prompt tokens in chunks of at most `rows` from the current position;
    /// returns the logits of the last token (borrowed until the next call).
    pub fn prefill(self: *Model, tokens: []const u32) Error![]const f32 {
        if (self.rows == 0) return error.PrefillDisabled;
        if (tokens.len == 0) return error.InvalidToken;
        for (tokens) |token| if (token >= config.vocab) return error.InvalidToken;
        if (tokens.len > self.state_layout.context - self.position) return error.ContextFull;
        var logits: []const f32 = undefined;
        var i: usize = 0;
        while (i < tokens.len) {
            const c = self.nextChunk(@intCast(tokens.len - i));
            logits = try self.runChunk(&self.prefill_commands[c.plan], c.plan, tokens[i..][0..c.rows]);
            i += c.rows;
        }
        return logits;
    }

    /// Index of the first recorded plan covering a chunk of `n` (1..rows) tokens.
    pub fn planFor(self: *const Model, n: u32) usize {
        return planIndex(self.plans[0..self.plan_count], n);
    }
    /// Next prefill chunk (`chunkFor` over the recorded plans).
    pub fn nextChunk(self: *const Model, remaining: u32) Chunk {
        return chunkFor(self.plans[0..self.plan_count], self.rows, remaining);
    }

    /// One chunk (1..plans[plan].rows tokens) through a command recorded by
    /// `recordPrefill(commands, plan, ...)`.
    pub fn runChunk(self: *Model, commands: *gpu.Commands, plan: usize, chunk: []const u32) Error![]const f32 {
        if (self.pending_verify != 0) return error.VerifyPending;
        if (self.rows == 0) return error.PrefillDisabled;
        if (plan >= self.plan_count) return error.InvalidPlan;
        if (chunk.len == 0 or chunk.len > self.plans[plan].rows) return error.InvalidToken;
        for (chunk) |token| if (token >= config.vocab) return error.InvalidToken;
        if (chunk.len > self.state_layout.context - self.position) return error.ContextFull;
        try self.ensureMapped(self.slot, @as(u64, self.position) + chunk.len);
        try self.flushMtp();
        const bytes = try self.io.mapped();
        const words = std.mem.bytesAsSlice(u32, bytes);
        words[layout.io.count] = @intCast(chunk.len);
        words[layout.io.p0] = self.position;
        @memcpy(words[layout.io.tokens..][0..chunk.len], chunk);
        const rope = layout.io.rope(self.rows);
        for (0..chunk.len) |r| {
            const pos: f64 = @floatFromInt(self.position + r);
            for (0..config.rope_dims / 2) |i| {
                const inv = std.math.pow(f64, self.hyper.rope_base, -@as(f64, @floatFromInt(2 * i)) / config.rope_dims);
                words[rope + r * 64 + i] = @bitCast(@as(f32, @floatCast(@cos(pos * inv))));
                words[rope + r * 64 + 32 + i] = @bitCast(@as(f32, @floatCast(@sin(pos * inv))));
            }
        }
        if (self.act_layout.mtp) |M| {
            // Batched catch-up (recorded in the prefill command): pending h in, the chunk's
            // last final-norm row out.
            const sp = layout.io.spec(self.rows);
            words[sp.src] = self.state_layout.hp.? | state_source;
            words[sp.src + 2] = M.hrows + @as(u32, @intCast(chunk.len)) * config.hidden;
        }
        try commands.run(self.options.timeout_ns);
        self.position += @intCast(chunk.len);
        if (self.options.mtp) self.mtp_pending = self.hpPending();
        const after = try self.io.mapped();
        const floats: []align(1) const f32 = std.mem.bytesAsSlice(f32, after[layout.io.logits * 4 ..][0 .. config.vocab * 4]);
        return @alignCast(floats);
    }

    /// Run a command recorded by `record` (the default step or a capture variant).
    pub fn run(self: *Model, commands: *gpu.Commands, token: u32) Error![]const f32 {
        if (self.pending_verify != 0) return error.VerifyPending;
        if (token >= config.vocab) return error.InvalidToken;
        if (self.position >= self.state_layout.context) return error.ContextFull;
        try self.ensureMapped(self.slot, @as(u64, self.position) + 1);
        const bytes = try self.io.mapped();
        const words = std.mem.bytesAsSlice(u32, bytes);
        words[layout.io.token] = token;
        words[layout.io.position] = self.position;
        for (0..config.rope_dims / 2) |i| {
            // FP64 angle and trig, rounded once to FP32.
            const inv = std.math.pow(f64, self.hyper.rope_base, -@as(f64, @floatFromInt(2 * i)) / config.rope_dims);
            const angle = @as(f64, @floatFromInt(self.position)) * inv;
            words[layout.io.cos + i] = @bitCast(@as(f32, @floatCast(@cos(angle))));
            words[layout.io.sin + i] = @bitCast(@as(f32, @floatCast(@sin(angle))));
        }
        try commands.run(self.options.timeout_ns);
        self.position += 1;
        const after = try self.io.mapped();
        const floats: []align(1) const f32 = std.mem.bytesAsSlice(f32, after[layout.io.logits * 4 ..][0 .. config.vocab * 4]);
        return @alignCast(floats);
    }

    const Rec = struct {
        model: *Model,
        c: *gpu.Commands,
        capture: ?*Capture,
        probe: ?Probe,
        plan: usize = 0,
        rows: u32 = 1,

        fn mark(r: Rec, phase: Phase, layer: i32) Error!void {
            const probe = r.probe orelse return;
            try probe.mark(probe.context, r.c, phase, layer);
        }
        fn bar(r: Rec) Error!void {
            try r.c.barrier(.compute, .compute);
        }
        fn kernel(r: Rec, id: KernelId, push: anytype, groups: [3]u32) Error!void {
            try r.c.dispatch(&r.model.kernels[@intFromEnum(id)], std.mem.asBytes(&push), groups);
        }
        /// A KV-cache kernel (`kv_kernel_ids`) for attention cache `cache`: the instance
        /// bound to that cache's KV buffer.
        fn attn(r: Rec, id: KernelId, cache: u32, push: anytype, groups: [3]u32) Error!void {
            try r.c.dispatch(r.model.kvKernel(id, cache), std.mem.asBytes(&push), groups);
        }
        fn hkernel(r: Rec, id: HKernel, push: anytype, groups: [3]u32) Error!void {
            try r.c.dispatch(&r.model.h_kernels[@intFromEnum(id)], std.mem.asBytes(&push), groups);
        }
        /// f16 copy's base (halves); only valid when a consumer has `x16` set.
        fn x16(r: Rec) u32 {
            return 2 * r.model.act_layout.x16.?;
        }
        /// The prefill norm into A.h; also writes its f16 copy when `need16`.
        fn normH(r: Rec, need16: bool, push: NormPush) Error!void {
            if (need16) return r.hkernel(.norm, NormHPush{ .base = push, .y16 = r.x16(), .stride16 = config.hidden }, .{ r.rows, 1, 1 });
            try r.kernel(.norm, push, .{ r.rows, 1, 1 });
        }
        fn proj(r: Rec, p: Proj) Error!void {
            try r.model.pipelines[p.pipe].record(r.c, p.geo);
        }
        /// Multi-row verify projection of `r.rows` rows.
        fn rproj(r: Rec, p: RProj) Error!void {
            try r.model.rpipes[p.pipe].record(r.c, p.geo, r.rows);
        }
        /// Batched decode projection of `r.rows` rows: balanced groups of at most
        /// `matvec.max_rows` rows at row offsets (each row's bits do not depend on the group).
        fn bproj(r: Rec, p: RProj) Error!void {
            var groups = RowGroups.init(r.rows, matvec.max_rows);
            while (groups.next()) |g| try r.model.rpipes[p.pipe].recordAt(r.c, p.geo, g.first, g.count);
        }
        /// Batched decode projection: the f16 WMMA kernel when the mode has one (every row of
        /// the span in one dispatch; rows are read up to the io count), else FP32 `bproj`.
        fn dproj(r: Rec, d: ?GProj, p: RProj) Error!void {
            const g = d orelse return r.bproj(p);
            try r.c.dispatch(&r.model.gemm_pipes[g.pipe], std.mem.asBytes(&g.push), g.groups);
        }
        /// The split-K reductions of `ds` (after their dispatches; a barrier first).
        fn dreduce(r: Rec, ds: []const ?GProj) Error!void {
            var any = false;
            for (ds) |d| any = any or (if (d) |g| g.reduce != null else false);
            if (!any) return;
            try r.bar();
            for (ds) |d| if (d) |g| if (g.reduce) |rp| try r.kernel(.reduce, rp, .{ reduceGroups(rp, r.rows), 1, 1 });
        }
        fn gphase(r: Rec, projs: []const *const GProj) Error!void {
            var any = false;
            for (projs) |g| {
                try r.c.dispatch(&r.model.gemm_pipes[g.pipe], std.mem.asBytes(&g.push), g.groups);
                any = any or g.reduce != null;
            }
            if (!any) return;
            try r.bar();
            for (projs) |g| if (g.reduce) |rp| try r.kernel(.reduce, rp, .{ reduceGroups(rp, r.rows), 1, 1 });
        }
        /// Prefill capture: `rows` rows of `width` words at `stride` (last_only: one row).
        fn snapRows(r: Rec, name: []const u8, layer: i32, word: u32, width: u32, stride: u32, last_only: bool) Error!void {
            const cap = r.capture orelse return;
            if (!cap.wants(name)) return;
            const rows: u32 = if (last_only) 1 else r.rows;
            const words = @as(u64, rows - 1) * stride + width;
            if (cap.count == cap.entries.len or (cap.used_words + words) * 4 > cap.buffer.size) return error.CaptureFull;
            try r.c.barrier(.compute, .transfer);
            try r.c.copy(&r.model.act, @as(u64, word) * 4, cap.buffer, cap.used_words * 4, words * 4);
            try r.c.barrier(.transfer, .compute);
            var e: Capture.Entry = .{ .layer = layer, .offset_words = cap.used_words, .words = width, .rows = rows, .stride = stride, .last_only = last_only };
            @memcpy(e.name[0..name.len], name);
            e.name_len = @intCast(name.len);
            cap.entries[cap.count] = e;
            cap.count += 1;
            cap.used_words += words;
        }
        fn snap(r: Rec, name: []const u8, layer: i32, word: u32, words: u32) Error!void {
            const cap = r.capture orelse return;
            if (!cap.wants(name)) return;
            if (cap.count == cap.entries.len or (cap.used_words + words) * 4 > cap.buffer.size) return error.CaptureFull;
            try r.c.barrier(.compute, .transfer);
            try r.c.copy(&r.model.act, @as(u64, word) * 4, cap.buffer, cap.used_words * 4, @as(u64, words) * 4);
            try r.c.barrier(.transfer, .compute);
            var e: Capture.Entry = .{ .layer = layer, .offset_words = cap.used_words, .words = words };
            @memcpy(e.name[0..name.len], name);
            e.name_len = @intCast(name.len);
            cap.entries[cap.count] = e;
            cap.count += 1;
            cap.used_words += words;
        }
    };

    /// Record one prefill chunk for plan `plan` (row count read from io at run time,
    /// at most `plans[plan].rows`) into `commands`.
    pub fn recordPrefill(self: *Model, commands: *gpu.Commands, plan: usize, hooks: Hooks) Error!void {
        if (self.rows == 0) return error.PrefillDisabled;
        if (plan >= self.plan_count) return error.InvalidPlan;
        const B = self.plans[plan].rows;
        const r: Rec = .{ .model = self, .c = commands, .capture = hooks.capture, .probe = hooks.probe, .plan = plan, .rows = B };
        const A = self.act_layout;
        const S = self.state_layout;
        const eps = self.hyper.eps;
        const H: u32 = config.hidden;
        const ctx = S.context;
        try commands.barrier(.transfer, .compute);
        try commands.barrier(.host, .compute);
        try r.bar();
        try r.kernel(.embed_b, EmbedBPush{ .tensor = self.embed_offset, .row_bytes = config.hidden / 32 * 18, .out = A.x, .columns = H, .tokens = layout.io.tokens, .out_rs = H }, .{ H / 256, B, 1 });
        try r.mark(.embed, -1);
        try r.snapRows("model.input_embed", -1, A.x, H, H, false);
        for (0..config.layers) |index| {
            const il: u32 = @intCast(index);
            const L = &self.layers[il];
            const li: i32 = @intCast(il);
            try r.bar();
            const in16 = if (config.isAttention(il)) any16(&.{ &L.g_q[plan], &L.g_k[plan], &L.g_v[plan] }) else any16(&.{ &L.g_qkv[plan], &L.g_z[plan], &L.g_alpha[plan], &L.g_beta[plan] });
            if (il == 0) {
                try r.normH(in16, .{ .x = A.x, .a = 0, .sum = 0, .y = A.h, .w = L.attn_norm, .stride = H, .flags = norm_rows_io, .eps = eps });
                try r.mark(.norm, li);
            } else {
                try r.normH(in16, .{ .x = A.r, .a = A.f, .sum = A.x, .y = A.h, .w = L.attn_norm, .stride = H, .flags = norm_add | norm_rows_io, .eps = eps });
                try r.mark(.norm, li);
                try r.snapRows("l_out", li - 1, A.x, H, H, false);
            }
            try r.snapRows("attn_norm", li, A.h, H, H, false);
            try r.bar();
            if (config.isAttention(il)) {
                const ai = config.attentionIndex(il);
                try r.gphase(&.{ &L.g_q[plan], &L.g_k[plan], &L.g_v[plan] });
                try r.mark(.attn_in, li);
                try r.snapRows("Qcur_full", li, A.qf, 12288, 12288, false);
                try r.snapRows("Kcur", li, A.kc, 1024, 1024, false);
                try r.snapRows("Vcur", li, A.vc, 1024, 1024, false);
                try r.bar();
                try r.attn(.qk_b, ai, QkBPush{ .slots = self.slot_io, .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = ctx, .rope = layout.io.rope(self.rows), .eps = eps, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ config.heads + config.kv_heads, B, 1 });
                try r.mark(.qkprep, li);
                try r.snapRows("Qcur_normed", li, A.qn, 6144, 6144, false);
                try r.snapRows("Kcur_normed", li, A.kn, 1024, 1024, false);
                try r.snapRows("Qcur", li, A.qr, 6144, 6144, false);
                try r.snapRows("Kcur_roped", li, A.kr, 1024, 1024, false);
                try r.bar();
                try r.attn(.flash, ai, FlashPush{ .slots = self.slot_io, .qr = A.qr, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .out = A.pregate, .ctx = ctx, .scale = 1.0 / 16.0, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ std.math.divCeil(u32, B, flash_rows) catch unreachable, config.heads / flash_groups, 1 });
                try r.mark(.attention, li);
                try r.bar();
                const gate_push: GatePush = .{ .pregate = A.pregate, .qf = A.qf, .gates = A.gates, .gated = A.gated };
                const gate_groups: [3]u32 = .{ @min(B * 6144 / 256, 4096), 1, 1 };
                if (L.g_out[plan].x16) try r.hkernel(.gate, GateHPush{ .base = gate_push, .gated16 = r.x16() }, gate_groups) else try r.kernel(.gate, gate_push, gate_groups);
                try r.mark(.gate, li);
                try r.snapRows("attn_pregate", li, A.pregate, 6144, 6144, false);
                try r.snapRows("gate_sigmoid", li, A.gates, 6144, 6144, false);
                try r.snapRows("attn_gated", li, A.gated, 6144, 6144, false);
                try r.bar();
                try r.gphase(&.{&L.g_out[plan]});
                try r.mark(.attn_out, li);
                try r.snapRows("attn_output", li, A.a, H, H, false);
            } else {
                const lin = config.linearIndex(il);
                try r.gphase(&.{ &L.g_qkv[plan], &L.g_z[plan], &L.g_alpha[plan], &L.g_beta[plan] });
                try r.mark(.lin_in, li);
                try r.snapRows("linear_attn_qkv_mixed", li, A.mixed, config.conv_channels, config.conv_channels, false);
                try r.snapRows("z", li, A.z, config.value_dim, config.value_dim, false);
                try r.snapRows("alpha", li, A.alpha, config.v_heads, config.v_heads, false);
                try r.snapRows("beta", li, A.beta_raw, config.v_heads, config.v_heads, false);
                try r.bar();
                try r.kernel(.conv_b, ConvPush{ .slots = self.slot_io, .mixed = A.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = A.conv_out, .w = L.conv_w, .conv = S.convLayer(lin) }, .{ config.conv_channels / 128, 1, 1 });
                try r.mark(.conv, li);
                try r.snapRows("conv_output_raw", li, A.conv_raw, config.conv_channels, config.conv_channels, false);
                try r.snapRows("conv_output_silu", li, A.conv_silu, config.conv_channels, config.conv_channels, false);
                try r.snapRows("q_conv_predelta", li, A.conv_out, config.key_dim, config.conv_channels, false);
                try r.snapRows("k_conv_predelta", li, A.conv_out + config.key_dim, config.key_dim, config.conv_channels, false);
                try r.bar();
                try r.kernel(self.deltaBKernel(), DeltaPush{ .slots = self.slot_io, .qk = A.conv_out, .v = A.conv_out + 2 * config.key_dim, .z = A.z, .beta_raw = A.beta_raw, .alpha = A.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(lin), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = eps, .state_out = S.ssmLayer(lin) }, .{ config.v_heads, 1, 1 });
                try r.bar();
                try r.kernel(.gnorm_b, GNormPush{ .o = A.o, .y = A.fo, .z = A.z, .norm_w = L.ssm_norm, .eps = eps }, .{ config.v_heads, r.rows, 1 });
                try r.mark(.delta, li);
                try r.snapRows("beta_sigmoid", li, A.beta, config.v_heads, config.v_heads, false);
                try r.snapRows("a_softplus", li, A.softplus, config.v_heads, config.v_heads, false);
                try r.snapRows("gate", li, A.gate, config.v_heads, config.v_heads, false);
                try r.snapRows("attn_output", li, A.o, config.value_dim, config.value_dim, false);
                try r.snapRows("final_output", li, A.fo, config.value_dim, config.value_dim, false);
                try r.bar();
                try r.gphase(&.{&L.g_ssm_out[plan]});
                try r.mark(.lin_out, li);
                try r.snapRows("linear_attn_out", li, A.a, H, H, false);
            }
            try r.bar();
            try r.normH(any16(&.{ &L.g_gate[plan], &L.g_up[plan] }), .{ .x = A.x, .a = A.a, .sum = A.r, .y = A.h, .w = L.post_norm, .stride = H, .flags = norm_add | norm_rows_io, .eps = eps });
            try r.mark(.norm, li);
            try r.snapRows("attn_residual", li, A.r, H, H, false);
            try r.snapRows("attn_post_norm", li, A.h, H, H, false);
            try r.bar();
            try r.gphase(&.{ &L.g_gate[plan], &L.g_up[plan] });
            try r.mark(.ffn_in, li);
            try r.snapRows("ffn_gate", li, A.fg, config.ffn, config.ffn, false);
            try r.snapRows("ffn_up", li, A.fu, config.ffn, config.ffn, false);
            try r.bar();
            const sw_push: SwigluPush = .{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn, .rows_io = 1 };
            const sw_groups: [3]u32 = .{ @min(B * config.ffn / 256, 4096), 1, 1 };
            if (L.g_down[plan].x16) try r.hkernel(.swiglu, SwigluHPush{ .base = sw_push, .y16 = r.x16() }, sw_groups) else try r.kernel(.swiglu, sw_push, sw_groups);
            try r.mark(.swiglu, li);
            try r.snapRows("ffn_swiglu", li, A.sw, config.ffn, config.ffn, false);
            try r.bar();
            try r.gphase(&.{&L.g_down[plan]});
            try r.mark(.ffn_down, li);
            try r.snapRows("ffn_out", li, A.f, H, H, false);
        }
        try r.bar();
        if (A.mtp) |M| {
            // MTP catch-up input: the final norm of every row into hrows rows 1.. (the head
            // needs only the last).
            try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = M.hrows + H, .w = self.output_norm, .stride = H, .flags = norm_add | norm_rows_io, .eps = eps }, .{ B, 1, 1 });
            try r.bar();
        }
        try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.hn, .w = self.output_norm, .stride = H, .flags = norm_add | norm_rows_io | norm_last_only, .eps = eps }, .{ B, 1, 1 });
        try r.mark(.norm, -1);
        try r.snapRows("l_out", config.layers - 1, A.x, H, H, false);
        try r.snapRows("result_norm", -1, A.hn, H, H, true);
        try r.bar();
        try r.proj(self.output);
        try r.mark(.output, -1);
        if (A.mtp != null) try self.recordCatchupBatch(r, plan);
        try commands.barrier(.compute, .host);
    }

    /// Batched MTP prompt catch-up of a prefill chunk (docs/specs/speculative.md): MTP row
    /// i holds (h of row i - 1, token i); only what the MTP KV depends on runs (embedding,
    /// enorm/hnorm, eh_proj, attn_norm, K and V, q/k norm and RoPE into KV cache 16).
    /// Host-set io words: spec src[0] = the pending h (state), src[2] = hrows row count.
    fn recordCatchupBatch(self: *Model, r: Rec, plan: usize) Error!void {
        const A = self.act_layout;
        const M = A.mtp.?;
        const S = self.state_layout;
        const L = &self.mtp_layer;
        const B = self.plans[plan].rows;
        const H: u32 = config.hidden;
        const eps = self.hyper.eps;
        const sp = layout.io.spec(self.rows);
        try r.bar();
        // hrows row 0 = the pending h; the embeddings into cat's first halves.
        try r.kernel(.rowcopy, RowCopyPush{ .src = sp.src, .dst = M.hrows, .words = H, .flags = 0 }, .{ H / 256, 1, 1 });
        try r.kernel(.embed_b, EmbedBPush{ .tensor = self.embed_offset, .row_bytes = config.hidden / 32 * 18, .out = M.cat, .columns = H, .tokens = layout.io.tokens, .out_rs = 2 * H }, .{ H / 256, B, 1 });
        try r.bar();
        try r.kernel(.copy2d, Copy2dPush{ .src = M.hrows, .dst = M.cat + H, .width = H, .src_stride = H, .dst_stride = 2 * H }, .{ H / 256, B, 1 });
        try r.bar();
        // In place (the norm loads its row before writing it).
        try r.kernel(.norm, NormPush{ .x = M.cat, .a = 0, .sum = 0, .y = M.cat, .w = L.enorm, .stride = 2 * H, .flags = norm_rows_io, .eps = eps }, .{ B, 1, 1 });
        try r.kernel(.norm, NormPush{ .x = M.cat + H, .a = 0, .sum = 0, .y = M.cat + H, .w = L.hnorm, .stride = 2 * H, .flags = norm_rows_io, .eps = eps }, .{ B, 1, 1 });
        try r.bar();
        try r.gphase(&.{&L.g_eh[plan]});
        try r.bar();
        try r.kernel(.norm, NormPush{ .x = A.x, .a = 0, .sum = 0, .y = A.h, .w = L.attn_norm, .stride = H, .flags = norm_rows_io, .eps = eps }, .{ B, 1, 1 });
        try r.bar();
        try r.gphase(&.{ &L.g_k[plan], &L.g_v[plan] });
        try r.bar();
        // Query heads process stale qf rows (finite, unused); the KV heads write cache 16.
        const ai = layout.mtp_attention;
        try r.attn(.qk_b, ai, QkBPush{ .slots = self.slot_io, .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = S.context, .rope = layout.io.rope(self.rows), .eps = eps, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ config.heads + config.kv_heads, B, 1 });
        try r.bar();
        try r.kernel(.rowcopy, RowCopyPush{ .src = sp.src + 2, .dst = S.hp.?, .words = H, .flags = 1 }, .{ H / 256, 1, 1 });
        try r.bar();
    }

    /// Record one complete decode step into `commands` (which must be recording).
    pub fn record(self: *Model, commands: *gpu.Commands, hooks: Hooks) Error!void {
        const r: Rec = .{ .model = self, .c = commands, .capture = hooks.capture, .probe = hooks.probe };
        const A = self.act_layout;
        const S = self.state_layout;
        const eps = self.hyper.eps;
        const H: u32 = config.hidden;
        try commands.barrier(.transfer, .compute);
        try r.bar();
        try r.kernel(.embed, EmbedPush{ .tensor = self.embed_offset, .row_bytes = config.hidden / 32 * 18, .out = A.x, .columns = H }, .{ H / 256, 1, 1 });
        try r.mark(.embed, -1);
        try r.snap("model.input_embed", -1, A.x, H);
        for (0..config.layers) |index| {
            const il: u32 = @intCast(index);
            const L = &self.layers[il];
            try r.bar();
            if (il == 0) {
                try r.kernel(.norm, NormPush{ .x = A.x, .a = 0, .sum = 0, .y = A.h, .w = L.attn_norm, .stride = 0, .flags = 0, .eps = eps }, .{ 1, 1, 1 });
                try r.mark(.norm, @intCast(il));
            } else {
                try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.h, .w = L.attn_norm, .stride = 0, .flags = 1, .eps = eps }, .{ 1, 1, 1 });
                try r.mark(.norm, @intCast(il));
                try r.snap("l_out", @intCast(il - 1), A.x, H);
            }
            try r.snap("attn_norm", @intCast(il), A.h, H);
            try r.bar();
            if (config.isAttention(il)) {
                const ai = config.attentionIndex(il);
                try r.proj(L.q);
                try r.proj(L.k);
                try r.proj(L.v);
                try r.mark(.attn_in, @intCast(il));
                try r.snap("Qcur_full", @intCast(il), A.qf, 2 * config.heads * config.head_dim);
                try r.snap("Kcur", @intCast(il), A.kc, config.kv_heads * config.head_dim);
                try r.snap("Vcur", @intCast(il), A.vc, config.kv_heads * config.head_dim);
                try r.bar();
                try r.attn(.qkprep, ai, QkPush{ .slots = self.slot_io, .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = S.context, .eps = eps, .rope = layout.io.cos, .pos = layout.io.position, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ config.heads + config.kv_heads, 1, 1 });
                try r.mark(.qkprep, @intCast(il));
                try r.snap("Qcur_normed", @intCast(il), A.qn, config.heads * config.head_dim);
                try r.snap("Kcur_normed", @intCast(il), A.kn, config.kv_heads * config.head_dim);
                try r.snap("Qcur", @intCast(il), A.qr, config.heads * config.head_dim);
                try r.snap("Kcur_roped", @intCast(il), A.kr, config.kv_heads * config.head_dim);
                try r.bar();
                try r.attn(.attn_scores, ai, attention.ScoresPush{ .slots = self.slot_io, .qr = A.qr, .scores = A.scores, .amax = A.amax, .kcache = S.kcache(ai), .ctx = S.context, .chunks = A.chunks, .scale = 1.0 / 16.0, .pos = layout.io.position, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.scores, A.chunks, 1));
                try r.mark(.scores, @intCast(il));
                try r.bar();
                try r.kernel(.attn_gmax, attention.GmaxPush{ .slots = self.slot_io, .amax = A.amax, .gmax = A.gmax, .chunks = A.chunks, .pos = layout.io.position }, attention.groups(.gmax, A.chunks, 1));
                try r.bar();
                try r.attn(.attn_pv, ai, attention.PvPush{ .slots = self.slot_io, .scores = A.scores, .amax = A.amax, .apart = A.apart, .asum = A.asum, .vcache = S.vcache(ai), .ctx = S.context, .chunks = A.chunks, .pos = layout.io.position, .gmax = A.gmax, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.pv, A.chunks, 1));
                try r.mark(.pv, @intCast(il));
                try r.bar();
                try r.kernel(.attn_cblock, attention.BlockPush{ .slots = self.slot_io, .apart = A.apart, .asum = A.asum, .bpart = A.bpart, .bsum = A.bsum, .chunks = A.chunks, .blocks = A.blocks, .pos = layout.io.position }, attention.groups(.block, A.chunks, 1));
                try r.bar();
                try r.kernel(.attn_combine, attention.CombinePush{ .slots = self.slot_io, .bpart = A.bpart, .bsum = A.bsum, .qf = A.qf, .pregate = A.pregate, .gates = A.gates, .gated = A.gated, .blocks = A.blocks, .pos = layout.io.position }, attention.groups(.combine, A.chunks, 1));
                try r.mark(.attention, @intCast(il));
                try r.snap("attn_pregate", @intCast(il), A.pregate, config.heads * config.head_dim);
                try r.snap("gate_sigmoid", @intCast(il), A.gates, config.heads * config.head_dim);
                try r.snap("attn_gated", @intCast(il), A.gated, config.heads * config.head_dim);
                try r.bar();
                try r.proj(L.out);
                try r.mark(.attn_out, @intCast(il));
                try r.snap("attn_output", @intCast(il), A.a, H);
            } else {
                const li = config.linearIndex(il);
                try r.proj(L.qkv);
                try r.proj(L.z);
                try r.proj(L.alpha);
                try r.proj(L.beta);
                try r.mark(.lin_in, @intCast(il));
                try r.snap("linear_attn_qkv_mixed", @intCast(il), A.mixed, config.conv_channels);
                try r.snap("z", @intCast(il), A.z, config.value_dim);
                try r.snap("alpha", @intCast(il), A.alpha, config.v_heads);
                try r.snap("beta", @intCast(il), A.beta_raw, config.v_heads);
                try r.bar();
                try r.kernel(.conv, ConvDPush{ .slots = self.slot_io, .mixed = A.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = A.conv_out, .w = L.conv_w, .conv = S.convLayer(li) }, .{ config.conv_channels / 128, 1, 1 });
                try r.mark(.conv, @intCast(il));
                try r.snap("conv_output_raw", @intCast(il), A.conv_raw, config.conv_channels);
                try r.snap("conv_output_silu", @intCast(il), A.conv_silu, config.conv_channels);
                try r.snap("q_conv_predelta", @intCast(il), A.conv_out, config.key_dim);
                try r.snap("k_conv_predelta", @intCast(il), A.conv_out + config.key_dim, config.key_dim);
                try r.bar();
                try r.kernel(self.deltaKernel(), DeltaDPush{ .slots = self.slot_io, .qk = A.conv_out, .v = A.conv_out + 2 * config.key_dim, .z = A.z, .beta_raw = A.beta_raw, .alpha = A.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(li), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = eps, .state_out = S.ssmLayer(li) }, .{ config.v_heads, 1, 1 });
                try r.mark(.delta, @intCast(il));
                try r.snap("beta_sigmoid", @intCast(il), A.beta, config.v_heads);
                try r.snap("a_softplus", @intCast(il), A.softplus, config.v_heads);
                try r.snap("gate", @intCast(il), A.gate, config.v_heads);
                try r.snap("attn_output", @intCast(il), A.o, config.value_dim);
                try r.snap("final_output", @intCast(il), A.fo, config.value_dim);
                try r.bar();
                try r.proj(L.ssm_out);
                try r.mark(.lin_out, @intCast(il));
                try r.snap("linear_attn_out", @intCast(il), A.a, H);
            }
            try r.bar();
            try r.kernel(.norm, NormPush{ .x = A.x, .a = A.a, .sum = A.r, .y = A.h, .w = L.post_norm, .stride = 0, .flags = 1, .eps = eps }, .{ 1, 1, 1 });
            try r.mark(.norm, @intCast(il));
            try r.snap("attn_residual", @intCast(il), A.r, H);
            try r.snap("attn_post_norm", @intCast(il), A.h, H);
            try r.bar();
            if (L.gate_up) |gu| {
                // One dispatch: gate, up and swiglu (the same values as the three steps).
                try r.model.swiglu_pipes[gu.pipe].record(r.c, gu.geo);
                try r.mark(.ffn_in, @intCast(il));
                try r.snap("ffn_gate", @intCast(il), A.fg, config.ffn);
                try r.snap("ffn_up", @intCast(il), A.fu, config.ffn);
                try r.snap("ffn_swiglu", @intCast(il), A.sw, config.ffn);
            } else {
                try r.proj(L.gate);
                try r.proj(L.up);
                try r.mark(.ffn_in, @intCast(il));
                try r.snap("ffn_gate", @intCast(il), A.fg, config.ffn);
                try r.snap("ffn_up", @intCast(il), A.fu, config.ffn);
                try r.bar();
                try r.kernel(.swiglu, SwigluPush{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn }, .{ config.ffn / 128, 1, 1 });
                try r.mark(.swiglu, @intCast(il));
                try r.snap("ffn_swiglu", @intCast(il), A.sw, config.ffn);
            }
            try r.bar();
            try r.proj(L.down);
            try r.mark(.ffn_down, @intCast(il));
            try r.snap("ffn_out", @intCast(il), A.f, H);
        }
        try r.bar();
        try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.hn, .w = self.output_norm, .stride = 0, .flags = 1, .eps = eps }, .{ 1, 1, 1 });
        try r.mark(.norm, -1);
        try r.snap("l_out", config.layers - 1, A.x, H);
        try r.snap("result_norm", -1, A.hn, H);
        try r.bar();
        try r.proj(self.output);
        try r.mark(.output, -1);
        try commands.barrier(.compute, .host);
    }

    /// Record the speculative verify of `n` rows (docs/specs/speculative.md): the decode
    /// step's operations with the same modules and per-row arithmetic, rows 0..n-1 of the
    /// activation regions, positions io position + r, tokens and RoPE rows from the io
    /// prefill regions. Linear layers keep their lin_in and conv outputs in their verify
    /// slot; conv and delta run without storing state. Logits go to `verify_logits`.
    fn recordVerify(self: *Model, commands: *gpu.Commands, n: u32) Error!void {
        const r: Rec = .{ .model = self, .c = commands, .capture = null, .probe = null, .rows = n };
        const A = self.act_layout;
        const S = self.state_layout;
        const eps = self.hyper.eps;
        const H: u32 = config.hidden;
        try commands.barrier(.transfer, .compute);
        try commands.barrier(.host, .compute);
        try r.bar();
        try r.kernel(.embed_b, EmbedBPush{ .tensor = self.embed_offset, .row_bytes = config.hidden / 32 * 18, .out = A.x, .columns = H, .tokens = layout.io.tokens, .out_rs = H }, .{ H / 256, n, 1 });
        for (0..config.layers) |index| {
            const il: u32 = @intCast(index);
            const L = &self.layers[il];
            try r.bar();
            if (il == 0) {
                try r.kernel(.norm, NormPush{ .x = A.x, .a = 0, .sum = 0, .y = A.h, .w = L.attn_norm, .stride = H, .flags = 0, .eps = eps }, .{ n, 1, 1 });
            } else {
                try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.h, .w = L.attn_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
            }
            try r.bar();
            if (config.isAttention(il)) {
                const ai = config.attentionIndex(il);
                try r.rproj(L.r_q);
                try r.rproj(L.r_k);
                try r.rproj(L.r_v);
                try r.bar();
                try r.attn(.qkprep, ai, QkPush{ .slots = self.slot_io, .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = S.context, .eps = eps, .rope = layout.io.rope(self.rows), .pos = layout.io.position, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ config.heads + config.kv_heads, n, 1 });
                try r.bar();
                try r.attn(.attn_scores, ai, attention.ScoresPush{ .slots = self.slot_io, .qr = A.qr, .scores = A.scores, .amax = A.amax, .kcache = S.kcache(ai), .ctx = S.context, .chunks = A.chunks, .scale = 1.0 / 16.0, .pos = layout.io.position, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.scores, A.chunks, n));
                try r.bar();
                try r.kernel(.attn_gmax, attention.GmaxPush{ .slots = self.slot_io, .amax = A.amax, .gmax = A.gmax, .chunks = A.chunks, .pos = layout.io.position }, attention.groups(.gmax, A.chunks, n));
                try r.bar();
                try r.attn(.attn_pv, ai, attention.PvPush{ .slots = self.slot_io, .scores = A.scores, .amax = A.amax, .apart = A.apart, .asum = A.asum, .vcache = S.vcache(ai), .ctx = S.context, .chunks = A.chunks, .pos = layout.io.position, .gmax = A.gmax, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.pv, A.chunks, n));
                try r.bar();
                try r.kernel(.attn_cblock, attention.BlockPush{ .slots = self.slot_io, .apart = A.apart, .asum = A.asum, .bpart = A.bpart, .bsum = A.bsum, .chunks = A.chunks, .blocks = A.blocks, .pos = layout.io.position }, attention.groups(.block, A.chunks, n));
                try r.bar();
                try r.kernel(.attn_combine, attention.CombinePush{ .slots = self.slot_io, .bpart = A.bpart, .bsum = A.bsum, .qf = A.qf, .pregate = A.pregate, .gates = A.gates, .gated = A.gated, .blocks = A.blocks, .pos = layout.io.position }, attention.groups(.combine, A.chunks, n));
                try r.bar();
                try r.rproj(L.r_out);
            } else {
                const li = config.linearIndex(il);
                const slot = A.specSlot(li);
                try r.rproj(L.r_qkv);
                try r.rproj(L.r_z);
                try r.rproj(L.r_alpha);
                try r.rproj(L.r_beta);
                try r.bar();
                try r.kernel(.conv, ConvDPush{ .slots = self.slot_io, .mixed = slot.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = slot.conv_out, .w = L.conv_w, .conv = S.convLayer(li), .rows = n, .commit = 0 }, .{ config.conv_channels / 128, 1, 1 });
                try r.bar();
                try r.kernel(self.deltaKernel(), DeltaDPush{ .slots = self.slot_io, .qk = slot.conv_out, .v = slot.conv_out + 2 * config.key_dim, .z = slot.z, .beta_raw = slot.beta, .alpha = slot.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(li), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = eps, .rows = n, .commit = 0, .state_out = S.ssmLayer(li) }, .{ config.v_heads, 1, 1 });
                try r.bar();
                try r.rproj(L.r_ssm_out);
            }
            try r.bar();
            try r.kernel(.norm, NormPush{ .x = A.x, .a = A.a, .sum = A.r, .y = A.h, .w = L.post_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
            try r.bar();
            const fused: ?SwRProj = if (L.r_gate_up) |gu| (if (self.swiglu_rpipes[gu.pipe].supports(n)) gu else null) else null;
            if (fused) |gu| {
                // One dispatch: gate, up and swiglu for all n rows (the same values).
                try r.model.swiglu_rpipes[gu.pipe].record(r.c, gu.geo, n);
            } else {
                try r.rproj(L.r_gate);
                try r.rproj(L.r_up);
                try r.bar();
                try r.kernel(.swiglu, SwigluPush{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn * n }, .{ config.ffn * n / 128, 1, 1 });
            }
            try r.bar();
            try r.rproj(L.r_down);
        }
        try r.bar();
        try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.hn, .w = self.output_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
        try r.bar();
        try r.rproj(self.output_rows);
        try commands.barrier(.compute, .host);
    }

    /// Record the batched decode of `n` rows (docs/specs/concurrent.md, "18b.2 design"): the
    /// verify pass's operations with the same modules, where row r takes its token from io
    /// tokens[r], its RoPE row from io rope + 64 r and its position, page table and state
    /// from its batch-table entry; conv and delta run one row per workgroup and store the
    /// state. Logits go to rows 0..n-1 of `verify_logits`.
    fn recordBatch(self: *Model, commands: *gpu.Commands, n: u32) Error!void {
        const r: Rec = .{ .model = self, .c = commands, .capture = null, .probe = null, .rows = n };
        const A = self.act_layout;
        const S = self.state_layout;
        const eps = self.hyper.eps;
        const H: u32 = config.hidden;
        const rows = @max(self.rows, 1);
        const table = layout.io.batch(rows);
        const rs = layout.io.slot_entry;
        try commands.barrier(.transfer, .compute);
        try commands.barrier(.host, .compute);
        try r.bar();
        try r.kernel(.embed_b, EmbedBPush{ .tensor = self.embed_offset, .row_bytes = config.hidden / 32 * 18, .out = A.x, .columns = H, .tokens = layout.io.tokens, .out_rs = H }, .{ H / 256, n, 1 });
        for (0..config.layers) |index| {
            const il: u32 = @intCast(index);
            const L = &self.layers[il];
            try r.bar();
            if (il == 0) {
                try r.kernel(.norm, NormPush{ .x = A.x, .a = 0, .sum = 0, .y = A.h, .w = L.attn_norm, .stride = H, .flags = 0, .eps = eps }, .{ n, 1, 1 });
            } else {
                try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.h, .w = L.attn_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
            }
            try r.bar();
            if (config.isAttention(il)) {
                const ai = config.attentionIndex(il);
                try r.dproj(L.d_q, L.r_q);
                try r.bproj(L.r_k);
                try r.bproj(L.r_v);
                try r.dreduce(&.{L.d_q});
                try r.bar();
                try r.attn(.qkprep, ai, QkPush{ .slots = table, .slot_rs = rs, .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = S.context, .eps = eps, .rope = layout.io.rope(rows), .pos = layout.io.position, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ config.heads + config.kv_heads, n, 1 });
                try r.bar();
                try r.attn(.attn_scores, ai, attention.ScoresPush{ .slots = table, .slot_rs = rs, .qr = A.qr, .scores = A.scores, .amax = A.amax, .kcache = S.kcache(ai), .ctx = S.context, .chunks = A.chunks, .scale = 1.0 / 16.0, .pos = layout.io.position, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.scores, A.chunks, n));
                try r.bar();
                try r.kernel(.attn_gmax, attention.GmaxPush{ .slots = table, .slot_rs = rs, .amax = A.amax, .gmax = A.gmax, .chunks = A.chunks, .pos = layout.io.position }, attention.groups(.gmax, A.chunks, n));
                try r.bar();
                try r.attn(.attn_pv, ai, attention.PvPush{ .slots = table, .slot_rs = rs, .scores = A.scores, .amax = A.amax, .apart = A.apart, .asum = A.asum, .vcache = S.vcache(ai), .ctx = S.context, .chunks = A.chunks, .pos = layout.io.position, .gmax = A.gmax, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.pv, A.chunks, n));
                try r.bar();
                try r.kernel(.attn_cblock, attention.BlockPush{ .slots = table, .slot_rs = rs, .apart = A.apart, .asum = A.asum, .bpart = A.bpart, .bsum = A.bsum, .chunks = A.chunks, .blocks = A.blocks, .pos = layout.io.position }, attention.groups(.block, A.chunks, n));
                try r.bar();
                try r.kernel(.attn_combine, attention.CombinePush{ .slots = table, .slot_rs = rs, .bpart = A.bpart, .bsum = A.bsum, .qf = A.qf, .pregate = A.pregate, .gates = A.gates, .gated = A.gated, .blocks = A.blocks, .pos = layout.io.position }, attention.groups(.combine, A.chunks, n));
                try r.bar();
                try r.dproj(L.d_out, L.r_out);
                try r.dreduce(&.{L.d_out});
            } else {
                const li = config.linearIndex(il);
                try r.dproj(L.d_qkv, L.b_qkv);
                try r.dproj(L.d_z, L.b_z);
                try r.bproj(L.b_alpha);
                try r.bproj(L.b_beta);
                try r.dreduce(&.{ L.d_qkv, L.d_z });
                try r.bar();
                try r.kernel(.conv, ConvDPush{ .slots = table, .slot_rs = rs, .mixed = A.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = A.conv_out, .w = L.conv_w, .conv = S.convLayer(li), .rows = 1, .commit = 1 }, .{ config.conv_channels / 128, n, 1 });
                try r.bar();
                try r.kernel(self.deltaKernel(), DeltaDPush{ .slots = table, .slot_rs = rs, .qk = A.conv_out, .v = A.conv_out + 2 * config.key_dim, .z = A.z, .beta_raw = A.beta_raw, .alpha = A.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(li), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = eps, .rows = 1, .commit = 1, .state_out = S.ssmLayer(li) }, .{ config.v_heads, n, 1 });
                try r.bar();
                try r.dproj(L.d_ssm_out, L.r_ssm_out);
                try r.dreduce(&.{L.d_ssm_out});
            }
            try r.bar();
            try r.kernel(.norm, NormPush{ .x = A.x, .a = A.a, .sum = A.r, .y = A.h, .w = L.post_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
            try r.bar();
            if (L.d_gate != null and L.d_up != null) {
                try r.dproj(L.d_gate, L.r_gate);
                try r.dproj(L.d_up, L.r_up);
                try r.dreduce(&.{ L.d_gate, L.d_up });
                try r.bar();
                try r.kernel(.swiglu, SwigluPush{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn * n }, .{ config.ffn * n / 128, 1, 1 });
            } else if (L.r_gate_up) |gu| {
                // Gate, up and swiglu fused, in groups of the largest supported count.
                const pipe = &self.swiglu_rpipes[gu.pipe];
                var cap: u32 = @min(pipe.max_count, n);
                while (!pipe.supports(cap)) cap -= 1;
                var groups = RowGroups.init(n, cap);
                while (groups.next()) |g| try pipe.recordAt(commands, gu.geo, g.first, g.count);
            } else {
                try r.bproj(L.r_gate);
                try r.bproj(L.r_up);
                try r.bar();
                try r.kernel(.swiglu, SwigluPush{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn * n }, .{ config.ffn * n / 128, 1, 1 });
            }
            try r.bar();
            try r.dproj(L.d_down, L.r_down);
            try r.dreduce(&.{L.d_down});
        }
        try r.bar();
        try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.hn, .w = self.output_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
        try r.bar();
        try r.bproj(self.output_rows);
        try commands.barrier(.compute, .host);
    }

    /// Record one MTP pass of `n` rows (docs/specs/speculative.md, "MTP runtime design"):
    /// tokens io[tok + r], positions io[pos] + r, RoPE rows at io rope + 64 r. `first`:
    /// the h rows are copied into `mh` from the io row-copy sources (row 0 from src[0],
    /// rows 1.. from src[1]); otherwise row r's h is at the static word `hsrc + r H`.
    /// Output: the MTP's h' rows in `mo`.
    fn recordMtpPass(self: *Model, commands: *gpu.Commands, n: u32, first: bool, hsrc: u32, tok: u32, pos: u32, rope: u32) Error!void {
        const r: Rec = .{ .model = self, .c = commands, .capture = null, .probe = null, .rows = n };
        const A = self.act_layout;
        const M = A.mtp.?;
        const S = self.state_layout;
        const L = &self.mtp_layer;
        const eps = self.hyper.eps;
        const H: u32 = config.hidden;
        const sp = layout.io.spec(@max(self.rows, 1));
        var hin = hsrc;
        try r.bar();
        if (first) {
            try r.kernel(.rowcopy, RowCopyPush{ .src = sp.src, .dst = M.mh, .words = H, .flags = 0 }, .{ H / 256, 1, 1 });
            if (n > 1) try r.kernel(.rowcopy, RowCopyPush{ .src = sp.src + 1, .dst = M.mh + H, .words = (n - 1) * H, .flags = 0 }, .{ (n - 1) * H / 256, 1, 1 });
            hin = M.mh;
        }
        try r.kernel(.embed_b, EmbedBPush{ .tensor = self.embed_offset, .row_bytes = config.hidden / 32 * 18, .out = A.x, .columns = H, .tokens = tok, .out_rs = H }, .{ H / 256, n, 1 });
        try r.bar();
        // e = enorm(embed), g = hnorm(h), concatenated per row (embedding first).
        for (0..n) |row| {
            const o: u32 = @intCast(row);
            try r.kernel(.norm, NormPush{ .x = A.x + o * H, .a = 0, .sum = 0, .y = M.cat + o * 2 * H, .w = L.enorm, .stride = 0, .flags = 0, .eps = eps }, .{ 1, 1, 1 });
            try r.kernel(.norm, NormPush{ .x = hin + o * H, .a = 0, .sum = 0, .y = M.cat + o * 2 * H + H, .w = L.hnorm, .stride = 0, .flags = 0, .eps = eps }, .{ 1, 1, 1 });
        }
        try r.bar();
        try r.rproj(L.eh);
        try r.bar();
        // The trunk's gated attention layer with the MTP's weights and KV cache.
        try r.kernel(.norm, NormPush{ .x = A.x, .a = 0, .sum = 0, .y = A.h, .w = L.attn_norm, .stride = H, .flags = 0, .eps = eps }, .{ n, 1, 1 });
        try r.bar();
        try r.rproj(L.q);
        try r.rproj(L.k);
        try r.rproj(L.v);
        try r.bar();
        const ai = layout.mtp_attention;
        try r.attn(.qkprep, ai, QkPush{ .slots = self.slot_io, .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = S.context, .eps = eps, .rope = rope, .pos = pos, .ptab = A.ptab, .pstride = S.pstride(ai) }, .{ config.heads + config.kv_heads, n, 1 });
        try r.bar();
        try r.attn(.attn_scores, ai, attention.ScoresPush{ .slots = self.slot_io, .qr = A.qr, .scores = A.scores, .amax = A.amax, .kcache = S.kcache(ai), .ctx = S.context, .chunks = A.chunks, .scale = 1.0 / 16.0, .pos = pos, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.scores, A.chunks, n));
        try r.bar();
        try r.kernel(.attn_gmax, attention.GmaxPush{ .slots = self.slot_io, .amax = A.amax, .gmax = A.gmax, .chunks = A.chunks, .pos = pos }, attention.groups(.gmax, A.chunks, n));
        try r.bar();
        try r.attn(.attn_pv, ai, attention.PvPush{ .slots = self.slot_io, .scores = A.scores, .amax = A.amax, .apart = A.apart, .asum = A.asum, .vcache = S.vcache(ai), .ctx = S.context, .chunks = A.chunks, .pos = pos, .gmax = A.gmax, .ptab = A.ptab, .pstride = S.pstride(ai) }, attention.groups(.pv, A.chunks, n));
        try r.bar();
        try r.kernel(.attn_cblock, attention.BlockPush{ .slots = self.slot_io, .apart = A.apart, .asum = A.asum, .bpart = A.bpart, .bsum = A.bsum, .chunks = A.chunks, .blocks = A.blocks, .pos = pos }, attention.groups(.block, A.chunks, n));
        try r.bar();
        try r.kernel(.attn_combine, attention.CombinePush{ .slots = self.slot_io, .bpart = A.bpart, .bsum = A.bsum, .qf = A.qf, .pregate = A.pregate, .gates = A.gates, .gated = A.gated, .blocks = A.blocks, .pos = pos }, attention.groups(.combine, A.chunks, n));
        try r.bar();
        try r.rproj(L.out);
        try r.bar();
        try r.kernel(.norm, NormPush{ .x = A.x, .a = A.a, .sum = A.r, .y = A.h, .w = L.post_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
        try r.bar();
        try r.rproj(L.gate);
        try r.rproj(L.up);
        try r.bar();
        try r.kernel(.swiglu, SwigluPush{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn * n }, .{ config.ffn * n / 128, 1, 1 });
        try r.bar();
        try r.rproj(L.down);
        try r.bar();
        try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = M.mo, .w = L.head_norm, .stride = H, .flags = norm_add, .eps = eps }, .{ n, 1, 1 });
        try r.bar();
    }

    /// Draft head of a pass of `n` rows: logits of `mo` row n-1, argmax into io[out].
    fn recordDraftHead(self: *Model, commands: *gpu.Commands, n: u32, out: u32) Error!void {
        const r: Rec = .{ .model = self, .c = commands, .capture = null, .probe = null };
        const M = self.act_layout.mtp.?;
        try r.proj(self.mtp_head[n - 1]);
        try r.bar();
        try r.kernel(.argmax_a, ArgmaxAPush{ .x = M.logits, .n = self.draftVocab(), .part = M.part }, .{ layout.argmax_groups, 1, 1 });
        try r.bar();
        try r.kernel(.argmax_b, ArgmaxBPush{ .part = M.part, .out = out }, .{ 1, 1, 1 });
        try r.bar();
    }

    /// Record a draft of `k` tokens after a first pass of `m` rows: pass, head; then k - 1
    /// chained one-row passes (h from the previous pass's last `mo` row, token from the
    /// previous draft), each with a head.
    fn recordDraft(self: *Model, commands: *gpu.Commands, m: u32, k: u32) Error!void {
        const sp = layout.io.spec(@max(self.rows, 1));
        const M = self.act_layout.mtp.?;
        try commands.barrier(.transfer, .compute);
        try commands.barrier(.host, .compute);
        try self.recordMtpPass(commands, m, true, 0, sp.tok, sp.pos, sp.rope);
        try self.recordDraftHead(commands, m, sp.draft);
        for (1..k) |j| {
            const row = m - 1 + @as(u32, @intCast(j)); // MTP row index of this chained pass
            const hsrc = M.mo + (if (j == 1) m - 1 else 0) * config.hidden;
            try self.recordMtpPass(commands, 1, false, hsrc, sp.draft + @as(u32, @intCast(j)) - 1, sp.pos + row, sp.rope + 64 * row);
            try self.recordDraftHead(commands, 1, sp.draft + @as(u32, @intCast(j)));
        }
        try commands.barrier(.compute, .host);
    }

    /// Record a catch-up pass of `n` rows (no head).
    fn recordCatchup(self: *Model, commands: *gpu.Commands, n: u32) Error!void {
        const sp = layout.io.spec(@max(self.rows, 1));
        try commands.barrier(.transfer, .compute);
        try commands.barrier(.host, .compute);
        try self.recordMtpPass(commands, n, true, 0, sp.tok, sp.pos, sp.rope);
        try commands.barrier(.compute, .host);
    }

    /// Record the commit of `m` verified rows: every linear layer's conv, then (after one
    /// barrier) every delta, re-run over the verify slots' first m rows with `commit`, which
    /// stores the state after row m - 1 (their outputs are rewritten with equal values).
    fn recordCommit(self: *Model, commands: *gpu.Commands, m: u32) Error!void {
        const r: Rec = .{ .model = self, .c = commands, .capture = null, .probe = null, .rows = m };
        const A = self.act_layout;
        const S = self.state_layout;
        try commands.barrier(.host, .compute);
        try r.bar();
        for (0..config.layers) |index| {
            const il: u32 = @intCast(index);
            if (config.isAttention(il)) continue;
            const L = &self.layers[il];
            const li = config.linearIndex(il);
            const slot = A.specSlot(li);
            try r.kernel(.conv, ConvDPush{ .slots = self.slot_io, .mixed = slot.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = slot.conv_out, .w = L.conv_w, .conv = S.convLayer(li), .rows = m, .commit = 1 }, .{ config.conv_channels / 128, 1, 1 });
        }
        try r.bar();
        for (0..config.layers) |index| {
            const il: u32 = @intCast(index);
            if (config.isAttention(il)) continue;
            const L = &self.layers[il];
            const li = config.linearIndex(il);
            const slot = A.specSlot(li);
            try r.kernel(self.deltaKernel(), DeltaDPush{ .slots = self.slot_io, .qk = slot.conv_out, .v = slot.conv_out + 2 * config.key_dim, .z = slot.z, .beta_raw = slot.beta, .alpha = slot.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(li), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = self.hyper.eps, .rows = m, .commit = 1, .state_out = S.ssmLayer(li) }, .{ config.v_heads, 1, 1 });
        }
        try r.bar();
    }
};
