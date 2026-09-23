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
    /// Weight bank and state arena upper bound (must stay below 4 GiB for uint32 addressing).
    bank_capacity: u64 = 0xf000_0000,
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
};
pub const Error = gpu.Error || matvec.Error || config.Error || layout.Error || gemm.Error || error{ InvalidToken, ContextFull, TooManyPipelines, InvalidTensorBytes, CaptureFull, PrefillDisabled, ProbeFailed, InvalidPlan, UnsupportedDevice, InvalidContext, InvalidSnapshot, InsufficientVram };

/// Bytes of one recurrent-state snapshot: every DeltaNet state and convolution history.
/// The attention KV is not copied; it stays valid in the state arena below `position`.
pub const snapshot_bytes: u64 = (@as(u64, layout.State.ssm_words) + layout.State.conv_words) * 4;
pub const max_snapshots = 64;
/// Device-local memory beyond the model's buffers that `init` requires to be free:
/// driver objects (pipelines, descriptors, command buffers) and allocation padding.
pub const vram_headroom: u64 = 256 * 1024 * 1024;

/// Recorded work groups, in execution order, reported to a `Probe`.
pub const Phase = enum { embed, norm, attn_in, qkprep, scores, softmax, pv, attention, gate, attn_out, lin_in, conv, delta, lin_out, ffn_in, swiglu, ffn_down, output };
/// Instrumentation hook called while recording, after each phase's dispatches and before
/// the following barrier (e.g. to write a GPU timestamp). Not called during execution.
pub const Probe = struct {
    context: *anyopaque,
    mark: *const fn (context: *anyopaque, commands: *gpu.Commands, phase: Phase, layer: i32) error{ProbeFailed}!void,
};
/// Optional recording extras; the production commands use none.
pub const Hooks = struct { capture: ?*Capture = null, probe: ?Probe = null };

const KernelId = enum { embed, norm, qkprep, conv, delta, swiglu, zero, reduce, embed_b, qk_b, softmax, gate, conv_b, delta_b, attn_scores, attn_pv, attn_combine, gnorm_b };
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
const QkPush = extern struct { qf: u32, kc: u32, vc: u32, qn: u32, qr: u32, kn: u32, kr: u32, qw: u32, kw: u32, kcache: u32, vcache: u32, ctx: u32, eps: f32 };
const ConvPush = extern struct { mixed: u32, raw: u32, silu: u32, out: u32, w: u32, conv: u32 };
const DeltaPush = extern struct { qk: u32, v: u32, z: u32, beta_raw: u32, alpha: u32, beta_out: u32, softplus_out: u32, g_out: u32, o: u32, y: u32, ssm: u32, a_w: u32, dt_w: u32, norm_w: u32, eps: f32 };
const SwigluPush = extern struct { g: u32, u: u32, y: u32, n: u32, rows_io: u32 = 0 };
const ZeroPush = extern struct { first: u32, count: u32 };
const EmbedBPush = extern struct { tensor: u32, row_bytes: u32, out: u32, columns: u32, tokens: u32, out_rs: u32 };
const QkBPush = extern struct { qf: u32, kc: u32, vc: u32, qn: u32, qr: u32, kn: u32, kr: u32, qw: u32, kw: u32, kcache: u32, vcache: u32, ctx: u32, rope: u32, eps: f32 };
const SoftmaxPush = extern struct { scores: u32, ctx: u32, head_stride: u32, scale: f32 };
const GatePush = extern struct { pregate: u32, qf: u32, gates: u32, gated: u32 };
/// Gated RMS norm of the batched DeltaNet output (model.comp K_GNORMB).
const GNormPush = extern struct { o: u32, y: u32, z: u32, norm_w: u32, eps: f32 };
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
const push_sizes = [kernel_count]u32{ @sizeOf(EmbedPush), @sizeOf(NormPush), @sizeOf(QkPush), @sizeOf(ConvPush), @sizeOf(DeltaPush), @sizeOf(SwigluPush), @sizeOf(ZeroPush), @sizeOf(gemm.ReducePush), @sizeOf(EmbedBPush), @sizeOf(QkBPush), @sizeOf(SoftmaxPush), @sizeOf(GatePush), @sizeOf(ConvPush), @sizeOf(DeltaPush), attention.pushBytes(.scores), attention.pushBytes(.pv), attention.pushBytes(.combine), @sizeOf(GNormPush) };
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

fn module(id: KernelId) []align(4) const u8 {
    const M = struct {
        const embed align(4) = @embedFile("shaders/embed.spv").*;
        const norm align(4) = @embedFile("shaders/norm.spv").*;
        const qkprep align(4) = @embedFile("shaders/qkprep.spv").*;
        const conv align(4) = @embedFile("shaders/conv.spv").*;
        const delta align(4) = @embedFile("shaders/delta.spv").*;
        const swiglu align(4) = @embedFile("shaders/swiglu.spv").*;
        const zero align(4) = @embedFile("shaders/zero.spv").*;
        const reduce align(4) = @embedFile("shaders/reduce.spv").*;
        const embed_b align(4) = @embedFile("shaders/embed_b.spv").*;
        const qk_b align(4) = @embedFile("shaders/qk_b.spv").*;
        const softmax align(4) = @embedFile("shaders/softmax.spv").*;
        const gate align(4) = @embedFile("shaders/gate.spv").*;
        const conv_b align(4) = @embedFile("shaders/conv_b.spv").*;
        const delta_b align(4) = @embedFile("shaders/delta_b.spv").*;
        const gnorm_b align(4) = @embedFile("shaders/gnorm_b.spv").*;
    };
    return switch (id) {
        .embed => &M.embed,
        .norm => &M.norm,
        .qkprep => &M.qkprep,
        .conv => &M.conv,
        .delta => &M.delta,
        .swiglu => &M.swiglu,
        .zero => &M.zero,
        .reduce => &M.reduce,
        .embed_b => &M.embed_b,
        .qk_b => &M.qk_b,
        .softmax => &M.softmax,
        .gate => &M.gate,
        .conv_b => &M.conv_b,
        .delta_b => &M.delta_b,
        .attn_scores => attention.module(.scores),
        .attn_pv => attention.module(.pv),
        .attn_combine => attention.module(.combine),
        .gnorm_b => &M.gnorm_b,
    };
}

const Proj = struct { pipe: u8 = 0, geo: matvec.Geometry = undefined };
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
    kernels: [kernel_count]gpu.Kernel = undefined,
    /// f16-mode producers; created only when the arena has the `x16` region.
    h_kernels: [h_kernel_count]gpu.Kernel = undefined,
    live_h_kernels: u8 = 0,
    pipelines: [max_pipelines]matvec.Pipeline = undefined,
    pipe_keys: [max_pipelines]u32 = undefined,
    pipeline_count: u8 = 0,
    layers: [config.layers]Layer = @splat(.{}),
    output: Proj = .{},
    output_norm: u32 = 0,
    embed_offset: u32 = 0,
    step_commands: gpu.Commands = undefined,
    reset_commands: gpu.Commands = undefined,
    prefill_commands: [max_plans]gpu.Commands = undefined,
    plans: [max_plans]Plan = undefined,
    plan_count: u8 = 0,
    gemm_pipes: [max_gemm_pipelines]gpu.Kernel = undefined,
    gemm_keys: [max_gemm_pipelines]u32 = undefined,
    gemm_count: u8 = 0,
    attn_gemm: gpu.Kernel = undefined,
    attn_gemm_live: bool = false,
    part_slots: [split_slots]u32 = @splat(0),
    rows: u32 = 0, // prefill chunk capacity (0 = disabled)
    position: u32 = 0,
    /// Device-local bytes `init` needed (buffers + `vram_headroom`) and found free; kept
    /// after an `InsufficientVram` failure for the error message. `free` is null when the
    /// driver cannot report its budget (then no check is made).
    vram: struct { needed: u64 = 0, free: ?u64 = null } = .{},
    snapshot_store: gpu.Buffer = undefined,
    snapshot_slots: u32 = 0,
    copy_commands: gpu.Commands = undefined,
    copy_live: bool = false,
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
            if (options.context % 32 != 0) return error.InvalidContext;
            // f16: gemm_f16 needs subgroups of 64 (the default); gemm_f16x requires size 32
            // with full subgroups.
            if (options.prefill_precision == .f16 and (!device.cooperative_matrix or device.subgroup.size != 64 or !device.full_subgroups or
                device.subgroup_sizes.min > gemm.f16x_subgroup or device.subgroup_sizes.max < gemm.f16x_subgroup)) return error.UnsupportedDevice;
            self.plan_count = makePlans(self.rows, &self.plans);
        }
        self.state_layout = try layout.state(options.context, capacity);
        // Split-K partial slots, sized from the fixed tensor inventory before any allocation.
        var slot_words: [split_slots]u64 = @splat(0);
        for (self.plans[0..self.plan_count]) |plan| for (&config.tensors()) |*s| {
            if (s.role != .matrix or std.mem.eql(u8, config.specName(s), "output.weight")) continue;
            const chunk = gemm.splitChunk(@intCast(s.rows), plan.rows, @intCast(s.k), split_target, gemm.tileFor(@intCast(s.rows), plan.rows));
            // (f16-mode projections never split; sizing their FP32 split slot is harmless.)
            if (chunk == 0) continue;
            const words = @as(u64, gemm.splitCount(@intCast(s.k), chunk)) * plan.rows * s.rows;
            const slot = splitSlot(config.specName(s));
            slot_words[slot] = @max(slot_words[slot], std.mem.alignForward(u64, words, 64));
        };
        var part_words: u64 = 0;
        for (slot_words) |w| part_words += w;
        var x16 = false;
        if (options.prefill_precision == .f16) for (self.plans[0..self.plan_count]) |plan| {
            x16 = x16 or plan.rows % gemm.f16x_tile_n == 0;
        };
        self.act_layout = try layout.act(options.context, @max(self.rows, 1), part_words, x16);
        var at: u64 = self.act_layout.part;
        for (&self.part_slots, slot_words) |*slot, w| {
            slot.* = @intCast(at);
            at += w;
        }

        const specs = config.tensors();
        var tensors: [config.tensor_count]*const gguf.Tensor = undefined;
        var items: [config.tensor_count]layout.Item = undefined;
        for (&specs, &tensors, &items) |*s, *t, *item| {
            t.* = container.findTensor(config.specName(s)) orelse return error.MissingTensor;
            try config.check(s, t.*);
            const format: matvec.Format = if (s.role == .embedding) .q4_0 else try config.matrixFormat(t.*.kind);
            const shape: matvec.Shape = .{ .format = format, .columns = @intCast(s.k), .rows = @intCast(s.rows) };
            try matvec.validateWeights(shape, t.*.data);
            if (t.*.data.len % 4 != 0) return error.InvalidTensorBytes;
            item.* = .{ .role = s.role, .bytes = t.*.data.len };
        }
        var placements: [config.tensor_count]layout.Placement = undefined;
        const banks = try layout.place(&items, capacity, &placements);

        // Refuse to load when the device-local heap cannot hold every buffer, e.g. because
        // another process uses the GPU: oversubscribed VRAM is evicted to host memory by
        // the kernel driver and has ended in compute timeouts and a lost device.
        if (options.snapshots > max_snapshots) return error.InvalidSnapshot;
        var needed: u64 = self.act_layout.words * 4 + self.state_layout.words * 4 + snapshot_bytes * options.snapshots + vram_headroom;
        for (banks.bytes[0..banks.count]) |bytes| needed += bytes;
        self.vram.needed = needed;
        if (try device.memoryBudget()) |budget| {
            self.vram.free = budget.free();
            if (needed > budget.free()) return error.InsufficientVram;
        }

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
        if (options.snapshots > 0) {
            self.copy_commands = try gpu.Commands.init(device);
            self.copy_live = true;
            self.snapshot_store = try gpu.Buffer.init(device, snapshot_bytes * options.snapshots, .device);
            self.snapshot_slots = options.snapshots;
        }

        try self.upload(&tensors, &placements);

        for (0..kernel_count) |i| {
            self.kernels[i] = try gpu.Kernel.init(device, module(@enumFromInt(i)), &.{ &self.banks[0], &self.act, &self.state, &self.io }, push_sizes[i]);
            self.live_kernels += 1;
        }
        if (self.act_layout.x16 != null) for (0..h_kernel_count) |i| {
            self.h_kernels[i] = try gpu.Kernel.init(device, hModule(@enumFromInt(i)), &.{ &self.banks[0], &self.act, &self.state, &self.io }, h_push_sizes[i]);
            self.live_h_kernels += 1;
        };
        if (self.rows > 0) {
            self.attn_gemm = try gpu.Kernel.init(device, try gemm.module(.f32_m, .narrow), &.{ &self.state, &self.act, &self.io, &self.act }, @sizeOf(gemm.Push));
            self.attn_gemm_live = true;
        }

        // Resolve parameter offsets and projections.
        for (&specs, tensors, placements) |*s, t, place| {
            const name = config.specName(s);
            const word: u32 = @intCast(place.offset / 4);
            if (s.role == .embedding) {
                self.embed_offset = @intCast(place.offset);
                continue;
            }
            if (std.mem.eql(u8, name, "output_norm.weight")) {
                self.output_norm = word;
                continue;
            }
            if (std.mem.eql(u8, name, "output.weight")) {
                self.output = try self.projection(s, t, place, self.act_layout.hn, null);
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
        }

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
        const zero_words = layout.State.ssm_words + layout.State.conv_words;
        try self.reset_commands.dispatch(&self.kernels[@intFromEnum(KernelId.zero)], std.mem.asBytes(&ZeroPush{ .first = self.state_layout.ssm, .count = zero_words }), .{ 4096, 1, 1 });
        try self.reset_commands.barrier(.compute, .compute);
        try self.reset_commands.end();
        for (0..self.plan_count) |i| {
            self.prefill_commands[i] = try gpu.Commands.init(device);
            self.live_prefill += 1;
            try self.prefill_commands[i].begin();
            try self.recordPrefill(&self.prefill_commands[i], i, .{});
            try self.prefill_commands[i].end();
        }
        try self.reset();
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
        for (self.pipelines[0..self.pipeline_count]) |*p| p.deinit() catch @panic("model pipeline in use");
        self.pipeline_count = 0;
        for (self.gemm_pipes[0..self.gemm_count]) |*k| k.deinit() catch @panic("model gemm in use");
        self.gemm_count = 0;
        if (self.attn_gemm_live) self.attn_gemm.deinit() catch @panic("attention gemm in use");
        self.attn_gemm_live = false;
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
        const arenas = [_]*gpu.Buffer{ &self.act, &self.state, &self.io };
        for (arenas[0..remaining]) |b| b.deinit() catch @panic("model arena in use");
        self.live_buffers = 0;
        self.bank_count = 0;
    }

    fn upload(self: *Model, tensors: []const *const gguf.Tensor, placements: []const layout.Placement) Error!void {
        var staging = try gpu.Buffer.init(self.device, self.options.staging_bytes, .host);
        defer staging.deinit() catch @panic("staging in use");
        var cmd = try gpu.Commands.init(self.device);
        defer cmd.deinit() catch @panic("upload command pending");
        const chunk = std.mem.alignBackward(u64, self.options.staging_bytes, 4);
        for (tensors, placements) |t, place| {
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

    fn projection(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: ?u32) Error!Proj {
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
            self.pipelines[i] = try matvec.Pipeline.init(format, aligned, &self.banks[place.bank], &self.act, if (to_io) &self.io else &self.act);
            self.pipe_keys[i] = key;
            self.pipeline_count += 1;
            index = i;
        }
        const shape: matvec.Shape = .{ .format = format, .columns = @intCast(s.k), .rows = @intCast(s.rows) };
        const out_bytes: u64 = if (output_word) |w| @as(u64, w) * 4 else layout.io.logits * 4;
        return .{ .pipe = index.?, .geo = try self.pipelines[index.?].projection(shape, place.offset, @as(u64, input_word) * 4, out_bytes) };
    }

    fn both(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: u32, g: *[max_plans]GProj) Error!Proj {
        for (self.plans[0..self.plan_count], g[0..self.plan_count]) |plan, *slot| slot.* = try self.gprojection(s, t, place, input_word, output_word, plan);
        return self.projection(s, t, place, input_word, output_word);
    }

    fn gprojection(self: *Model, s: *const config.Spec, t: *const gguf.Tensor, place: layout.Placement, input_word: u32, output_word: u32, plan: Plan) Error!GProj {
        const format = try config.matrixFormat(t.kind);
        const v = gemm.variant(format);
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
                .x16 => try gpu.Kernel.initWith(self.device, try gemm.moduleF16x(v), &buffers, @sizeOf(gemm.Push), .{ .subgroup_size = gemm.f16x_subgroup, .full_subgroups = true }),
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
        try self.reset_commands.run(self.options.timeout_ns);
        self.position = 0;
    }

    /// Copy the recurrent and convolution state (as of `position`) into snapshot `slot`.
    pub fn saveSnapshot(self: *Model, slot: u32) Error!void {
        if (slot >= self.snapshot_slots) return error.InvalidSnapshot;
        try self.snapshotCopy(&self.state, @as(u64, self.state_layout.ssm) * 4, &self.snapshot_store, slot * snapshot_bytes);
    }

    /// Restore snapshot `slot`, saved when `position` tokens had been processed; the next
    /// token is at `position`. The caller guarantees the attention KV below `position` is
    /// unchanged since the save (only positions at or above it were written).
    pub fn loadSnapshot(self: *Model, slot: u32, position: u32) Error!void {
        if (slot >= self.snapshot_slots) return error.InvalidSnapshot;
        if (position > self.state_layout.context) return error.ContextFull;
        try self.snapshotCopy(&self.snapshot_store, slot * snapshot_bytes, &self.state, @as(u64, self.state_layout.ssm) * 4);
        self.position = position;
    }

    fn snapshotCopy(self: *Model, source: *gpu.Buffer, source_offset: u64, destination: *gpu.Buffer, destination_offset: u64) Error!void {
        const c = &self.copy_commands;
        try c.reset();
        try c.begin();
        try c.barrier(.compute, .transfer);
        try c.copy(source, source_offset, destination, destination_offset, snapshot_bytes);
        try c.barrier(.transfer, .compute);
        try c.end();
        try c.run(self.options.timeout_ns);
    }

    /// Advance one token at `self.position`; logits are borrowed until the next call.
    pub fn step(self: *Model, token: u32) Error![]const f32 {
        return self.run(&self.step_commands, token);
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
        if (self.rows == 0) return error.PrefillDisabled;
        if (plan >= self.plan_count) return error.InvalidPlan;
        if (chunk.len == 0 or chunk.len > self.plans[plan].rows) return error.InvalidToken;
        for (chunk) |token| if (token >= config.vocab) return error.InvalidToken;
        if (chunk.len > self.state_layout.context - self.position) return error.ContextFull;
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
        try commands.run(self.options.timeout_ns);
        self.position += @intCast(chunk.len);
        const after = try self.io.mapped();
        const floats: []align(1) const f32 = std.mem.bytesAsSlice(f32, after[layout.io.logits * 4 ..][0 .. config.vocab * 4]);
        return @alignCast(floats);
    }

    /// Run a command recorded by `record` (the default step or a capture variant).
    pub fn run(self: *Model, commands: *gpu.Commands, token: u32) Error![]const f32 {
        if (token >= config.vocab) return error.InvalidToken;
        if (self.position >= self.state_layout.context) return error.ContextFull;
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
                try r.kernel(.qk_b, QkBPush{ .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = ctx, .rope = layout.io.rope(self.rows), .eps = eps }, .{ config.heads + config.kv_heads, B, 1 });
                try r.mark(.qkprep, li);
                try r.snapRows("Qcur_normed", li, A.qn, 6144, 6144, false);
                try r.snapRows("Kcur_normed", li, A.kn, 1024, 1024, false);
                try r.snapRows("Qcur", li, A.qr, 6144, 6144, false);
                try r.snapRows("Kcur_roped", li, A.kr, 1024, 1024, false);
                try r.bar();
                const qk: gemm.Push = .{ .a_base = S.kcache(ai), .a_rs = 1, .a_cs = ctx, .a_bs = config.head_dim * ctx, .a_group = config.heads / config.kv_heads, .x_base = A.qr, .x_rs = 6144, .x_bs = config.head_dim, .y_base = A.scores, .y_rs = ctx, .y_bs = B * ctx, .m = 0, .k = config.head_dim, .flags = gemm.m_from_keys };
                try commands.dispatch(&self.attn_gemm, std.mem.asBytes(&qk), try gemm.validate(.f32_m, .narrow, qk, .{ .rows = B, .batches = config.heads, .max_keys = ctx }, self.state.size, self.act.size));
                try r.mark(.scores, li);
                try r.bar();
                try r.kernel(.softmax, SoftmaxPush{ .scores = A.scores, .ctx = ctx, .head_stride = B * ctx, .scale = 1.0 / 16.0 }, .{ B, config.heads, 1 });
                try r.mark(.softmax, li);
                try r.bar();
                const pv: gemm.Push = .{ .a_base = S.vcache(ai), .a_rs = 1, .a_cs = config.head_dim, .a_bs = ctx * config.head_dim, .a_group = config.heads / config.kv_heads, .x_base = A.scores, .x_rs = ctx, .x_bs = B * ctx, .y_base = A.pregate, .y_rs = 6144, .y_bs = config.head_dim, .m = config.head_dim, .k = 0, .flags = gemm.k_from_keys };
                try commands.dispatch(&self.attn_gemm, std.mem.asBytes(&pv), try gemm.validate(.f32_m, .narrow, pv, .{ .rows = B, .batches = config.heads, .max_keys = ctx }, self.state.size, self.act.size));
                try r.mark(.pv, li);
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
                try r.kernel(.conv_b, ConvPush{ .mixed = A.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = A.conv_out, .w = L.conv_w, .conv = S.convLayer(lin) }, .{ config.conv_channels / 128, 1, 1 });
                try r.mark(.conv, li);
                try r.snapRows("conv_output_raw", li, A.conv_raw, config.conv_channels, config.conv_channels, false);
                try r.snapRows("conv_output_silu", li, A.conv_silu, config.conv_channels, config.conv_channels, false);
                try r.snapRows("q_conv_predelta", li, A.conv_out, config.key_dim, config.conv_channels, false);
                try r.snapRows("k_conv_predelta", li, A.conv_out + config.key_dim, config.key_dim, config.conv_channels, false);
                try r.bar();
                try r.kernel(.delta_b, DeltaPush{ .qk = A.conv_out, .v = A.conv_out + 2 * config.key_dim, .z = A.z, .beta_raw = A.beta_raw, .alpha = A.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(lin), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = eps }, .{ config.v_heads, 1, 1 });
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
        try r.kernel(.norm, NormPush{ .x = A.r, .a = A.f, .sum = A.x, .y = A.hn, .w = self.output_norm, .stride = H, .flags = norm_add | norm_rows_io | norm_last_only, .eps = eps }, .{ B, 1, 1 });
        try r.mark(.norm, -1);
        try r.snapRows("l_out", config.layers - 1, A.x, H, H, false);
        try r.snapRows("result_norm", -1, A.hn, H, H, true);
        try r.bar();
        try r.proj(self.output);
        try r.mark(.output, -1);
        try commands.barrier(.compute, .host);
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
                try r.kernel(.qkprep, QkPush{ .qf = A.qf, .kc = A.kc, .vc = A.vc, .qn = A.qn, .qr = A.qr, .kn = A.kn, .kr = A.kr, .qw = L.q_norm, .kw = L.k_norm, .kcache = S.kcache(ai), .vcache = S.vcache(ai), .ctx = S.context, .eps = eps }, .{ config.heads + config.kv_heads, 1, 1 });
                try r.mark(.qkprep, @intCast(il));
                try r.snap("Qcur_normed", @intCast(il), A.qn, config.heads * config.head_dim);
                try r.snap("Kcur_normed", @intCast(il), A.kn, config.kv_heads * config.head_dim);
                try r.snap("Qcur", @intCast(il), A.qr, config.heads * config.head_dim);
                try r.snap("Kcur_roped", @intCast(il), A.kr, config.kv_heads * config.head_dim);
                try r.bar();
                try r.kernel(.attn_scores, attention.ScoresPush{ .qr = A.qr, .scores = A.scores, .amax = A.amax, .kcache = S.kcache(ai), .ctx = S.context, .chunks = A.chunks, .scale = 1.0 / 16.0 }, attention.groups(.scores, A.chunks));
                try r.mark(.scores, @intCast(il));
                try r.bar();
                try r.kernel(.attn_pv, attention.PvPush{ .scores = A.scores, .amax = A.amax, .apart = A.apart, .asum = A.asum, .vcache = S.vcache(ai), .ctx = S.context, .chunks = A.chunks }, attention.groups(.pv, A.chunks));
                try r.mark(.pv, @intCast(il));
                try r.bar();
                try r.kernel(.attn_combine, attention.CombinePush{ .apart = A.apart, .asum = A.asum, .qf = A.qf, .pregate = A.pregate, .gates = A.gates, .gated = A.gated, .chunks = A.chunks }, attention.groups(.combine, A.chunks));
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
                try r.kernel(.conv, ConvPush{ .mixed = A.mixed, .raw = A.conv_raw, .silu = A.conv_silu, .out = A.conv_out, .w = L.conv_w, .conv = S.convLayer(li) }, .{ config.conv_channels / 128, 1, 1 });
                try r.mark(.conv, @intCast(il));
                try r.snap("conv_output_raw", @intCast(il), A.conv_raw, config.conv_channels);
                try r.snap("conv_output_silu", @intCast(il), A.conv_silu, config.conv_channels);
                try r.snap("q_conv_predelta", @intCast(il), A.conv_out, config.key_dim);
                try r.snap("k_conv_predelta", @intCast(il), A.conv_out + config.key_dim, config.key_dim);
                try r.bar();
                try r.kernel(.delta, DeltaPush{ .qk = A.conv_out, .v = A.conv_out + 2 * config.key_dim, .z = A.z, .beta_raw = A.beta_raw, .alpha = A.alpha, .beta_out = A.beta, .softplus_out = A.softplus, .g_out = A.gate, .o = A.o, .y = A.fo, .ssm = S.ssmLayer(li), .a_w = L.ssm_a, .dt_w = L.dt, .norm_w = L.ssm_norm, .eps = eps }, .{ config.v_heads, 1, 1 });
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
            try r.proj(L.gate);
            try r.proj(L.up);
            try r.mark(.ffn_in, @intCast(il));
            try r.snap("ffn_gate", @intCast(il), A.fg, config.ffn);
            try r.snap("ffn_up", @intCast(il), A.fu, config.ffn);
            try r.bar();
            try r.kernel(.swiglu, SwigluPush{ .g = A.fg, .u = A.fu, .y = A.sw, .n = config.ffn }, .{ config.ffn / 128, 1, 1 });
            try r.mark(.swiglu, @intCast(il));
            try r.snap("ffn_swiglu", @intCast(il), A.sw, config.ffn);
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
};
