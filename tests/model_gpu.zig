const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const m = zerv.matvec;
const gemm = zerv.model.gemm;
const w = @import("matvec_workload.zig");
const t = std.testing;

const Harness = struct {
    a: gpu.Buffer,
    act: gpu.Buffer,
    io: gpu.Buffer,
    kernel: gpu.Kernel,
    reduce: gpu.Kernel,
    cmd: gpu.Commands,

    tile: gemm.Tile = .narrow,

    fn init(self: *Harness, device: *gpu.Device, v: gemm.Variant, a_bytes: u64, act_bytes: u64) !void {
        return self.initTile(device, v, .narrow, a_bytes, act_bytes);
    }
    fn initTile(self: *Harness, device: *gpu.Device, v: gemm.Variant, tile: gemm.Tile, a_bytes: u64, act_bytes: u64) !void {
        self.tile = tile;
        self.a = try gpu.Buffer.init(device, a_bytes, .host);
        self.act = try gpu.Buffer.init(device, act_bytes, .host);
        self.io = try gpu.Buffer.init(device, 1024, .host);
        self.kernel = try gpu.Kernel.init(device, try gemm.module(v, tile), &.{ &self.a, &self.act, &self.io, &self.act }, @sizeOf(gemm.Push));
        self.reduce = try gpu.Kernel.init(device, gemm.reduceModule(), &.{ &self.a, &self.act, &self.act, &self.io }, @sizeOf(gemm.ReducePush));
        self.cmd = try gpu.Commands.init(device);
    }
    fn deinit(self: *Harness) void {
        self.cmd.deinit() catch @panic("cmd");
        self.reduce.deinit() catch @panic("reduce");
        self.kernel.deinit() catch @panic("kernel");
        self.io.deinit() catch @panic("io");
        self.act.deinit() catch @panic("act");
        self.a.deinit() catch @panic("a");
    }
    fn run(self: *Harness, v: gemm.Variant, push: gemm.Push, lim: gemm.Limits, count: u32, p0: u32) !void {
        return self.runSplit(v, push, lim, count, p0, null);
    }
    fn runSplit(self: *Harness, v: gemm.Variant, push: gemm.Push, lim: gemm.Limits, count: u32, p0: u32, reduce: ?gemm.ReducePush) !void {
        const groups = try gemm.validate(v, self.tile, push, lim, self.a.size, self.act.size);
        const io = std.mem.bytesAsSlice(u32, try self.io.mapped());
        io[2] = count;
        io[3] = p0;
        try self.cmd.reset();
        try self.cmd.begin();
        try self.cmd.barrier(.host, .compute);
        try self.cmd.dispatch(&self.kernel, std.mem.asBytes(&push), groups);
        if (reduce) |rp| {
            try self.cmd.barrier(.compute, .compute);
            try self.cmd.dispatch(&self.reduce, std.mem.asBytes(&rp), .{ 64, 1, 1 });
        }
        try self.cmd.barrier(.compute, .host);
        try self.cmd.end();
        try self.cmd.run(w.timeout_ns);
    }
};

const exponents = [_]i32{ 0, 1, -1, 3, -2, 5, -4, 2, 0, 4, -3, 1 };
fn scale(row: usize) ?f32 {
    if (row % 11 == 7) return null; // zero row
    return std.math.ldexp(@as(f32, 1), exponents[row % exponents.len]);
}

test "prefill GEMM: independent matvec fixtures with scaled rows, tails, all finite halves" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 * 1024 });
    defer device.deinit() catch @panic("live gemm resources");
    var checked: usize = 0;
    // Narrow-tile outputs (per pass) that the wide tile must reproduce bit for bit.
    var narrow_out: [2]std.ArrayList(u8) = .{ .empty, .empty };
    defer for (&narrow_out) |*o| o.deinit(t.allocator);
    var wide_checked: usize = 0;
    for (parsed.value.cases) |case| for ([_]gemm.Tile{ .narrow, .wide }) |tile| {
        if (tile == .wide and case.format == .f32) continue;
        const raw = try w.fixtureBytes(t.allocator, case);
        defer t.allocator.free(raw);
        const data = try w.parseCase(raw);
        const shape = data.shape;
        const v = try gemm.variant(shape.format);
        const rows: u32 = if (case.kind == .explicit) 67 else 5; // not a multiple of the tile
        const K = shape.columns;
        const M = shape.rows;
        // 16-byte weight base (Q5_K aligned loads) and a 4-aligned X row stride whose
        // padding holds +inf, so any read past K that escapes the k guard poisons y.
        const a_off: u32 = if (shape.format == .f32) 4 else 16;
        const xs: u32 = std.mem.alignForward(u32, K, 4) + 4;
        var h: Harness = undefined;
        const x_base: u32 = 16;
        const y_base: u32 = x_base + rows * xs + 16;
        const split = case.kind == .explicit and K >= 512;
        const splits: u32 = if (split) (K + 255) / 256 else 1;
        const part_base: u32 = y_base + rows * M + 16;
        const act_words: u64 = @as(u64, part_base) + @as(u64, splits) * rows * M + 16;
        try h.initTile(&device, v, tile, a_off + data.weights.len + 64, act_words * 4);
        defer h.deinit();
        const a = try h.a.mapped();
        @memset(a, 0);
        @memcpy(a[a_off..][0..data.weights.len], data.weights);
        const act = std.mem.bytesAsSlice(f32, try h.act.mapped());
        @memset(act, std.math.nan(f32)); // sentinel
        const x: []align(1) const f32 = std.mem.bytesAsSlice(f32, data.input);
        for (0..rows) |r| {
            const s = scale(r);
            for (0..K) |k| act[x_base + r * xs + k] = if (s) |f| x[k] * f else 0;
            for (K..xs) |k| act[x_base + r * xs + k] = std.math.inf(f32);
        }
        const push: gemm.Push = if (shape.format == .f32)
            .{ .a_base = a_off / 4, .a_rs = K, .a_cs = 1, .x_base = x_base, .x_rs = xs, .y_base = y_base, .y_rs = M, .m = M, .k = K }
        else
            .{ .a_base = a_off, .a_rs = @intCast(data.weights.len / M), .x_base = x_base, .x_rs = xs, .y_base = y_base, .y_rs = M, .m = M, .k = K };
        const count = rows - 2; // the last two rows must stay untouched
        for (0..@as(usize, if (split) 2 else 1)) |pass| {
            if (pass == 1) {
                // Split-K into 256-wide chunks + fixed-order reduction must meet the same bound.
                const y = std.mem.bytesAsSlice(f32, try h.act.mapped())[y_base..][0 .. rows * M];
                @memset(y, std.math.nan(f32));
                var sp = push;
                sp.k_chunk = 256;
                sp.y_base = part_base;
                sp.y_bs = rows * M;
                try h.runSplit(v, sp, .{ .rows = rows, .batches = splits }, count, 0, .{ .part = part_base, .splits = splits, .part_bs = rows * M, .y = y_base, .y_rs = M, .m = M });
            } else try h.run(v, push, .{ .rows = rows }, count, 0);
            const out = std.mem.bytesAsSlice(f32, try h.act.mapped());
            const all_y = std.mem.sliceAsBytes(out[y_base..][0 .. rows * M]);
            if (tile == .narrow) {
                narrow_out[pass].clearRetainingCapacity();
                try narrow_out[pass].appendSlice(t.allocator, all_y);
            } else {
                try t.expectEqualSlices(u8, narrow_out[pass].items, all_y);
                wide_checked += 1;
            }
            var row_bytes = try t.allocator.alloc(u8, M * 4);
            defer t.allocator.free(row_bytes);
            for (0..rows) |r| {
                const y = out[y_base + r * M ..][0..M];
                if (r >= count) {
                    for (y) |value| try t.expect(std.math.isNan(value));
                    continue;
                }
                const s = scale(r) orelse {
                    for (y) |value| try t.expectEqual(@as(f32, 0), value);
                    continue;
                };
                if (case.kind == .explicit) {
                    for (y, 0..) |value, i| {
                        try t.expect(std.math.isFinite(value));
                        const expected = case.ideal[i] * s;
                        const err = @abs(@as(f64, value) - expected);
                        const bound: f64 = if (case.exact) 0 else (2e-6 + 4e-6 * case.sumabs[i]) * s;
                        if (err > bound) {
                            std.debug.print("gemm mismatch {s} row={d} col={d} got={d} want={d} err={e} bound={e}\n", .{ case.name, r, i, value, expected, err, bound });
                            return error.IndependentGoldenMismatch;
                        }
                    }
                } else {
                    // Exact scaling back to the independent canonical output fingerprint.
                    for (y, 0..) |value, i| std.mem.writeInt(u32, row_bytes[i * 4 ..][0..4], @bitCast(value / s), .little);
                    try w.check(case, row_bytes);
                }
                if (tile == .narrow) checked += M;
            }
        }
    };
    try t.expect(checked > 1453147); // 7 half fields x 63488 x 3 + ramp + explicit rows, plus split-K passes
    try t.expect(wide_checked > 20); // every quantized case, both passes where split
    try t.expectEqual(@as(u64, 0), device.allocated_bytes);
}

test "prefill GEMM: F32 strided/batched attention-style products against FP64" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live gemm resources");
    var prng = std.Random.DefaultPrng.init(99);
    const r = prng.random();
    const Z = 6;
    const group = 3;
    const ctx = 160;
    const rows = 70;
    const count = 69;
    const p0 = 81; // keys = 150
    const keys = p0 + count;
    for ([_]bool{ false, true }) |k_from_keys| {
        // QK-like: M = keys (runtime), K = 72 (tail). PV-like: K = keys (runtime), M = 72.
        const D = 72;
        const M: u32 = if (k_from_keys) D else keys;
        const K: u32 = if (k_from_keys) keys else D;
        const a_rows_stride: u32 = 1; // m contiguous
        const a_cs: u32 = if (k_from_keys) D else ctx; // k stride
        const a_bs: u32 = ctx * D + 8;
        const x_rs: u32 = Z * @max(K, ctx) + 4; // 16-byte X rows
        const x_bs: u32 = @max(K, ctx);
        const y_rs: u32 = Z * ctx + 5;
        const y_bs: u32 = ctx;
        const a_words: u64 = @as(u64, Z / group) * a_bs + 64;
        const x_base: u32 = 8;
        const y_base: u32 = x_base + rows * x_rs + 32;
        const act_words: u64 = @as(u64, y_base) + @as(u64, rows) * y_rs + 64;
        var h: Harness = undefined;
        try h.init(&device, .f32_m, a_words * 4, act_words * 4);
        defer h.deinit();
        const a = std.mem.bytesAsSlice(f32, try h.a.mapped());
        for (a) |*v| v.* = r.floatNorm(f32);
        const act = std.mem.bytesAsSlice(f32, try h.act.mapped());
        for (act) |*v| v.* = r.floatNorm(f32);
        const sentinel = std.math.nan(f32);
        for (act[y_base..]) |*v| v.* = sentinel;
        const push: gemm.Push = .{ .a_base = 3, .a_rs = a_rows_stride, .a_cs = a_cs, .a_bs = a_bs, .a_group = group, .x_base = x_base, .x_rs = x_rs, .x_bs = x_bs, .y_base = y_base, .y_rs = y_rs, .y_bs = y_bs, .m = if (k_from_keys) M else 0, .k = if (k_from_keys) 0 else K, .flags = if (k_from_keys) gemm.k_from_keys else gemm.m_from_keys };
        try h.run(.f32_m, push, .{ .rows = rows, .batches = Z, .max_keys = ctx }, count, p0);
        const out = std.mem.bytesAsSlice(f32, try h.act.mapped());
        const x = act; // unchanged inputs (outputs live after y_base)
        for (0..Z) |z| for (0..rows) |tt| for (0..y_bs) |mm| {
            const y = out[y_base + z * y_bs + tt * y_rs + mm];
            if (tt >= count or mm >= M) {
                try t.expect(std.math.isNan(y));
                continue;
            }
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..K) |k| {
                const av: f64 = a[3 + (z / group) * a_bs + mm * a_rows_stride + k * a_cs];
                const xv: f64 = x[x_base + z * x_bs + tt * x_rs + k];
                sum += av * xv;
                abs += @abs(av * xv);
            }
            try t.expect(@abs(@as(f64, y) - sum) <= 2e-6 + 4e-6 * abs);
        };
    }
    // Extent validation rejects out-of-range pushes before recording.
    const bad: gemm.Push = .{ .a_base = 0, .a_rs = 1, .a_cs = 64, .x_base = 0, .x_rs = 64, .y_base = 0, .y_rs = 64, .m = 64, .k = 64 };
    try t.expectError(error.InvalidRange, gemm.validate(.f32_m, .narrow, bad, .{ .rows = 64 }, 64 * 63 * 4, 1 << 20));
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, .{ .a_base = 0, .a_rs = 17, .x_base = 0, .x_rs = 32, .y_base = 0, .y_rs = 1, .m = 1, .k = 32 }, .{ .rows = 1 }, 1 << 20, 1 << 20));
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, .{ .a_base = 1, .a_rs = 18, .x_base = 0, .x_rs = 32, .y_base = 0, .y_rs = 1, .m = 1, .k = 32 }, .{ .rows = 1 }, 1 << 20, 1 << 20));
    // X rows must be 16-byte aligned; Q5_K weight rows 16-byte aligned.
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, .{ .a_base = 0, .a_rs = 18, .x_base = 2, .x_rs = 32, .y_base = 64, .y_rs = 1, .m = 1, .k = 32 }, .{ .rows = 1 }, 1 << 20, 1 << 20));
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, .{ .a_base = 0, .a_rs = 18, .x_base = 0, .x_rs = 34, .y_base = 64, .y_rs = 1, .m = 1, .k = 32 }, .{ .rows = 2 }, 1 << 20, 1 << 20));
    try t.expectError(error.InvalidShape, gemm.validate(.q5_k, .narrow, .{ .a_base = 8, .a_rs = 176, .x_base = 0, .x_rs = 256, .y_base = 512, .y_rs = 1, .m = 1, .k = 256 }, .{ .rows = 1 }, 1 << 20, 1 << 20));
}

test "split-K decode attention against FP64 (positions across chunk edges, stale keys ignored)" {
    try decodeAttention(.f32, 256);
}

// `--kv-page-tokens` (docs/specs/concurrent.md): 128-token pages and one page holding the
// whole context (the layout before paging) meet the same bound.
test "split-K decode attention with 128-token pages and with one context-sized page" {
    try decodeAttention(.f32, 128);
    try decodeAttention(.f16, 4096);
}

// Block 17c: the f16 KV variants read halves and compute exactly as the f32 kernels, so
// against FP64 over the same (f16-representable) K/V values they meet the same bound.
test "split-K decode attention with an f16 KV cache against FP64 over the f16 values" {
    try decodeAttention(.f16, 256);
}

/// KV cache values as the kernels see them: FP32 words or f16 halves (element offsets).
const KvView = struct {
    f32s: []align(1) f32 = &.{},
    f16s: []align(1) f16 = &.{},
    fn of(buffer: *gpu.Buffer, kv: zerv.model.KvType) !KvView {
        const bytes = try buffer.mapped();
        return if (kv == .f16) .{ .f16s = std.mem.bytesAsSlice(f16, bytes) } else .{ .f32s = std.mem.bytesAsSlice(f32, bytes) };
    }
    /// Stores `v` (rounded to f16 in f16 mode) and returns the value stored.
    fn set(self: KvView, i: usize, v: f32) f32 {
        if (self.f16s.len > 0) {
            self.f16s[i] = @floatCast(v);
            return self.f16s[i];
        }
        self.f32s[i] = v;
        return v;
    }
    fn fill(self: KvView, v: f32) void {
        if (self.f16s.len > 0) @memset(self.f16s, @as(f16, @floatCast(v))) else @memset(self.f32s, v);
    }
    fn get(self: KvView, i: usize) f32 {
        return if (self.f16s.len > 0) self.f16s[i] else self.f32s[i];
    }
};

/// Paged KV addressing (docs/specs/concurrent.md, "Addressing"), independent of the
/// kernels: logical page q of the sequence is physical page `table[q]`; a page is `pstride`
/// elements; the layer's K and V pieces start at `kcache`/`vcache` inside every page, K as
/// [head][dim][page token], V as [head][page token][dim]. The tests use a permuted table
/// over more physical pages than the sequence needs and a layer that is not first in its
/// buffer, so identity-only addressing fails them.
const Paged = struct {
    table: []const u32,
    /// Tokens per page: the kernels' specialization constant 0.
    page: u32,
    pstride: u32,
    kcache: u32,
    vcache: u32,
    /// Physical pages: at least 3 more than the sequence uses and coprime to 7, so the
    /// table below is a permutation into them (no assert: tests also run in ReleaseFast,
    /// where a failed assert is undefined behaviour, not a failure).
    physical: u32,
    /// `table.len` logical pages of `page` tokens, layer 1 of 2 in the buffer.
    fn init(table: []u32, page: u32) Paged {
        var physical: u32 = @intCast(table.len + 3);
        if (physical % 7 == 0) physical += 1;
        for (table, 0..) |*e, q| e.* = @intCast((q * 7 + 3) % physical);
        const piece = zerv.model.layout.kv_token_elements * page;
        return .{ .table = table, .page = page, .physical = physical, .pstride = 2 * piece, .kcache = piece, .vcache = piece + 4 * 256 * page };
    }
    fn k(self: Paged, g: usize, d: usize, j: usize) usize {
        return self.table[j / self.page] * @as(usize, self.pstride) + self.kcache + (g * 256 + d) * self.page + j % self.page;
    }
    fn v(self: Paged, g: usize, d: usize, j: usize) usize {
        return self.table[j / self.page] * @as(usize, self.pstride) + self.vcache + (g * self.page + j % self.page) * 256 + d;
    }
    /// Kernel options carrying the page size.
    fn options(self: *const Paged) gpu.Kernel.Options {
        return .{ .constants = (&self.page)[0..1] };
    }
    /// Writes the table as u32 words at word `at` of `act`.
    fn store(self: Paged, act: *gpu.Buffer, at: u32) !void {
        @memcpy(std.mem.bytesAsSlice(u32, try act.mapped())[at..][0..self.table.len], self.table);
    }
};

fn decodeAttention(kv_type: zerv.model.KvType, comptime page: u32) !void {
    const att = zerv.model.attention;
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024, .storage16 = kv_type == .f16 });
    defer device.deinit() catch @panic("live attention resources");
    const ctx: u32 = 4096;
    const H = 24;
    const G = 4;
    const D = 256;
    const chunks: u32 = ctx / att.chunk;
    // Activation arena regions (words).
    const qr: u32 = 0;
    const qf: u32 = qr + H * D;
    const scores: u32 = qf + H * 2 * D;
    const amax: u32 = scores + H * ctx;
    const apart: u32 = amax + H * chunks;
    const asum: u32 = apart + H * chunks * D;
    const pregate: u32 = asum + H * chunks;
    const gates: u32 = pregate + H * D;
    const gated: u32 = gates + H * D;
    const gmax: u32 = gated + H * D;
    const blocks: u32 = (chunks + 7) / 8;
    const bpart: u32 = gmax + H;
    const bsum: u32 = bpart + H * blocks * D;
    const ptab: u32 = bsum + H * blocks;
    const act_words: u64 = ptab + ctx / page;
    var table: [ctx / page]u32 = undefined;
    const pg = Paged.init(&table, page);
    const state_words: u64 = @as(u64, pg.pstride) * pg.physical;

    var params = try gpu.Buffer.init(&device, 256, .host);
    defer params.deinit() catch @panic("params");
    var act = try gpu.Buffer.init(&device, act_words * 4, .host);
    defer act.deinit() catch @panic("act");
    var state = try gpu.Buffer.init(&device, state_words * kv_type.bytes(), .host);
    defer state.deinit() catch @panic("state");
    var io = try gpu.Buffer.init(&device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var kernels: [5]gpu.Kernel = undefined;
    for (&kernels, [_]att.Pass{ .scores, .pv, .combine, .gmax, .block }) |*k, pass| k.* = try gpu.Kernel.initWith(&device, att.module(pass, kv_type), &.{ &params, &act, &state, &io }, att.pushBytes(pass), pg.options());
    defer for (&kernels) |*k| k.deinit() catch @panic("kernel");
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&kernels[0], std.mem.asBytes(&att.ScoresPush{ .qr = qr, .scores = scores, .amax = amax, .kcache = pg.kcache, .ctx = ctx, .chunks = chunks, .scale = 1.0 / 16.0, .pos = 1, .ptab = ptab, .pstride = pg.pstride }), att.groups(.scores, chunks, 1));
    try cmd.barrier(.compute, .compute);
    try cmd.dispatch(&kernels[3], std.mem.asBytes(&att.GmaxPush{ .amax = amax, .gmax = gmax, .chunks = chunks, .pos = 1 }), att.groups(.gmax, chunks, 1));
    try cmd.barrier(.compute, .compute);
    try cmd.dispatch(&kernels[1], std.mem.asBytes(&att.PvPush{ .scores = scores, .amax = amax, .apart = apart, .asum = asum, .vcache = pg.vcache, .ctx = ctx, .chunks = chunks, .pos = 1, .gmax = gmax, .ptab = ptab, .pstride = pg.pstride }), att.groups(.pv, chunks, 1));
    try cmd.barrier(.compute, .compute);
    try cmd.dispatch(&kernels[4], std.mem.asBytes(&att.BlockPush{ .apart = apart, .asum = asum, .bpart = bpart, .bsum = bsum, .chunks = chunks, .blocks = blocks, .pos = 1 }), att.groups(.block, chunks, 1));
    try cmd.barrier(.compute, .compute);
    try cmd.dispatch(&kernels[2], std.mem.asBytes(&att.CombinePush{ .bpart = bpart, .bsum = bsum, .qf = qf, .pregate = pregate, .gates = gates, .gated = gated, .blocks = blocks, .pos = 1 }), att.groups(.combine, chunks, 1));
    try cmd.barrier(.compute, .host);
    try cmd.end();

    var prng = std.Random.DefaultPrng.init(1313);
    const r = prng.random();
    const K = try t.allocator.alloc(f32, G * D * ctx); // [g][d][j]
    defer t.allocator.free(K);
    const V = try t.allocator.alloc(f32, G * ctx * D); // [g][j][d]
    defer t.allocator.free(V);
    // f16 mode: values rounded to f16 once, so the FP64 reference sees what is stored.
    for (K) |*v| v.* = if (kv_type == .f16) @as(f16, @floatCast(r.float(f32) * 2 - 1)) else r.float(f32) * 2 - 1;
    for (V) |*v| v.* = if (kv_type == .f16) @as(f16, @floatCast(r.float(f32) * 2 - 1)) else r.float(f32) * 2 - 1;
    const u: f64 = std.math.ldexp(@as(f64, 1), -24);
    const gamma256 = 256 * u / (1 - 256 * u);
    // Stale keys past the position: a huge finite FP32 value, NaN in f16 (either shows if read).
    const huge: f32 = if (kv_type == .f16) std.math.nan(f32) else 1e30;
    const s64 = try t.allocator.alloc(f64, ctx);
    defer t.allocator.free(s64);
    var worst: f64 = 0;
    for ([_]u32{ 0, 1, 62, 63, 64, 65, 127, 1000, 4095 }) |pos| {
        const a = std.mem.bytesAsSlice(f32, try act.mapped());
        for (a[0..qf]) |*v| v.* = r.float(f32) * 6 - 3;
        for (a[qf..scores]) |*v| v.* = r.float(f32) * 8 - 4;
        for (a[scores..]) |*v| v.* = std.math.nan(f32); // scratch/outputs: stale reads would show
        try pg.store(&act, ptab);
        const st = try KvView.of(&state, kv_type);
        // Unmapped pages and the other layer's pieces hold the stale value too.
        st.fill(huge);
        for (0..G) |g| for (0..D) |d| for (0..pos + 1) |j| {
            _ = st.set(pg.k(g, d, j), K[(g * D + d) * ctx + j]);
            _ = st.set(pg.v(g, d, j), V[(g * ctx + j) * D + d]);
        };
        std.mem.bytesAsSlice(u32, try io.mapped())[1] = pos;
        try cmd.run(w.timeout_ns);
        const out = std.mem.bytesAsSlice(f32, try act.mapped());
        const n = pos + 1;
        const live = (n + att.chunk - 1) / att.chunk;
        for (0..H) |h| {
            const g = h / att.heads_per_kv;
            var mx: f64 = -std.math.inf(f64);
            var S: f64 = 0;
            for (0..n) |j| {
                var dot: f64 = 0;
                var abs: f64 = 0;
                for (0..D) |d| {
                    const p = @as(f64, a[qr + h * D + d]) * K[(g * D + d) * ctx + j];
                    dot += p;
                    abs += @abs(p);
                }
                s64[j] = dot / 16;
                S = @max(S, gamma256 * abs / 16);
                mx = @max(mx, s64[j]);
            }
            var total: f64 = 0;
            for (0..n) |j| total += @exp(s64[j] - mx);
            for (0..D) |d| {
                var o64: f64 = 0;
                for (0..n) |j| o64 += @exp(s64[j] - mx) / total * V[(g * ctx + j) * D + d];
                var bound: f64 = 1e-30;
                for (0..n) |j| {
                    const pj = @exp(s64[j] - mx) / total;
                    const x = s64[j] - mx;
                    bound += pj * (@abs(V[(g * ctx + j) * D + d]) + @abs(o64)) * (2 * S + (2 * @abs(x) + @as(f64, @floatFromInt(att.chunk + live + 16))) * u);
                }
                const o = out[pregate + h * D + d];
                if (pos == 0) try t.expectEqual(V[(g * ctx) * D + d], o);
                const err = @abs(@as(f64, o) - o64);
                if (!(err <= bound)) {
                    std.debug.print("pos {d} head {d} dim {d}: o {e} ref {e} err {e} bound {e}\n", .{ pos, h, d, o, o64, err, bound });
                    return error.TestUnexpectedResult;
                }
                worst = @max(worst, err / bound);
                const x: f64 = a[qf + h * 2 * D + D + d];
                const g64 = 1 / (1 + @exp(-x));
                const gv = out[gates + h * D + d];
                try t.expect(@abs(@as(f64, gv) - g64) <= (4 + 2 * @abs(x)) * std.math.ldexp(@as(f64, 1), -23) * g64);
                try t.expectEqual(o * gv, out[gated + h * D + d]);
            }
        }
    }
    try t.expect(worst > 0 and worst < 0.5); // measured 0.0012 (bound is first-order worst case)
    std.debug.print("split-K decode attention ({s} KV, {d}-token pages): worst error / first-order bound {e}\n", .{ @tagName(kv_type), page, worst });
}

// Block 17b (batched MTP catch-up): the Q8_0 GEMM (eh_proj) against FP64 on exactly
// decoded random weights; blocks at both alignments (34-byte blocks alternate 0 / 2 mod 4,
// base 2 mod 4); the wide tile bit-identical to the narrow one; split-K within the bound;
// rows >= count untouched.
test "prefill GEMM: Q8_0 against FP64, wide = narrow, split-K" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("live gemm resources");
    var prng = std.Random.DefaultPrng.init(0x8008);
    const random = prng.random();
    const K: u32 = 1024;
    const M: u32 = 300; // not a multiple of the 256-row tile
    const rows: u32 = 70;
    const count: u32 = 67;
    const row_bytes: u32 = K / 32 * 34;
    const a_off: u32 = 6; // 2 mod 4
    const packed_bytes = try t.allocator.alloc(u8, @as(usize, M) * row_bytes);
    defer t.allocator.free(packed_bytes);
    const values = try t.allocator.alloc(f64, @as(usize, M) * K);
    defer t.allocator.free(values);
    for (0..@as(usize, M) * K / 32) |b| {
        const d: f16 = @floatCast((random.float(f32) - 0.5) * 0.05);
        std.mem.writeInt(u16, packed_bytes[b * 34 ..][0..2], @bitCast(d), .little);
        for (0..32) |j| {
            const q: i8 = random.int(i8);
            packed_bytes[b * 34 + 2 + j] = @bitCast(q);
            values[b * 32 + j] = @as(f64, @floatCast(d)) * @as(f64, @floatFromInt(q));
        }
    }
    const x_base: u32 = 16;
    const y_base: u32 = x_base + rows * K + 16;
    const part_base: u32 = y_base + rows * M + 16;
    const splits: u32 = K / 256;
    const act_words: u64 = @as(u64, part_base) + @as(u64, splits) * rows * M + 16;
    var outputs: [3][]u8 = undefined;
    for (0..3) |pass| {
        const tile: gemm.Tile = if (pass == 1) .wide else .narrow;
        var h: Harness = undefined;
        try h.initTile(&device, .q8_0, tile, a_off + packed_bytes.len + 64, act_words * 4);
        defer h.deinit();
        const a = try h.a.mapped();
        @memset(a, 0);
        @memcpy(a[a_off..][0..packed_bytes.len], packed_bytes);
        const act = std.mem.bytesAsSlice(f32, try h.act.mapped());
        @memset(act, std.math.nan(f32));
        var xprng = std.Random.DefaultPrng.init(0x8009);
        for (0..rows) |r| for (0..K) |k| {
            act[x_base + r * K + k] = xprng.random().floatNorm(f32);
        };
        const push: gemm.Push = .{ .a_base = a_off, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y_base, .y_rs = M, .m = M, .k = K };
        if (pass == 2) {
            var sp = push;
            sp.k_chunk = 256;
            sp.y_base = part_base;
            sp.y_bs = rows * M;
            try h.runSplit(.q8_0, sp, .{ .rows = rows, .batches = splits }, count, 0, .{ .part = part_base, .splits = splits, .part_bs = rows * M, .y = y_base, .y_rs = M, .m = M });
        } else try h.run(.q8_0, push, .{ .rows = rows }, count, 0);
        const out = std.mem.bytesAsSlice(f32, try h.act.mapped());
        for (0..rows) |r| for (0..M) |i| {
            const got = out[y_base + r * M + i];
            if (r >= count) {
                try t.expect(std.math.isNan(got));
                continue;
            }
            var exact: f64 = 0;
            var magnitude: f64 = 0;
            for (0..K) |k| {
                const term = values[i * K + k] * @as(f64, act[x_base + r * K + k]);
                exact += term;
                magnitude += @abs(term);
            }
            try t.expect(@abs(@as(f64, got) - exact) <= @as(f64, K + 64) * 0x1p-24 * magnitude);
        };
        outputs[pass] = try t.allocator.dupe(u8, std.mem.sliceAsBytes(out[y_base..][0 .. rows * M]));
    }
    defer for (outputs) |o| t.allocator.free(o);
    try t.expectEqualSlices(u8, outputs[0], outputs[1]);
    try t.expectError(error.InvalidShape, gemm.validate(.q8_0, .narrow, .{ .a_base = 6, .a_rs = row_bytes + 2, .x_base = 16, .x_rs = K, .y_base = 0, .y_rs = M, .m = M, .k = K }, .{ .rows = 4 }, 1 << 30, 1 << 30));
}

// Block 16a: fused prefill attention (flash.comp) against FP64. Chunks of `count` rows at
// positions p0.. (row r sees keys 0..p0+r) across row-block (64) and key-tile (64) edges,
// a context that is not a multiple of 64; cache entries at or past p0 + count and query
// rows past count hold NaN (never read: outputs stay finite), output rows past count stay
// untouched.
test "fused prefill attention against FP64 (row blocks, tile edges, causal mask, GQA)" {
    try fusedAttention(.f32, 256);
}

// 128-token pages, and one page of the whole 4064-token context (not a multiple of 128).
test "fused prefill attention with 128-token pages and with one context-sized page" {
    try fusedAttention(.f32, 128);
    try fusedAttention(.f16, 4064);
}

test "fused prefill attention with an f16 KV cache against FP64 over the f16 values" {
    try fusedAttention(.f16, 256);
}

fn fusedAttention(kv_type: zerv.model.KvType, comptime page: u32) !void {
    const builtin = @import("builtin");
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024, .storage16 = kv_type == .f16 });
    defer device.deinit() catch @panic("live flash resources");
    const ctx: u32 = 4064;
    const H = 24;
    const G = 4;
    const D = 256;
    const rows_max: u32 = 256;
    const qr: u32 = 0;
    const out: u32 = qr + rows_max * H * D;
    const ptab: u32 = out + rows_max * H * D;
    const pages = (ctx + page - 1) / page;
    const act_words: u64 = ptab + pages;
    var table: [pages]u32 = undefined;
    const pg = Paged.init(&table, page);
    const kv_words: u64 = @as(u64, pg.pstride) * pg.physical;
    var params = try gpu.Buffer.init(&device, 256, .host);
    defer params.deinit() catch @panic("params");
    var act = try gpu.Buffer.init(&device, act_words * 4, .host);
    defer act.deinit() catch @panic("act");
    var kv = try gpu.Buffer.init(&device, kv_words * kv_type.bytes(), .host);
    defer kv.deinit() catch @panic("kv");
    var io = try gpu.Buffer.init(&device, 1024, .host);
    defer io.deinit() catch @panic("io");
    const FlashPush = extern struct { qr: u32, kcache: u32, vcache: u32, out: u32, ctx: u32, scale: f32, ptab: u32, pstride: u32 };
    var kernel = try gpu.Kernel.initWith(&device, zerv.model.flashModule(kv_type), &.{ &params, &act, &kv, &io }, @sizeOf(FlashPush), pg.options());
    defer kernel.deinit() catch @panic("kernel");
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");
    var prng = std.Random.DefaultPrng.init(0x16a);
    const r = prng.random();
    const u: f64 = std.math.ldexp(@as(f64, 1), -24);
    const gamma256 = 256 * u / (1 - 256 * u);
    const s64 = try t.allocator.alloc(f64, ctx);
    defer t.allocator.free(s64);
    var worst: f64 = 0;
    const Case = struct { p0: u32, count: u32 };
    for ([_]Case{ .{ .p0 = 0, .count = 1 }, .{ .p0 = 0, .count = 64 }, .{ .p0 = 0, .count = 70 }, .{ .p0 = 37, .count = 5 }, .{ .p0 = 100, .count = 130 }, .{ .p0 = 1000, .count = 64 }, .{ .p0 = 3800, .count = 256 }, .{ .p0 = 4063, .count = 1 } }) |c| {
        const a = std.mem.bytesAsSlice(f32, try act.mapped());
        for (0..rows_max) |row| for (0..H * D) |i| {
            a[qr + row * H * D + i] = if (row < c.count) r.float(f32) * 6 - 3 else std.math.nan(f32);
        };
        @memset(a[out..ptab], std.math.nan(f32));
        try pg.store(&act, ptab);
        const st = try KvView.of(&kv, kv_type);
        const live = c.p0 + c.count;
        st.fill(std.math.nan(f32)); // unmapped pages, the other layer and entries past live
        for (0..G) |g| for (0..D) |d| for (0..live) |j| {
            _ = st.set(pg.k(g, d, j), r.float(f32) * 2 - 1);
            _ = st.set(pg.v(g, d, j), r.float(f32) * 2 - 1);
        };
        const words = std.mem.bytesAsSlice(u32, try io.mapped());
        words[2] = c.count;
        words[3] = c.p0;
        try cmd.reset();
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        try cmd.dispatch(&kernel, std.mem.asBytes(&FlashPush{ .qr = qr, .kcache = pg.kcache, .vcache = pg.vcache, .out = out, .ctx = ctx, .scale = 1.0 / 16.0, .ptab = ptab, .pstride = pg.pstride }), .{ (c.count + zerv.model.flash_rows - 1) / zerv.model.flash_rows, H / zerv.model.flash_groups, 1 });
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(w.timeout_ns);
        const o_all = std.mem.bytesAsSlice(f32, try act.mapped());
        for (c.count..rows_max) |row| for (0..H * D) |i| try t.expect(std.math.isNan(o_all[out + row * H * D + i]));
        // Rows checked (all in ReleaseFast; edges in Debug).
        var rows_buf: [256]u32 = undefined;
        var n_rows: usize = 0;
        for (0..c.count) |row| {
            const edge = row < 2 or row + 2 >= c.count or row % 8 == 7 or row % 64 == 0 or row == c.count / 2;
            if (builtin.mode == .Debug and !edge) continue;
            rows_buf[n_rows] = @intCast(row);
            n_rows += 1;
        }
        const heads = if (builtin.mode == .Debug) &[_]u32{ 0, 5, 6, 23 } else &[_]u32{ 0, 1, 5, 6, 11, 12, 17, 18, 23 };
        for (rows_buf[0..n_rows]) |row| for (heads) |h| {
            const g = h / 6;
            const n = c.p0 + row + 1;
            var mx: f64 = -std.math.inf(f64);
            var S: f64 = 0;
            for (0..n) |j| {
                var dot: f64 = 0;
                var abs: f64 = 0;
                for (0..D) |d| {
                    const pr = @as(f64, a[qr + row * H * D + h * D + d]) * st.get(pg.k(g, d, j));
                    dot += pr;
                    abs += @abs(pr);
                }
                s64[j] = dot / 16;
                S = @max(S, gamma256 * abs / 16);
                mx = @max(mx, s64[j]);
            }
            var total: f64 = 0;
            for (0..n) |j| total += @exp(s64[j] - mx);
            const tiles: f64 = @floatFromInt((n + 63) / 64);
            for (0..D) |d| {
                var o64: f64 = 0;
                for (0..n) |j| o64 += @exp(s64[j] - mx) / total * st.get(pg.v(g, d, j));
                var bound: f64 = 1e-30;
                for (0..n) |j| {
                    const pj = @exp(s64[j] - mx) / total;
                    const x = s64[j] - mx;
                    bound += pj * (@abs(st.get(pg.v(g, d, j))) + @abs(o64)) * (2 * S + (2 * @abs(x) + @as(f64, @floatFromInt(n)) + 2 * tiles + 16) * u);
                }
                const o = o_all[out + row * H * D + h * D + d];
                const err = @abs(@as(f64, o) - o64);
                if (!(err <= bound)) {
                    std.debug.print("p0 {d} count {d} row {d} head {d} dim {d}: o {e} ref {e} err {e} bound {e}\n", .{ c.p0, c.count, row, h, d, o, o64, err, bound });
                    return error.TestUnexpectedResult;
                }
                worst = @max(worst, err / bound);
            }
        };
    }
    try t.expect(worst > 0 and worst < 0.5);
    std.debug.print("fused prefill attention ({s} KV, {d}-token pages): worst error / first-order bound {e}\n", .{ @tagName(kv_type), page, worst });
}

// Block 17c gate 2 (docs/specs/model.md, "KV precision"): the f16 KV writers store exactly
// f16_RNE(x) of the FP32 values the f32 writers store: qkprep (decode rows, 2 rows as in
// verify) and qk_b (prefill rows). V is copied raw, so its inputs carry the rounding edge
// cases: ties to even, subnormal results (including ties at the subnormal scale), values
// that round up to the next binade, overflow to infinity and signed zeros. Zig's
// @floatCast f32 -> f16 is round-to-nearest-even with subnormals, the reference.
test "f16 KV writers store round-to-nearest-even halves of the FP32 values" {
    try kvWriters(256);
    try kvWriters(128);
}

fn kvWriters(comptime page: u32) !void {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024, .storage16 = true });
    defer device.deinit() catch @panic("live writer resources");
    // Two rows at positions page - 1 and page: the writes cross a page boundary.
    const ctx: u32 = 2 * page;
    const rows: u32 = 2;
    const G = 4;
    const D = 256;
    // Activation arena (words): inputs qf, kc, vc; outputs qn, qr, kn, kr; page table.
    const qf: u32 = 0;
    const kc: u32 = qf + rows * 12288;
    const vc: u32 = kc + rows * 1024;
    const qn: u32 = vc + rows * 1024;
    const qr: u32 = qn + rows * 6144;
    const kn: u32 = qr + rows * 6144;
    const kr: u32 = kn + rows * 1024;
    const ptab: u32 = kr + rows * 1024;
    const act_words: u32 = ptab + ctx / page;
    var table: [ctx / page]u32 = undefined;
    const pg = Paged.init(&table, page);
    const kv_elems: u32 = pg.pstride * pg.physical;
    var params = try gpu.Buffer.init(&device, 2 * 256 * 4, .host);
    defer params.deinit() catch @panic("params");
    var act = try gpu.Buffer.init(&device, act_words * 4, .host);
    defer act.deinit() catch @panic("act");
    var kv32 = try gpu.Buffer.init(&device, kv_elems * 4, .host);
    defer kv32.deinit() catch @panic("kv32");
    var kv16 = try gpu.Buffer.init(&device, kv_elems * 2, .host);
    defer kv16.deinit() catch @panic("kv16");
    var io = try gpu.Buffer.init(&device, 4096, .host);
    defer io.deinit() catch @panic("io");
    const model = zerv.model;
    var prng = std.Random.DefaultPrng.init(0x17c);
    const r = prng.random();
    {
        const weights = std.mem.bytesAsSlice(f32, try params.mapped());
        for (weights) |*v| v.* = 0.5 + r.float(f32) * 2;
    }
    // V edge cases (f32 inputs; the RNE result is checked, whatever it is).
    const ulp16 = std.math.ldexp(@as(f32, 1), -10); // f16 ulp at 1
    const sub = std.math.ldexp(@as(f32, 1), -24); // smallest f16 subnormal
    const edges = [_]f32{ 1 + ulp16 / 2, 1 + 3 * ulp16 / 2, -(1 + ulp16 / 2), 2048 + 1, 2048 + 3, sub, sub / 2, 3 * sub / 2, 5 * sub / 2, -sub / 2, 1e-6, 6.1e-5, 6.103515625e-05 - sub / 2, 65504, 65519.99, 65520, 70000, -70000, 0.0, -0.0, 1e-30, 113.00391 };
    const Writer = enum { qkprep, qk_b };
    for ([_]Writer{ .qkprep, .qk_b }) |writer| {
        const push_size: u32 = if (writer == .qkprep) @sizeOf(model.QkPush) else @sizeOf(model.QkBPush);
        var k32 = try gpu.Kernel.initWith(&device, if (writer == .qkprep) model.qkprepModule(.f32) else model.qkBModule(.f32), &.{ &params, &act, &kv32, &io }, push_size, pg.options());
        defer k32.deinit() catch @panic("k32");
        var k16 = try gpu.Kernel.initWith(&device, if (writer == .qkprep) model.qkprepModule(.f16) else model.qkBModule(.f16), &.{ &params, &act, &kv16, &io }, push_size, pg.options());
        defer k16.deinit() catch @panic("k16");
        try pg.store(&act, ptab);
        const a = std.mem.bytesAsSlice(f32, try act.mapped());
        for (a[qf..kc]) |*v| v.* = r.float(f32) * 4 - 2;
        for (a[kc..vc]) |*v| v.* = r.float(f32) * 4 - 2;
        for (a[vc..qn], 0..) |*v, i| v.* = if (i % 3 == 0) edges[(i / 3) % edges.len] else (r.float(f32) * 2 - 1) * 120;
        const words = std.mem.bytesAsSlice(u32, try io.mapped());
        const p0: u32 = page - 1;
        words[1] = p0; // decode position (qkprep reads io[pos] = word 1)
        words[2] = rows; // prefill count
        words[3] = p0; // prefill first position
        const rope: u32 = 64;
        for (0..rows) |row| for (0..32) |i| {
            const angle = @as(f64, @floatFromInt(p0 + row)) * std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(2 * i)) / 64);
            words[rope + row * 64 + i] = @bitCast(@as(f32, @floatCast(@cos(angle))));
            words[rope + row * 64 + 32 + i] = @bitCast(@as(f32, @floatCast(@sin(angle))));
        };
        @memset(std.mem.bytesAsSlice(u32, try kv32.mapped()), 0x7fc00000);
        @memset(std.mem.bytesAsSlice(u16, try kv16.mapped()), 0x7e00);
        var kr32: [rows * 1024]f32 = undefined;
        for ([_]*gpu.Kernel{ &k32, &k16 }, 0..) |kernel, pass| {
            var cmd = try gpu.Commands.init(&device);
            defer cmd.deinit() catch @panic("cmd");
            try cmd.begin();
            try cmd.barrier(.host, .compute);
            if (writer == .qkprep) {
                try cmd.dispatch(kernel, std.mem.asBytes(&model.QkPush{ .qf = qf, .kc = kc, .vc = vc, .qn = qn, .qr = qr, .kn = kn, .kr = kr, .qw = 0, .kw = 256, .kcache = pg.kcache, .vcache = pg.vcache, .ctx = ctx, .eps = 1e-6, .rope = rope, .pos = 1, .ptab = ptab, .pstride = pg.pstride }), .{ 28, rows, 1 });
            } else {
                try cmd.dispatch(kernel, std.mem.asBytes(&model.QkBPush{ .qf = qf, .kc = kc, .vc = vc, .qn = qn, .qr = qr, .kn = kn, .kr = kr, .qw = 0, .kw = 256, .kcache = pg.kcache, .vcache = pg.vcache, .ctx = ctx, .rope = rope, .eps = 1e-6, .ptab = ptab, .pstride = pg.pstride }), .{ 28, rows, 1 });
            }
            try cmd.barrier(.compute, .host);
            try cmd.end();
            try cmd.run(w.timeout_ns);
            const out = std.mem.bytesAsSlice(f32, try act.mapped());
            if (pass == 0) @memcpy(&kr32, out[kr..][0 .. rows * 1024]) else {
                // The FP32 outputs of both variants are the same computation.
                for (kr32, out[kr..][0 .. rows * 1024]) |x, y| try t.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
            }
        }
        const c32 = std.mem.bytesAsSlice(f32, try kv32.mapped());
        const c16 = std.mem.bytesAsSlice(f16, try kv16.mapped());
        var checked: usize = 0;
        var subnormal: usize = 0;
        var ties: usize = 0;
        for (0..rows) |row| for (0..G) |g| for (0..D) |d| {
            const pos = p0 + @as(u32, @intCast(row));
            for ([_]usize{ pg.k(g, d, pos), pg.v(g, d, pos) }) |i| {
                const x = c32[i];
                const want: f16 = @floatCast(x);
                const got = c16[i];
                if (@as(u16, @bitCast(got)) != @as(u16, @bitCast(want))) {
                    std.debug.print("{s} element {d}: f32 {e} ({x}) -> f16 {x}, want {x}\n", .{ @tagName(writer), i, x, @as(u32, @bitCast(x)), @as(u16, @bitCast(got)), @as(u16, @bitCast(want)) });
                    return error.TestUnexpectedResult;
                }
                checked += 1;
                if (want != 0 and @abs(@as(f32, want)) < 6.103515625e-05) subnormal += 1;
                const back: f32 = want;
                if (@abs(x - back) * 2 == @abs(@as(f32, std.math.nextAfter(f16, want, std.math.inf(f16))) - back)) ties += 1;
            }
        };
        // Every written K and V element compared; the edge cases were exercised.
        try t.expectEqual(@as(usize, 2 * rows * G * D), checked);
        try t.expect(subnormal > 0 and ties > 0);
        // Nothing else written: every other element keeps its NaN sentinel (the outputs are
        // finite or infinite, never NaN).
        var written32: usize = 0;
        var written16: usize = 0;
        for (c32[0..kv_elems], c16[0..kv_elems]) |x, y| {
            written32 += @intFromBool(!std.math.isNan(x));
            written16 += @intFromBool(!std.math.isNan(y));
        }
        try t.expectEqual(checked, written32);
        try t.expectEqual(checked, written16);
    }
}
