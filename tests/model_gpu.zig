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
        const v = gemm.variant(shape.format);
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
    const att = zerv.model.attention;
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
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
    const act_words: u64 = gated + H * D;
    const kcache: u32 = 0;
    const vcache: u32 = G * D * ctx;
    const state_words: u64 = 2 * @as(u64, G) * D * ctx;

    var params = try gpu.Buffer.init(&device, 256, .host);
    defer params.deinit() catch @panic("params");
    var act = try gpu.Buffer.init(&device, act_words * 4, .host);
    defer act.deinit() catch @panic("act");
    var state = try gpu.Buffer.init(&device, state_words * 4, .host);
    defer state.deinit() catch @panic("state");
    var io = try gpu.Buffer.init(&device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var kernels: [3]gpu.Kernel = undefined;
    for (&kernels, [_]att.Pass{ .scores, .pv, .combine }) |*k, pass| k.* = try gpu.Kernel.init(&device, att.module(pass), &.{ &params, &act, &state, &io }, att.pushBytes(pass));
    defer for (&kernels) |*k| k.deinit() catch @panic("kernel");
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&kernels[0], std.mem.asBytes(&att.ScoresPush{ .qr = qr, .scores = scores, .amax = amax, .kcache = kcache, .ctx = ctx, .chunks = chunks, .scale = 1.0 / 16.0 }), att.groups(.scores, chunks));
    try cmd.barrier(.compute, .compute);
    try cmd.dispatch(&kernels[1], std.mem.asBytes(&att.PvPush{ .scores = scores, .amax = amax, .apart = apart, .asum = asum, .vcache = vcache, .ctx = ctx, .chunks = chunks }), att.groups(.pv, chunks));
    try cmd.barrier(.compute, .compute);
    try cmd.dispatch(&kernels[2], std.mem.asBytes(&att.CombinePush{ .apart = apart, .asum = asum, .qf = qf, .pregate = pregate, .gates = gates, .gated = gated, .chunks = chunks }), att.groups(.combine, chunks));
    try cmd.barrier(.compute, .host);
    try cmd.end();

    var prng = std.Random.DefaultPrng.init(1313);
    const r = prng.random();
    const K = try t.allocator.alloc(f32, G * D * ctx); // [g][d][j]
    defer t.allocator.free(K);
    const V = try t.allocator.alloc(f32, G * ctx * D); // [g][j][d]
    defer t.allocator.free(V);
    for (K) |*v| v.* = r.float(f32) * 2 - 1;
    for (V) |*v| v.* = r.float(f32) * 2 - 1;
    const u: f64 = std.math.ldexp(@as(f64, 1), -24);
    const gamma256 = 256 * u / (1 - 256 * u);
    const huge: f32 = 1e30;
    const s64 = try t.allocator.alloc(f64, ctx);
    defer t.allocator.free(s64);
    var worst: f64 = 0;
    for ([_]u32{ 0, 1, 62, 63, 64, 65, 127, 1000, 4095 }) |pos| {
        const a = std.mem.bytesAsSlice(f32, try act.mapped());
        for (a[0..qf]) |*v| v.* = r.float(f32) * 6 - 3;
        for (a[qf..scores]) |*v| v.* = r.float(f32) * 8 - 4;
        for (a[scores..]) |*v| v.* = std.math.nan(f32); // scratch/outputs: stale reads would show
        const st = std.mem.bytesAsSlice(f32, try state.mapped());
        for (0..G) |g| for (0..D) |d| for (0..ctx) |j| {
            st[kcache + (g * D + d) * ctx + j] = if (j <= pos) K[(g * D + d) * ctx + j] else huge;
            st[vcache + (g * ctx + j) * D + d] = if (j <= pos) V[(g * ctx + j) * D + d] else huge;
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
}
