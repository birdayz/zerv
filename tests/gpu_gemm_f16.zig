//! f16 WMMA prefill GEMM (block 14, gemm_f16.comp; explicit hardware test).
//! 1. Conversion: X -> f16 must round to nearest even. Single positive products (exact on
//!    this hardware's WMMA) make y equal f16(x) bit for bit.
//! 2. Every format (Q4_0, Q4_1, Q5_K), random weights, a row tail (n < tile): results
//!    against an FP64 CPU sum of the *same* f16-rounded inputs,
//!    |y - ref| <= 2^-16 * sum|w x| + 1e-6 (spec bound for the WMMA f32 accumulation);
//!    rows beyond the plan are untouched.
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const gemm = zerv.model.gemm;
const quant = zerv.quant;
const t = std.testing;

const M: u32 = 4096; // f16Eligible: M >= 4096, M % 128 == 0
const rows: u32 = 128; // plan rows (one 128-row tile)
const timeout_ns = 30 * std.time.ns_per_s;

const Run = struct {
    device: *gpu.Device,
    a: gpu.Buffer,
    act: gpu.Buffer,
    io: gpu.Buffer,
    kernel: gpu.Kernel,
    cmd: gpu.Commands,

    fn init(self: *Run, device: *gpu.Device, v: gemm.Variant, a_bytes: u64, act_words: u64) !void {
        self.device = device;
        self.a = try gpu.Buffer.init(device, a_bytes, .host);
        self.act = try gpu.Buffer.init(device, act_words * 4, .host);
        self.io = try gpu.Buffer.init(device, 1024, .host);
        self.kernel = try gpu.Kernel.init(device, try gemm.moduleF16(v), &.{ &self.a, &self.act, &self.io, &self.act }, @sizeOf(gemm.Push));
        self.cmd = try gpu.Commands.init(device);
    }
    fn deinit(self: *Run) void {
        self.cmd.deinit() catch @panic("cmd");
        self.kernel.deinit() catch @panic("kernel");
        self.io.deinit() catch @panic("io");
        self.act.deinit() catch @panic("act");
        self.a.deinit() catch @panic("a");
    }
    fn dispatch(self: *Run, v: gemm.Variant, push: gemm.Push, n: u32) !void {
        const groups = try gemm.validateF16(v, push, .{ .rows = rows }, self.a.size, self.act.size);
        std.mem.bytesAsSlice(u32, try self.io.mapped())[2] = n;
        try self.cmd.reset();
        try self.cmd.begin();
        try self.cmd.barrier(.host, .compute);
        try self.cmd.dispatch(&self.kernel, std.mem.asBytes(&push), groups);
        try self.cmd.barrier(.compute, .host);
        try self.cmd.end();
        try self.cmd.run(timeout_ns);
    }
};

fn half(x: f32) f16 {
    return @floatCast(x); // Zig float casts round to nearest even
}

fn openDevice() !?gpu.Device {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 * 1024, .cooperative_matrix = true });
    if (device.subgroup.size != 64) {
        try device.deinit();
        return null;
    }
    return device;
}

test "f16 GEMM: activations are rounded to nearest even (exact single products)" {
    var device = (try openDevice()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    const K: u32 = 32;
    const x_base: u32 = 16;
    const y_base: u32 = x_base + rows * K + 16;
    var run: Run = undefined;
    try run.init(&device, .q4_0, @as(u64, M) * 18 + 64, y_base + rows * M + 64);
    defer run.deinit();
    // W[m][k] = 1 if k == m % 32 else 0: d = 1.0, q = 9 at that k, 8 elsewhere.
    const a = try run.a.mapped();
    @memset(a, 0);
    for (0..M) |m| {
        const blk = a[m * 18 ..][0..18];
        std.mem.writeInt(u16, blk[0..2], @bitCast(@as(f16, 1.0)), .little);
        for (0..16) |j| {
            const lo: u8 = if (j == m % 32) 9 else 8;
            const hi: u8 = if (j + 16 == m % 32) 9 else 8;
            blk[2 + j] = lo | (hi << 4);
        }
    }
    // Positive x in [1, 2): mantissa low 13 bits cover ties (even and odd f16 targets),
    // just below/above ties, and random values.
    const act = std.mem.bytesAsSlice(f32, try run.act.mapped());
    @memset(act, std.math.nan(f32));
    var prng = std.Random.DefaultPrng.init(0x14f16);
    const patterns = [_]u32{ 0x1000, 0x3000, 0x0fff, 0x1001, 0x2fff, 0x3001, 0x0000, 0x1fff };
    for (0..rows) |r| for (0..K) |k| {
        const base = 0x3f800000 | (prng.random().int(u32) & 0x007fe000);
        const low = if ((r + k) % 3 == 0) prng.random().int(u32) & 0x1fff else patterns[(r * 7 + k) % patterns.len];
        act[x_base + r * K + k] = @bitCast(base | low);
    };
    const tail = act[y_base + rows * M ..][0..64];
    @memset(tail, 12345.0);
    try run.dispatch(.q4_0, .{ .a_base = 0, .a_rs = 18, .x_base = x_base, .x_rs = K, .y_base = y_base, .y_rs = M, .m = M, .k = K }, rows);
    var ties_even: usize = 0;
    for (0..rows) |r| for (0..M) |m| {
        const x = act[x_base + r * K + m % 32];
        const want: f32 = half(x);
        const got = act[y_base + r * M + m];
        if (got != want) {
            std.debug.print("f16 rounding mismatch row={d} m={d} x=0x{x} got {e} want {e}\n", .{ r, m, @as(u32, @bitCast(x)), got, want });
            return error.RoundingMismatch;
        }
        if (@as(u32, @bitCast(x)) & 0x1fff == 0x1000) ties_even += 1;
    };
    try t.expect(ties_even > 0);
    for (tail) |value| try t.expectEqual(@as(f32, 12345.0), value);
}

fn randomBlocks(comptime format: quant.Format, bytes: []u8, random: std.Random) void {
    const width = comptime quant.blockBytes(format);
    random.bytes(bytes);
    var at: usize = 0;
    while (at < bytes.len) : (at += width) {
        const blk = bytes[at..][0..width];
        // Realistic finite scales (the model's d are ~1e-3..1e-2).
        const d: f16 = @floatCast(std.math.ldexp(@as(f32, 1.0) + random.float(f32), -8 + @as(i32, random.intRangeAtMost(u3, 0, 4))));
        std.mem.writeInt(u16, blk[0..2], @bitCast(d), .little);
        if (format == .q4_1 or format == .q5_k) {
            const m: f16 = @floatCast((random.float(f32) - 0.5) * 0.05);
            std.mem.writeInt(u16, blk[2..4], @bitCast(if (format == .q5_k) @abs(m) else m), .little);
        }
    }
}

fn checkFormat(comptime format: quant.Format, v: gemm.Variant, device: *gpu.Device, K: u32, n: u32) !void {
    const width = comptime quant.blockBytes(format);
    const elements = comptime quant.blockElements(format);
    const row_bytes: u32 = K / @as(u32, elements) * @as(u32, width);
    const x_base: u32 = 16;
    const y_base: u32 = x_base + rows * K + 16;
    var run: Run = undefined;
    try run.init(device, v, @as(u64, M) * row_bytes + 64, y_base + rows * M + 64);
    defer run.deinit();
    var prng = std.Random.DefaultPrng.init(0xf16 ^ @as(u64, @intFromEnum(format)));
    const random = prng.random();
    const a = try run.a.mapped();
    @memset(a, 0);
    randomBlocks(format, a[0 .. M * row_bytes], random);
    const act = std.mem.bytesAsSlice(f32, try run.act.mapped());
    @memset(act, std.math.nan(f32));
    for (0..rows) |r| for (0..K) |k| {
        act[x_base + r * K + k] = if (r < n) random.floatNorm(f32) else std.math.inf(f32); // rows >= n must not be read
    };
    const tail = act[y_base + rows * M ..][0..64];
    @memset(tail, 12345.0);
    try run.dispatch(v, .{ .a_base = 0, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y_base, .y_rs = M, .m = M, .k = K }, n);

    const w = try t.allocator.alloc(f32, K);
    defer t.allocator.free(w);
    var worst: f64 = 0;
    for (0..M) |m| {
        try quant.decode(format, a[m * row_bytes ..][0..row_bytes], w);
        for (0..n) |r| {
            var sum: f64 = 0;
            var sumabs: f64 = 0;
            for (0..K) |k| {
                const p = @as(f64, half(w[k])) * @as(f64, half(act[x_base + r * K + k]));
                sum += p;
                sumabs += @abs(p);
            }
            const got: f64 = act[y_base + r * M + m];
            const err = @abs(got - sum);
            const bound = std.math.ldexp(sumabs, -16) + 1e-6;
            if (!(err <= bound)) {
                std.debug.print("f16 GEMM {s} row={d} m={d} got {e} want {e} err {e} bound {e}\n", .{ @tagName(format), r, m, got, sum, err, bound });
                return error.BoundExceeded;
            }
            worst = @max(worst, err / (sumabs + 1e-30));
        }
    }
    for (tail) |value| try t.expectEqual(@as(f32, 12345.0), value);
    std.debug.print("f16 GEMM {s} K={d} n={d}: worst |err|/sum|wx| = {e}\n", .{ @tagName(format), K, n, worst });
}

test "f16 GEMM: Q4_0, Q4_1 and Q5_K against an FP64 sum of the same f16-rounded inputs" {
    var device = (try openDevice()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    try checkFormat(.q4_0, .q4_0, &device, 512, 100);
    try checkFormat(.q4_1, .q4_1, &device, 512, 128);
    try checkFormat(.q5_k, .q5_k, &device, 512, 77);
    // Validation: shapes the f16 kernel does not cover are rejected.
    const ok: gemm.Push = .{ .a_base = 0, .a_rs = 18 * 16, .x_base = 0, .x_rs = 512, .y_base = 1 << 20, .y_rs = M, .m = M, .k = 512 };
    var bad = ok;
    bad.m = 1024;
    try t.expectError(error.InvalidShape, gemm.validateF16(.q4_0, bad, .{ .rows = rows }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validateF16(.q4_0, ok, .{ .rows = 96 }, 1 << 30, 1 << 30));
    bad = ok;
    bad.k_chunk = 256;
    try t.expectError(error.InvalidShape, gemm.validateF16(.q4_0, bad, .{ .rows = rows, .batches = 2 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validateF16(.q6_k, ok, .{ .rows = rows }, 1 << 30, 1 << 30));
    _ = try gemm.validateF16(.q4_0, ok, .{ .rows = rows }, 1 << 30, 1 << 30);
    try t.expectError(error.InvalidRange, gemm.validateF16(.q4_0, ok, .{ .rows = rows }, 1 << 30, (1 << 20) * 4));
}

// ---- gemm_f16n (block 18e, --decode-precision f16): 128 x 16 tile -----------------------
// Spec gate (docs/specs/concurrent.md, "18e design"): every row equals gemm_f16's result for
// the same X row, bit for bit (so a decode row does not depend on its batch), for Q4_0, Q4_1
// and Q5_K, row counts 1..40 (tails and several 16-row tiles), several K; rows past the
// validated span untouched.
fn checkSmallN(comptime format: quant.Format, v: gemm.Variant, device: *gpu.Device, K: u32, n: u32, seed: u64) !void {
    const width = comptime quant.blockBytes(format);
    const elements = comptime quant.blockElements(format);
    const row_bytes: u32 = K / @as(u32, elements) * @as(u32, width);
    const span: u32 = std.mem.alignForward(u32, n, gemm.f16n_tile_n); // rows the small kernel may write
    const x_base: u32 = 16;
    const y_ref: u32 = x_base + rows * K + 16;
    const y_small: u32 = y_ref + rows * M + 64;
    const total: u32 = y_small + span * M + 64;
    var a = try gpu.Buffer.init(device, @as(u64, M) * row_bytes + 64, .host);
    defer a.deinit() catch @panic("a");
    var act = try gpu.Buffer.init(device, @as(u64, total) * 4, .host);
    defer act.deinit() catch @panic("act");
    var io = try gpu.Buffer.init(device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var big = try gpu.Kernel.init(device, try gemm.moduleF16(v), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer big.deinit() catch @panic("big");
    var small = try gpu.Kernel.init(device, try gemm.moduleF16n(v), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer small.deinit() catch @panic("small");
    var cmd = try gpu.Commands.init(device);
    defer cmd.deinit() catch @panic("cmd");
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const wa = try a.mapped();
    @memset(wa, 0);
    randomBlocks(format, wa[0 .. M * row_bytes], random);
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    @memset(words, 12345.0);
    for (0..rows) |r| for (0..K) |k| {
        words[x_base + r * K + k] = if (r < n) random.floatNorm(f32) else std.math.inf(f32);
    };
    std.mem.bytesAsSlice(u32, try io.mapped())[2] = n;
    const p_ref: gemm.Push = .{ .a_base = 0, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y_ref, .y_rs = M, .m = M, .k = K };
    var p_small = p_ref;
    p_small.y_base = y_small;
    const g_ref = try gemm.validateF16(v, p_ref, .{ .rows = rows }, a.size, act.size);
    const g_small = try gemm.validateF16n(v, p_small, .{ .rows = span }, a.size, act.size);
    try t.expectEqual([3]u32{ M / 128, span / 16, 1 }, g_small);
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&big, std.mem.asBytes(&p_ref), g_ref);
    try cmd.dispatch(&small, std.mem.asBytes(&p_small), g_small);
    try cmd.barrier(.compute, .host);
    try cmd.end();
    try cmd.run(timeout_ns);
    const got = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_small..][0 .. n * M]));
    const want = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_ref..][0 .. n * M]));
    for (got, want, 0..) |g, w_, i| if (g != w_) {
        std.debug.print("gemm_f16n {s} K={d} n={d}: row {d} m {d}: 0x{x} != gemm_f16 0x{x}\n", .{ @tagName(format), K, n, i / M, i % M, g, w_ });
        return error.NotBitwiseEqual;
    };
    for (words[total - 64 .. total]) |value| try t.expectEqual(@as(f32, 12345.0), value);
    for (words[y_small - 64 .. y_small]) |value| try t.expectEqual(@as(f32, 12345.0), value);
}

/// Split-K (k_chunk != 0): part z of the split dispatch equals the unsplit kernel over the
/// K range of part z (weights and X offset to its first block), bit for bit; so the split
/// arithmetic is exactly "chains per part, then the parts added in order" (the reduce).
fn checkSmallNSplit(comptime format: quant.Format, v: gemm.Variant, device: *gpu.Device, K: u32, chunk: u32, n: u32, seed: u64) !void {
    const width = comptime quant.blockBytes(format);
    const elements = comptime quant.blockElements(format);
    const row_bytes: u32 = K / @as(u32, elements) * @as(u32, width);
    const span: u32 = std.mem.alignForward(u32, n, gemm.f16n_tile_n);
    const parts: u32 = gemm.splitCount(K, chunk);
    const x_base: u32 = 16;
    const y_split: u32 = x_base + span * K + 64;
    const y_parts: u32 = y_split + parts * span * M + 64;
    const total: u32 = y_parts + parts * span * M + 64;
    var a = try gpu.Buffer.init(device, @as(u64, M) * row_bytes + 64, .host);
    defer a.deinit() catch @panic("a");
    var act = try gpu.Buffer.init(device, @as(u64, total) * 4, .host);
    defer act.deinit() catch @panic("act");
    var io = try gpu.Buffer.init(device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var small = try gpu.Kernel.init(device, try gemm.moduleF16n(v), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer small.deinit() catch @panic("small");
    var cmd = try gpu.Commands.init(device);
    defer cmd.deinit() catch @panic("cmd");
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const wa = try a.mapped();
    @memset(wa, 0);
    randomBlocks(format, wa[0 .. M * row_bytes], random);
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    @memset(words, 12345.0);
    for (0..span) |r| for (0..K) |k| {
        words[x_base + r * K + k] = if (r < n) random.floatNorm(f32) else std.math.inf(f32);
    };
    std.mem.bytesAsSlice(u32, try io.mapped())[2] = n;
    const p_split: gemm.Push = .{ .a_base = 0, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y_split, .y_rs = M, .y_bs = span * M, .m = M, .k = K, .k_chunk = chunk };
    const g_split = try gemm.validateF16n(v, p_split, .{ .rows = span, .batches = parts }, a.size, act.size);
    try t.expectEqual([3]u32{ M / 128, span / 16, parts }, g_split);
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&small, std.mem.asBytes(&p_split), g_split);
    for (0..parts) |z| {
        const kb: u32 = @as(u32, @intCast(z)) * chunk;
        const kl: u32 = @min(K, kb + chunk) - kb;
        const p_part: gemm.Push = .{ .a_base = kb / @as(u32, elements) * @as(u32, width), .a_rs = row_bytes, .x_base = x_base + kb, .x_rs = K, .y_base = y_parts + @as(u32, @intCast(z)) * span * M, .y_rs = M, .m = M, .k = kl };
        // A K sub-range of full-width rows (a_rs > the range's bytes): not a shape the
        // production validator accepts, so this reference dispatch gives its grid directly
        // (every read lies inside the rows validated by the split dispatch above).
        try cmd.dispatch(&small, std.mem.asBytes(&p_part), .{ M / 128, span / 16, 1 });
    }
    try cmd.barrier(.compute, .host);
    try cmd.end();
    try cmd.run(timeout_ns);
    for (0..parts) |z| {
        const got = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_split + z * span * M ..][0 .. n * M]));
        const want = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_parts + z * span * M ..][0 .. n * M]));
        for (got, want, 0..) |g, w_, i| if (g != w_) {
            std.debug.print("gemm_f16n split {s} K={d} chunk={d} part {d}: row {d} m {d}: 0x{x} != 0x{x}\n", .{ @tagName(format), K, chunk, z, i / M, i % M, g, w_ });
            return error.NotBitwiseEqual;
        };
    }
    for (words[total - 64 .. total]) |value| try t.expectEqual(@as(f32, 12345.0), value);
}

test "f16 GEMM n16: every row bitwise equal to gemm_f16 (Q4_0, Q4_1, Q5_K; 1..40 rows)" {
    var device = (try openDevice()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    var checked: u32 = 0;
    for ([_]u32{ 1, 5, 16, 17, 40 }, 0..) |n, i| {
        try checkSmallN(.q4_0, .q4_0, &device, 5120, n, 0x18e0 + i);
        try checkSmallN(.q4_1, .q4_1, &device, 512, n, 0x18e1 + i);
        try checkSmallN(.q5_k, .q5_k, &device, 1024, n, 0x18e2 + i);
        checked += 3;
    }
    // Split-K parts (the decode split rule's chunks and a short last part).
    try checkSmallNSplit(.q4_0, .q4_0, &device, 5120, 1280, 16, 0x18e7);
    try checkSmallNSplit(.q4_0, .q4_0, &device, 17408, gemm.f16nChunk(5120, 17408, 384), 7, 0x18e8);
    try checkSmallNSplit(.q4_1, .q4_1, &device, 1280, 512, 3, 0x18e9);
    try checkSmallNSplit(.q5_k, .q5_k, &device, 1536, 512, 17, 0x18ea);
    try t.expectEqual(@as(u32, 1792), gemm.f16nChunk(5120, 17408, 384)); // 40 tiles: 10 parts
    try t.expectEqual(@as(u32, 0), gemm.f16nChunk(49152, 5120, 384)); // 384 tiles: no split
    // Validation: row spans must be whole 16-row tiles; ineligible shapes are refused.
    const ok: gemm.Push = .{ .a_base = 0, .a_rs = 18 * 16, .x_base = 0, .x_rs = 512, .y_base = 1 << 20, .y_rs = M, .m = M, .k = 512 };
    _ = try gemm.validateF16n(.q4_0, ok, .{ .rows = 16 }, 1 << 30, 1 << 30);
    try t.expectError(error.InvalidShape, gemm.validateF16n(.q4_0, ok, .{ .rows = 8 }, 1 << 30, 1 << 30));
    var bad = ok;
    bad.m = 1024;
    try t.expectError(error.InvalidShape, gemm.validateF16n(.q4_0, bad, .{ .rows = 16 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validateF16n(.q6_k, ok, .{ .rows = 16 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidRange, gemm.validateF16n(.q4_0, ok, .{ .rows = 32 }, 1 << 30, (1 << 20) * 4));
    std.debug.print("gemm_f16n: {d} (format, rows) cases bitwise equal to gemm_f16\n", .{checked});
}

// ---- gemm_f16d (block 18e, decode v2): wave32 streaming tiles -------------------------------
// Spec gate (docs/specs/concurrent.md, "18e design", kernel v2): every row of every split part
// equals gemm_f16n's (v1) bit for bit, same push (the model's split chunks, unsplit, 1..32
// rows, K 512..17408, subnormal-scale weights); words around the output untouched.
fn checkDecodeV2(device: *gpu.Device, M_: u32, K: u32, chunk: u32, n: u32, seed: u64) !void {
    const row_bytes: u32 = K / 32 * 18;
    const span: u32 = std.mem.alignForward(u32, n, gemm.f16n_tile_n);
    const parts: u32 = gemm.splitCount(K, chunk);
    const x_base: u32 = 16;
    const y1: u32 = x_base + span * K + 64;
    const y2: u32 = y1 + parts * span * M_ + 64;
    const total: u32 = y2 + parts * span * M_ + 64;
    var a = try gpu.Buffer.init(device, @as(u64, M_) * row_bytes + 64, .host);
    defer a.deinit() catch @panic("a");
    var act = try gpu.Buffer.init(device, @as(u64, total) * 4, .host);
    defer act.deinit() catch @panic("act");
    var io = try gpu.Buffer.init(device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var v1 = try gpu.Kernel.init(device, try gemm.moduleF16n(.q4_0), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer v1.deinit() catch @panic("v1");
    var v2 = try gpu.Kernel.initWith(device, try gemm.moduleF16d(.q4_0), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push), .{ .subgroup_size = gemm.f16d_subgroup, .full_subgroups = true });
    defer v2.deinit() catch @panic("v2");
    var cmd = try gpu.Commands.init(device);
    defer cmd.deinit() catch @panic("cmd");
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const wa = try a.mapped();
    @memset(wa, 0);
    q4Blocks(wa[0 .. M_ * row_bytes], random);
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    @memset(words, 12345.0);
    for (0..span) |r| for (0..K) |k| {
        words[x_base + r * K + k] = if (r < n) random.floatNorm(f32) else std.math.inf(f32);
    };
    std.mem.bytesAsSlice(u32, try io.mapped())[2] = n;
    const p1: gemm.Push = .{ .a_base = 0, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y1, .y_rs = M_, .y_bs = span * M_, .m = M_, .k = K, .k_chunk = chunk };
    var p2 = p1;
    p2.y_base = y2;
    const lim: gemm.Limits = .{ .rows = span, .batches = parts };
    const g1 = try gemm.validateF16n(.q4_0, p1, lim, a.size, act.size);
    const g2 = try gemm.validateF16d(.q4_0, p2, lim, a.size, act.size);
    try t.expectEqual([3]u32{ M_ / 64, span / 16, parts }, g2);
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&v1, std.mem.asBytes(&p1), g1);
    try cmd.dispatch(&v2, std.mem.asBytes(&p2), g2);
    try cmd.barrier(.compute, .host);
    try cmd.end();
    try cmd.run(timeout_ns);
    for (0..parts) |z| {
        const got = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y2 + z * span * M_ ..][0 .. n * M_]));
        const want = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y1 + z * span * M_ ..][0 .. n * M_]));
        for (got, want, 0..) |g, w_, i| if (g != w_) {
            std.debug.print("gemm_f16d M={d} K={d} chunk={d} n={d} part {d}: row {d} m {d}: 0x{x} != gemm_f16n 0x{x}\n", .{ M_, K, chunk, n, z, i / M_, i % M_, g, w_ });
            return error.NotBitwiseEqual;
        };
    }
    for (words[total - 64 .. total]) |value| try t.expectEqual(@as(f32, 12345.0), value);
    for (words[y2 - 64 .. y2]) |value| try t.expectEqual(@as(f32, 12345.0), value);
}

test "f16 decode v2: every row and split part bitwise equal to gemm_f16n (Q4_0, 1..32 rows)" {
    var device = (try openDeviceX()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    var cases: u32 = 0;
    for ([_]u32{ 1, 3, 8, 16, 17, 32 }, 0..) |n, i| {
        try checkDecodeV2(&device, 5120, 5120, gemm.f16nChunk(5120, 5120, 384), n, 0x18d0 + i);
        try checkDecodeV2(&device, 17408, 5120, gemm.f16nChunk(17408, 5120, 384), n, 0x18d8 + i);
        cases += 2;
    }
    try checkDecodeV2(&device, 5120, 17408, gemm.f16nChunk(5120, 17408, 384), 8, 0x18e0); // ffn_down, 10 parts
    try checkDecodeV2(&device, 6144, 512, 0, 5, 0x18e1); // unsplit
    try checkDecodeV2(&device, 12288, 5120, gemm.f16nChunk(12288, 5120, 384), 8, 0x18e2);
    cases += 3;
    // Selection and validation.
    try t.expect(gemm.f16dEligible(.q4_0, 5120, 17408, 1792));
    try t.expect(!gemm.f16dEligible(.q4_1, 5120, 17408, 1792));
    try t.expect(!gemm.f16dEligible(.q4_0, 5120, 5120 + 32, 0));
    try t.expect(!gemm.f16dEligible(.q4_0, 4096 + 128 * 3 - 64, 5120, 0) or (4096 + 128 * 3 - 64) % 64 == 0);
    const ok: gemm.Push = .{ .a_base = 0, .a_rs = 18 * 16, .x_base = 0, .x_rs = 512, .y_base = 1 << 20, .y_rs = M, .m = M, .k = 512 };
    _ = try gemm.validateF16d(.q4_0, ok, .{ .rows = 16 }, 1 << 30, 1 << 30);
    var bad = ok;
    bad.x_rs = 514;
    try t.expectError(error.InvalidShape, gemm.validateF16d(.q4_0, bad, .{ .rows = 16 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validateF16d(.q4_1, ok, .{ .rows = 16 }, 1 << 30, 1 << 30));
    std.debug.print("gemm_f16d: {d} (shape, rows) cases bitwise equal to gemm_f16n\n", .{cases});
}

// ---- gemm_f16m (block 18c.2, --f16-small-tile): 32 x 128 tile ---------------------------
// Spec gate (docs/specs/concurrent.md, "18c.2 design"): every valid row equals gemm_f16's
// result bit for bit (Q4_0, Q4_1, Q5_K; one and two 128-row tiles, row tails, K 512..5120);
// words around the output untouched.
fn checkSmallM(comptime format: quant.Format, v: gemm.Variant, device: *gpu.Device, K: u32, plan: u32, n: u32, seed: u64) !void {
    const width = comptime quant.blockBytes(format);
    const elements = comptime quant.blockElements(format);
    const row_bytes: u32 = K / @as(u32, elements) * @as(u32, width);
    const x_base: u32 = 16;
    const y_ref: u32 = x_base + plan * K + 16;
    const y_m: u32 = y_ref + plan * M + 64;
    const total: u32 = y_m + plan * M + 64;
    var a = try gpu.Buffer.init(device, @as(u64, M) * row_bytes + 64, .host);
    defer a.deinit() catch @panic("a");
    var act = try gpu.Buffer.init(device, @as(u64, total) * 4, .host);
    defer act.deinit() catch @panic("act");
    var io = try gpu.Buffer.init(device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var big = try gpu.Kernel.init(device, try gemm.moduleF16(v), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer big.deinit() catch @panic("big");
    var small = try gpu.Kernel.init(device, try gemm.moduleF16m(v), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer small.deinit() catch @panic("small");
    var cmd = try gpu.Commands.init(device);
    defer cmd.deinit() catch @panic("cmd");
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const wa = try a.mapped();
    @memset(wa, 0);
    randomBlocks(format, wa[0 .. M * row_bytes], random);
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    @memset(words, 12345.0);
    for (0..plan) |r| for (0..K) |k| {
        words[x_base + r * K + k] = if (r < n) random.floatNorm(f32) else std.math.inf(f32);
    };
    std.mem.bytesAsSlice(u32, try io.mapped())[2] = n;
    const p_ref: gemm.Push = .{ .a_base = 0, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y_ref, .y_rs = M, .m = M, .k = K };
    var p_m = p_ref;
    p_m.y_base = y_m;
    const g_ref = try gemm.validateF16(v, p_ref, .{ .rows = plan }, a.size, act.size);
    const g_m = try gemm.validateF16m(v, p_m, .{ .rows = plan }, a.size, act.size);
    try t.expectEqual([3]u32{ M / gemm.f16m_tile_m, plan / 128, 1 }, g_m);
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&big, std.mem.asBytes(&p_ref), g_ref);
    try cmd.dispatch(&small, std.mem.asBytes(&p_m), g_m);
    try cmd.barrier(.compute, .host);
    try cmd.end();
    try cmd.run(timeout_ns);
    const got = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_m..][0 .. n * M]));
    const want = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_ref..][0 .. n * M]));
    for (got, want, 0..) |g, w_, i| if (g != w_) {
        std.debug.print("gemm_f16m {s} K={d} n={d}: row {d} m {d}: 0x{x} != gemm_f16 0x{x}\n", .{ @tagName(format), K, n, i / M, i % M, g, w_ });
        return error.NotBitwiseEqual;
    };
    for (words[total - 64 .. total]) |value| try t.expectEqual(@as(f32, 12345.0), value);
    for (words[y_m - 64 .. y_m]) |value| try t.expectEqual(@as(f32, 12345.0), value);
}

test "f16 GEMM m32: every row bitwise equal to gemm_f16 (Q4_0, Q4_1, Q5_K; row tails, 1-2 tiles)" {
    var device = (try openDevice()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    try checkSmallM(.q4_0, .q4_0, &device, 5120, 128, 100, 0x18c0);
    try checkSmallM(.q4_0, .q4_0, &device, 512, 128, 128, 0x18c1);
    try checkSmallM(.q4_0, .q4_0, &device, 1024, 256, 200, 0x18c2);
    try checkSmallM(.q4_1, .q4_1, &device, 512, 128, 1, 0x18c3);
    try checkSmallM(.q4_1, .q4_1, &device, 5120, 256, 256, 0x18c4);
    try checkSmallM(.q5_k, .q5_k, &device, 1024, 128, 77, 0x18c5);
    try checkSmallM(.q5_k, .q5_k, &device, 5120, 256, 129, 0x18c6);
    std.debug.print("gemm_f16m: 7 (format, K, rows) cases bitwise equal to gemm_f16\n", .{});
}

test "f16 GEMM m32: selection rule" {
    // Short plans whose 128-tile grid leaves the device idle take the 32-row tile...
    try t.expectEqual(gemm.F16Kernel.m32, gemm.f16KernelFor(.q4_0, 5120, 17408, 128, true).?);
    try t.expectEqual(gemm.F16Kernel.m32, gemm.f16KernelFor(.q4_1, 8064, 5120, 128, true).?);
    try t.expectEqual(gemm.F16Kernel.m32, gemm.f16KernelFor(.q5_k, 5120, 5120, 128, true).?);
    // ...unless the grid is already large, the plan is long, or the knob is off.
    try t.expectEqual(gemm.F16Kernel.x16, gemm.f16KernelFor(.q4_0, 5120, 17408, 256, true).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16KernelFor(.q5_k, 5120, 5120, 256, true).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16KernelFor(.q4_1, 8192, 5120, 128, true).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16KernelFor(.q4_0, 17408, 5120, 128, true).?);
    try t.expectEqual(gemm.F16Kernel.x16, gemm.f16KernelFor(.q4_0, 17408, 5120, 256, true).?);
    try t.expectEqual(gemm.F16Kernel.x16, gemm.f16KernelFor(.q4_0, 5120, 17408, 512, true).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16KernelFor(.q4_0, 5120, 17408, 128, false).?);
    try t.expectEqual(gemm.F16Kernel.x16, gemm.f16KernelFor(.q4_0, 5120, 17408, 256, false).?);
    try t.expect(gemm.f16KernelFor(.q6_k, 5120, 5120, 128, true) == null);
    try t.expect(gemm.f16KernelFor(.q4_0, 5120, 5120, 96, true) == null);
    // Validation is gemm_f16's (whole 128-row tiles, eligible shapes), grid M / 32.
    const ok: gemm.Push = .{ .a_base = 0, .a_rs = 18 * 16, .x_base = 0, .x_rs = 512, .y_base = 1 << 20, .y_rs = M, .m = M, .k = 512 };
    try t.expectEqual([3]u32{ M / 32, 2, 1 }, try gemm.validateF16m(.q4_0, ok, .{ .rows = 256 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validateF16m(.q4_0, ok, .{ .rows = 96 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validateF16m(.q6_k, ok, .{ .rows = 128 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidRange, gemm.validateF16m(.q4_0, ok, .{ .rows = 128 }, 1 << 30, (1 << 20) * 4));
}

// ---- gemm_f16x (block 16b): wave32, 128 x 256 tile, reads an f16 copy of X ----------------
// Spec gates (docs/specs/prefill.md, "f16 GEMM, wave32 kernel with f16 X"):
// 1. bitwise-equal to gemm_f16 (f32 X holding the same values) on random Q4_0 weights,
//    including scales whose f16 products are subnormal, with row tails and several K; the
//    block-14 FP64 bound; rows beyond the plan untouched.
// 2. the f16-mode producers' f16 copy equals f16(y) of their FP32 output bit for bit.

const rows_x: u32 = gemm.f16x_tile_n; // plan rows (one 256-row tile)

/// Device for both f16 kernels, or null (skip) when the features are missing.
fn openDeviceX() !?gpu.Device {
    var device = gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 * 1024, .cooperative_matrix = true, .subgroup_size_control = true }) catch |e| switch (e) {
        error.UnsupportedFeature => return null,
        else => return e,
    };
    if (device.subgroup.size != 64 or !device.full_subgroups or device.subgroup_sizes.min > gemm.f16x_subgroup or device.subgroup_sizes.max < gemm.f16x_subgroup) {
        try device.deinit();
        return null;
    }
    return device;
}

/// Random Q4_0 blocks; the scales cover the model's range plus f16-subnormal products
/// (and subnormal d), large and negative scales.
fn q4Blocks(bytes: []u8, random: std.Random) void {
    random.bytes(bytes);
    var at: usize = 0;
    var i: usize = 0;
    while (at < bytes.len) : ({
        at += 18;
        i += 1;
    }) {
        const exponent: i32 = if (i % 7 == 3) -20 else if (i % 11 == 5) -17 else if (i % 13 == 7) 9 else -8 + @as(i32, random.intRangeAtMost(u3, 0, 4));
        const sign: f32 = if (i % 5 == 1) -1.0 else 1.0;
        const d: f16 = @floatCast(sign * std.math.ldexp(@as(f32, 1.0) + random.float(f32), exponent));
        std.mem.writeInt(u16, bytes[at..][0..2], @bitCast(d), .little);
    }
}

fn checkX(device: *gpu.Device, K: u32, n: u32, seed: u64) !void {
    const row_bytes: u32 = K / 32 * 18;
    const x_base: u32 = 16; // f32 X (gemm_f16), words
    const x16_word: u32 = std.mem.alignForward(u32, x_base + rows_x * K, 64); // f16 X (gemm_f16x)
    const y_ref: u32 = x16_word + rows_x * K / 2 + 64;
    const y_x: u32 = y_ref + rows_x * M + 64;
    const total: u32 = y_x + rows_x * M + 64;
    var a = try gpu.Buffer.init(device, @as(u64, M) * row_bytes + 64, .host);
    defer a.deinit() catch @panic("a");
    var act = try gpu.Buffer.init(device, @as(u64, total) * 4, .host);
    defer act.deinit() catch @panic("act");
    var io = try gpu.Buffer.init(device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var k64 = try gpu.Kernel.init(device, try gemm.moduleF16(.q4_0), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push));
    defer k64.deinit() catch @panic("k64");
    var kx = try gpu.Kernel.initWith(device, try gemm.moduleF16x(.q4_0), &.{ &a, &act, &io, &act }, @sizeOf(gemm.Push), .{ .subgroup_size = gemm.f16x_subgroup, .full_subgroups = true });
    defer kx.deinit() catch @panic("kx");
    var cmd = try gpu.Commands.init(device);
    defer cmd.deinit() catch @panic("cmd");

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const wa = try a.mapped();
    @memset(wa, 0);
    q4Blocks(wa[0 .. M * row_bytes], random);
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    @memset(words, 12345.0);
    const x16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(words[x16_word..][0 .. rows_x * K / 2]));
    for (0..rows_x) |r| for (0..K) |k| {
        // Rows >= n are never read (they read row n - 1): inf would poison the outputs.
        const x: f32 = if (r < n) random.floatNorm(f32) else std.math.inf(f32);
        words[x_base + r * K + k] = x;
        x16[r * K + k] = @bitCast(half(x));
    };
    std.mem.bytesAsSlice(u32, try io.mapped())[2] = n;

    const p_ref: gemm.Push = .{ .a_base = 0, .a_rs = row_bytes, .x_base = x_base, .x_rs = K, .y_base = y_ref, .y_rs = M, .m = M, .k = K };
    var p_x = p_ref;
    p_x.x_base = 2 * x16_word;
    p_x.y_base = y_x;
    const g_ref = try gemm.validateF16(.q4_0, p_ref, .{ .rows = rows_x }, a.size, act.size);
    const g_x = try gemm.validateF16x(.q4_0, p_x, .{ .rows = rows_x }, a.size, act.size);
    try t.expectEqual([3]u32{ M / 128, 1, 1 }, g_x);
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(&k64, std.mem.asBytes(&p_ref), g_ref);
    try cmd.dispatch(&kx, std.mem.asBytes(&p_x), g_x);
    try cmd.barrier(.compute, .host);
    try cmd.end();
    try cmd.run(timeout_ns);

    // Bitwise against gemm_f16 on the valid rows. Rows >= n are plan rows nobody reads:
    // gemm_f16 skips 128-row tiles at or past n, gemm_f16x computes them from row n - 1.
    const got = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_x..][0 .. rows_x * M]));
    const want = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_ref..][0 .. rows_x * M]));
    for (got[0 .. n * M], want[0 .. n * M], 0..) |g, w_, i| if (g != w_) {
        std.debug.print("gemm_f16x K={d} n={d}: row {d} m {d}: 0x{x} != gemm_f16 0x{x}\n", .{ K, n, i / M, i % M, g, w_ });
        return error.NotBitwiseEqual;
    };
    for (words[total - 64 .. total]) |v| try t.expectEqual(@as(f32, 12345.0), v);
    for (words[y_x - 64 .. y_x]) |v| try t.expectEqual(@as(f32, 12345.0), v);

    // FP64 bound (block 14) on a sample: rows 0, n/2, n-1; every 7th m.
    const w = try t.allocator.alloc(f32, K);
    defer t.allocator.free(w);
    var subnormal_weights: usize = 0;
    var m: usize = 0;
    while (m < M) : (m += 7) {
        try quant.decode(.q4_0, wa[m * row_bytes ..][0..row_bytes], w);
        for (w) |value| {
            const h = half(value);
            if (h != 0 and @abs(h) < std.math.floatMin(f16)) subnormal_weights += 1;
        }
        for ([_]u32{ 0, n / 2, n - 1 }) |r| {
            var sum: f64 = 0;
            var sumabs: f64 = 0;
            for (0..K) |k| {
                const p = @as(f64, half(w[k])) * @as(f64, half(words[x_base + r * K + k]));
                sum += p;
                sumabs += @abs(p);
            }
            const y: f64 = @as(f32, @bitCast(got[r * M + m]));
            if (!(@abs(y - sum) <= std.math.ldexp(sumabs, -16) + 1e-6)) {
                std.debug.print("gemm_f16x K={d} row={d} m={d} got {e} want {e}\n", .{ K, r, m, y, sum });
                return error.BoundExceeded;
            }
        }
    }
    try t.expect(subnormal_weights > 0);
    std.debug.print("gemm_f16x K={d} n={d}: bitwise equal to gemm_f16 ({d} sampled subnormal f16 weights)\n", .{ K, n, subnormal_weights });
}

test "f16 GEMM x16: bitwise equal to gemm_f16 (Q4_0, subnormal weights, row tails, K 64..5120)" {
    var device = (try openDeviceX()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    try checkX(&device, 64, rows_x, 0x16b1);
    try checkX(&device, 512, 200, 0x16b2);
    try checkX(&device, 5120, 1, 0x16b3);
    try checkX(&device, 5120, 256, 0x16b4);
}

test "f16 GEMM x16: selection and validation" {
    try t.expectEqual(gemm.F16Kernel.x16, gemm.f16Kernel(.q4_0, 4096, 5120, 512).?);
    try t.expectEqual(gemm.F16Kernel.x16, gemm.f16Kernel(.q4_0, 4096, 5120, 256).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16Kernel(.q4_0, 4096, 5120, 128).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16Kernel(.q4_1, 4096, 5120, 512).?);
    try t.expectEqual(gemm.F16Kernel.wave64, gemm.f16Kernel(.q4_0, 4096, 32, 512).?);
    try t.expect(gemm.f16Kernel(.q4_0, 1024, 5120, 512) == null);
    try t.expect(gemm.f16Kernel(.q6_k, 4096, 5120, 512) == null);
    const ok: gemm.Push = .{ .a_base = 0, .a_rs = 18 * 16, .x_base = 64, .x_rs = 512, .y_base = 1 << 20, .y_rs = M, .m = M, .k = 512 };
    const big: u64 = 1 << 30;
    try t.expectEqual([3]u32{ M / 128, 2, 1 }, try gemm.validateF16x(.q4_0, ok, .{ .rows = 512 }, big, big));
    try t.expectError(error.InvalidShape, gemm.validateF16x(.q4_0, ok, .{ .rows = 128 }, big, big));
    try t.expectError(error.InvalidShape, gemm.validateF16x(.q4_1, ok, .{ .rows = 256 }, big, big));
    var bad = ok;
    bad.x_base = 4;
    try t.expectError(error.InvalidShape, gemm.validateF16x(.q4_0, bad, .{ .rows = 256 }, big, big));
    bad = ok;
    bad.x_rs = 504;
    try t.expectError(error.InvalidShape, gemm.validateF16x(.q4_0, bad, .{ .rows = 256 }, big, big));
    bad = ok;
    bad.k_chunk = 256;
    try t.expectError(error.InvalidShape, gemm.validateF16x(.q4_0, bad, .{ .rows = 256 }, big, big));
    bad = ok;
    bad.a_rs = 18 * 15;
    try t.expectError(error.InvalidShape, gemm.validateF16x(.q4_0, bad, .{ .rows = 256 }, big, big));
    try t.expectError(error.InvalidRange, gemm.validateF16x(.q4_0, ok, .{ .rows = 256 }, @as(u64, M) * 18 * 16 - 4, big));
    // Y ends at word (1 << 20) + 255 * M + M; X (halves) fits in far fewer bytes.
    try t.expectError(error.InvalidRange, gemm.validateF16x(.q4_0, ok, .{ .rows = 256 }, big, ((1 << 20) + 256 * M) * 4 - 4));
    _ = try gemm.validateF16x(.q4_0, ok, .{ .rows = 256 }, big, ((1 << 20) + 256 * M) * 4);
}

test "full-subgroup kernels need a device opened with subgroup size control" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 64 * 1024 * 1024, .cooperative_matrix = true });
    defer device.deinit() catch @panic("device");
    try t.expect(!device.full_subgroups);
    var buffer = try gpu.Buffer.init(&device, 1024, .host);
    defer buffer.deinit() catch @panic("buffer");
    try t.expectError(error.UnsupportedFeature, gpu.Kernel.initWith(&device, try gemm.moduleF16x(.q4_0), &.{ &buffer, &buffer, &buffer, &buffer }, @sizeOf(gemm.Push), .{ .full_subgroups = true }));
}

// Producer pushes: the FP32 kernel's push, then the f16 copy's offsets (model.comp F16OUT).
const NormHPush = extern struct { x: u32, a: u32, sum: u32, y: u32, w: u32, width: u32, stride: u32, flags: u32, eps: f32, y16: u32, stride16: u32 };
const SwigluHPush = extern struct { g: u32, u: u32, y: u32, n: u32, rows_io: u32, y16: u32 };
const GateHPush = extern struct { pregate: u32, qf: u32, gates: u32, gated: u32, gated16: u32 };

/// Every f16 copy element equals f16(y) of the FP32 output at the same index; `span`
/// halves after the copy stay untouched (0xffff sentinel).
fn expectCopy(words: []align(1) const f32, y: u32, halves: []align(1) const u16, count: usize, span: usize) !void {
    for (0..count) |i| {
        const want: u16 = @bitCast(half(words[y + i]));
        if (halves[i] != want) {
            std.debug.print("f16 copy {d}: 0x{x} != f16(0x{x}) = 0x{x}\n", .{ i, halves[i], @as(u32, @bitCast(words[y + i])), want });
            return error.CopyMismatch;
        }
    }
    for (halves[count..][0..span]) |h| try t.expectEqual(@as(u16, 0xffff), h);
}

test "f16-mode producers: the f16 copy is f16(y) of the FP32 output, bit for bit" {
    var device = (try openDeviceX()) orelse return error.SkipZigTest;
    defer device.deinit() catch @panic("device");
    const model = zerv.model;
    const H: u32 = model.config.hidden;
    const words_total: u32 = 1 << 20;
    var params = try gpu.Buffer.init(&device, 64 * 1024, .host);
    defer params.deinit() catch @panic("params");
    var act = try gpu.Buffer.init(&device, @as(u64, words_total) * 4, .host);
    defer act.deinit() catch @panic("act");
    var state = try gpu.Buffer.init(&device, 1024, .host);
    defer state.deinit() catch @panic("state");
    var io = try gpu.Buffer.init(&device, 1024, .host);
    defer io.deinit() catch @panic("io");
    var prng = std.Random.DefaultPrng.init(0x16b5);
    const random = prng.random();
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    const ios = std.mem.bytesAsSlice(u32, try io.mapped());
    const copy_word: u32 = 1 << 19; // f16 copies start here (halves 2 * copy_word)

    // norm_h: 3 rows (io count), with the residual add; stride16 = H.
    {
        var k = try gpu.Kernel.init(&device, model.f16ProducerModule(.norm), &.{ &params, &act, &state, &io }, @sizeOf(NormHPush));
        defer k.deinit() catch @panic("norm_h");
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        const pw = std.mem.bytesAsSlice(f32, try params.mapped());
        for (pw[0..H]) |*v| v.* = 0.5 + random.float(f32);
        for (words) |*v| v.* = random.floatNorm(f32);
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(words[copy_word..]));
        @memset(halves[0 .. 4 * H], 0xffff);
        ios[2] = 3;
        const push: NormHPush = .{ .x = 0, .a = 4 * H, .sum = 8 * H, .y = 12 * H, .w = 0, .width = H, .stride = H, .flags = 1 | 2, .eps = 1e-6, .y16 = 2 * copy_word, .stride16 = H };
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        try cmd.dispatch(&k, std.mem.asBytes(&push), .{ 4, 1, 1 }); // row 3 >= count: no output
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(timeout_ns);
        try expectCopy(words, 12 * H, halves, 3 * H, H);
        // FP32 output against an FP64 reference.
        for (0..3) |r| {
            var ss: f64 = 0;
            for (0..H) |i| {
                const v: f64 = @as(f64, words[r * H + i]) + words[4 * H + r * H + i];
                ss += v * v;
            }
            const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(H)) + 1e-6);
            for (0..H) |i| {
                const v: f64 = @as(f64, words[r * H + i]) + words[4 * H + r * H + i];
                const want = v * inv * pw[i];
                try t.expect(@abs(words[12 * H + r * H + i] - want) <= 1e-5 * (@abs(want) + 1e-3));
            }
        }
    }
    // swiglu_h: 3 rows of n = 1000 (rows_io), y16 at an odd half offset.
    {
        var k = try gpu.Kernel.init(&device, model.f16ProducerModule(.swiglu), &.{ &params, &act, &state, &io }, @sizeOf(SwigluHPush));
        defer k.deinit() catch @panic("swiglu_h");
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        for (words[0..copy_word]) |*v| v.* = random.floatNorm(f32) * 4;
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(words[copy_word..]));
        @memset(halves[0..4000], 0xffff);
        ios[2] = 3;
        const push: SwigluHPush = .{ .g = 0, .u = 4096, .y = 8192, .n = 1000, .rows_io = 1, .y16 = 2 * copy_word + 1 };
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        try cmd.dispatch(&k, std.mem.asBytes(&push), .{ 12, 1, 1 });
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(timeout_ns);
        try t.expectEqual(@as(u16, 0xffff), halves[0]);
        try expectCopy(words, 8192, halves[1..], 3000, 500);
        for (0..3000) |i| {
            const g: f64 = words[i];
            const want = g / (1 + @exp(-g)) * words[4096 + i];
            try t.expect(@abs(words[8192 + i] - want) <= 1e-5 * (@abs(want) + 1e-3));
        }
    }
    // gate_h: 2 rows (io count) of 6144.
    {
        var k = try gpu.Kernel.init(&device, model.f16ProducerModule(.gate), &.{ &params, &act, &state, &io }, @sizeOf(GateHPush));
        defer k.deinit() catch @panic("gate_h");
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        for (words[0..copy_word]) |*v| v.* = random.floatNorm(f32) * 3;
        const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(words[copy_word..]));
        @memset(halves[0 .. 3 * 6144], 0xffff);
        ios[2] = 2;
        const push: GateHPush = .{ .pregate = 0, .qf = 16384, .gates = 3 * 16384, .gated = 4 * 16384, .gated16 = 2 * copy_word };
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        try cmd.dispatch(&k, std.mem.asBytes(&push), .{ 48, 1, 1 });
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(timeout_ns);
        try expectCopy(words, 4 * 16384, halves, 2 * 6144, 6144);
        for (0..2 * 6144) |i| {
            const r = i / 6144;
            const c = i % 6144;
            const q: f64 = words[16384 + r * 12288 + (c / 256) * 512 + 256 + c % 256];
            const want = words[i] / (1 + @exp(-q));
            try t.expect(@abs(words[4 * 16384 + i] - want) <= 1e-5 * (@abs(want) + 1e-3));
        }
    }
}

// ---- native gemm_f16x machine code (docs/specs/prefill.md, "Native gemm_f16x machine code") --
// Gate 2: the kernel created from our pipeline binary equals the SPIR-V kernel bit for bit on
// every plan row (rows >= n included, they read row n - 1); a wrong global key falls back to
// SPIR-V; malformed binaries are rejected before any driver call.

fn nativeBinary() gpu.Kernel.Binary {
    const n = gemm.nativeF16x(.q4_0).?;
    return .{ .data = n.data, .key = n.key, .global_key = n.global_key };
}

fn checkNative(device: *gpu.Device, native: *gpu.Kernel, spirv: *gpu.Kernel, a: *gpu.Buffer, act: *gpu.Buffer, io: *gpu.Buffer, K: u32, n: u32, a_base: u32, seed: u64) !void {
    const row_bytes: u32 = K / 32 * 18;
    const x16_word: u32 = 64;
    const y_s: u32 = x16_word + rows_x * K / 2 + 64;
    const y_n: u32 = y_s + rows_x * M + 64;
    const total: u32 = y_n + rows_x * M + 64;
    if (@as(u64, total) * 4 > act.size or @as(u64, M) * row_bytes + a_base > a.size) return error.TestBuffersTooSmall;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const wa = try a.mapped();
    random.bytes(wa);
    q4Blocks(wa[a_base..][0 .. M * row_bytes], random);
    const words = std.mem.bytesAsSlice(f32, try act.mapped());
    @memset(words[0..total], 12345.0);
    const x16 = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(words[x16_word..][0 .. rows_x * K / 2]));
    for (0..rows_x) |r| for (0..K) |k| {
        x16[r * K + k] = @bitCast(half(if (r < n) random.floatNorm(f32) else std.math.inf(f32)));
    };
    std.mem.bytesAsSlice(u32, try io.mapped())[2] = n;
    const p_s: gemm.Push = .{ .a_base = a_base, .a_rs = row_bytes, .x_base = 2 * x16_word, .x_rs = K, .y_base = y_s, .y_rs = M, .m = M, .k = K };
    var p_n = p_s;
    p_n.y_base = y_n;
    const groups = try gemm.validateF16x(.q4_0, p_s, .{ .rows = rows_x }, a.size, act.size);
    var cmd = try gpu.Commands.init(device);
    defer cmd.deinit() catch @panic("cmd");
    try cmd.begin();
    try cmd.barrier(.host, .compute);
    try cmd.dispatch(spirv, std.mem.asBytes(&p_s), groups);
    try cmd.dispatch(native, std.mem.asBytes(&p_n), groups);
    try cmd.barrier(.compute, .host);
    try cmd.end();
    try cmd.run(timeout_ns);
    const got = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_n..][0 .. rows_x * M]));
    const want = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(words[y_s..][0 .. rows_x * M]));
    for (got, want, 0..) |g, w_, i| if (g != w_) {
        std.debug.print("native gemm_f16x K={d} n={d} a_base={d}: row {d} m {d}: 0x{x} != SPIR-V 0x{x}\n", .{ K, n, a_base, i / M, i % M, g, w_ });
        return error.NotBitwiseEqual;
    };
    for (words[total - 64 .. total]) |v| try t.expectEqual(@as(f32, 12345.0), v);
    for (words[y_n - 64 .. y_n]) |v| try t.expectEqual(@as(f32, 12345.0), v);
}

test "native gemm_f16x: bitwise equal to the SPIR-V kernel; fallback on another driver key" {
    var device = gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 * 1024, .cooperative_matrix = true, .subgroup_size_control = true, .pipeline_binaries = true }) catch |e| switch (e) {
        error.UnsupportedFeature => return error.SkipZigTest,
        else => return e,
    };
    defer device.deinit() catch @panic("device");
    if (!device.full_subgroups or device.subgroup_sizes.min > gemm.f16x_subgroup or device.subgroup_sizes.max < gemm.f16x_subgroup) return error.SkipZigTest;
    const binary = nativeBinary();
    const key = device.pipeline_key orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, &key, binary.global_key)) {
        std.debug.print("native gemm_f16x: driver key differs from the binary's (another Mesa build?); skipped\n", .{});
        return error.SkipZigTest;
    }
    var a = try gpu.Buffer.init(&device, @as(u64, M) * (5120 / 32 * 18) + 64, .host);
    defer a.deinit() catch @panic("a");
    var act = try gpu.Buffer.init(&device, (64 + rows_x * 5120 / 2 + 2 * rows_x * M + 256) * 4, .host);
    defer act.deinit() catch @panic("act");
    var io = try gpu.Buffer.init(&device, 1024, .host);
    defer io.deinit() catch @panic("io");
    const buffers = [_]*gpu.Buffer{ &a, &act, &io, &act };
    const opts: gpu.Kernel.Options = .{ .subgroup_size = gemm.f16x_subgroup, .full_subgroups = true };
    var spirv = try gpu.Kernel.initWith(&device, try gemm.moduleF16x(.q4_0), &buffers, @sizeOf(gemm.Push), opts);
    defer spirv.deinit() catch @panic("spirv");
    var with_binary = opts;
    with_binary.binary = binary;
    var native = try gpu.Kernel.initWith(&device, try gemm.moduleF16x(.q4_0), &buffers, @sizeOf(gemm.Push), with_binary);
    defer native.deinit() catch @panic("native");
    try t.expect(!spirv.native);
    try t.expect(native.native);
    try checkNative(&device, &native, &spirv, &a, &act, &io, 64, rows_x, 0, 0x5a1);
    try checkNative(&device, &native, &spirv, &a, &act, &io, 192, 200, 2, 0x5a2);
    try checkNative(&device, &native, &spirv, &a, &act, &io, 512, 1, 0, 0x5a3);
    try checkNative(&device, &native, &spirv, &a, &act, &io, 5120, rows_x, 2, 0x5a4);
    try checkNative(&device, &native, &spirv, &a, &act, &io, 5120, 129, 0, 0x5a5);

    // Another driver's key: the SPIR-V kernel, same results.
    var other_key = binary.global_key.*;
    other_key[7] ^= 0x40;
    var mismatch = binary;
    mismatch.global_key = &other_key;
    var fallback_opts = opts;
    fallback_opts.binary = mismatch;
    var fallback = try gpu.Kernel.initWith(&device, try gemm.moduleF16x(.q4_0), &buffers, @sizeOf(gemm.Push), fallback_opts);
    defer fallback.deinit() catch @panic("fallback");
    try t.expect(!fallback.native);
    try checkNative(&device, &fallback, &spirv, &a, &act, &io, 512, 300 % rows_x, 2, 0x5a6);

    // Malformed binaries: rejected before any driver call.
    const kernels = device.kernels;
    for ([_]gpu.Kernel.Binary{
        .{ .data = binary.data[0..0], .key = binary.key, .global_key = binary.global_key },
        .{ .data = binary.data, .key = binary.key[0..0], .global_key = binary.global_key },
        .{ .data = binary.data, .key = &([_]u8{1} ** 33), .global_key = binary.global_key },
    }) |bad| {
        var bad_opts = opts;
        bad_opts.binary = bad;
        try t.expectError(error.InvalidShader, gpu.Kernel.initWith(&device, try gemm.moduleF16x(.q4_0), &buffers, @sizeOf(gemm.Push), bad_opts));
    }
    try t.expectEqual(kernels, device.kernels);
    std.debug.print("native gemm_f16x: bitwise equal to SPIR-V on 5 cases; key mismatch falls back\n", .{});
}

test "native gemm_f16x: a device without pipeline binaries uses the SPIR-V" {
    var device = gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024, .cooperative_matrix = true, .subgroup_size_control = true }) catch |e| switch (e) {
        error.UnsupportedFeature => return error.SkipZigTest,
        else => return e,
    };
    defer device.deinit() catch @panic("device");
    if (!device.full_subgroups) return error.SkipZigTest;
    try t.expect(device.pipeline_key == null);
    var buffer = try gpu.Buffer.init(&device, 4096, .device);
    defer buffer.deinit() catch @panic("buffer");
    var k = try gpu.Kernel.initWith(&device, try gemm.moduleF16x(.q4_0), &.{ &buffer, &buffer, &buffer, &buffer }, @sizeOf(gemm.Push), .{ .subgroup_size = gemm.f16x_subgroup, .full_subgroups = true, .binary = nativeBinary() });
    defer k.deinit() catch @panic("kernel");
    try t.expect(!k.native);
}
