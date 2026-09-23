//! Research probe (block 13g): runs D = A*B + C for N independent 16x16x16 cases on the
//! device's cooperative-matrix (WMMA) path. Numerics analysis happens elsewhere
//! (bench/coopmat_numerics.py).
//! Usage: zerv-coopmat-probe SPV INPUT OUTPUT [i8]
//!   INPUT: u32 N, then N*256 f16 A (row-major), N*256 f16 B (column-major),
//!          N*256 f32 C (row-major). With `i8`: int8 A/B and int32 C.
//!   OUTPUT: N*256 f32 (or int32) D (row-major).
//! Usage: zerv-coopmat-probe peak SPV GROUPS ITERS CHAINS SAMPLES
//!   Times bench/coopmat/peak.comp; prints one JSON line (median TFLOP/s counting
//!   2*16*16*16 per matrix op).
const std = @import("std");
const gpu = @import("zerv").gpu;
const w = @import("matvec_workload");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 7 and std.mem.eql(u8, args[1], "peak")) return peak(a, io, args[2..]);
    if (args.len != 4 and !(args.len == 5 and std.mem.eql(u8, args[4], "i8"))) return error.Usage;
    const element_bytes: usize = if (args.len == 5) 1 else 2;
    const code_raw = try std.Io.Dir.cwd().readFileAlloc(io, args[1], a, .limited(1 << 20));
    const code = try a.alignedAlloc(u8, .@"4", code_raw.len);
    @memcpy(code, code_raw);
    const input = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 30));
    if (input.len < 4) return error.InvalidInput;
    const n = std.mem.readInt(u32, input[0..4], .little);
    if (n == 0 or n > 65535) return error.InvalidInput;
    const half_bytes = @as(usize, n) * 256 * element_bytes;
    const float_bytes = @as(usize, n) * 256 * 4;
    if (input.len != 4 + 2 * half_bytes + float_bytes) return error.InvalidInput;

    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1 << 30, .cooperative_matrix = true });
    defer device.deinit() catch @panic("live");
    var ba = try gpu.Buffer.init(&device, half_bytes, .host);
    defer ba.deinit() catch @panic("a");
    var bb = try gpu.Buffer.init(&device, half_bytes, .host);
    defer bb.deinit() catch @panic("b");
    var bc = try gpu.Buffer.init(&device, float_bytes, .host);
    defer bc.deinit() catch @panic("c");
    var bd = try gpu.Buffer.init(&device, float_bytes, .host);
    defer bd.deinit() catch @panic("d");
    @memcpy(try ba.mapped(), input[4..][0..half_bytes]);
    @memcpy(try bb.mapped(), input[4 + half_bytes ..][0..half_bytes]);
    @memcpy(try bc.mapped(), input[4 + 2 * half_bytes ..][0..float_bytes]);
    @memset(try bd.mapped(), 0xff);
    var kernel = try gpu.Kernel.init(&device, code, &.{ &ba, &bb, &bc, &bd }, 0);
    defer kernel.deinit() catch @panic("k");
    var commands = try gpu.Commands.init(&device);
    defer commands.deinit() catch @panic("c");
    try commands.begin();
    try commands.dispatch(&kernel, &.{}, .{ n, 1, 1 });
    try commands.barrier(.compute, .host);
    try commands.end();
    try commands.run(60 * std.time.ns_per_s);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = try bd.mapped() });
}

fn peak(a: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const code_raw = try std.Io.Dir.cwd().readFileAlloc(io, args[0], a, .limited(1 << 20));
    const code = try a.alignedAlloc(u8, .@"4", code_raw.len);
    @memcpy(code, code_raw);
    const groups = try std.fmt.parseInt(u32, args[1], 10);
    const iters = try std.fmt.parseInt(u32, args[2], 10);
    const chains = try std.fmt.parseInt(u32, args[3], 10);
    const sample_count = try std.fmt.parseInt(usize, args[4], 10);
    if (groups == 0 or iters == 0 or chains == 0 or sample_count == 0) return error.Usage;
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1 << 30, .cooperative_matrix = true });
    defer device.deinit() catch @panic("live");
    var input = try gpu.Buffer.init(&device, 4096, .host);
    defer input.deinit() catch @panic("in");
    // Small nonzero values that stay finite under repeated accumulation.
    const words: []align(1) u32 = @ptrCast(try input.mapped());
    for (words, 0..) |*word, i| word.* = if (i < 64) 0x3c003c00 else 0x01010101;
    for (words[0..64]) |*word| word.* = 0x3c003c00; // f16 1.0 pairs / f32 bit pattern ~0.0078
    var output = try gpu.Buffer.init(&device, @as(u64, groups) * 1024, .device);
    defer output.deinit() catch @panic("out");
    var kernel = try gpu.Kernel.init(&device, code, &.{ &input, &output }, 4);
    defer kernel.deinit() catch @panic("k");
    var timing = try w.Timing.init(&device);
    defer timing.deinit() catch @panic("timing");
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");
    try cmd.begin();
    try timing.begin(&cmd);
    try cmd.dispatch(&kernel, std.mem.asBytes(&iters), .{ groups, 1, 1 });
    try timing.end();
    try cmd.end();
    for (0..3) |_| try cmd.run(w.timeout_ns);
    const samples = try a.alloc(f64, sample_count);
    for (samples) |*sample| {
        const start = std.Io.Clock.awake.now(io);
        try cmd.run(w.timeout_ns);
        sample.* = try timing.read(@floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds));
    }
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    const median = samples[samples.len / 2];
    const flops = 2.0 * 16 * 16 * 16 * @as(f64, @floatFromInt(groups)) * @as(f64, @floatFromInt(iters)) * @as(f64, @floatFromInt(chains));
    var buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buffer);
    try std.json.Stringify.value(.{ .spv = args[0], .groups = groups, .iters = iters, .chains = chains, .samples = sample_count, .gpu_ns_median = median, .gpu_ns_min = samples[0], .tflops = flops / median / 1e3 }, .{}, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}
