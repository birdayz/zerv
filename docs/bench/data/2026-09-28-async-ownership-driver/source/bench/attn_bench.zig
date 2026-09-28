//! Prefill attention component benchmark (docs/research/2026-09-28-prefill-attention.md):
//! times one attention layer (24 query heads, 4 KV heads, head dim 256, f16 KV cache,
//! 128-token pages, identity page table) for `rows` query rows at positions p0.. with random
//! data, sustained (warm-up, then batches of dispatches between fences).
//! Usage: zerv-attn-bench SPV fp32|wmma6|wmma3 ROWS P0 [REPS] [BATCH] [WARM_S]
//!   fp32: flash.comp's f16-KV module layout (8 rows x 6 heads per workgroup);
//!   wmmaH: flash_w.comp built with HEADS=H (16 rows x H heads per workgroup, wave32).
//! Prints one JSON line: median/min ms per dispatch and TFLOP/s (4 * rows * keys * 256 * 24,
//! keys = p0 + rows / 2 on average).
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;

const H = 24;
const G = 4;
const D = 256;
const PAGE = 128;

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) return error.Usage;
    const code_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], a, .limited(1 << 24));
    const code = try a.alignedAlloc(u8, .@"4", code_bytes.len);
    @memcpy(code, code_bytes);
    const kind = args[2];
    const wmma_heads: u32 = if (std.mem.eql(u8, kind, "wmma6")) 6 else if (std.mem.eql(u8, kind, "wmma3")) 3 else if (std.mem.eql(u8, kind, "fp32")) 0 else return error.Usage;
    const rows = try std.fmt.parseInt(u32, args[3], 10);
    const p0 = try std.fmt.parseInt(u32, args[4], 10);
    const reps = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else 21;
    const batch = if (args.len > 6) try std.fmt.parseInt(u32, args[6], 10) else 5;
    const warm_s = if (args.len > 7) try std.fmt.parseInt(u32, args[7], 10) else 5;
    const ctx = std.mem.alignForward(u32, p0 + rows, PAGE);
    const pages = ctx / PAGE;

    var device = try gpu.Device.open(.{ .max_allocated_bytes = 8 << 30, .storage16 = true, .cooperative_matrix = wmma_heads != 0, .subgroup_size_control = wmma_heads != 0 });
    defer device.deinit() catch @panic("live resources");
    // act: Q [rows][6144] f32 at 0, O at qr + rows * 6144, the page table after it.
    const qr: u32 = 0;
    const out: u32 = rows * 6144;
    const ptab: u32 = out + rows * 6144;
    var params = try gpu.Buffer.init(&device, 256, .device);
    defer params.deinit() catch @panic("params");
    var act = try gpu.Buffer.init(&device, (@as(u64, ptab) + pages) * 4, .device);
    defer act.deinit() catch @panic("act");
    const pstride: u32 = 2 * G * D * PAGE; // halves per page (K then V)
    var kv = try gpu.Buffer.init(&device, @as(u64, pstride) * pages * 2, .device);
    defer kv.deinit() catch @panic("kv");
    var iob = try gpu.Buffer.init(&device, 1024, .host);
    defer iob.deinit() catch @panic("io");
    // Random contents through a host staging buffer.
    {
        const total = @max(act.size, kv.size);
        var st = try gpu.Buffer.init(&device, total, .host);
        defer st.deinit() catch @panic("staging");
        var prng = std.Random.DefaultPrng.init(0x16c);
        const r = prng.random();
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        const f: []align(1) f32 = @ptrCast(try st.mapped());
        for (f[0..ptab]) |*x| x.* = r.float(f32) * 6 - 3;
        const tab: []align(1) u32 = @ptrCast(f[ptab..][0..pages]);
        for (tab, 0..) |*x, i| x.* = @intCast(i);
        try cmd.begin();
        try cmd.copy(&st, 0, &act, 0, act.size);
        try cmd.end();
        try cmd.run(10 * std.time.ns_per_s);
        const hh: []align(1) f16 = @ptrCast(try st.mapped());
        for (hh[0 .. kv.size / 2]) |*x| x.* = @floatCast(r.float(f32) * 2 - 1);
        try cmd.reset();
        try cmd.begin();
        try cmd.copy(&st, 0, &kv, 0, kv.size);
        try cmd.end();
        try cmd.run(10 * std.time.ns_per_s);
    }
    const words: []align(1) u32 = @ptrCast(try iob.mapped());
    words[2] = rows;
    words[3] = p0;
    const Push = extern struct { qr: u32, kcache: u32, vcache: u32, out: u32, ctx: u32, scale: f32, ptab: u32, pstride: u32, slots: u32 };
    const constants = [_]u32{PAGE};
    var kernel = try gpu.Kernel.initWith(&device, code, &.{ &params, &act, &kv, &iob }, @sizeOf(Push), if (wmma_heads != 0) .{ .constants = &constants, .subgroup_size = 32, .full_subgroups = true } else .{ .constants = &constants });
    defer kernel.deinit() catch @panic("kernel");
    const push: Push = .{ .qr = qr, .kcache = 0, .vcache = G * D * PAGE, .out = out, .ctx = ctx, .scale = 1.0 / 16.0, .ptab = ptab, .pstride = pstride, .slots = 0 };
    const groups: [3]u32 = if (wmma_heads != 0) .{ (rows + 15) / 16, H / wmma_heads, 1 } else .{ (rows + 7) / 8, G, 1 };
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");
    try cmd.begin();
    for (0..batch) |_| {
        try cmd.dispatch(&kernel, std.mem.asBytes(&push), groups);
        try cmd.barrier(.compute, .compute);
    }
    try cmd.end();
    // Warm-up (clocks to steady state), then timed reps.
    const t_warm = std.Io.Clock.awake.now(io).nanoseconds;
    while (std.Io.Clock.awake.now(io).nanoseconds - t_warm < @as(i96, warm_s) * std.time.ns_per_s) try cmd.run(60 * std.time.ns_per_s);
    const samples = try a.alloc(f64, reps);
    for (samples) |*s| {
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        try cmd.run(60 * std.time.ns_per_s);
        s.* = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6 / @as(f64, @floatFromInt(batch));
    }
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    const med = samples[samples.len / 2];
    const keys = @as(f64, @floatFromInt(p0)) + @as(f64, @floatFromInt(rows)) / 2;
    const flops = 4 * @as(f64, @floatFromInt(rows)) * keys * D * H;
    var buf: [512]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print("{{\"spv\":\"{s}\",\"kind\":\"{s}\",\"rows\":{d},\"p0\":{d},\"ms_median\":{d:.4},\"ms_min\":{d:.4},\"tflops\":{d:.2}}}\n", .{ args[1], kind, rows, p0, med, samples[0], flops / med / 1e9 });
    try w.interface.flush();
}
