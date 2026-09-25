//! Research microbenchmark: GPU time of a chain of dependent single-kernel phases, as in
//! the decode step (dispatch, barrier, dispatch, ...). Uses the shipped model shaders read
//! from src/model/shaders at run time. Prints one JSON line per variant:
//!   norm_add: RMS norm with residual add over 5120 values (1 workgroup), with barriers
//!   norm_add_nobarrier: the same dispatches without barriers (overlap allowed)
//!   zero_link: an empty dispatch (zero kernel, count 0) with barriers = link overhead
//!   rowsio_{host,device}_{1,96}: the norm gated on io[IO_COUNT] = 1 (ROWS_IO, as the model's
//!     row-count-gated kernels), io in host-visible (as the model) or device-local memory,
//!     1 or 96 workgroups (95 read io and return): the cost of the io read at dispatch start
//! Usage: zerv-kernel-chain [N [NORM_SPV]]  (default 129 = norms per decode step)
//!        zerv-kernel-chain compare SPV_A SPV_B ROWS   (bitwise norm output comparison
//!        on ROWS random rows with residual add; prints the differing rows)
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const w = @import("matvec_workload");

const NormPush = extern struct { x: u32, a: u32, sum: u32, y: u32, w: u32, width: u32 = 5120, stride: u32, flags: u32, eps: f32 };
const ZeroPush = extern struct { first: u32, count: u32 };

fn spv(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]align(4) u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 * 1024 * 1024));
    const aligned = try a.alignedAlloc(u8, .@"4", bytes.len);
    @memcpy(aligned, bytes);
    return aligned;
}

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 5 and std.mem.eql(u8, args[1], "compare")) return compare(a, io, args[2], args[3], try std.fmt.parseInt(u32, args[4], 10));
    const n: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 129;
    if (n == 0 or n > 4000) return error.Usage;
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live");
    // Device-local like the model's arenas (host-visible memory would be read over PCIe).
    var params = try gpu.Buffer.init(&device, 1 << 20, .device);
    defer params.deinit() catch @panic("p");
    var act = try gpu.Buffer.init(&device, 1 << 20, .device);
    defer act.deinit() catch @panic("a");
    var state = try gpu.Buffer.init(&device, 1 << 20, .device);
    defer state.deinit() catch @panic("s");
    var io_buf = try gpu.Buffer.init(&device, 4096, .host);
    defer io_buf.deinit() catch @panic("io");
    var io_dev = try gpu.Buffer.init(&device, 4096, .device);
    defer io_dev.deinit() catch @panic("iod");
    @memset(std.mem.bytesAsSlice(u32, try io_buf.mapped()), 0);
    std.mem.bytesAsSlice(u32, try io_buf.mapped())[2] = 1; // IO_COUNT
    var staging = try gpu.Buffer.init(&device, 1 << 20, .host);
    defer staging.deinit() catch @panic("st");
    var prng = std.Random.DefaultPrng.init(7);
    for (std.mem.bytesAsSlice(f32, try staging.mapped())) |*v| v.* = prng.random().floatNorm(f32);
    {
        var up = try gpu.Commands.init(&device);
        defer up.deinit() catch @panic("up");
        try up.begin();
        try up.copy(&staging, 0, &params, 0, 1 << 20);
        try up.copy(&staging, 0, &act, 0, 1 << 20);
        try up.copy(&io_buf, 0, &io_dev, 0, 4096);
        try up.barrier(.transfer, .compute);
        try up.end();
        try up.run(w.timeout_ns);
    }
    const binds = [_]*gpu.Buffer{ &params, &act, &state, &io_buf };
    const norm_path = if (args.len > 2) args[2] else "src/model/shaders/norm.spv";
    var norm = try gpu.Kernel.init(&device, try spv(a, io, norm_path), &binds, @sizeOf(NormPush));
    defer norm.deinit() catch @panic("n");
    var zero = try gpu.Kernel.init(&device, try spv(a, io, "src/model/shaders/zero.spv"), &binds, @sizeOf(ZeroPush));
    defer zero.deinit() catch @panic("z");
    const binds_dev = [_]*gpu.Buffer{ &params, &act, &state, &io_dev };
    var norm_dev = try gpu.Kernel.init(&device, try spv(a, io, norm_path), &binds_dev, @sizeOf(NormPush));
    defer norm_dev.deinit() catch @panic("nd");
    var out_buf: [1024]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const push: NormPush = .{ .x = 0, .a = 8192, .sum = 16384, .y = 24576, .w = 0, .stride = 0, .flags = 1, .eps = 1e-6 };
    var push_rows = push;
    push_rows.flags |= 2; // ROWS_IO
    const order_env = std.c.getenv("CHAIN_ORDER");
    const reversed = order_env != null;
    const fwd = [_][]const u8{ "norm_add", "norm_add_nobarrier", "zero_link", "rowsio_host_1", "rowsio_device_1", "rowsio_host_96", "rowsio_device_96" };
    const rev = [_][]const u8{ "rowsio_device_96", "rowsio_host_96", "rowsio_device_1", "rowsio_host_1", "zero_link", "norm_add_nobarrier", "norm_add" };
    for (if (reversed) rev else fwd) |variant| {
        var timing = try w.Timing.init(&device);
        defer timing.deinit() catch @panic("t");
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("c");
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        try timing.begin(&cmd);
        for (0..n) |_| {
            if (std.mem.eql(u8, variant, "zero_link")) {
                try cmd.dispatch(&zero, std.mem.asBytes(&ZeroPush{ .first = 0, .count = 0 }), .{ 1, 1, 1 });
            } else if (std.mem.startsWith(u8, variant, "rowsio_")) {
                const groups: u32 = if (std.mem.endsWith(u8, variant, "_96")) 96 else 1;
                const k = if (std.mem.indexOf(u8, variant, "device") != null) &norm_dev else &norm;
                try cmd.dispatch(k, std.mem.asBytes(&push_rows), .{ groups, 1, 1 });
            } else try cmd.dispatch(&norm, std.mem.asBytes(&push), .{ 1, 1, 1 });
            if (!std.mem.eql(u8, variant, "norm_add_nobarrier")) try cmd.barrier(.compute, .compute);
        }
        try timing.end();
        try cmd.end();
        for (0..5) |_| try cmd.run(w.timeout_ns);
        var samples: [21]f64 = undefined;
        for (&samples) |*s| {
            const start = std.Io.Clock.awake.now(io);
            try cmd.run(w.timeout_ns);
            s.* = try timing.read(@floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds));
        }
        std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
        try stdout.interface.print("{{\"variant\":\"{s}\",\"n\":{d},\"gpu_us_median\":{d:.2},\"us_per_link\":{d:.3}}}\n", .{ variant, n, samples[10] / 1e3, samples[10] / 1e3 / @as(f64, @floatFromInt(n)) });
        try stdout.interface.flush();
    }
}

fn compare(a: std.mem.Allocator, io: std.Io, path_a: []const u8, path_b: []const u8, rows: u32) !void {
    if (rows == 0 or rows > 1000) return error.Usage;
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("live");
    const H = 5120;
    const words: u64 = @as(u64, rows) * H;
    var params = try gpu.Buffer.init(&device, H * 4, .host);
    defer params.deinit() catch @panic("p");
    var act = try gpu.Buffer.init(&device, words * 4 * 4, .host); // x, a, sum, y
    defer act.deinit() catch @panic("a");
    var state = try gpu.Buffer.init(&device, 256, .host);
    defer state.deinit() catch @panic("s");
    var io_buf = try gpu.Buffer.init(&device, 4096, .host);
    defer io_buf.deinit() catch @panic("io");
    var io_dev = try gpu.Buffer.init(&device, 4096, .device);
    defer io_dev.deinit() catch @panic("iod");
    @memset(std.mem.bytesAsSlice(u32, try io_buf.mapped()), 0);
    std.mem.bytesAsSlice(u32, try io_buf.mapped())[2] = 1; // IO_COUNT
    std.mem.bytesAsSlice(u32, try io_buf.mapped())[2] = rows;
    var prng = std.Random.DefaultPrng.init(12345);
    for (std.mem.bytesAsSlice(f32, try params.mapped())) |*v| v.* = 0.5 + prng.random().float(f32);
    const binds = [_]*gpu.Buffer{ &params, &act, &state, &io_buf };
    var outputs: [2][]u32 = undefined;
    for ([_][]const u8{ path_a, path_b }, 0..) |path, which| {
        var k = try gpu.Kernel.init(&device, try spv(a, io, path), &binds, @sizeOf(NormPush));
        defer k.deinit() catch @panic("k");
        var fill = std.Random.DefaultPrng.init(777);
        const f = std.mem.bytesAsSlice(f32, try act.mapped());
        for (f[0 .. 2 * words]) |*v| v.* = fill.random().floatNorm(f32) * std.math.pow(f32, 4, fill.random().float(f32) * 4 - 2);
        @memset(f[2 * words ..], 0);
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("c");
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        const push: NormPush = .{ .x = 0, .a = @intCast(words), .sum = @intCast(2 * words), .y = @intCast(3 * words), .w = 0, .stride = H, .flags = 1 | 2, .eps = 9.99999997e-7 };
        try cmd.dispatch(&k, std.mem.asBytes(&push), .{ rows, 1, 1 });
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(w.timeout_ns);
        const raw = std.mem.bytesAsSlice(u32, try act.mapped())[2 * words ..];
        outputs[which] = try a.alloc(u32, raw.len);
        for (outputs[which], raw) |*d, v| d.* = v;
    }
    var differing: u32 = 0;
    for (0..rows) |r| {
        const sa = outputs[0][r * H ..][0..H];
        const sb = outputs[1][r * H ..][0..H];
        const ya = outputs[0][words + r * H ..][0..H];
        const yb = outputs[1][words + r * H ..][0..H];
        if (!std.mem.eql(u32, sa, sb) or !std.mem.eql(u32, ya, yb)) {
            differing += 1;
            var n_diff: u32 = 0;
            for (ya, yb) |x, y| n_diff += @intFromBool(x != y);
            if (differing <= 5) std.debug.print("row {d}: sum identical={} y elements differing={d}\n", .{ r, std.mem.eql(u32, sa, sb), n_diff });
            if (differing == 1) {
                // Dump the row (sum = v, y from both kernels) for offline analysis.
                const file = try std.Io.Dir.cwd().createFile(io, ".tools/norm-diff-row.bin", .{});
                defer file.close(io);
                var buf: [4096]u8 = undefined;
                var fw: std.Io.File.Writer = .init(file, io, &buf);
                try fw.interface.writeAll(std.mem.sliceAsBytes(sa));
                try fw.interface.writeAll(std.mem.sliceAsBytes(ya));
                try fw.interface.writeAll(std.mem.sliceAsBytes(yb));
                try fw.interface.flush();
            }
        }
    }
    std.debug.print("{d} of {d} rows differ\n", .{ differing, rows });
}
