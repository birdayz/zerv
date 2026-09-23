//! Direct submit/fence wall-time benchmark; no model inference or GPU-only timing claim.
const std = @import("std");
const gpu = @import("zerv").gpu;
const workload = @import("gpu_workload");

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const bench = args.len == 2 and std.mem.eql(u8, args[1], "--bench");
    if (args.len != 1 and !bench) return error.ExpectedOptionalBench;
    const parsed = try workload.goldens(init.arena.allocator());
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("device still owns resources");
    std.debug.print("device={s} vendor={x} id={x} api={d} driver={d} family={d}\n", .{ device.name(), device.properties.vendorID, device.properties.deviceID, device.properties.apiVersion, device.properties.driverVersion, device.family });
    var buffer: [8192]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    for (parsed.value.cases) |case| {
        if (bench and case.kind == .affine and case.count != 65 and case.count != 5120 and case.count != 1048576) continue;
        var run: workload.Workload = undefined;
        try run.init(&device, case.kind, case.count);
        defer run.deinit();
        std.debug.print("allocation kind={s} count={d} bytes={d} allocation_sizes={d},{d},{d},{d} memory_types={d},{d},{d},{d}\n", .{ @tagName(case.kind), case.count, run.bytes, run.input.allocation_size, run.readback.allocation_size, run.a.allocation_size, run.b.allocation_size, run.input.memory_type, run.readback.memory_type, run.a.memory_type, run.b.memory_type });
        try run.execute();
        if (bench) {
            const iterations: usize = if (case.kind == .affine) (if (case.count >= 1048576) 100 else 1000) else (if (case.count >= 67108864) 10 else if (case.count >= 1048576) 100 else 1000);
            for (0..3) |_| try run.execute();
            for (0..7) |trial| {
                const start = std.Io.Clock.awake.now(init.io);
                for (0..iterations) |_| try run.execute();
                const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
                if (elapsed <= 0) return error.InvalidElapsed;
                try std.json.Stringify.value(.{ .kind = "timing", .workload = @tagName(case.kind), .count = case.count, .trial = trial, .iterations = iterations, .elapsed_ns = elapsed }, .{}, &stdout.interface);
                try stdout.interface.writeByte('\n');
            }
        }
        const result = try run.finish();
        if (!std.mem.eql(u8, case.input_sha256, &result.input_sha256) or !std.mem.eql(u8, case.output_sha256, &result.output_sha256)) return error.IndependentGoldenMismatch;
        try std.json.Stringify.value(result, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
    }
    try stdout.interface.flush();
}
