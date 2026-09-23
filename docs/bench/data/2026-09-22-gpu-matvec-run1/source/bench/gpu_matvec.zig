//! Resident single-vector submit/fence benchmark; output validation is external and untimed.
const std = @import("std");
const zerv = @import("zerv");
const w = @import("matvec_workload");

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.ExpectedCaseOutputIterations;
    const iterations = try std.fmt.parseInt(usize, args[3], 10);
    if (iterations > 100000) return error.InvalidIterations;
    var mapped = try zerv.artifact.MappedFile.open(init.io, args[1], 4 * 1024 * 1024 * 1024);
    defer mapped.deinit();
    const data = try w.parseCase(mapped.bytes);
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 3 * 1024 * 1024 * 1024 });
    defer device.deinit() catch @panic("live device resources");
    std.debug.print("device={s} vendor={x} id={x} api={d} driver={d} family={d}\n", .{ device.name(), device.properties.vendorID, device.properties.deviceID, device.properties.apiVersion, device.properties.driverVersion, device.family });
    var run: w.Workload = undefined;
    try run.init(&device, data);
    defer run.deinit();
    std.debug.print("allocation_bytes={d} sizes={d},{d},{d} memory_types={d},{d},{d} groups={d},{d},{d}\n", .{ device.allocated_bytes, run.upload.allocation_size, run.resident.allocation_size, run.readback.allocation_size, run.upload.memory_type, run.resident.memory_type, run.readback.memory_type, run.plan.layout.groups[0], run.plan.layout.groups[1], run.plan.layout.groups[2] });
    try run.execute();
    var stdout_buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    if (iterations != 0) {
        for (0..3) |_| try run.execute();
        for (0..7) |trial| {
            const start = std.Io.Clock.awake.now(init.io);
            for (0..iterations) |_| try run.execute();
            const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            if (elapsed <= 0) return error.InvalidElapsed;
            try std.json.Stringify.value(.{ .trial = trial, .iterations = iterations, .elapsed_ns = elapsed }, .{}, &stdout.interface);
            try stdout.interface.writeByte('\n');
        }
    }
    const output = try run.finish();
    const file = try std.Io.Dir.cwd().createFile(init.io, args[2], .{ .exclusive = true });
    defer file.close(init.io);
    var output_buffer: [4096]u8 = undefined;
    var writer: std.Io.File.Writer = .init(file, init.io, &output_buffer);
    try writer.interface.writeAll(output);
    try writer.interface.flush();
    try stdout.interface.flush();
}
