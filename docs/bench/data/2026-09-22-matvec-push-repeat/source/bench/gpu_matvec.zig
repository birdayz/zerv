//! Resident single-vector submit/fence benchmark; output validation is external and untimed.
const std = @import("std");
const zerv = @import("zerv");
const w = @import("matvec_workload");

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const raw_args = try init.minimal.args.toSlice(init.arena.allocator());
    var count = raw_args.len;
    var aligned = false;
    var timestamps = false;
    while (count > 1) {
        if (std.mem.eql(u8, raw_args[count - 1], "--aligned")) {
            if (aligned) return error.DuplicateOption;
            aligned = true;
        } else if (std.mem.eql(u8, raw_args[count - 1], "--timestamps")) {
            if (timestamps) return error.DuplicateOption;
            timestamps = true;
        } else break;
        count -= 1;
    }
    const args = raw_args[0..count];
    if ((args.len == 4 or args.len == 5) and std.mem.eql(u8, args[1], "--fixtures")) {
        const only: ?zerv.matvec.Format = if (args.len == 5) (std.meta.stringToEnum(zerv.matvec.Format, args[4]) orelse return error.UnknownFormat) else null;
        return fixtures(init, args[2], try std.fmt.parseInt(u32, args[3], 10), aligned, only);
    }
    if (args.len != 4 and args.len != 6) return error.ExpectedCaseOutputIterations;
    const iterations = try std.fmt.parseInt(usize, args[3], 10);
    if (iterations > 100000) return error.InvalidIterations;
    var mapped = try zerv.artifact.MappedFile.open(init.io, args[1], 4 * 1024 * 1024 * 1024);
    defer mapped.deinit();
    const data = try w.parseCase(mapped.bytes);
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 3 * 1024 * 1024 * 1024 });
    defer device.deinit() catch @panic("live device resources");
    std.debug.print("device={s} vendor={x} id={x} api={d} driver={d} family={d}\n", .{ device.name(), device.properties.vendorID, device.properties.deviceID, device.properties.apiVersion, device.properties.driverVersion, device.family });
    var run: w.Workload = undefined;
    if (aligned) try run.initAligned(&device, data) else try run.init(&device, data);
    defer run.deinit();
    if (args.len == 6) {
        var module = try zerv.artifact.MappedFile.open(init.io, args[4], 1024 * 1024);
        defer module.deinit();
        try run.replaceKernel(module.bytes, try std.fmt.parseInt(u32, args[5], 10));
    }
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
    if (timestamps) try profile(init.io, &run, &stdout.interface);
    const output = try run.finish();
    const file = try std.Io.Dir.cwd().createFile(init.io, args[2], .{ .exclusive = true });
    defer file.close(init.io);
    var output_buffer: [4096]u8 = undefined;
    var writer: std.Io.File.Writer = .init(file, init.io, &output_buffer);
    try writer.interface.writeAll(output);
    try writer.interface.flush();
    try stdout.interface.flush();
}

fn fixtures(init: std.process.Init, directory: []const u8, rows_per_group: u32, aligned: bool, only: ?zerv.matvec.Format) !void {
    const a = init.arena.allocator();
    const parsed = try w.goldens(a);
    defer parsed.deinit();
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live candidate fixture resources");
    var verified: usize = 0;
    for (parsed.value.cases) |case| {
        if (only) |format| if (case.format != format) continue;
        verified += 1;
        const raw = try w.fixtureBytes(a, case);
        defer a.free(raw);
        var run: w.Workload = undefined;
        const data = try w.parseCase(raw);
        if (aligned) try run.initAligned(&device, data) else try run.init(&device, data);
        defer run.deinit();
        const name = try std.fmt.allocPrint(a, "{s}/{s}.spv", .{ directory, @tagName(case.format) });
        defer a.free(name);
        var module = try zerv.artifact.MappedFile.open(init.io, name, 1024 * 1024);
        defer module.deinit();
        try run.replaceKernel(module.bytes, rows_per_group);
        try run.execute();
        try w.check(case, try run.finish());
        try run.execute();
        try w.check(case, try run.finish());
        try run.zeroInput();
        try run.execute();
        const zeros = try run.finish();
        for (0..case.rows) |r| {
            const value: f32 = @bitCast(std.mem.readInt(u32, zeros[r * 4 ..][0..4], .little));
            if (value != 0) return error.ChangedInputMismatch;
        }
    }
    if (verified == 0) return error.NoFixtures;
    std.debug.print("verified {d} independent cases, replay, guards and changed input\n", .{verified});
}

fn profile(io: std.Io, run: *w.Workload, writer: *std.Io.Writer) !void {
    var timing = try w.Timing.init(run.resident.device);
    defer timing.deinit() catch @panic("live timestamp command");
    var command = try zerv.gpu.Commands.init(run.resident.device);
    defer command.deinit() catch @panic("pending timestamp command");
    try command.begin();
    try timing.begin(&command);
    try run.plan.record(&command);
    try timing.end();
    try command.end();
    for (0..4) |_| try command.run(w.timeout_ns);
    for (0..32) |trial| {
        const start = std.Io.Clock.awake.now(io);
        try command.run(w.timeout_ns);
        const wall_ns = start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        const gpu_ns = try timing.read(@floatFromInt(wall_ns));
        if (gpu_ns <= 0 or gpu_ns > @as(f64, @floatFromInt(wall_ns)) + 2 * timing.period) return error.InvalidTimestampInterval;
        try std.json.Stringify.value(.{ .profile_trial = trial, .wall_ns = wall_ns, .gpu_ns = gpu_ns, .timestamp_bits = timing.bits, .timestamp_period_ns = timing.period }, .{}, writer);
        try writer.writeByte('\n');
    }
}
