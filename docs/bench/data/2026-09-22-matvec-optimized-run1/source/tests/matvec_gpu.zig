const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const m = zerv.matvec;
const w = @import("matvec_workload.zig");
const t = std.testing;

test "native matvec independent exact/reduction fixtures, guards, replay and changed input" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 48), parsed.value.cases.len);
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live matvec resources");
    for (parsed.value.cases) |case| {
        const raw = try w.fixtureBytes(t.allocator, case);
        defer t.allocator.free(raw);
        const data = try w.parseCase(raw);
        var run: w.Workload = undefined;
        try run.init(&device, data);
        defer run.deinit();
        if (case.kind == .row_ramp) try t.expectEqualSlices(u32, &.{ 32769, 2, 1 }, &run.plan.layout.groups);
        try t.expectError(error.ResourceInUse, run.plan.deinit());
        try t.expectError(error.ResourceInUse, run.resident.deinit());
        try run.execute();
        try w.check(case, try run.finish());
        try run.execute(); // Previously downloaded output must not interfere with replay.
        try w.check(case, try run.finish());
        try run.zeroInput();
        try run.execute();
        const zeros = try run.finish();
        for (0..case.rows) |r| {
            const actual: f32 = @bitCast(std.mem.readInt(u32, zeros[r * 4 ..][0..4], .little));
            try t.expectEqual(@as(f32, 0), actual);
        }
    }
    try t.expectEqual(@as(u64, 0), device.allocated_bytes);
    try t.expectEqual(@as(u32, 0), device.pending);
}

test "matvec workload partial allocation failure rolls back all resources" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 600 });
    defer device.deinit() catch @panic("live rollback resources");
    const bytes: [128]u8 = @splat(0);
    var run: w.Workload = undefined;
    try t.expectError(error.ResourceLimit, run.init(&device, .{ .shape = .{ .format = .f32, .columns = 32, .rows = 1 }, .weights = &bytes, .input = &bytes }));
    try t.expectEqual(@as(u64, 0), device.allocated_bytes);
    try t.expectEqual(@as(u32, 0), device.buffers);
    try t.expectEqual(@as(u32, 0), device.kernels);
    try t.expectEqual(@as(u32, 0), device.commands);
}

test "matvec setup alias/device/extent failures do not retain or modify buffers" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 });
    defer device.deinit() catch @panic("live device resources");
    var buffer = try gpu.Buffer.init(&device, 1024, .host);
    defer buffer.deinit() catch @panic("live buffer resources");
    @memset(try buffer.mapped(), 0xcd);
    const shape: m.Shape = .{ .format = .q4_0, .columns = 32, .rows = 3 };
    const weights: m.View = .{ .buffer = &buffer, .offset = 2 };
    const input: m.View = .{ .buffer = &buffer, .offset = 64 };
    const output: m.View = .{ .buffer = &buffer, .offset = 192 };
    try t.expectError(error.AliasedOutput, m.Plan.init(shape, weights, input, .{ .buffer = &buffer, .offset = 52 }));
    try t.expectError(error.AliasedOutput, m.Plan.init(shape, weights, input, .{ .buffer = &buffer, .offset = 188 }));
    try t.expectError(error.InvalidRange, m.Plan.init(shape, .{ .buffer = &buffer, .offset = 1 }, input, output));
    try t.expectError(error.InvalidRange, m.Plan.init(shape, weights, input, .{ .buffer = &buffer, .offset = 1016 }));
    try t.expectEqual(@as(u32, 0), buffer.references);
    try t.expectEqual(@as(u32, 0), device.kernels);
    for (try buffer.mapped()) |b| try t.expectEqual(@as(u8, 0xcd), b);
    var other = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 });
    defer other.deinit() catch @panic("live other device");
    var foreign = try gpu.Buffer.init(&other, 1024, .device);
    defer foreign.deinit() catch @panic("live other buffer");
    try t.expectError(error.WrongDevice, m.Plan.init(shape, weights, input, .{ .buffer = &foreign }));
    var plan = try m.Plan.init(shape, weights, input, output);
    defer plan.deinit() catch @panic("live plan");
    try t.expectEqual(@as(u32, 3), buffer.references); // Three disjoint views, one buffer owner.
    var cmd = try gpu.Commands.init(&other);
    defer cmd.deinit() catch @panic("live command");
    try cmd.begin();
    try t.expectError(error.WrongDevice, plan.record(&cmd));
    try t.expectEqual(@as(usize, 0), cmd.kernel_count);
    try cmd.end();
}

test "F32 local-size fallback retains independent results and benchmark replacement errors retain ownership" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live fallback resources");
    const actual_limits = device.properties.limits;
    var module = try zerv.artifact.MappedFile.open(t.io, "src/matvec/shaders/f32_small.spv", 1024 * 1024);
    defer module.deinit();
    // Restrict the host planner only: test each reason for choosing the core64
    // path without claiming this machine physically has these lower limits.
    for (0..3) |limited| {
        device.properties.limits = actual_limits;
        switch (limited) {
            0 => device.properties.limits.maxComputeWorkGroupInvocations = 64,
            1 => device.properties.limits.maxComputeWorkGroupSize[0] = 64,
            2 => device.properties.limits.maxComputeSharedMemorySize = 256,
            else => unreachable,
        }
        for (parsed.value.cases) |case| {
            if (case.format != .f32) continue;
            const raw = try w.fixtureBytes(t.allocator, case);
            defer t.allocator.free(raw);
            var run: w.Workload = undefined;
            try run.init(&device, try w.parseCase(raw));
            defer run.deinit();
            try t.expectError(error.InvalidCase, run.replaceKernel(module.bytes, 0));
            try t.expectError(error.InvalidCase, run.replaceKernel(module.bytes, 33));
            try t.expectError(error.InvalidShader, run.replaceKernel(&.{}, 1));
            try run.execute();
            try w.check(case, try run.finish());
            try run.replaceKernel(module.bytes, 1);
            try run.execute();
            try w.check(case, try run.finish());
        }
    }
    device.properties.limits = actual_limits;
    try t.expectEqual(@as(u32, 0), device.kernels);
    try t.expectEqual(@as(u64, 0), device.allocated_bytes);
}
