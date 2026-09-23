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
    for ([_]bool{ false, true }) |aligned| {
        for (parsed.value.cases) |case| {
            const raw = try w.fixtureBytes(t.allocator, case);
            defer t.allocator.free(raw);
            const data = try w.parseCase(raw);
            var run: w.Workload = undefined;
            if (aligned) try run.initAligned(&device, data) else try run.init(&device, data);
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

test "diagnostic timestamp query reset replay brackets independently checked matvec" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    const case = parsed.value.cases[0];
    const raw = try w.fixtureBytes(t.allocator, case);
    defer t.allocator.free(raw);
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live timestamp device");
    var run: w.Workload = undefined;
    try run.init(&device, try w.parseCase(raw));
    defer run.deinit();
    var timing = try w.Timing.init(&device);
    defer timing.deinit() catch @panic("live timestamp command");
    var command = try gpu.Commands.init(&device);
    defer command.deinit() catch @panic("pending timestamp command");
    try t.expectError(error.InvalidState, timing.begin(&command));
    try command.begin();
    try timing.begin(&command);
    try t.expectError(error.ResourceInUse, timing.deinit());
    try t.expectError(error.InvalidState, timing.begin(&command));
    try run.plan.record(&command);
    try timing.end();
    try t.expectError(error.InvalidState, timing.end());
    try command.end();
    for (0..3) |_| {
        const start = std.Io.Clock.awake.now(t.io);
        try command.run(w.timeout_ns);
        const elapsed = start.durationTo(std.Io.Clock.awake.now(t.io)).nanoseconds;
        const ns = try timing.read(@floatFromInt(elapsed));
        try t.expect(ns >= 0 and ns <= @as(f64, @floatFromInt(elapsed)) + 2 * timing.period);
        try w.check(case, try run.finish());
    }
}

test "shared matvec pipeline: many independent projections in one command, both alignments" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live pipeline resources");
    inline for ([_]m.Format{ .f32, .q4_0, .q4_1, .q5_k, .q6_k }) |format| {
        for ([_]bool{ false, true }) |aligned| {
            if (aligned and format != .q4_1 and format != .q5_k) continue;
            // Pack every explicit case of this format into one buffer at distinct offsets.
            var total: usize = 64;
            var cases: [48]*const w.Case = undefined;
            var count: usize = 0;
            for (parsed.value.cases) |*case| {
                if (case.format != format or case.kind != .explicit) continue;
                cases[count] = case;
                count += 1;
                const shape: m.Shape = .{ .format = format, .columns = case.columns, .rows = case.rows };
                total += std.mem.alignForward(usize, @intCast(try shape.weightBytes()), 4) + 8 + case.columns * 4 + case.rows * 4 + 64;
            }
            try t.expect(count >= 4);
            var upload = try gpu.Buffer.init(&device, total, .host);
            defer upload.deinit() catch @panic("upload");
            var resident = try gpu.Buffer.init(&device, total, .device);
            defer resident.deinit() catch @panic("resident");
            var readback = try gpu.Buffer.init(&device, total, .host);
            defer readback.deinit() catch @panic("readback");
            var pipeline = try m.Pipeline.init(format, aligned, &resident, &resident, &resident);
            defer pipeline.deinit() catch @panic("pipeline");
            const bytes = try upload.mapped();
            @memset(bytes, 0xcd);
            var geometry: [48]m.Geometry = undefined;
            var outputs: [48]usize = undefined;
            var at: usize = if (aligned or format == .f32) 0 else 2; // 2-byte starts for generic modules
            for (cases[0..count], 0..) |case, i| {
                const raw = try w.fixtureBytes(t.allocator, case.*);
                defer t.allocator.free(raw);
                const data = try w.parseCase(raw);
                const woff = at;
                @memcpy(bytes[woff..][0..data.weights.len], data.weights);
                const xoff = std.mem.alignForward(usize, woff + data.weights.len, 4) + 4;
                @memcpy(bytes[xoff..][0..data.input.len], data.input);
                outputs[i] = xoff + data.input.len + 8;
                geometry[i] = try pipeline.projection(data.shape, woff, xoff, outputs[i]);
                at = std.mem.alignForward(usize, outputs[i] + @as(usize, case.rows) * 4 + 16, 4) + (if (aligned or format == .f32) @as(usize, 0) else 2);
            }
            // Rejections are validated before any recording.
            const first = cases[0];
            const shape: m.Shape = .{ .format = format, .columns = first.columns, .rows = first.rows };
            if (aligned) try t.expectError(error.InvalidRange, pipeline.projection(shape, 2, 1024, 2048));
            try t.expectError(error.AliasedOutput, pipeline.projection(shape, geometry[0].push.weight_offset, geometry[0].push.input_offset * 4, geometry[0].push.input_offset * 4));
            const other: m.Format = if (format == .f32) .q4_0 else .f32;
            try t.expectError(error.InvalidShape, pipeline.projection(.{ .format = other, .columns = 32, .rows = 1 }, 0, 1024, 4096));
            var cmd = try gpu.Commands.init(&device);
            defer cmd.deinit() catch @panic("command");
            try cmd.begin();
            try cmd.copy(&upload, 0, &resident, 0, total);
            try cmd.barrier(.transfer, .compute);
            for (geometry[0..count]) |g| try pipeline.record(&cmd, g);
            try cmd.barrier(.compute, .transfer);
            try cmd.copy(&resident, 0, &readback, 0, total);
            try cmd.barrier(.transfer, .host);
            try cmd.end();
            try cmd.run(w.timeout_ns);
            const back = try readback.mapped();
            for (cases[0..count], outputs[0..count]) |case, o| try w.check(case.*, back[o..][0 .. @as(usize, case.rows) * 4]);
            try t.expectEqual(@as(u32, 1), pipeline.kernel.references);
        }
    }
    try t.expectError(error.InvalidShape, blk: {
        var buffer = try gpu.Buffer.init(&device, 1024, .device);
        defer buffer.deinit() catch @panic("buffer");
        break :blk m.Pipeline.init(.q4_0, true, &buffer, &buffer, &buffer);
    });
    try t.expectEqual(@as(u32, 0), device.kernels);
}
