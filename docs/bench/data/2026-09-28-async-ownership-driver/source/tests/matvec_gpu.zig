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
    // Both accumulation module sets (docs/specs/matvec-push.md) meet every fixture.
    for ([_]m.Accumulation{ .fma, .separate }) |accumulation| for ([_]bool{ false, true }) |aligned| {
        for (parsed.value.cases) |case| {
            const raw = try w.fixtureBytes(t.allocator, case);
            defer t.allocator.free(raw);
            const data = try w.parseCase(raw);
            var run: w.Workload = undefined;
            try run.initWith(&device, data, aligned, accumulation);
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
    };
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

// Block 17b gate 1 (docs/specs/speculative.md): the multi-row module equals the single-row
// module row by row, bit for bit, on every independent fixture (all formats, row ramps,
// odd shapes), for 1..max_rows input rows at unaligned and aligned weight offsets; rows
// past `count` are not written. Inputs: row 0 the fixture input, row r a permuted, sign-
// flipped copy (finite, same magnitudes).
test "multi-row matvec equals the single-row module per row, bitwise, on all fixtures" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("live matvec resources");
    const R = m.max_rows;
    for ([_]m.Accumulation{ .fma, .separate }) |accumulation| for ([_]bool{ false, true }) |aligned_offset| {
        for (parsed.value.cases) |case| {
            const raw = try w.fixtureBytes(t.allocator, case);
            defer t.allocator.free(raw);
            const data = try w.parseCase(raw);
            const shape = data.shape;
            const K = shape.columns;
            const M = shape.rows;
            const weight_offset: u64 = if (aligned_offset) 0 else if (shape.format == .f32) 4 else 2;
            var weights = try gpu.Buffer.init(&device, std.mem.alignForward(u64, weight_offset + data.weights.len, 4) + 64, .host);
            defer weights.deinit() catch @panic("w");
            @memcpy((try weights.mapped())[weight_offset..][0..data.weights.len], data.weights);
            var input = try gpu.Buffer.init(&device, @as(u64, R) * K * 4, .host);
            defer input.deinit() catch @panic("x");
            const xs = std.mem.bytesAsSlice(f32, try input.mapped());
            const x0 = std.mem.bytesAsSlice(f32, data.input);
            for (0..R) |r| for (0..K) |i| {
                const v: f32 = @bitCast(std.mem.readInt(u32, data.input[((i * (2 * r + 1)) % K) * 4 ..][0..4], .little));
                xs[r * K + i] = if (r == 0) @bitCast(std.mem.readInt(u32, std.mem.sliceAsBytes(x0)[i * 4 ..][0..4], .little)) else if (r % 2 == 1) -v else v;
            };
            // Outputs: single-row results rows 0..R-1, multi-row results rows R..2R-1.
            var output = try gpu.Buffer.init(&device, @as(u64, 2 * R) * M * 4, .host);
            defer output.deinit() catch @panic("y");
            const aligned_words = aligned_offset and (shape.format == .q4_1 or shape.format == .q5_k);
            var single = try m.Pipeline.initWith(shape.format, aligned_words, accumulation, &weights, &input, &output);
            defer single.deinit() catch @panic("single");
            var multi = try m.RowsPipeline.initWith(shape.format, aligned_words, accumulation, R, &weights, &input, &output);
            defer multi.deinit() catch @panic("multi");
            const geo = try multi.projection(shape, weight_offset, 0, @as(u64, R) * M * 4, K, M, R);
            for (1..R + 1) |count| {
                @memset(try output.mapped(), 0xff);
                var cmd = try gpu.Commands.init(&device);
                defer cmd.deinit() catch @panic("cmd");
                try cmd.begin();
                try cmd.barrier(.host, .compute);
                for (0..R) |r| try single.record(&cmd, try single.projection(shape, weight_offset, @as(u64, r) * K * 4, @as(u64, r) * M * 4));
                try multi.record(&cmd, geo, @intCast(count));
                try cmd.barrier(.compute, .host);
                try cmd.end();
                try cmd.run(w.timeout_ns);
                const ys = std.mem.bytesAsSlice(u32, try output.mapped());
                for (0..R) |r| for (0..M) |row| {
                    const got = ys[(R + r) * M + row];
                    if (r < count) {
                        if (got != ys[r * M + row]) {
                            std.debug.print("{s} {s} count={d} r={d} row={d}: 0x{x} != 0x{x}\n", .{ case.name, @tagName(shape.format), count, r, row, got, ys[r * M + row] });
                            return error.NotBitwiseEqual;
                        }
                    } else try t.expectEqual(@as(u32, 0xffffffff), got);
                };
            }
        }
    };
    // Validation: count, strides, extents and aliasing.
    var buffer = try gpu.Buffer.init(&device, 4096, .host);
    defer buffer.deinit() catch @panic("b");
    var other = try gpu.Buffer.init(&device, 4096, .host);
    defer other.deinit() catch @panic("o");
    try t.expectError(error.InvalidShape, m.RowsPipeline.init(.f32, false, 0, &buffer, &other, &other));
    try t.expectError(error.InvalidShape, m.RowsPipeline.init(.f32, false, R + 1, &buffer, &other, &other));
    var p = try m.RowsPipeline.init(.f32, false, R, &buffer, &other, &other);
    defer p.deinit() catch @panic("p");
    const s: m.Shape = .{ .format = .f32, .columns = 32, .rows = 4 };
    _ = try p.projection(s, 0, 0, 2048, 32, 4, R);
    try t.expectError(error.InvalidShape, p.projection(s, 0, 0, 2048, 32, 4, 0));
    try t.expectError(error.InvalidShape, p.projection(s, 0, 0, 2048, 32, 4, R + 1));
    try t.expectError(error.InvalidShape, p.projection(s, 0, 0, 2048, 31, 4, 2));
    try t.expectError(error.InvalidShape, p.projection(s, 0, 0, 2048, 32, 3, 2));
    try t.expectError(error.InvalidRange, p.projection(s, 0, 0, 4096 - 16 * 4, 32, 4, R));
    try t.expectError(error.AliasedOutput, p.projection(s, 0, 0, 32 * 4, 32, 4, 2));
    try t.expectError(error.InvalidShape, m.Pipeline.init(.q8_0, false, &buffer, &other, &other));
}

// Block 17c (verify FFN fusion): the fused multi-row module computes, per input row, the
// same g, u and y bits as the single-row fused module (itself tested against the
// single-row projections above); rows at or past `count` are not written.
test "fused multi-row gate/up/swiglu equals the single-row fused module per row, bitwise" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("live matvec resources");
    const R = m.max_rows;
    var checked: usize = 0;
    for ([_]m.Accumulation{ .fma, .separate }) |accumulation| for ([_]bool{ false, true }) |aligned_offset| for (parsed.value.cases) |case| {
        const raw = try w.fixtureBytes(t.allocator, case);
        defer t.allocator.free(raw);
        const data = try w.parseCase(raw);
        const shape = data.shape;
        if (shape.format == .f32) continue;
        const K = shape.columns;
        const M = shape.rows;
        const row_bytes = data.weights.len / M;
        const gate_offset: u64 = if (aligned_offset) @as(u64, 0) else 2;
        const up_offset: u64 = std.mem.alignForward(u64, gate_offset + data.weights.len, 4) + @as(u64, if (aligned_offset) 0 else 2);
        var weights = try gpu.Buffer.init(&device, std.mem.alignForward(u64, up_offset + data.weights.len, 4) + 64, .host);
        defer weights.deinit() catch @panic("w");
        const wb = try weights.mapped();
        @memcpy(wb[gate_offset..][0..data.weights.len], data.weights);
        for (0..M) |row| @memcpy(wb[up_offset + row * row_bytes ..][0..row_bytes], data.weights[((row + 1) % M) * row_bytes ..][0..row_bytes]);
        var input = try gpu.Buffer.init(&device, @as(u64, R) * K * 4, .host);
        defer input.deinit() catch @panic("x");
        const xs = std.mem.bytesAsSlice(f32, try input.mapped());
        for (0..R) |r| for (0..K) |i| {
            const v: f32 = @bitCast(std.mem.readInt(u32, data.input[((i * (2 * r + 1)) % K) * 4 ..][0..4], .little));
            xs[r * K + i] = if (r % 2 == 1) -v else v;
        };
        // Outputs: single-row fused g/u/y for row r at (3r + 0/1/2) M; multi-row g, u and
        // y rows (stride M) at 3RM, 4RM and 5RM.
        var output = try gpu.Buffer.init(&device, @as(u64, 6 * R) * M * 4, .host);
        defer output.deinit() catch @panic("y");
        const aligned_words = aligned_offset and (shape.format == .q4_1 or shape.format == .q5_k);
        var single = try m.SwigluPipeline.initWith(shape.format, aligned_words, accumulation, &weights, &input, &output);
        defer single.deinit() catch @panic("single");
        var multi = try m.SwigluRowsPipeline.init(shape.format, aligned_words, accumulation, R, &weights, &input, &output);
        defer multi.deinit() catch @panic("multi");
        const base = @as(u64, R) * M * 4;
        const geo = try multi.projection(shape, gate_offset, up_offset, 0, 3 * base, 4 * base, 5 * base, K, M, R);
        for (1..R + 1) |count| {
            if (!multi.supports(@intCast(count))) {
                var cmd = try gpu.Commands.init(&device);
                defer cmd.deinit() catch @panic("cmd");
                try cmd.begin();
                try t.expectError(error.InvalidShape, multi.record(&cmd, geo, @intCast(count)));
                try cmd.end();
                continue;
            }
            @memset(try output.mapped(), 0xff);
            var cmd = try gpu.Commands.init(&device);
            defer cmd.deinit() catch @panic("cmd");
            try cmd.begin();
            try cmd.barrier(.host, .compute);
            for (0..R) |r| {
                const o = @as(u64, 3 * r) * M * 4;
                try single.record(&cmd, try single.projection(shape, gate_offset, up_offset, @as(u64, r) * K * 4, o, o + M * 4, o + 2 * M * 4));
            }
            try multi.record(&cmd, geo, @intCast(count));
            try cmd.barrier(.compute, .host);
            try cmd.end();
            try cmd.run(w.timeout_ns);
            const ys = std.mem.bytesAsSlice(u32, try output.mapped());
            for (0..R) |r| for (0..3) |kind| for (0..M) |row| {
                const got = ys[(3 * R + kind * R + r) * M + row];
                const want = if (r < count) ys[(3 * r + kind) * M + row] else 0xffffffff;
                if (got != want) {
                    std.debug.print("{s} {s} {s} aligned={} count={d} r={d} output={d} row={d}: 0x{x} != 0x{x}\n", .{ case.name, @tagName(shape.format), @tagName(accumulation), aligned_offset, count, r, kind, row, got, want });
                    return error.NotBitwiseEqual;
                }
                checked += 1;
            };
        }
    };
    try t.expect(checked > 0);
    // Validation: formats, counts, aliasing.
    var buffer = try gpu.Buffer.init(&device, 1 << 20, .host);
    defer buffer.deinit() catch @panic("b");
    try t.expectError(error.InvalidShape, m.SwigluRowsPipeline.init(.f32, false, .fma, R, &buffer, &buffer, &buffer));
    try t.expectError(error.InvalidShape, m.SwigluRowsPipeline.init(.q8_0, false, .fma, R, &buffer, &buffer, &buffer));
    try t.expectError(error.InvalidShape, m.SwigluRowsPipeline.init(.q4_0, false, .fma, R + 1, &buffer, &buffer, &buffer));
    var p = try m.SwigluRowsPipeline.init(.q4_0, false, .fma, R, &buffer, &buffer, &buffer);
    defer p.deinit() catch @panic("p");
    const s: m.Shape = .{ .format = .q4_0, .columns = 64, .rows = 8 };
    _ = try p.projection(s, 0, 1024, 4096, 8192, 12288, 16384, 64, 8, R);
    try t.expectError(error.InvalidShape, p.projection(s, 0, 1024, 4096, 8192, 12288, 16384, 64, 8, R + 1));
    try t.expectError(error.AliasedOutput, p.projection(s, 0, 1024, 4096, 8192, 8192 + 64, 16384, 64, 8, R));
    try t.expectError(error.AliasedOutput, p.projection(s, 0, 1024, 4096, 4096 + 256, 12288, 16384, 64, 8, R));
    try t.expectError(error.InvalidRange, p.projection(s, 0, 1024, 4100, 8192, 12288, 16384, 64, 8, R));
}

// Block 18b.2 (batched decode, docs/specs/concurrent.md): a span of more rows than one
// module handles runs in groups at row offsets (`recordAt`); every row equals the single-row
// module's result, bitwise, whatever group and count computed it. Plain and fused SwiGLU.
test "multi-row matvec over a span in groups at row offsets equals the single-row module, bitwise" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("live matvec resources");
    const R = m.max_rows;
    const S: u32 = 2 * R + 1; // span: groups 5 + 3 + 3 (plain), fused: supported counts only
    var checked: usize = 0;
    for (parsed.value.cases) |case| {
        const raw = try w.fixtureBytes(t.allocator, case);
        defer t.allocator.free(raw);
        const data = try w.parseCase(raw);
        const shape = data.shape;
        const K = shape.columns;
        const M = shape.rows;
        const fused = shape.format != .f32 and shape.format != .q8_0;
        // Weights: gate at 0, up (rows rotated by one) after it.
        const row_bytes = data.weights.len / M;
        const up_offset: u64 = std.mem.alignForward(u64, data.weights.len, 4);
        var weights = try gpu.Buffer.init(&device, up_offset + std.mem.alignForward(u64, data.weights.len, 4) + 64, .host);
        defer weights.deinit() catch @panic("w");
        const wb = try weights.mapped();
        @memcpy(wb[0..data.weights.len], data.weights);
        for (0..M) |row| @memcpy(wb[up_offset + row * row_bytes ..][0..row_bytes], data.weights[((row + 1) % M) * row_bytes ..][0..row_bytes]);
        var input = try gpu.Buffer.init(&device, @as(u64, S) * K * 4, .host);
        defer input.deinit() catch @panic("x");
        const xs = std.mem.bytesAsSlice(f32, try input.mapped());
        for (0..S) |r| for (0..K) |i| {
            const v: f32 = @bitCast(std.mem.readInt(u32, data.input[((i * (2 * r + 1)) % K) * 4 ..][0..4], .little));
            xs[r * K + i] = if (r % 2 == 1) -v else v;
        };
        // Output rows (stride M): single-row y (S), multi-row y (S); fused: single g/u/y
        // (3S), multi g/u/y (3S).
        var output = try gpu.Buffer.init(&device, @as(u64, 8 * S) * M * 4, .host);
        defer output.deinit() catch @panic("y");
        var single = try m.Pipeline.init(shape.format, false, &weights, &input, &output);
        defer single.deinit() catch @panic("single");
        var multi = try m.RowsPipeline.init(shape.format, false, R, &weights, &input, &output);
        defer multi.deinit() catch @panic("multi");
        const rowb = @as(u64, M) * 4;
        const geo = try multi.projectionSpan(shape, 0, 0, S * rowb, K, M, R, S);
        try t.expectError(error.InvalidShape, multi.projectionSpan(shape, 0, 0, S * rowb, K, M, R, R - 1));
        var fsingle: ?m.SwigluPipeline = if (fused) try m.SwigluPipeline.init(shape.format, false, &weights, &input, &output) else null;
        defer if (fsingle) |*f| f.deinit() catch @panic("fsingle");
        var fmulti: ?m.SwigluRowsPipeline = if (fused) try m.SwigluRowsPipeline.init(shape.format, false, .fma, R, &weights, &input, &output) else null;
        defer if (fmulti) |*f| f.deinit() catch @panic("fmulti");
        const fbase = 2 * S * rowb; // single g/u/y rows 3r..3r+2, multi g, u, y spans after
        const fgeo = if (fmulti) |*f| try f.projectionSpan(shape, 0, up_offset, 0, fbase + 3 * S * rowb, fbase + 4 * S * rowb, fbase + 5 * S * rowb, K, M, R, S) else undefined;
        @memset(try output.mapped(), 0xff);
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        for (0..S) |r| {
            try single.record(&cmd, try single.projection(shape, 0, @as(u64, r) * K * 4, r * rowb));
            if (fsingle) |*f| {
                const o = fbase + 3 * r * rowb;
                try f.record(&cmd, try f.projection(shape, 0, up_offset, @as(u64, r) * K * 4, o, o + rowb, o + 2 * rowb));
            }
        }
        const plain_groups = [_][2]u32{ .{ 0, 5 }, .{ 5, 3 }, .{ 8, 3 } };
        for (plain_groups) |grp| try multi.recordAt(&cmd, geo, grp[0], grp[1]);
        if (fmulti) |*f| {
            // Rows in groups of the largest supported count that fits.
            var first: u32 = 0;
            while (first < S) {
                var n: u32 = @min(R, S - first);
                while (!f.supports(n)) n -= 1;
                try f.recordAt(&cmd, fgeo, first, n);
                first += n;
            }
        }
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(w.timeout_ns);
        const ys = std.mem.bytesAsSlice(u32, try output.mapped());
        for (0..S) |r| for (0..M) |row| {
            const want = ys[r * M + row];
            const got = ys[(S + r) * M + row];
            if (got != want) {
                std.debug.print("{s} {s} r={d} row={d}: 0x{x} != 0x{x}\n", .{ case.name, @tagName(shape.format), r, row, got, want });
                return error.NotBitwiseEqual;
            }
            checked += 1;
            if (fused) for (0..3) |kind| {
                const fw = ys[(2 * S + 3 * r + kind) * M + row];
                const fg = ys[(2 * S + 3 * S + kind * S + r) * M + row];
                if (fg != fw) {
                    std.debug.print("fused {s} {s} r={d} output={d} row={d}: 0x{x} != 0x{x}\n", .{ case.name, @tagName(shape.format), r, kind, row, fg, fw });
                    return error.NotBitwiseEqual;
                }
                checked += 1;
            };
        };
        {
            var bad = try gpu.Commands.init(&device);
            defer bad.deinit() catch @panic("bad");
            try bad.begin();
            try t.expectError(error.InvalidShape, multi.recordAt(&bad, geo, S - 2, 3));
            try t.expectError(error.InvalidShape, multi.recordAt(&bad, geo, S, 1));
            try t.expectError(error.InvalidShape, multi.recordAt(&bad, geo, 0, R + 1));
            try bad.end();
        }
        // The span must fit the output buffer.
        try t.expectError(error.InvalidRange, multi.projectionSpan(shape, 0, 0, 7 * S * rowb, K, M, R, S + 1));
    }
    try t.expect(checked > 0);
    std.debug.print("multi-row span groups: {d} values bitwise equal\n", .{checked});
}

// Block 17b (the MTP's eh_proj): the Q8_0 multi-row module against an FP64 reference of
// the exactly decoded weights (d * q is exact in FP32), and each row independent of the
// row count, bitwise. Shapes: an odd M (the last workgroup's second weight row is past
// the end), several K tails, 2-byte-aligned block starts (34-byte blocks).
test "multi-row Q8_0 matvec: FP64 bound and row-count invariance" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 256 * 1024 * 1024 });
    defer device.deinit() catch @panic("live matvec resources");
    var prng = std.Random.DefaultPrng.init(0x5108);
    const random = prng.random();
    const R = m.max_rows;
    for ([_][2]u32{ .{ 32, 1 }, .{ 96, 7 }, .{ 544, 33 }, .{ 10240, 129 } }) |shape_km| {
        const K = shape_km[0];
        const M = shape_km[1];
        const shape: m.Shape = .{ .format = .q8_0, .columns = K, .rows = M };
        const bytes = try shape.weightBytes();
        const packed_bytes = try t.allocator.alloc(u8, bytes);
        defer t.allocator.free(packed_bytes);
        const values = try t.allocator.alloc(f64, @as(usize, K) * M);
        defer t.allocator.free(values);
        for (0..@as(usize, K) * M / 32) |b| {
            const d: f16 = @floatCast(random.float(f32) * 0.02 - 0.01);
            std.mem.writeInt(u16, packed_bytes[b * 34 ..][0..2], @bitCast(d), .little);
            for (0..32) |j| {
                const q: i8 = random.int(i8);
                packed_bytes[b * 34 + 2 + j] = @bitCast(q);
                values[b * 32 + j] = @as(f64, @floatCast(d)) * @as(f64, @floatFromInt(q));
            }
        }
        try m.validateWeights(shape, packed_bytes);
        var weights = try gpu.Buffer.init(&device, std.mem.alignForward(u64, bytes, 4) + 64, .host);
        defer weights.deinit() catch @panic("w");
        @memcpy((try weights.mapped())[0..bytes], packed_bytes);
        var input = try gpu.Buffer.init(&device, @as(u64, R) * K * 4, .host);
        defer input.deinit() catch @panic("x");
        const xs = std.mem.bytesAsSlice(f32, try input.mapped());
        for (xs) |*x| x.* = random.floatNorm(f32);
        // Outputs: count c's rows at [(c - 1) * R * M ..].
        var output = try gpu.Buffer.init(&device, @as(u64, R) * R * M * 4, .host);
        defer output.deinit() catch @panic("y");
        var p = try m.RowsPipeline.init(.q8_0, false, R, &weights, &input, &output);
        defer p.deinit() catch @panic("p");
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        for (1..R + 1) |count| try p.record(&cmd, try p.projection(shape, 0, 0, @as(u64, count - 1) * R * M * 4, K, M, @intCast(count)), @intCast(count));
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(w.timeout_ns);
        const ys = std.mem.bytesAsSlice(f32, try output.mapped());
        for (0..R) |r| for (0..M) |row| {
            var exact: f64 = 0;
            var magnitude: f64 = 0;
            for (0..K) |c| {
                const term = values[row * K + c] * @as(f64, xs[r * K + c]);
                exact += term;
                magnitude += @abs(term);
            }
            const got = ys[(R - 1) * R * M + r * M + row];
            // FP32 accumulation bound: K roundings of partial sums bounded by the magnitude.
            try t.expect(@abs(@as(f64, got) - exact) <= @as(f64, @floatFromInt(K + 8)) * 0x1p-24 * magnitude);
            for (r + 1..R + 1) |count| try t.expectEqual(@as(u32, @bitCast(got)), @as(u32, @bitCast(ys[(count - 1) * R * M + r * M + row])));
        };
    }
}

// Block 17c fused decode FFN input: rows of gate and up equal the single-row module
// bitwise (same accumulation), and y = silu(g) * u within a few ulp of FP64. Up is the
// gate weights rotated by one row, so an offset or row mix-up cannot pass.
test "fused gate/up/swiglu matvec equals the single-row module per row" {
    const parsed = try w.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("live matvec resources");
    var checked: usize = 0;
    for ([_]m.Accumulation{ .fma, .separate }) |accumulation| for (parsed.value.cases) |case| {
        const raw = try w.fixtureBytes(t.allocator, case);
        defer t.allocator.free(raw);
        const data = try w.parseCase(raw);
        const shape = data.shape;
        if (shape.format == .f32 or shape.rows < 2) continue;
        const K = shape.columns;
        const M = shape.rows;
        const row_bytes = data.weights.len / M;
        const up_offset = std.mem.alignForward(u64, data.weights.len + 2, 4) + 2; // odd half-word
        var weights = try gpu.Buffer.init(&device, std.mem.alignForward(u64, up_offset + data.weights.len, 4) + 64, .host);
        defer weights.deinit() catch @panic("w");
        const wb = try weights.mapped();
        @memcpy(wb[2..][0..data.weights.len], data.weights); // gate at offset 2
        for (0..M) |row| @memcpy(wb[up_offset + row * row_bytes ..][0..row_bytes], data.weights[((row + 1) % M) * row_bytes ..][0..row_bytes]);
        var input = try gpu.Buffer.init(&device, @as(u64, K) * 4 + 64, .host);
        defer input.deinit() catch @panic("x");
        @memcpy((try input.mapped())[0 .. K * 4], data.input);
        var output = try gpu.Buffer.init(&device, @as(u64, 5) * M * 4, .host);
        defer output.deinit() catch @panic("y");
        var single = try m.Pipeline.initWith(shape.format, false, accumulation, &weights, &input, &output);
        defer single.deinit() catch @panic("single");
        var fused = try m.SwigluPipeline.initWith(shape.format, false, accumulation, &weights, &input, &output);
        defer fused.deinit() catch @panic("fused");
        const geo = try fused.projection(shape, 2, up_offset, 0, 2 * M * 4, 3 * M * 4, 4 * M * 4);
        @memset(try output.mapped(), 0xff);
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("cmd");
        try cmd.begin();
        try cmd.barrier(.host, .compute);
        try single.record(&cmd, try single.projection(shape, 2, 0, 0));
        try single.record(&cmd, try single.projection(shape, up_offset, 0, M * 4));
        try fused.record(&cmd, geo);
        try cmd.barrier(.compute, .host);
        try cmd.end();
        try cmd.run(w.timeout_ns);
        const ys = std.mem.bytesAsSlice(f32, try output.mapped());
        for (0..M) |row| {
            const g = ys[row];
            const u = ys[M + row];
            try t.expectEqual(@as(u32, @bitCast(g)), @as(u32, @bitCast(ys[2 * M + row])));
            try t.expectEqual(@as(u32, @bitCast(u)), @as(u32, @bitCast(ys[3 * M + row])));
            const g64: f64 = g;
            const want = g64 / (1 + @exp(-g64)) * @as(f64, u);
            const y: f64 = ys[4 * M + row];
            // GPU exp: relative error grows with |x| (x log2 e is rounded), as in the gate
            // bound of the attention test.
            const bound = (8 + 2 * @abs(g64)) * std.math.ldexp(@as(f64, 1), -23) * @abs(want) + 1e-30;
            if (!(@abs(y - want) <= bound)) {
                std.debug.print("{s} row {d}: g {e} u {e} y {e} want {e}\n", .{ case.name, row, g, u, y, want });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        }
    };
    try t.expect(checked > 0);
    // Validation: outputs may not overlap each other or the input.
    var buffer = try gpu.Buffer.init(&device, 1 << 20, .host);
    defer buffer.deinit() catch @panic("b");
    var p = try m.SwigluPipeline.init(.q4_0, false, &buffer, &buffer, &buffer);
    defer p.deinit() catch @panic("p");
    const shape: m.Shape = .{ .format = .q4_0, .columns = 64, .rows = 8 };
    try t.expectError(error.AliasedOutput, p.projection(shape, 0, 1024, 4096, 8192, 8192 + 16, 16384));
    try t.expectError(error.AliasedOutput, p.projection(shape, 0, 1024, 4096, 4096 + 64, 16384, 20480));
    _ = try p.projection(shape, 0, 1024, 4096, 8192, 12288, 16384);
    try t.expectError(error.InvalidShape, m.SwigluPipeline.init(.f32, false, &buffer, &buffer, &buffer));
}
