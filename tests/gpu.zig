//! Explicit hardware tests: bazel test //tests:gpu, not part of driver-free CPU tests.
const std = @import("std");
const gpu = @import("zerv").gpu;
const workload = @import("gpu_workload.zig");
const t = std.testing;

test {
    _ = @import("matvec_gpu.zig");
    _ = @import("model_gpu.zig");
    _ = @import("gpu_coopmat.zig");
    _ = @import("gpu_gemm_f16.zig");
}

test "native Vulkan real transfers, affine dispatch, partial groups and sentinel goldens" {
    const parsed = try workload.goldens(t.allocator);
    defer parsed.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 512 * 1024 * 1024 });
    defer device.deinit() catch @panic("device still has children");
    for (parsed.value.cases) |case| {
        var run: workload.Workload = undefined;
        try run.init(&device, case.kind, case.count);
        defer run.deinit();
        try run.execute();
        _ = try run.finish(); // Also replay after a transfer readback of the compute output.
        // Replay without command rebuilding and ensure full output still matches.
        try run.execute();
        const result = try run.finish();
        try t.expectEqual(case.bytes, result.bytes);
        try t.expectEqualStrings(case.input_sha256, &result.input_sha256);
        try t.expectEqualStrings(case.output_sha256, &result.output_sha256);
    }
    try t.expectEqual(@as(u64, 0), device.allocated_bytes);
    try t.expectEqual(@as(u32, 0), device.pending);
}

test "memory budget: the device-local heap reports this process's own allocations" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 2 * 1024 * 1024 * 1024 });
    defer device.deinit() catch @panic("device still has children");
    // Required on the target (RADV); the server's free-VRAM check depends on it.
    const before = (try device.memoryBudget()) orelse return error.NoMemoryBudget;
    try t.expect(before.size > 0 and before.budget <= before.size and before.usage <= before.budget);
    const bytes: u64 = 1024 * 1024 * 1024;
    var buffer = try gpu.Buffer.init(&device, bytes, .device);
    const during = (try device.memoryBudget()).?;
    try buffer.deinit();
    const after = (try device.memoryBudget()).?;
    try t.expectEqual(before.heap, during.heap);
    // Usage grows by the allocation (allocation padding is small), free space shrinks.
    try t.expect(during.usage >= before.usage + bytes and during.usage <= before.usage + bytes + 64 * 1024 * 1024);
    try t.expect(during.free() + bytes <= before.free() + 64 * 1024 * 1024);
    try t.expect(after.usage <= before.usage + 64 * 1024 * 1024);
}

test "native Vulkan ownership, budget rollback and command state guards" {
    try t.expectError(error.InvalidLimit, gpu.Device.open(.{ .max_allocated_bytes = 0 }));
    try t.expectError(error.NoDevice, gpu.Device.open(.{ .device_index = 1000, .max_allocated_bytes = 4096 }));
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 });
    defer device.deinit() catch @panic("device still has children");
    try t.expectError(error.InvalidRange, gpu.Buffer.init(&device, 0, .host));
    var input = try gpu.Buffer.init(&device, 512, .host);
    defer input.deinit() catch @panic("input still referenced");
    var output = try gpu.Buffer.init(&device, 512, .device);
    defer output.deinit() catch @panic("output still referenced");
    try t.expectError(error.NotHostVisible, output.mapped());
    try t.expectEqual(@as(u32, 0), device.memory.memoryTypes[input.memory_type].propertyFlags & ~@as(u32, 15));
    try t.expectEqual(@as(u32, 0), device.memory.memoryTypes[output.memory_type].propertyFlags & ~@as(u32, 15));
    try t.expectError(error.ResourceInUse, device.deinit());
    const allocated = device.allocated_bytes;
    device.budget = allocated;
    try t.expectError(error.ResourceLimit, gpu.Buffer.init(&device, 4, .device));
    try t.expectEqual(allocated, device.allocated_bytes);
    try t.expectEqual(@as(u32, 2), device.buffers);
    device.budget = 1024 * 1024;
    var kernel = try gpu.Kernel.init(&device, &workload.shader, &.{ &input, &output }, 4);
    defer kernel.deinit() catch @panic("kernel still referenced");
    try t.expectError(error.ResourceInUse, input.deinit());
    try t.expectError(error.InvalidShader, gpu.Kernel.init(&device, &.{}, &.{ &input, &output }, 4));
    try t.expectError(error.InvalidLayout, gpu.Kernel.init(&device, &workload.shader, &.{}, 4));
    try t.expectError(error.InvalidLayout, gpu.Kernel.init(&device, &workload.shader, &.{&input}, 3));
    try t.expectError(error.InvalidLayout, gpu.Kernel.init(&device, &workload.shader, &.{&input}, 132));
    var commands = try gpu.Commands.init(&device);
    defer commands.deinit() catch @panic("commands still pending");
    try t.expectError(error.InvalidState, commands.end());
    try t.expectError(error.InvalidState, commands.submit());
    try t.expectError(error.InvalidState, commands.wait(0));
    try commands.begin();
    try t.expectError(error.InvalidState, commands.begin());
    try t.expectError(error.InvalidRange, commands.copy(&input, 0, &output, 0, 0));
    try t.expectError(error.InvalidRange, commands.copy(&input, 2, &output, 0, 4));
    try t.expectError(error.InvalidRange, commands.copy(&input, 0, &output, 512, 4));
    try t.expectError(error.InvalidRange, commands.copy(&input, 0, &input, 0, 4));
    const push = [_]u8{ 1, 0, 0, 0 };
    try t.expectError(error.InvalidLayout, commands.dispatch(&kernel, &.{}, .{ 1, 1, 1 }));
    try t.expectError(error.InvalidDispatch, commands.dispatch(&kernel, &push, .{ 0, 1, 1 }));
    for (device.properties.limits.maxComputeWorkGroupCount, 0..) |limit, axis| {
        if (limit == std.math.maxInt(u32)) continue;
        var groups = [_]u32{ 1, 1, 1 };
        groups[axis] = limit + 1;
        try t.expectError(error.InvalidDispatch, commands.dispatch(&kernel, &push, groups));
    }
    @memset(try input.mapped(), 0);
    try commands.dispatch(&kernel, &push, .{ 1, 1, 1 });
    try t.expectError(error.ResourceInUse, kernel.deinit());
    try commands.end();
    try commands.submit();
    try t.expectError(error.InvalidState, commands.submit());
    try t.expectError(error.ResourceInUse, commands.reset());
    try t.expectError(error.ResourceInUse, commands.deinit());
    try t.expectError(error.ResourceInUse, input.mapped());
    try t.expectError(error.InvalidLimit, commands.wait(std.math.maxInt(u64)));
    commands.wait(0) catch |err| {
        try t.expectEqual(error.Timeout, err);
        try t.expectEqual(gpu.Commands.State.pending, commands.state);
        try commands.wait(workload.timeout_ns);
    };
    try t.expectEqual(gpu.Commands.State.executable, commands.state);
    try commands.reset();
    try t.expectEqual(@as(u32, 0), kernel.references);
    try t.expectEqual(@as(u32, 0), device.pending);
    try commands.begin();
    try commands.barrier(.compute, .transfer);
    try commands.copy(&input, 0, &output, 0, 4);
    try commands.end();
    try commands.run(workload.timeout_ns);
    try commands.reset();
}

test "native Vulkan bounded children, retained references and cross-device rejection" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 16 * 1024 * 1024 });
    defer device.deinit() catch @panic("device still has children");
    var buffers: [128]gpu.Buffer = undefined;
    var buffer_count: usize = 0;
    defer for (buffers[0..buffer_count]) |*buffer| buffer.deinit() catch @panic("buffer cleanup");
    for (&buffers) |*buffer| {
        buffer.* = try gpu.Buffer.init(&device, 16, .device);
        buffer_count += 1;
    }
    try t.expectError(error.ResourceLimit, gpu.Buffer.init(&device, 16, .device));
    var kernels: [gpu.max_kernels]gpu.Kernel = undefined;
    var kernel_count: usize = 0;
    defer for (kernels[0..kernel_count]) |*kernel| kernel.deinit() catch @panic("kernel cleanup");
    for (&kernels) |*kernel| {
        kernel.* = try gpu.Kernel.init(&device, &workload.shader, &.{ &buffers[0], &buffers[1] }, 4);
        kernel_count += 1;
    }
    try t.expectError(error.ResourceLimit, gpu.Kernel.init(&device, &workload.shader, &.{ &buffers[0], &buffers[1] }, 4));
    var commands: [gpu.max_commands]gpu.Commands = undefined;
    var command_count: usize = 0;
    defer for (commands[0..command_count]) |*cmd| cmd.deinit() catch @panic("command cleanup");
    for (&commands) |*cmd| {
        cmd.* = try gpu.Commands.init(&device);
        command_count += 1;
    }
    try t.expectError(error.ResourceLimit, gpu.Commands.init(&device));
    const cmd = &commands[0];
    try cmd.begin();
    for (1..64) |i| try cmd.copy(&buffers[0], 0, &buffers[i], 0, 4);
    try t.expectError(error.ResourceLimit, cmd.copy(&buffers[0], 0, &buffers[64], 0, 4));
    try t.expectEqual(@as(u32, 0), buffers[64].references);
    const n = gpu.max_command_kernels;
    for (kernels[0..n]) |*kernel| try cmd.dispatch(kernel, &.{ 1, 0, 0, 0 }, .{ 1, 1, 1 });
    try t.expectError(error.ResourceLimit, cmd.dispatch(&kernels[n], &.{ 1, 0, 0, 0 }, .{ 1, 1, 1 }));
    try t.expectEqual(@as(u32, 0), kernels[n].references);
    try cmd.reset();
    try t.expectEqual(@as(u32, 0), buffers[63].references);
    try t.expectEqual(@as(u32, 0), kernels[n - 1].references);
    var other = try gpu.Device.open(.{ .max_allocated_bytes = 1024 * 1024 });
    defer other.deinit() catch @panic("other device cleanup");
    var other_buffer = try gpu.Buffer.init(&other, 16, .device);
    defer other_buffer.deinit() catch @panic("other buffer cleanup");
    try cmd.begin();
    try t.expectError(error.WrongDevice, cmd.copy(&buffers[0], 0, &other_buffer, 0, 4));
    // Use the second device (which has spare kernel capacity) to reach device validation.
    try t.expectError(error.WrongDevice, gpu.Kernel.init(&other, &workload.shader, &.{ &other_buffer, &buffers[0] }, 4));
    try t.expectEqual(@as(usize, 0), cmd.buffer_count);
}
