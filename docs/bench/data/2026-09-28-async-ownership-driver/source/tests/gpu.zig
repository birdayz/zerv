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

test "imported host memory (VK_EXT_external_memory_host): device round trip, bounds, ownership" {
    const linux = std.os.linux;
    const size: usize = 4 << 20;
    const rc = linux.mmap(null, 2 * size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try t.expectEqual(linux.E.SUCCESS, linux.errno(rc));
    const mem = @as([*]u8, @ptrFromInt(rc))[0 .. 2 * size];
    defer _ = linux.munmap(mem.ptr, mem.len);
    // Without the option the entry point is absent.
    {
        var plain = try gpu.Device.open(.{ .max_allocated_bytes = 64 << 20 });
        defer plain.deinit() catch @panic("device still has children");
        try t.expectError(error.UnsupportedFeature, gpu.Buffer.initImported(&plain, mem[0..size]));
    }
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 64 << 20, .host_import = true });
    defer device.deinit() catch @panic("device still has children");
    try t.expect(device.host_import_alignment > 0);
    try t.expectError(error.InvalidRange, gpu.Buffer.initImported(&device, mem[0..0]));
    try t.expectError(error.InvalidRange, gpu.Buffer.initImported(&device, mem[1..][0..4096])); // unaligned start
    try t.expectError(error.InvalidRange, gpu.Buffer.initImported(&device, mem[0..100])); // unaligned length
    for (mem[0..size], 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 5);
    var a = try gpu.Buffer.initImported(&device, mem[0..size]);
    const budget = device.budget;
    device.budget = size + size / 2;
    try t.expectError(error.ResourceLimit, gpu.Buffer.initImported(&device, mem[size..][0..size]));
    try t.expectEqual(@as(u64, size), device.allocated_bytes);
    try t.expectEqual(@as(u32, 1), device.buffers);
    device.budget = budget;
    var b = try gpu.Buffer.initImported(&device, mem[size..][0..size]);
    var d = try gpu.Buffer.init(&device, size, .device);
    try t.expectEqual(@as(u64, 3 * size), device.allocated_bytes);
    try t.expect((try a.mapped()).ptr == mem.ptr);
    {
        var c = try gpu.Commands.init(&device);
        defer c.deinit() catch @panic("commands");
        try c.reset();
        try c.begin();
        try c.copy(&a, 0, &d, 0, size);
        try t.expectError(error.ResourceInUse, a.deinit());
        try c.barrier(.transfer, .transfer);
        try c.copy(&d, 0, &b, 0, size);
        try c.barrier(.transfer, .host);
        try c.end();
        try c.run(5_000_000_000);
    }
    try t.expect(std.mem.eql(u8, mem[0..size], mem[size..][0..size]));
    try d.deinit();
    try b.deinit();
    try a.deinit();
    try t.expectEqual(@as(u64, 0), device.allocated_bytes);
    // The memory stays the caller's: still mapped and writable after deinit.
    mem[0] = 7;
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
    try t.expectError(error.InvalidState, commands.poll());
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
    // This command only dispatches: descriptor bindings, not direct copy refs.
    try t.expectEqual(@as(u32, 1), input.pending_uses);
    try t.expectEqual(@as(u32, 1), output.pending_uses);
    try t.expectError(error.InvalidLimit, commands.wait(std.math.maxInt(u64)));
    commands.wait(0) catch |err| {
        try t.expectEqual(error.Timeout, err);
        try t.expectEqual(gpu.Commands.State.pending, commands.state);
        try commands.wait(workload.timeout_ns);
    };
    try t.expectEqual(gpu.Commands.State.executable, commands.state);
    try t.expectError(error.InvalidState, commands.poll());
    try t.expectEqual(@as(u32, 0), input.pending_uses);
    try t.expectEqual(@as(u32, 0), output.pending_uses);
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

test "asynchronous fences retain only their buffers, multiple owners, either acknowledgment order and replay" {
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 8 << 20 });
    defer device.deinit() catch @panic("live device");
    var input = try gpu.Buffer.init(&device, 1 << 20, .host);
    defer input.deinit() catch @panic("live input");
    var a = try gpu.Buffer.init(&device, 1 << 20, .host);
    defer a.deinit() catch @panic("live output a");
    var b = try gpu.Buffer.init(&device, 1 << 20, .host);
    defer b.deinit() catch @panic("live output b");
    var unrelated = try gpu.Buffer.init(&device, 4096, .host);
    defer unrelated.deinit() catch @panic("live unrelated");
    const bytes = try input.mapped();
    for (bytes, 0..) |*v, i| v.* = @truncate(i *% 2654435761 >> 5);
    var commands: [2]gpu.Commands = .{ try gpu.Commands.init(&device), try gpu.Commands.init(&device) };
    defer for (&commands) |*c| c.deinit() catch @panic("pending DMA");
    for (&commands, [_]*gpu.Buffer{ &a, &b }) |*c, target| {
        try c.begin();
        try c.barrier(.host, .transfer);
        try c.copy(&input, 0, target, 0, input.size);
        try c.barrier(.transfer, .host);
        try c.end();
    }
    // Do not depend on how fast the GPU finishes: guards last until acknowledgment.
    for ([_]usize{ 0, 1, 0, 1 }) |first| {
        for (&commands) |*c| try c.submit();
        try t.expectEqual(@as(u32, 2), device.pending);
        try t.expectEqual(@as(u32, 2), input.pending_uses);
        try t.expectError(error.ResourceInUse, input.mapped());
        try t.expectError(error.ResourceInUse, a.mapped());
        try t.expectError(error.ResourceInUse, b.mapped());
        @memset(try unrelated.mapped(), 0xa5);
        try pollUntilDone(&commands[first]);
        try t.expectEqual(@as(u32, 1), input.pending_uses);
        try t.expectError(error.ResourceInUse, input.mapped());
        const ready = if (first == 0) &a else &b;
        for (try ready.mapped(), 0..) |v, i| try t.expectEqual(@as(u8, @truncate(i *% 2654435761 >> 5)), v);
        try t.expectError(error.InvalidState, commands[first].poll());
        try t.expectError(error.ResourceInUse, commands[1 - first].reset());
        try pollUntilDone(&commands[1 - first]);
        try t.expectEqual(@as(u32, 0), device.pending);
        try t.expectEqual(@as(u32, 0), input.pending_uses);
        try t.expectEqualSlices(u8, try input.mapped(), try a.mapped());
        try t.expectEqualSlices(u8, try input.mapped(), try b.mapped());
        try t.expectError(error.ResourceInUse, input.deinit()); // recorded refs remain
    }
}

fn pollUntilDone(c: *gpu.Commands) !void {
    for (0..10000) |_| {
        if (try c.poll()) return;
        try t.expectEqual(gpu.Commands.State.pending, c.state);
        try std.Io.sleep(t.io, .fromMicroseconds(100), .awake);
    }
    @panic("GPU did not complete; cannot release pending DMA");
}
