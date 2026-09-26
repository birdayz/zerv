const std = @import("std");
const gpu = @import("gpu");
const vk = gpu.testing.abi;
const t = std.testing;

test "Vulkan C ABI: every scoped structure, field offset and numeric constant" {
    @setEvalBranchQuota(10000);
    if (@import("builtin").cpu.arch != .x86_64 or @import("builtin").os.tag != .linux) return error.SkipZigTest;
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, @embedFile("fixtures/gpu/abi.json"), .{});
    defer parsed.deinit();
    const structs = parsed.value.object.get("structs").?.array.items;
    const constants = parsed.value.object.get("constants").?.object;
    var checked_structs: usize = 0;
    var checked_constants: usize = 0;
    inline for (comptime std.meta.declarations(vk)) |decl| {
        // Do not take addresses of extern functions: even Debug info can require linkage.
        if (comptime std.mem.startsWith(u8, decl.name, "vk")) continue;
        const value = @field(vk, decl.name);
        if (@TypeOf(value) == type) {
            if (@typeInfo(value) == .@"struct") {
                var found = false;
                for (structs) |item| {
                    if (!std.mem.eql(u8, item.object.get("name").?.string, decl.name)) continue;
                    found = true;
                    try t.expectEqual(@as(i64, @sizeOf(value)), item.object.get("size").?.integer);
                    try t.expectEqual(@as(i64, @alignOf(value)), item.object.get("alignment").?.integer);
                    const fields = item.object.get("fields").?.object;
                    try t.expectEqual(std.meta.fields(value).len, fields.count());
                    inline for (comptime std.meta.fields(value)) |field| {
                        try t.expectEqual(@as(i64, @offsetOf(value, field.name)), fields.get(field.name).?.integer);
                    }
                }
                try t.expect(found);
                checked_structs += 1;
            }
        } else if (@TypeOf(value) == comptime_int) {
            try t.expectEqual(@as(i64, value), constants.get(decl.name).?.integer);
            checked_constants += 1;
        }
    }
    try t.expectEqual(@as(usize, 64), checked_structs);
    try t.expectEqual(structs.len, checked_structs);
    try t.expectEqual(@as(usize, 86), checked_constants);
    try t.expectEqual(constants.count(), checked_constants);
}

test "GPU memory selection requires coherent host mapping and compatible type bits" {
    var properties: vk.VkPhysicalDeviceMemoryProperties = .{ .memoryTypeCount = 4, .memoryHeapCount = 2 };
    properties.memoryTypes[0] = .{ .propertyFlags = 1, .heapIndex = 0 };
    properties.memoryTypes[1] = .{ .propertyFlags = 7, .heapIndex = 0 };
    properties.memoryTypes[2] = .{ .propertyFlags = 6, .heapIndex = 1 };
    properties.memoryTypes[3] = .{ .propertyFlags = 2, .heapIndex = 1 };
    try t.expectEqual(@as(u32, 0), try gpu.testing.memoryType(&properties, 15, .device));
    try t.expectEqual(@as(u32, 2), try gpu.testing.memoryType(&properties, 15, .host));
    try t.expectEqual(@as(u32, 1), try gpu.testing.memoryType(&properties, 2, .host));
    try t.expectError(error.UnsupportedMemoryType, gpu.testing.memoryType(&properties, 8, .host));
    try t.expectError(error.UnsupportedMemoryType, gpu.testing.memoryType(&properties, 0, .device));
    // Advertised optional memory flags are NOT automatically enabled capabilities.
    for ([_]u32{ 0x10, 0x20, 0x40, 0x80, 0x100, 0x80000000 }) |extra| {
        properties.memoryTypes[3].propertyFlags = 6 | extra;
        try t.expectEqual(@as(u32, 2), try gpu.testing.memoryType(&properties, 15, .host));
        try t.expectError(error.UnsupportedMemoryType, gpu.testing.memoryType(&properties, 8, .host));
        properties.memoryTypes[3].propertyFlags = 1 | extra;
        try t.expectError(error.UnsupportedMemoryType, gpu.testing.memoryType(&properties, 8, .device));
    }
    properties.memoryTypes[3].heapIndex = 2;
    try t.expectError(error.InvalidDriverProperties, gpu.testing.memoryType(&properties, 15, .host));
    properties.memoryTypeCount = 33;
    try t.expectError(error.InvalidDriverProperties, gpu.testing.memoryType(&properties, 15, .host));
}

test "GPU copy extent arithmetic and trusted module header guards" {
    try t.expect(gpu.testing.validRange(64, 4, 60));
    for ([_][3]u64{ .{ 64, 0, 0 }, .{ 64, 2, 4 }, .{ 64, 4, 5 }, .{ 64, 64, 4 }, .{ 64, 68, 4 }, .{ 64, 4, std.math.maxInt(u64) - 3 } }) |args|
        try t.expect(!gpu.testing.validRange(args[0], args[1], args[2]));
    try gpu.testing.validateCode(@embedFile("fixtures/gpu/affine.spv"));
    var code: [20]u8 = @splat(0);
    try t.expectError(error.InvalidShader, gpu.testing.validateCode(&code));
    @memcpy(&code, @embedFile("fixtures/gpu/affine.spv")[0..20]);
    for ([_]usize{ 0, 4, 12, 16 }) |at| {
        var changed = code;
        std.mem.writeInt(u32, changed[at..][0..4], if (at == 16) 1 else 0, .little);
        try t.expectError(error.InvalidShader, gpu.testing.validateCode(&changed));
    }
    try t.expectError(error.InvalidShader, gpu.testing.validateCode(code[0..19]));
}

test "GPU driver errors preserve VkResult and device loss without hardware" {
    var device: gpu.Device = .{ .budget = 1 };
    try device.check(vk.VK_SUCCESS);
    try t.expectError(error.DeviceOutOfMemory, device.check(vk.VK_ERROR_OUT_OF_DEVICE_MEMORY));
    try t.expectEqual(@as(i32, vk.VK_ERROR_OUT_OF_DEVICE_MEMORY), device.last_result);
    try t.expectError(error.Timeout, device.check(vk.VK_TIMEOUT));
    try t.expect(!device.lost);
    try t.expectError(error.DeviceLost, device.check(vk.VK_ERROR_DEVICE_LOST));
    try t.expect(device.lost);
    try t.expectEqual(@as(i32, vk.VK_ERROR_DEVICE_LOST), device.last_result);
}
