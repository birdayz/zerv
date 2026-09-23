//! Private test/benchmark instrumentation. No production GPU API or timing claim.
const std = @import("std");
const gpu = @import("zerv").gpu;
const vk = gpu.testing.abi;
const QueryPool = ?*opaque {};
const QueryInfo = extern struct {
    sType: i32 = 11,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queryType: i32 = 2,
    queryCount: u32 = 2,
    pipelineStatistics: u32 = 0,
};
extern fn vkCreateQueryPool(vk.VkDevice, *const QueryInfo, ?*const anyopaque, *QueryPool) callconv(.c) i32;
extern fn vkDestroyQueryPool(vk.VkDevice, QueryPool, ?*const anyopaque) callconv(.c) void;
extern fn vkCmdResetQueryPool(vk.VkCommandBuffer, QueryPool, u32, u32) callconv(.c) void;
extern fn vkCmdWriteTimestamp(vk.VkCommandBuffer, u32, QueryPool, u32) callconv(.c) void;
extern fn vkGetQueryPoolResults(vk.VkDevice, QueryPool, u32, u32, usize, *anyopaque, u64, u32) callconv(.c) i32;

pub fn ticks(start: u64, end: u64, bits: u32) !u64 {
    if (bits == 0 or bits > 64) return error.InvalidTimestampBits;
    const mask = if (bits == 64) std.math.maxInt(u64) else (@as(u64, 1) << @as(u6, @intCast(bits))) - 1;
    return (end -% start) & mask;
}

/// Stable addresses, externally serialized. The recorded command must be
/// destroyed before this pool, and this pool before its device.
pub const Timing = struct {
    device: *gpu.Device,
    pool: QueryPool,
    bits: u32,
    period: f64,
    command: ?*gpu.Commands = null,
    ended: bool = false,

    pub fn init(device: *gpu.Device) !Timing {
        try device.ready();
        var count: u32 = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(device.physical, &count, null);
        if (count > 32 or device.family >= count) return error.InvalidQueueCount;
        var families: [32]vk.VkQueueFamilyProperties = undefined;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(device.physical, &count, &families);
        if (count > 32 or device.family >= count) return error.InvalidQueueCount;
        const bits = families[device.family].timestampValidBits;
        _ = try ticks(0, 0, bits);
        const period = device.properties.limits.timestampPeriod;
        if (!std.math.isFinite(period) or period <= 0) return error.InvalidTimestampPeriod;
        var pool: QueryPool = null;
        try device.check(vkCreateQueryPool(device.handle, &.{}, null, &pool));
        return .{ .device = device, .pool = pool, .bits = bits, .period = period };
    }
    pub fn begin(self: *Timing, command: *gpu.Commands) !void {
        if (self.pool == null or self.command != null or command.state != .recording) return error.InvalidState;
        if (command.device != self.device) return error.WrongDevice;
        self.command = command;
        vkCmdResetQueryPool(command.handle, self.pool, 0, 2);
        vkCmdWriteTimestamp(command.handle, 1, self.pool, 0); // TOP_OF_PIPE
    }
    pub fn end(self: *Timing) !void {
        const command = self.command orelse return error.InvalidState;
        if (self.ended or command.state != .recording) return error.InvalidState;
        vkCmdWriteTimestamp(command.handle, 8192, self.pool, 1); // BOTTOM_OF_PIPE
        self.ended = true;
    }
    /// Caller must have successfully waited for the command's fence. No WAIT
    /// flag: a missing/unavailable result fails rather than blocking forever.
    pub fn read(self: *Timing, wall_ns: f64) !f64 {
        const command = self.command orelse return error.InvalidState;
        if (!self.ended or command.state != .executable or !std.math.isFinite(wall_ns) or wall_ns <= 0) return error.InvalidState;
        const wrap_ns = std.math.pow(f64, 2, @floatFromInt(self.bits)) * self.period;
        if (wall_ns >= wrap_ns) return error.AmbiguousTimestampWrap;
        var values: [2]u64 = undefined;
        try self.device.check(vkGetQueryPoolResults(self.device.handle, self.pool, 0, 2, @sizeOf(@TypeOf(values)), &values, 8, 1));
        return @as(f64, @floatFromInt(try ticks(values[0], values[1], self.bits))) * self.period;
    }
    pub fn deinit(self: *Timing) !void {
        if (self.pool == null) return error.InvalidState;
        if (self.command) |command| if (command.pool != null) return error.ResourceInUse;
        vkDestroyQueryPool(self.device.handle, self.pool, null);
        self.pool = null;
    }
};

/// Up to `capacity` BOTTOM_OF_PIPE timestamps in one recorded command: `begin` resets
/// the pool and writes mark 0; each `mark` writes the next. Same lifetime rules as Timing.
pub const Marks = struct {
    device: *gpu.Device,
    pool: QueryPool,
    bits: u32,
    period: f64,
    capacity: u32,
    count: u32 = 0,

    pub fn init(device: *gpu.Device, capacity: u32) !Marks {
        if (capacity < 2) return error.InvalidCapacity;
        const probe = try Timing.init(device); // validates bits/period on this queue
        var timing = probe;
        const bits = timing.bits;
        const period = timing.period;
        try timing.deinit();
        var pool: QueryPool = null;
        try device.check(vkCreateQueryPool(device.handle, &.{ .queryCount = capacity }, null, &pool));
        return .{ .device = device, .pool = pool, .bits = bits, .period = period, .capacity = capacity };
    }
    pub fn begin(self: *Marks, command: *gpu.Commands) !void {
        if (self.pool == null or command.state != .recording) return error.InvalidState;
        vkCmdResetQueryPool(command.handle, self.pool, 0, self.capacity);
        vkCmdWriteTimestamp(command.handle, 1, self.pool, 0);
        self.count = 1;
    }
    pub fn mark(self: *Marks, command: *gpu.Commands) !u32 {
        if (self.pool == null or command.state != .recording or self.count == 0) return error.InvalidState;
        if (self.count == self.capacity) return error.TooManyMarks;
        vkCmdWriteTimestamp(command.handle, 8192, self.pool, self.count);
        self.count += 1;
        return self.count - 1;
    }
    /// Nanoseconds between consecutive marks (out.len == count - 1), after a waited run.
    pub fn read(self: *Marks, out: []f64) !void {
        if (self.count < 2 or out.len != self.count - 1) return error.InvalidState;
        var values: [4096]u64 = undefined;
        if (self.count > values.len) return error.TooManyMarks;
        try self.device.check(vkGetQueryPoolResults(self.device.handle, self.pool, 0, self.count, self.count * 8, &values, 8, 1));
        for (out, values[0 .. self.count - 1], values[1..self.count]) |*o, a, b| o.* = @as(f64, @floatFromInt(try ticks(a, b, self.bits))) * self.period;
    }
    pub fn deinit(self: *Marks) void {
        if (self.pool != null) vkDestroyQueryPool(self.device.handle, self.pool, null);
        self.pool = null;
    }
};

test "private timestamp binding matches independent Khronos C ABI" {
    const fixture = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, @embedFile("fixtures/gpu/timing-abi.json"), .{});
    defer fixture.deinit();
    const obj = fixture.value.object;
    try std.testing.expectEqual(@as(i64, @sizeOf(QueryInfo)), obj.get("size").?.integer);
    try std.testing.expectEqual(@as(i64, @alignOf(QueryInfo)), obj.get("alignment").?.integer);
    inline for (std.meta.fields(QueryInfo)) |field| try std.testing.expectEqual(@as(i64, @offsetOf(QueryInfo, field.name)), obj.get(field.name).?.integer);
    for ([_][]const u8{ "VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO", "VK_QUERY_TYPE_TIMESTAMP", "VK_QUERY_RESULT_64_BIT", "VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT", "VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT" }, [_]i64{ 11, 2, 1, 1, 8192 }) |name, value| try std.testing.expectEqual(value, obj.get(name).?.integer);
    try std.testing.expectEqual(@as(u64, 6), try ticks(253, 259, 8));
    try std.testing.expectEqual(@as(u64, 6), try ticks(253, 3, 8));
    try std.testing.expectEqual(@as(u64, 6), try ticks(std.math.maxInt(u64) - 2, 3, 64));
    try std.testing.expectError(error.InvalidTimestampBits, ticks(0, 1, 0));
    try std.testing.expectError(error.InvalidTimestampBits, ticks(0, 1, 65));
}
