const vk = @import("vk.zig");
const driver = @import("device.zig");
const Device = driver.Device;
const Error = driver.Error;

pub const Buffer = struct {
    device: *Device,
    handle: vk.VkBuffer = null,
    allocation: vk.VkDeviceMemory = null,
    size: u64,
    allocation_size: u64 = 0,
    memory_type: u32 = 0,
    mapping: ?*anyopaque = null,
    references: u32 = 0,

    pub fn init(device: *Device, size: u64, location: driver.Location) Error!Buffer {
        try device.ready();
        if (size == 0 or size > device.budget) return error.InvalidRange;
        if (device.buffers >= 128 or device.buffers >= device.properties.limits.maxMemoryAllocationCount) return error.ResourceLimit;
        var self: Buffer = .{ .device = device, .size = size };
        const info: vk.VkBufferCreateInfo = .{ .size = size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE };
        try device.check(vk.vkCreateBuffer(device.handle, &info, null, &self.handle));
        errdefer if (self.handle != null) vk.vkDestroyBuffer(device.handle, self.handle, null);
        var requirements: vk.VkMemoryRequirements = .{};
        vk.vkGetBufferMemoryRequirements(device.handle, self.handle, &requirements);
        if (requirements.size < size or requirements.alignment == 0) return error.InvalidDriverProperties;
        if (requirements.size > device.budget - device.allocated_bytes) return error.ResourceLimit;
        self.memory_type = try driver.memoryType(&device.memory, requirements.memoryTypeBits, location);
        self.allocation_size = requirements.size;
        const allocation: vk.VkMemoryAllocateInfo = .{ .allocationSize = requirements.size, .memoryTypeIndex = self.memory_type };
        try device.check(vk.vkAllocateMemory(device.handle, &allocation, null, &self.allocation));
        // A bound buffer must be destroyed before its memory is freed, even on map failure.
        errdefer {
            vk.vkDestroyBuffer(device.handle, self.handle, null);
            self.handle = null;
            vk.vkFreeMemory(device.handle, self.allocation, null);
        }
        try device.check(vk.vkBindBufferMemory(device.handle, self.handle, self.allocation, 0));
        if (location == .host) try device.check(vk.vkMapMemory(device.handle, self.allocation, 0, size, 0, &self.mapping));
        device.allocated_bytes += requirements.size;
        device.buffers += 1;
        return self;
    }

    /// Borrowed coherent span. Never access retained spans while the device has pending commands.
    pub fn mapped(self: *Buffer) Error![]u8 {
        try self.device.ready();
        if (self.handle == null) return error.InvalidState;
        if (self.device.pending != 0) return error.ResourceInUse;
        const pointer: [*]u8 = @ptrCast(self.mapping orelse return error.NotHostVisible);
        return pointer[0..@intCast(self.size)];
    }

    pub fn deinit(self: *Buffer) Error!void {
        if (self.handle == null) return error.InvalidState;
        if (self.references != 0) return error.ResourceInUse;
        if (self.mapping != null) vk.vkUnmapMemory(self.device.handle, self.allocation);
        vk.vkDestroyBuffer(self.device.handle, self.handle, null);
        vk.vkFreeMemory(self.device.handle, self.allocation, null);
        self.device.allocated_bytes -= self.allocation_size;
        self.device.buffers -= 1;
        self.handle = null;
        self.allocation = null;
        self.mapping = null;
    }
};

pub fn validRange(total: u64, offset: u64, size: u64) bool {
    return size != 0 and size % 4 == 0 and offset % 4 == 0 and offset <= total and size <= total - offset;
}
