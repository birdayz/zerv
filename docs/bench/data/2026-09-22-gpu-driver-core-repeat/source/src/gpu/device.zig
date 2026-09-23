const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vk.zig");

pub const Error = error{ UnsupportedTarget, InvalidLimit, EnumerationLimit, NoDevice, InvalidDriverProperties, DriverError, HostOutOfMemory, DeviceOutOfMemory, DeviceLost, Timeout, ResourceLimit, ResourceInUse, InvalidState, InvalidRange, UnsupportedMemoryType, NotHostVisible, WrongDevice, InvalidShader, InvalidLayout, InvalidDispatch };
pub const Location = enum { device, host };
pub const Options = struct { device_index: ?u32 = null, max_allocated_bytes: u64 };

/// Stable-address owner. Externally serialize this device and all its children.
pub const Device = struct {
    instance: vk.VkInstance = null,
    physical: vk.VkPhysicalDevice = null,
    handle: vk.VkDevice = null,
    queue: vk.VkQueue = null,
    family: u32 = 0,
    properties: vk.VkPhysicalDeviceProperties = .{},
    memory: vk.VkPhysicalDeviceMemoryProperties = .{},
    budget: u64,
    allocated_bytes: u64 = 0,
    buffers: u32 = 0,
    kernels: u32 = 0,
    commands: u32 = 0,
    pending: u32 = 0,
    last_result: i32 = 0,
    lost: bool = false,

    pub fn open(options: Options) Error!Device {
        if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.UnsupportedTarget;
        if (options.max_allocated_bytes == 0) return error.InvalidLimit;
        var self: Device = .{ .budget = options.max_allocated_bytes };
        const app: vk.VkApplicationInfo = .{ .pApplicationName = "zerv", .apiVersion = vk.VK_API_VERSION_1_1 };
        const info: vk.VkInstanceCreateInfo = .{ .pApplicationInfo = &app };
        try self.check(vk.vkCreateInstance(&info, null, &self.instance));
        errdefer vk.vkDestroyInstance(self.instance, null);
        var devices: [16]vk.VkPhysicalDevice = undefined;
        var count: u32 = devices.len;
        const result = vk.vkEnumeratePhysicalDevices(self.instance, &count, &devices);
        if (result == vk.VK_INCOMPLETE) return error.EnumerationLimit;
        try self.check(result);
        if (count > devices.len) return error.InvalidDriverProperties;
        for (devices[0..count], 0..) |physical, index| {
            if (options.device_index) |wanted| if (wanted != index) continue;
            var properties: vk.VkPhysicalDeviceProperties = .{};
            vk.vkGetPhysicalDeviceProperties(physical, &properties);
            if (properties.apiVersion < vk.VK_API_VERSION_1_1) continue;
            if (options.device_index == null and properties.deviceType != vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            var queue_count: u32 = 0;
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, null);
            if (queue_count > 32) return error.EnumerationLimit;
            var families: [32]vk.VkQueueFamilyProperties = undefined;
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, &families);
            if (queue_count > families.len) return error.InvalidDriverProperties;
            var selected: ?u32 = null;
            for (families[0..queue_count], 0..) |family, i| {
                if (family.queueCount == 0 or family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT == 0) continue;
                if (selected == null or family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT == 0) selected = @intCast(i);
            }
            self.family = selected orelse continue;
            self.physical = physical;
            self.properties = properties;
            break;
        }
        if (self.physical == null) return error.NoDevice;
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical, &self.memory);
        if (self.memory.memoryTypeCount > 32 or self.memory.memoryHeapCount > 16) return error.InvalidDriverProperties;
        const priority: f32 = 1;
        const queue_info: vk.VkDeviceQueueCreateInfo = .{ .queueFamilyIndex = self.family, .queueCount = 1, .pQueuePriorities = &priority };
        const device_info: vk.VkDeviceCreateInfo = .{ .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info };
        try self.check(vk.vkCreateDevice(self.physical, &device_info, null, &self.handle));
        vk.vkGetDeviceQueue(self.handle, self.family, 0, &self.queue);
        return self;
    }

    pub fn name(self: *const Device) []const u8 {
        return std.mem.sliceTo(&self.properties.deviceName, 0);
    }

    pub fn deinit(self: *Device) Error!void {
        if (self.handle == null) return error.InvalidState;
        if (self.buffers != 0 or self.kernels != 0 or self.commands != 0 or self.pending != 0) return error.ResourceInUse;
        vk.vkDestroyDevice(self.handle, null);
        vk.vkDestroyInstance(self.instance, null);
        self.handle = null;
        self.instance = null;
    }

    pub fn ready(self: *const Device) Error!void {
        if (self.handle == null) return error.InvalidState;
        if (self.lost) return error.DeviceLost;
    }

    pub fn check(self: *Device, result: vk.VkResult) Error!void {
        if (result == vk.VK_SUCCESS) return;
        self.last_result = result;
        if (result == vk.VK_ERROR_DEVICE_LOST) self.lost = true;
        return switch (result) {
            vk.VK_ERROR_OUT_OF_HOST_MEMORY => error.HostOutOfMemory,
            vk.VK_ERROR_OUT_OF_DEVICE_MEMORY => error.DeviceOutOfMemory,
            vk.VK_ERROR_DEVICE_LOST => error.DeviceLost,
            vk.VK_TIMEOUT => error.Timeout,
            else => error.DriverError,
        };
    }
};

pub fn memoryType(properties: *const vk.VkPhysicalDeviceMemoryProperties, allowed: u32, location: Location) Error!u32 {
    if (properties.memoryTypeCount > 32 or properties.memoryHeapCount > 16) return error.InvalidDriverProperties;
    const required: u32 = if (location == .host) vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT else vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    var selected: ?u32 = null;
    for (properties.memoryTypes[0..properties.memoryTypeCount], 0..) |item, i| {
        if (item.heapIndex >= properties.memoryHeapCount) return error.InvalidDriverProperties;
        const base_flags: u32 = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        // No protected/lazy/extension memory features are enabled on this device.
        if (item.propertyFlags & ~base_flags != 0) continue;
        if (allowed & (@as(u32, 1) << @intCast(i)) == 0 or item.propertyFlags & required != required) continue;
        if (selected == null or (location == .host and item.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT == 0)) selected = @intCast(i);
    }
    return selected orelse error.UnsupportedMemoryType;
}
