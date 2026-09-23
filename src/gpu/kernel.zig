const std = @import("std");
const vk = @import("vk.zig");
const driver = @import("device.zig");
const Device = driver.Device;
const Error = driver.Error;
const Buffer = @import("buffer.zig").Buffer;

/// Kernels a device may hold at once (bounded child count; docs/specs/gpu-driver.md).
pub const max_kernels = 128;

pub const Kernel = struct {
    device: *Device,
    handle: vk.VkPipeline = null,
    layout: vk.VkPipelineLayout = null,
    set_layout: vk.VkDescriptorSetLayout = null,
    pool: vk.VkDescriptorPool = null,
    set: vk.VkDescriptorSet = null,
    buffers: [8]*Buffer = undefined,
    buffer_count: usize = 0,
    push_bytes: u32,
    references: u32 = 0,

    pub const Options = struct {
        /// Require this subgroup size (a power of two within `device.subgroup_sizes`;
        /// needs a device opened with `subgroup_size_control`). Null: driver's choice.
        subgroup_size: ?u32 = null,
        /// Launch only full subgroups (`REQUIRE_FULL_SUBGROUPS`; needs
        /// `device.full_subgroups`). The local size X must be a multiple of the subgroup
        /// size (the required one, else the device maximum); the SPIR-V is trusted.
        full_subgroups: bool = false,
    };

    /// Trusted, validated SPIR-V only; callers own the shader/layout/shape safety contract.
    pub fn init(device: *Device, code: []align(4) const u8, buffers: []const *Buffer, push_bytes: u32) Error!Kernel {
        return initWith(device, code, buffers, push_bytes, .{});
    }

    pub fn initWith(device: *Device, code: []align(4) const u8, buffers: []const *Buffer, push_bytes: u32, options: Options) Error!Kernel {
        try device.ready();
        if (options.subgroup_size) |size| {
            if (device.subgroup_sizes.max == 0) return error.UnsupportedFeature;
            if (!std.math.isPowerOfTwo(size) or size < device.subgroup_sizes.min or size > device.subgroup_sizes.max) return error.InvalidLayout;
        }
        if (options.full_subgroups and !device.full_subgroups) return error.UnsupportedFeature;
        try validateCode(code);
        if (buffers.len == 0 or buffers.len > 8 or buffers.len > device.properties.limits.maxPerStageDescriptorStorageBuffers or buffers.len > device.properties.limits.maxDescriptorSetStorageBuffers or push_bytes > 128 or push_bytes > device.properties.limits.maxPushConstantsSize or push_bytes % 4 != 0) return error.InvalidLayout;
        if (device.kernels >= max_kernels) return error.ResourceLimit;
        for (buffers) |buffer| {
            if (buffer.device != device) return error.WrongDevice;
            if (buffer.handle == null) return error.InvalidState;
            if (buffer.size > device.properties.limits.maxStorageBufferRange) return error.InvalidLayout;
        }
        var self: Kernel = .{ .device = device, .push_bytes = push_bytes };
        var shader: vk.VkShaderModule = null;
        const shader_info: vk.VkShaderModuleCreateInfo = .{ .codeSize = code.len, .pCode = @ptrCast(code.ptr) };
        try device.check(vk.vkCreateShaderModule(device.handle, &shader_info, null, &shader));
        defer vk.vkDestroyShaderModule(device.handle, shader, null);
        var bindings: [8]vk.VkDescriptorSetLayoutBinding = undefined;
        for (bindings[0..buffers.len], 0..) |*binding, i| binding.* = .{ .binding = @intCast(i), .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT };
        const set_info: vk.VkDescriptorSetLayoutCreateInfo = .{ .bindingCount = @intCast(buffers.len), .pBindings = &bindings };
        try device.check(vk.vkCreateDescriptorSetLayout(device.handle, &set_info, null, &self.set_layout));
        errdefer vk.vkDestroyDescriptorSetLayout(device.handle, self.set_layout, null);
        const push: vk.VkPushConstantRange = .{ .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .size = push_bytes };
        const layout_info: vk.VkPipelineLayoutCreateInfo = .{ .setLayoutCount = 1, .pSetLayouts = &self.set_layout, .pushConstantRangeCount = if (push_bytes == 0) 0 else 1, .pPushConstantRanges = if (push_bytes == 0) null else &push };
        try device.check(vk.vkCreatePipelineLayout(device.handle, &layout_info, null, &self.layout));
        errdefer vk.vkDestroyPipelineLayout(device.handle, self.layout, null);
        const required: vk.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = .{ .requiredSubgroupSize = options.subgroup_size orelse 0 };
        const pipeline_info: vk.VkComputePipelineCreateInfo = .{ .stage = .{ .pNext = if (options.subgroup_size != null) &required else null, .flags = if (options.full_subgroups) vk.VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT else 0, .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" }, .layout = self.layout, .basePipelineIndex = -1 };
        // Vulkan can return partial handles on a failed multi-pipeline creation.
        errdefer if (self.handle != null) vk.vkDestroyPipeline(device.handle, self.handle, null);
        try device.check(vk.vkCreateComputePipelines(device.handle, null, 1, &pipeline_info, null, &self.handle));
        const pool_size: vk.VkDescriptorPoolSize = .{ .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = @intCast(buffers.len) };
        const pool_info: vk.VkDescriptorPoolCreateInfo = .{ .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
        try device.check(vk.vkCreateDescriptorPool(device.handle, &pool_info, null, &self.pool));
        errdefer vk.vkDestroyDescriptorPool(device.handle, self.pool, null);
        const set_allocation: vk.VkDescriptorSetAllocateInfo = .{ .descriptorPool = self.pool, .descriptorSetCount = 1, .pSetLayouts = &self.set_layout };
        try device.check(vk.vkAllocateDescriptorSets(device.handle, &set_allocation, &self.set));
        var infos: [8]vk.VkDescriptorBufferInfo = undefined;
        var writes: [8]vk.VkWriteDescriptorSet = undefined;
        for (buffers, 0..) |buffer, i| {
            infos[i] = .{ .buffer = buffer.handle, .range = buffer.size };
            writes[i] = .{ .dstSet = self.set, .dstBinding = @intCast(i), .descriptorCount = 1, .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i] };
        }
        vk.vkUpdateDescriptorSets(device.handle, @intCast(buffers.len), &writes, 0, null);
        for (buffers, 0..) |buffer, i| {
            self.buffers[i] = buffer;
            buffer.references += 1;
        }
        self.buffer_count = buffers.len;
        device.kernels += 1;
        return self;
    }

    pub fn deinit(self: *Kernel) Error!void {
        if (self.handle == null) return error.InvalidState;
        if (self.references != 0) return error.ResourceInUse;
        vk.vkDestroyDescriptorPool(self.device.handle, self.pool, null);
        vk.vkDestroyPipeline(self.device.handle, self.handle, null);
        vk.vkDestroyPipelineLayout(self.device.handle, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device.handle, self.set_layout, null);
        for (self.buffers[0..self.buffer_count]) |buffer| buffer.references -= 1;
        self.device.kernels -= 1;
        self.handle = null;
    }
};

pub fn validateCode(code: []const u8) Error!void {
    if (code.len < 20 or code.len > 1024 * 1024 or code.len % 4 != 0) return error.InvalidShader;
    const magic = std.mem.readInt(u32, code[0..4], .little);
    const version = std.mem.readInt(u32, code[4..8], .little);
    if (magic != 0x07230203 or version < 0x10000 or version > 0x10300 or version & 0xff != 0 or std.mem.readInt(u32, code[12..16], .little) == 0 or std.mem.readInt(u32, code[16..20], .little) != 0) return error.InvalidShader;
}
