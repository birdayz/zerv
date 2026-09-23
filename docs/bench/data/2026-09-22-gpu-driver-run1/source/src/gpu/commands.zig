const vk = @import("vk.zig");
const driver = @import("device.zig");
const Device = driver.Device;
const Error = driver.Error;
const buffers = @import("buffer.zig");
const Buffer = buffers.Buffer;
const Kernel = @import("kernel.zig").Kernel;

pub const Scope = enum {
    transfer,
    compute,
    host,

    fn stage(self: Scope) u32 {
        return switch (self) {
            .transfer => vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            .compute => vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .host => vk.VK_PIPELINE_STAGE_HOST_BIT,
        };
    }
    fn access(self: Scope) u32 {
        return switch (self) {
            .transfer => vk.VK_ACCESS_TRANSFER_READ_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            .compute => vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            .host => vk.VK_ACCESS_HOST_READ_BIT | vk.VK_ACCESS_HOST_WRITE_BIT,
        };
    }
};

pub const Commands = struct {
    pub const State = enum { initial, recording, executable, pending, invalid };
    device: *Device,
    pool: vk.VkCommandPool = null,
    handle: vk.VkCommandBuffer = null,
    fence: vk.VkFence = null,
    state: State = .initial,
    retained_buffers: [64]*Buffer = undefined,
    buffer_count: usize = 0,
    retained_kernels: [32]*Kernel = undefined,
    kernel_count: usize = 0,

    pub fn init(device: *Device) Error!Commands {
        try device.ready();
        if (device.commands >= 32) return error.ResourceLimit;
        var self: Commands = .{ .device = device };
        const pool_info: vk.VkCommandPoolCreateInfo = .{ .queueFamilyIndex = device.family };
        try device.check(vk.vkCreateCommandPool(device.handle, &pool_info, null, &self.pool));
        errdefer vk.vkDestroyCommandPool(device.handle, self.pool, null);
        const allocation: vk.VkCommandBufferAllocateInfo = .{ .commandPool = self.pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        try device.check(vk.vkAllocateCommandBuffers(device.handle, &allocation, &self.handle));
        const fence_info: vk.VkFenceCreateInfo = .{};
        try device.check(vk.vkCreateFence(device.handle, &fence_info, null, &self.fence));
        device.commands += 1;
        return self;
    }

    pub fn begin(self: *Commands) Error!void {
        try self.expect(.initial);
        const info: vk.VkCommandBufferBeginInfo = .{};
        try self.device.check(vk.vkBeginCommandBuffer(self.handle, &info));
        self.state = .recording;
    }

    pub fn end(self: *Commands) Error!void {
        try self.expect(.recording);
        self.device.check(vk.vkEndCommandBuffer(self.handle)) catch |err| {
            self.state = .invalid;
            return err;
        };
        self.state = .executable;
    }

    pub fn reset(self: *Commands) Error!void {
        try self.device.ready();
        if (self.pool == null) return error.InvalidState;
        if (self.state == .pending) return error.ResourceInUse;
        try self.device.check(vk.vkResetCommandPool(self.device.handle, self.pool, 0));
        self.release();
        self.state = .initial;
    }

    pub fn copy(self: *Commands, source: *Buffer, source_offset: u64, destination: *Buffer, destination_offset: u64, size: u64) Error!void {
        try self.expect(.recording);
        if (source.device != self.device or destination.device != self.device) return error.WrongDevice;
        if (source.handle == null or destination.handle == null) return error.InvalidState;
        if (!buffers.validRange(source.size, source_offset, size) or !buffers.validRange(destination.size, destination_offset, size)) return error.InvalidRange;
        if (source == destination and source_offset < destination_offset + size and destination_offset < source_offset + size) return error.InvalidRange;
        const new_source = !self.hasBuffer(source);
        const new_destination = source != destination and !self.hasBuffer(destination);
        if (self.buffer_count + @as(usize, @intFromBool(new_source)) + @as(usize, @intFromBool(new_destination)) > self.retained_buffers.len) return error.ResourceLimit;
        if (new_source) self.keepBuffer(source);
        if (new_destination) self.keepBuffer(destination);
        const region: vk.VkBufferCopy = .{ .srcOffset = source_offset, .dstOffset = destination_offset, .size = size };
        vk.vkCmdCopyBuffer(self.handle, source.handle, destination.handle, 1, &region);
    }

    pub fn barrier(self: *Commands, source: Scope, destination: Scope) Error!void {
        try self.expect(.recording);
        const dependency: vk.VkMemoryBarrier = .{ .srcAccessMask = source.access(), .dstAccessMask = destination.access() };
        vk.vkCmdPipelineBarrier(self.handle, source.stage(), destination.stage(), 0, 1, &dependency, 0, null, 0, null);
    }

    pub fn dispatch(self: *Commands, kernel: *Kernel, push: []const u8, groups: [3]u32) Error!void {
        try self.expect(.recording);
        if (kernel.device != self.device) return error.WrongDevice;
        if (kernel.handle == null) return error.InvalidState;
        if (push.len != kernel.push_bytes) return error.InvalidLayout;
        for (groups, self.device.properties.limits.maxComputeWorkGroupCount) |count, limit|
            if (count == 0 or count > limit) return error.InvalidDispatch;
        var found = false;
        for (self.retained_kernels[0..self.kernel_count]) |item| {
            if (item == kernel) found = true;
        }
        if (!found) {
            if (self.kernel_count == self.retained_kernels.len) return error.ResourceLimit;
            self.retained_kernels[self.kernel_count] = kernel;
            self.kernel_count += 1;
            kernel.references += 1;
        }
        vk.vkCmdBindPipeline(self.handle, vk.VK_PIPELINE_BIND_POINT_COMPUTE, kernel.handle);
        vk.vkCmdBindDescriptorSets(self.handle, vk.VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout, 0, 1, &kernel.set, 0, null);
        if (push.len != 0) vk.vkCmdPushConstants(self.handle, kernel.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push.len), push.ptr);
        vk.vkCmdDispatch(self.handle, groups[0], groups[1], groups[2]);
    }

    pub fn submit(self: *Commands) Error!void {
        try self.expect(.executable);
        try self.device.check(vk.vkResetFences(self.device.handle, 1, &self.fence));
        const submission: vk.VkSubmitInfo = .{ .commandBufferCount = 1, .pCommandBuffers = &self.handle };
        self.device.check(vk.vkQueueSubmit(self.device.queue, 1, &submission, self.fence)) catch |err| {
            self.state = .invalid;
            return err;
        };
        self.state = .pending;
        self.device.pending += 1;
    }

    /// A timeout does not release resources or permit reset/resubmission.
    pub fn wait(self: *Commands, timeout_ns: u64) Error!void {
        if (self.pool == null or self.state != .pending) return error.InvalidState;
        if (timeout_ns == @import("std").math.maxInt(u64)) return error.InvalidLimit;
        self.device.check(vk.vkWaitForFences(self.device.handle, 1, &self.fence, 1, timeout_ns)) catch |err| {
            if (self.device.lost) {
                self.state = .invalid;
                self.device.pending -= 1;
            }
            return err;
        };
        self.device.pending -= 1;
        self.state = .executable;
    }

    pub fn run(self: *Commands, timeout_ns: u64) Error!void {
        try self.submit();
        try self.wait(timeout_ns);
    }

    pub fn deinit(self: *Commands) Error!void {
        if (self.pool == null) return error.InvalidState;
        if (self.state == .pending) {
            if (!self.device.lost) return error.ResourceInUse;
            self.device.pending -= 1;
        }
        vk.vkDestroyFence(self.device.handle, self.fence, null);
        vk.vkDestroyCommandPool(self.device.handle, self.pool, null);
        self.release();
        self.device.commands -= 1;
        self.pool = null;
        self.handle = null;
        self.state = .invalid;
    }

    fn expect(self: *const Commands, state: State) Error!void {
        try self.device.ready();
        if (self.pool == null or self.state != state) return error.InvalidState;
    }
    fn hasBuffer(self: *const Commands, buffer: *Buffer) bool {
        for (self.retained_buffers[0..self.buffer_count]) |item| if (item == buffer) return true;
        return false;
    }
    fn keepBuffer(self: *Commands, buffer: *Buffer) void {
        self.retained_buffers[self.buffer_count] = buffer;
        self.buffer_count += 1;
        buffer.references += 1;
    }
    fn release(self: *Commands) void {
        for (self.retained_buffers[0..self.buffer_count]) |buffer| buffer.references -= 1;
        for (self.retained_kernels[0..self.kernel_count]) |kernel| kernel.references -= 1;
        self.buffer_count = 0;
        self.kernel_count = 0;
    }
};
