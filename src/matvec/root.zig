//! Resident packed-weight, FP32-input single-vector projections. No model/session logic.
const std = @import("std");
const gpu = @import("../gpu/root.zig");

pub const Error = gpu.Error || error{ InvalidShape, InvalidWeights, NonFiniteWeight, UnsupportedSubnormal, AliasedOutput, UnsupportedDevice };
pub const Format = enum(u32) {
    f32 = 0,
    q4_0 = 2,
    q4_1 = 3,
    q5_k = 13,
    q6_k = 14,

    pub fn blockElements(self: Format) u32 {
        return switch (self) {
            .f32 => 1,
            .q4_0, .q4_1 => 32,
            .q5_k, .q6_k => 256,
        };
    }
    pub fn blockBytes(self: Format) u32 {
        return switch (self) {
            .f32 => 4,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_k => 176,
            .q6_k => 210,
        };
    }
};
pub const Shape = struct {
    format: Format,
    columns: u32,
    rows: u32,

    pub fn weightBytes(self: Shape) Error!u64 {
        if (self.columns == 0 or self.columns > 32768 or self.rows == 0 or self.rows > 1048576 or self.columns % self.format.blockElements() != 0) return error.InvalidShape;
        const bytes = @as(u64, self.columns / self.format.blockElements()) * self.format.blockBytes() * self.rows;
        if (bytes > std.math.maxInt(u32)) return error.InvalidRange;
        return bytes;
    }
};
pub const Region = struct { offset: u64, buffer_bytes: u64 };
pub const Push = extern struct { columns: u32, rows: u32, row_bytes: u32, weight_offset: u32, input_offset: u32, output_offset: u32, groups_x: u32 };
pub const Geometry = struct { push: Push, groups: [3]u32, bytes: [3]u64 };

/// Driver-free arithmetic gate. Region order: weights, FP32 input, FP32 output.
pub fn geometry(shape: Shape, regions: [3]Region, group_limits: [3]u32) Error!Geometry {
    const sizes: [3]u64 = .{ try shape.weightBytes(), @as(u64, shape.columns) * 4, @as(u64, shape.rows) * 4 };
    for (regions, sizes, 0..) |region, size, index| {
        const alignment: u64 = if (index == 0 and shape.format != .f32) 2 else 4;
        if (region.buffer_bytes == 0 or region.buffer_bytes > std.math.maxInt(u32) or region.buffer_bytes % 4 != 0 or region.offset % alignment != 0 or region.offset > region.buffer_bytes or size > region.buffer_bytes - region.offset) return error.InvalidRange;
    }
    if (group_limits[0] == 0 or group_limits[1] == 0 or group_limits[2] == 0) return error.InvalidDispatch;
    const max_x = @min(group_limits[0], 65535);
    const gy = (shape.rows + max_x - 1) / max_x;
    if (gy > group_limits[1]) return error.InvalidDispatch;
    const gx = (shape.rows + gy - 1) / gy;
    return .{ .push = .{ .columns = shape.columns, .rows = shape.rows, .row_bytes = @intCast(sizes[0] / shape.rows), .weight_offset = @intCast(regions[0].offset), .input_offset = @intCast(regions[1].offset / 4), .output_offset = @intCast(regions[2].offset / 4), .groups_x = gx }, .groups = .{ gx, gy, 1 }, .bytes = sizes };
}

/// Required before upload. Does not allocate or write. Resident content remains caller-owned.
pub fn validateWeights(shape: Shape, bytes: []const u8) Error!void {
    if (bytes.len != try shape.weightBytes()) return error.InvalidWeights;
    if (shape.format == .f32) {
        var offset: usize = 0;
        while (offset < bytes.len) : (offset += 4) {
            const bits = std.mem.readInt(u32, bytes[offset..][0..4], .little) & 0x7fffffff;
            if (bits >= 0x7f800000) return error.NonFiniteWeight;
            if (bits != 0 and bits < 0x00800000) return error.UnsupportedSubnormal;
        }
    } else {
        var offset: usize = 0;
        while (offset < bytes.len) : (offset += shape.format.blockBytes()) {
            const field_offset = offset + @as(usize, if (shape.format == .q6_k) 208 else 0);
            const fields: usize = if (shape.format == .q4_1 or shape.format == .q5_k) 2 else 1;
            for (0..fields) |field| {
                const bits = std.mem.readInt(u16, bytes[field_offset + field * 2 ..][0..2], .little);
                if (bits & 0x7c00 == 0x7c00) return error.NonFiniteWeight;
            }
        }
    }
}

pub const View = struct { buffer: *gpu.Buffer, offset: u64 = 0 };
pub const Plan = struct {
    kernel: gpu.Kernel,
    layout: Geometry,

    /// Stable address after recording. Buffers must contain previously validated weights/data.
    pub fn init(shape: Shape, weights: View, input: View, output: View) Error!Plan {
        const device = weights.buffer.device;
        const views = [_]View{ weights, input, output };
        var regions: [3]Region = undefined;
        for (views, 0..) |view, i| {
            if (view.buffer.device != device) return error.WrongDevice;
            if (view.buffer.handle == null) return error.InvalidState;
            regions[i] = .{ .offset = view.offset, .buffer_bytes = view.buffer.size };
        }
        const limits = device.properties.limits;
        const layout = try geometry(shape, regions, limits.maxComputeWorkGroupCount);
        for (views[0..2], layout.bytes[0..2]) |view, bytes| {
            if (view.buffer == output.buffer and overlaps(view.offset, bytes, output.offset, layout.bytes[2])) return error.AliasedOutput;
        }
        if (limits.maxComputeWorkGroupInvocations < 64 or limits.maxComputeWorkGroupSize[0] < 64 or limits.maxComputeWorkGroupSize[1] < 1 or limits.maxComputeWorkGroupSize[2] < 1 or limits.maxComputeSharedMemorySize < 256) return error.UnsupportedDevice;
        const wide_f32 = shape.format == .f32 and shape.columns >= 1024 and limits.maxComputeWorkGroupInvocations >= 256 and limits.maxComputeWorkGroupSize[0] >= 256 and limits.maxComputeSharedMemorySize >= 1024;
        const kernel = try gpu.Kernel.init(device, shader(shape.format, wide_f32, weights.offset % 4 == 0), &.{ weights.buffer, input.buffer, output.buffer }, @sizeOf(Push));
        return .{ .kernel = kernel, .layout = layout };
    }

    /// Conservative dependencies cover upload, previous compute and prior readback/replay.
    pub fn record(self: *Plan, commands: *gpu.Commands) Error!void {
        if (commands.device != self.kernel.device) return error.WrongDevice;
        if (self.kernel.handle == null) return error.InvalidState;
        try commands.barrier(.transfer, .compute);
        try commands.barrier(.compute, .compute);
        try commands.dispatch(&self.kernel, std.mem.asBytes(&self.layout.push), self.layout.groups);
    }
    pub fn deinit(self: *Plan) Error!void {
        try self.kernel.deinit();
    }
};

/// Shared projection pipeline for graph execution: one kernel per (module, buffer
/// triple), many validated projections. Recording dispatches only; the caller owns
/// all barriers. Stable address after the first recording; buffers outlive it.
pub const Pipeline = struct {
    kernel: gpu.Kernel,
    format: Format,
    aligned_words: bool,
    buffers: [3]*gpu.Buffer,

    /// `aligned_words` selects the direct-word Q4_1/Q5_K module; every projection must
    /// then have a 4-byte-aligned weight offset. F32 uses the 256-lane module when the
    /// device permits (valid for any K), otherwise the core 64-lane module.
    pub fn init(format: Format, aligned_words: bool, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!Pipeline {
        if (aligned_words and format != .q4_1 and format != .q5_k) return error.InvalidShape;
        const device = weights.device;
        for ([_]*gpu.Buffer{ weights, input, output }) |buffer| {
            if (buffer.device != device) return error.WrongDevice;
            if (buffer.handle == null) return error.InvalidState;
        }
        const limits = device.properties.limits;
        if (limits.maxComputeWorkGroupInvocations < 64 or limits.maxComputeWorkGroupSize[0] < 64 or limits.maxComputeSharedMemorySize < 256) return error.UnsupportedDevice;
        const wide_f32 = format == .f32 and limits.maxComputeWorkGroupInvocations >= 256 and limits.maxComputeWorkGroupSize[0] >= 256 and limits.maxComputeSharedMemorySize >= 1024;
        const kernel = try gpu.Kernel.init(device, shader(format, wide_f32, aligned_words), &.{ weights, input, output }, @sizeOf(Push));
        return .{ .kernel = kernel, .format = format, .aligned_words = aligned_words, .buffers = .{ weights, input, output } };
    }

    /// Driver-free validation of one projection against this pipeline's buffers.
    pub fn projection(self: *const Pipeline, shape: Shape, weight_offset: u64, input_offset: u64, output_offset: u64) Error!Geometry {
        if (shape.format != self.format) return error.InvalidShape;
        if (self.aligned_words and weight_offset % 4 != 0) return error.InvalidRange;
        const offsets = [3]u64{ weight_offset, input_offset, output_offset };
        var regions: [3]Region = undefined;
        for (&regions, self.buffers, offsets) |*region, buffer, offset| region.* = .{ .offset = offset, .buffer_bytes = buffer.size };
        const layout = try geometry(shape, regions, self.kernel.device.properties.limits.maxComputeWorkGroupCount);
        for (0..2) |i| {
            if (self.buffers[i] == self.buffers[2] and overlaps(offsets[i], layout.bytes[i], output_offset, layout.bytes[2])) return error.AliasedOutput;
        }
        return layout;
    }

    pub fn record(self: *Pipeline, commands: *gpu.Commands, layout: Geometry) Error!void {
        if (commands.device != self.kernel.device) return error.WrongDevice;
        try commands.dispatch(&self.kernel, std.mem.asBytes(&layout.push), layout.groups);
    }
    pub fn deinit(self: *Pipeline) Error!void {
        try self.kernel.deinit();
    }
};

fn overlaps(a: u64, a_bytes: u64, b: u64, b_bytes: u64) bool {
    // Regions have already passed bounded extent checks.
    return a < b + b_bytes and b < a + a_bytes;
}
fn shader(format: Format, wide_f32: bool, aligned_words: bool) []align(4) const u8 {
    const Modules = struct {
        const f32_code align(4) = @embedFile("shaders/f32.spv").*;
        const f32_small_code align(4) = @embedFile("shaders/f32_small.spv").*;
        const q4_0_code align(4) = @embedFile("shaders/q4_0.spv").*;
        const q4_1_code align(4) = @embedFile("shaders/q4_1.spv").*;
        const q5_k_code align(4) = @embedFile("shaders/q5_k.spv").*;
        const q4_1_aligned_code align(4) = @embedFile("shaders/q4_1_aligned.spv").*;
        const q5_k_aligned_code align(4) = @embedFile("shaders/q5_k_aligned.spv").*;
        const q6_k_code align(4) = @embedFile("shaders/q6_k.spv").*;
    };
    return switch (format) {
        .f32 => if (wide_f32) &Modules.f32_code else &Modules.f32_small_code,
        .q4_0 => &Modules.q4_0_code,
        .q4_1 => if (aligned_words) &Modules.q4_1_aligned_code else &Modules.q4_1_code,
        .q5_k => if (aligned_words) &Modules.q5_k_aligned_code else &Modules.q5_k_code,
        .q6_k => &Modules.q6_k_code,
    };
}
