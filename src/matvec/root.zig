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
    /// Multi-row module only (the MTP layer's eh_proj; docs/specs/speculative.md).
    q8_0 = 8,

    pub fn blockElements(self: Format) u32 {
        return switch (self) {
            .f32 => 1,
            .q4_0, .q4_1, .q8_0 => 32,
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
            .q8_0 => 34,
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
/// Dot-product accumulation of the decode and verify projections
/// (docs/specs/matvec-push.md): `.fma` (default) one rounding per step; `.separate` the
/// previous strict multiply then add, with its own verify table.
pub const Accumulation = enum { fma, separate };
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
        return initWith(shape, weights, input, output, .fma);
    }
    pub fn initWith(shape: Shape, weights: View, input: View, output: View, accumulation: Accumulation) Error!Plan {
        if (shape.format == .q8_0) return error.InvalidShape; // no single-row module
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
        const kernel = try gpu.Kernel.init(device, shader(shape.format, wide_f32, weights.offset % 4 == 0, accumulation), &.{ weights.buffer, input.buffer, output.buffer }, @sizeOf(Push));
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
        return initWith(format, aligned_words, .fma, weights, input, output);
    }
    pub fn initWith(format: Format, aligned_words: bool, accumulation: Accumulation, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!Pipeline {
        if (aligned_words and format != .q4_1 and format != .q5_k) return error.InvalidShape;
        if (format == .q8_0) return error.InvalidShape; // no single-row module
        const device = weights.device;
        for ([_]*gpu.Buffer{ weights, input, output }) |buffer| {
            if (buffer.device != device) return error.WrongDevice;
            if (buffer.handle == null) return error.InvalidState;
        }
        const limits = device.properties.limits;
        if (limits.maxComputeWorkGroupInvocations < 64 or limits.maxComputeWorkGroupSize[0] < 64 or limits.maxComputeSharedMemorySize < 256) return error.UnsupportedDevice;
        const wide_f32 = format == .f32 and limits.maxComputeWorkGroupInvocations >= 256 and limits.maxComputeWorkGroupSize[0] >= 256 and limits.maxComputeSharedMemorySize >= 1024;
        const kernel = try gpu.Kernel.init(device, shader(format, wide_f32, aligned_words, accumulation), &.{ weights, input, output }, @sizeOf(Push));
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

/// Fused FFN input for decode (block 17c; `matvec.comp` with SWIGLU): workgroup i computes
/// row i of two same-shape projections of one weight buffer (gate at `weight_offset`, up at
/// `up_offset`) with the single-row module's per-row arithmetic, writing g, u and
/// silu(g) * u (FP32 word offsets `output_offset`, `u_offset`, `y_offset`).
pub const SwigluPush = extern struct { columns: u32, rows: u32, row_bytes: u32, weight_offset: u32, input_offset: u32, output_offset: u32, groups_x: u32, up_offset: u32, u_offset: u32, y_offset: u32 };
pub const SwigluGeometry = struct { push: SwigluPush, groups: [3]u32 };
pub const SwigluPipeline = struct {
    kernel: gpu.Kernel,
    format: Format,
    aligned_words: bool,
    buffers: [3]*gpu.Buffer,

    /// Quantized formats only (`InvalidShape` for F32 and Q8_0): the caller falls back to
    /// separate projections and the swiglu kernel.
    pub fn init(format: Format, aligned_words: bool, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!SwigluPipeline {
        return initWith(format, aligned_words, .fma, weights, input, output);
    }
    pub fn initWith(format: Format, aligned_words: bool, accumulation: Accumulation, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!SwigluPipeline {
        if (aligned_words and format != .q4_1 and format != .q5_k) return error.InvalidShape;
        const code = swigluShader(format, aligned_words, accumulation) orelse return error.InvalidShape;
        const device = weights.device;
        for ([_]*gpu.Buffer{ weights, input, output }) |buffer| {
            if (buffer.device != device) return error.WrongDevice;
            if (buffer.handle == null) return error.InvalidState;
        }
        const limits = device.properties.limits;
        if (limits.maxComputeWorkGroupInvocations < 64 or limits.maxComputeWorkGroupSize[0] < 64 or limits.maxComputeSharedMemorySize < 512) return error.UnsupportedDevice;
        const kernel = try gpu.Kernel.init(device, code, &.{ weights, input, output }, @sizeOf(SwigluPush));
        return .{ .kernel = kernel, .format = format, .aligned_words = aligned_words, .buffers = .{ weights, input, output } };
    }

    /// Driver-free validation: both weight regions, the input and the three outputs are
    /// in bounds (as `geometry`), and no output overlaps the input or another output.
    pub fn projection(self: *const SwigluPipeline, shape: Shape, gate_offset: u64, up_offset: u64, input_offset: u64, g_offset: u64, u_offset: u64, y_offset: u64) Error!SwigluGeometry {
        if (shape.format != self.format) return error.InvalidShape;
        if (self.aligned_words and (gate_offset % 4 != 0 or up_offset % 4 != 0)) return error.InvalidRange;
        const limits = self.kernel.device.properties.limits.maxComputeWorkGroupCount;
        const outputs = [3]u64{ g_offset, u_offset, y_offset };
        var gate: Geometry = undefined;
        for (outputs, 0..) |out, i| {
            const regions = [3]Region{ .{ .offset = if (i == 1) up_offset else gate_offset, .buffer_bytes = self.buffers[0].size }, .{ .offset = input_offset, .buffer_bytes = self.buffers[1].size }, .{ .offset = out, .buffer_bytes = self.buffers[2].size } };
            const g = try geometry(shape, regions, limits);
            if (i == 0) gate = g;
            if (self.buffers[1] == self.buffers[2] and overlaps(input_offset, g.bytes[1], out, g.bytes[2])) return error.AliasedOutput;
            if (self.buffers[0] == self.buffers[2] and overlaps(if (i == 1) up_offset else gate_offset, g.bytes[0], out, g.bytes[2])) return error.AliasedOutput;
        }
        const out_bytes = gate.bytes[2];
        for (outputs[0..2], 0..) |a, i| for (outputs[i + 1 ..]) |b| if (overlaps(a, out_bytes, b, out_bytes)) return error.AliasedOutput;
        const p = gate.push;
        return .{ .push = .{ .columns = p.columns, .rows = p.rows, .row_bytes = p.row_bytes, .weight_offset = p.weight_offset, .input_offset = p.input_offset, .output_offset = p.output_offset, .groups_x = p.groups_x, .up_offset = @intCast(up_offset), .u_offset = @intCast(u_offset / 4), .y_offset = @intCast(y_offset / 4) }, .groups = gate.groups };
    }

    pub fn record(self: *SwigluPipeline, commands: *gpu.Commands, layout: SwigluGeometry) Error!void {
        if (commands.device != self.kernel.device) return error.WrongDevice;
        try commands.dispatch(&self.kernel, std.mem.asBytes(&layout.push), layout.groups);
    }
    pub fn deinit(self: *SwigluPipeline) Error!void {
        try self.kernel.deinit();
    }
};

fn swigluShader(format: Format, aligned_words: bool, accumulation: Accumulation) ?[]align(4) const u8 {
    return switch (accumulation) {
        .fma => swigluIn(SwigluModules("shaders/"), format, aligned_words),
        .separate => swigluIn(SwigluModules("shaders/separate/"), format, aligned_words),
    };
}
fn SwigluModules(comptime dir: []const u8) type {
    return struct {
        const q4_0 align(4) = @embedFile(dir ++ "swiglu_q4_0.spv").*;
        const q4_1 align(4) = @embedFile(dir ++ "swiglu_q4_1.spv").*;
        const q4_1_aligned align(4) = @embedFile(dir ++ "swiglu_q4_1_aligned.spv").*;
        const q5_k align(4) = @embedFile(dir ++ "swiglu_q5_k.spv").*;
        const q5_k_aligned align(4) = @embedFile(dir ++ "swiglu_q5_k_aligned.spv").*;
        const q6_k align(4) = @embedFile(dir ++ "swiglu_q6_k.spv").*;
    };
}
fn swigluIn(comptime Modules: type, format: Format, aligned_words: bool) ?[]align(4) const u8 {
    return switch (format) {
        .q4_0 => &Modules.q4_0,
        .q4_1 => if (aligned_words) &Modules.q4_1_aligned else &Modules.q4_1,
        .q5_k => if (aligned_words) &Modules.q5_k_aligned else &Modules.q5_k,
        .q6_k => &Modules.q6_k,
        .f32, .q8_0 => null,
    };
}

fn overlaps(a: u64, a_bytes: u64, b: u64, b_bytes: u64) bool {
    // Regions have already passed bounded extent checks.
    return a < b + b_bytes and b < a + a_bytes;
}
fn shader(format: Format, wide_f32: bool, aligned_words: bool, accumulation: Accumulation) []align(4) const u8 {
    return switch (accumulation) {
        .fma => shaderIn(SingleModules("shaders/"), format, wide_f32, aligned_words),
        .separate => shaderIn(SingleModules("shaders/separate/"), format, wide_f32, aligned_words),
    };
}
fn SingleModules(comptime dir: []const u8) type {
    return struct {
        const f32_code align(4) = @embedFile(dir ++ "f32.spv").*;
        const f32_small_code align(4) = @embedFile(dir ++ "f32_small.spv").*;
        const q4_0_code align(4) = @embedFile(dir ++ "q4_0.spv").*;
        const q4_1_code align(4) = @embedFile(dir ++ "q4_1.spv").*;
        const q5_k_code align(4) = @embedFile(dir ++ "q5_k.spv").*;
        const q4_1_aligned_code align(4) = @embedFile(dir ++ "q4_1_aligned.spv").*;
        const q5_k_aligned_code align(4) = @embedFile(dir ++ "q5_k_aligned.spv").*;
        const q6_k_code align(4) = @embedFile(dir ++ "q6_k.spv").*;
    };
}
fn shaderIn(comptime Modules: type, format: Format, wide_f32: bool, aligned_words: bool) []align(4) const u8 {
    return switch (format) {
        .f32 => if (wide_f32) &Modules.f32_code else &Modules.f32_small_code,
        .q4_0 => &Modules.q4_0_code,
        .q4_1 => if (aligned_words) &Modules.q4_1_aligned_code else &Modules.q4_1_code,
        .q5_k => if (aligned_words) &Modules.q5_k_aligned_code else &Modules.q5_k_code,
        .q6_k => &Modules.q6_k_code,
        .q8_0 => unreachable, // rejected by the callers
    };
}

/// Input rows a multi-row projection processes at once (speculative verification).
pub const max_rows = 5;
/// Weight rows per multi-row workgroup for each exact row count 1..`max_rows` (GROUP in
/// matvec_rows.comp; each lane's X loads serve this many weight rows). Must match
/// tools/compile_matvec.py (the GPU equality test fails otherwise: rows go missing).
/// Measured per count: docs/bench/2026-09-24-fma-matvec.md (fma) and
/// docs/bench/2026-09-24-spec-verify.md (separate).
pub fn rowsGroups(accumulation: Accumulation) [max_rows]u32 {
    return switch (accumulation) {
        .fma => .{ 1, 3, 3, 3, 3 },
        .separate => .{ 1, 2, 2, 3, 2 },
    };
}
pub const RowsPush = extern struct { base: Push, input_stride: u32, output_stride: u32, count: u32 };
/// Validated multi-row projection; `groups[c - 1]` is the dispatch for count c (the
/// modules differ in weight rows per workgroup). `span`: input/output rows validated from
/// the offsets (>= push.count); `recordAt` runs up to push.count of them at a row offset.
pub const RowsGeometry = struct { push: RowsPush, groups: [max_rows][3]u32, span: u32 };

/// Multi-row projection pipeline (matvec_rows.comp): Y[r] = W X[r] for r < count <=
/// `max_rows`, each row bitwise equal to the single-row `Pipeline` result (same module
/// family, lane partition and summation order). One kernel per row count 1..`max_count`
/// (the modules are compiled per count) for one (format, alignment, buffer triple); the
/// caller owns barriers. Same buffer and offset rules as `Pipeline`.
pub const RowsPipeline = struct {
    kernels: [max_rows]gpu.Kernel,
    max_count: u32,
    format: Format,
    aligned_words: bool,
    wide_f32: bool,
    accumulation: Accumulation,
    buffers: [3]*gpu.Buffer,

    pub fn init(format: Format, aligned_words: bool, max_count: u32, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!RowsPipeline {
        return initWith(format, aligned_words, .fma, max_count, weights, input, output);
    }
    pub fn initWith(format: Format, aligned_words: bool, accumulation: Accumulation, max_count: u32, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!RowsPipeline {
        if (aligned_words and format != .q4_1 and format != .q5_k) return error.InvalidShape;
        if (max_count == 0 or max_count > max_rows) return error.InvalidShape;
        const device = weights.device;
        for ([_]*gpu.Buffer{ weights, input, output }) |buffer| {
            if (buffer.device != device) return error.WrongDevice;
            if (buffer.handle == null) return error.InvalidState;
        }
        const limits = device.properties.limits;
        var shared_bytes: u64 = 0;
        for (rowsGroups(accumulation), 1..) |g, n| shared_bytes = @max(shared_bytes, @as(u64, g) * n * 256 * 4);
        if (limits.maxComputeWorkGroupInvocations < 64 or limits.maxComputeWorkGroupSize[0] < 64 or limits.maxComputeSharedMemorySize < shared_bytes) return error.UnsupportedDevice;
        // Same lane count as the single-row pipeline for this device (the F32 lane
        // partition decides the summation order).
        const wide_f32 = format == .f32 and limits.maxComputeWorkGroupInvocations >= 256 and limits.maxComputeWorkGroupSize[0] >= 256 and limits.maxComputeSharedMemorySize >= 1024;
        var self: RowsPipeline = .{ .kernels = undefined, .max_count = 0, .format = format, .aligned_words = aligned_words, .wide_f32 = wide_f32, .accumulation = accumulation, .buffers = .{ weights, input, output } };
        errdefer self.deinit() catch {};
        for (0..max_count) |i| {
            self.kernels[i] = try gpu.Kernel.init(device, rowsShader(format, wide_f32, aligned_words, accumulation, @intCast(i + 1)), &.{ weights, input, output }, @sizeOf(RowsPush));
            self.max_count += 1;
        }
        return self;
    }

    /// Driver-free validation of `count` rows: input rows at `input_offset + r *
    /// input_stride` floats-apart (bytes for the offsets, floats for the strides, as the
    /// kernel reads them), outputs likewise. Every row must lie inside its buffer and no
    /// input row may overlap an output row.
    pub fn projection(self: *const RowsPipeline, shape: Shape, weight_offset: u64, input_offset: u64, output_offset: u64, input_stride: u32, output_stride: u32, count: u32) Error!RowsGeometry {
        return self.projectionSpan(shape, weight_offset, input_offset, output_offset, input_stride, output_stride, count, count);
    }

    /// `projection` for `span` >= `count` rows (docs/specs/concurrent.md, batched decode):
    /// all `span` input and output rows are validated; each dispatch covers at most `count`.
    pub fn projectionSpan(self: *const RowsPipeline, shape: Shape, weight_offset: u64, input_offset: u64, output_offset: u64, input_stride: u32, output_stride: u32, count: u32, span: u32) Error!RowsGeometry {
        if (shape.format != self.format) return error.InvalidShape;
        if (count == 0 or count > self.max_count or span < count or input_stride < shape.columns or output_stride < shape.rows) return error.InvalidShape;
        // Quantized modules read X as vec4: 16-byte aligned rows.
        if (shape.format != .f32 and (input_offset % 16 != 0 or input_stride % 4 != 0)) return error.InvalidRange;
        if (self.aligned_words and weight_offset % 4 != 0) return error.InvalidRange;
        const in_span = (@as(u64, span - 1) * input_stride + shape.columns) * 4;
        const out_span = (@as(u64, span - 1) * output_stride + shape.rows) * 4;
        // Row 0 through the single-row rules (weights, alignment); then the full spans.
        const regions = [3]Region{
            .{ .offset = weight_offset, .buffer_bytes = self.buffers[0].size },
            .{ .offset = input_offset, .buffer_bytes = self.buffers[1].size },
            .{ .offset = output_offset, .buffer_bytes = self.buffers[2].size },
        };
        const layout = try geometry(shape, regions, self.kernels[0].device.properties.limits.maxComputeWorkGroupCount);
        if (input_offset + in_span > self.buffers[1].size or output_offset + out_span > self.buffers[2].size) return error.InvalidRange;
        if (self.buffers[0] == self.buffers[2] and overlaps(weight_offset, layout.bytes[0], output_offset, out_span)) return error.AliasedOutput;
        if (self.buffers[1] == self.buffers[2] and overlaps(input_offset, in_span, output_offset, out_span)) return error.AliasedOutput;
        // Count c: `rowsGroups(acc)[c - 1]` weight rows per workgroup, ceil(M / G) workgroups
        // split over x/y as `geometry` splits M.
        const limits = self.kernels[0].device.properties.limits.maxComputeWorkGroupCount;
        var groups: [max_rows][3]u32 = @splat(.{ 0, 0, 0 });
        const table = rowsGroups(self.accumulation);
        for (groups[0..count], table[0..count]) |*slot, g| {
            const total = (shape.rows + g - 1) / g;
            const max_x = @min(limits[0], 65535);
            const gy = (total + max_x - 1) / max_x;
            if (gy > limits[1]) return error.InvalidDispatch;
            slot.* = .{ (total + gy - 1) / gy, gy, 1 };
        }
        return .{ .push = .{ .base = layout.push, .input_stride = input_stride, .output_stride = output_stride, .count = count }, .groups = groups, .span = span };
    }

    /// `record` of rows `first` .. `first + count - 1` of the validated span.
    pub fn recordAt(self: *RowsPipeline, commands: *gpu.Commands, rows: RowsGeometry, first: u32, count: u32) Error!void {
        if (first >= rows.span or count > rows.span - first) return error.InvalidShape;
        var shifted = rows; // push offsets are in words
        shifted.push.base.input_offset += first * rows.push.input_stride;
        shifted.push.base.output_offset += first * rows.push.output_stride;
        try self.record(commands, shifted, count);
    }

    /// Record with `count` (1..the count validated in `projection`) input rows: the
    /// module compiled for exactly `count` rows.
    pub fn record(self: *RowsPipeline, commands: *gpu.Commands, rows: RowsGeometry, count: u32) Error!void {
        if (commands.device != self.kernels[0].device) return error.WrongDevice;
        if (count == 0 or count > rows.push.count) return error.InvalidShape;
        var push = rows.push;
        push.count = count;
        push.base.groups_x = rows.groups[count - 1][0];
        try commands.dispatch(&self.kernels[count - 1], std.mem.asBytes(&push), rows.groups[count - 1]);
    }
    pub fn deinit(self: *RowsPipeline) Error!void {
        while (self.max_count > 0) {
            try self.kernels[self.max_count - 1].deinit();
            self.max_count -= 1;
        }
    }
};

/// Gate rows per fused multi-row workgroup (GROUP in matvec_rows.comp with SWIGLU: the
/// workgroup computes these rows of gate and the same rows of up) for each exact row count;
/// 0: no fused module for that count (`SwigluRowsPipeline.supports`). Must match
/// tools/compile_matvec.py (`SWIGLU_ROWS_CONFIG*`); measured in
/// docs/bench/2026-09-24-verify-fusion.md (separate: the same table, not tuned). K-quants
/// stop at count 2: their 4-weight-row modules spill VGPRs, and ACO's VGPR-to-LDS
/// spilling miscompiled one such module (no shipped matvec module may spill).
pub fn swigluRowsGroups(accumulation: Accumulation, format: Format) [max_rows]u32 {
    if (format == .q5_k or format == .q6_k) return .{ 1, 1, 0, 0, 0 };
    return switch (accumulation) {
        .fma => .{ 1, 1, 2, 2, 0 },
        .separate => .{ 1, 1, 2, 2, 0 },
    };
}
pub const SwigluRowsPush = extern struct { rows: RowsPush, up_offset: u32, u_offset: u32, y_offset: u32 };
pub const SwigluRowsGeometry = struct { push: SwigluRowsPush, groups: [max_rows][3]u32, span: u32 };

/// Fused verify FFN input (block 17c; matvec_rows.comp with SWIGLU): for r < count, g[r] =
/// Wg X[r] and u[r] = Wu X[r] (two same-shape regions of one weight buffer), each bitwise
/// equal to the multi-row (and single-row) module's result, and y[r] = silu(g) * u with the
/// single-row `SwigluPipeline`'s expression. Outputs are rows `output_stride` floats apart at
/// the byte offsets `g_offset`, `u_offset` and `y_offset`. Quantized formats only
/// (`InvalidShape` for F32 and Q8_0). One kernel per supported count 1..`max_count`
/// (`supports`); the caller records the separate path for the others and owns barriers.
pub const SwigluRowsPipeline = struct {
    kernels: [max_rows]gpu.Kernel,
    max_count: u32,
    format: Format,
    aligned_words: bool,
    accumulation: Accumulation,
    buffers: [3]*gpu.Buffer,

    pub fn init(format: Format, aligned_words: bool, accumulation: Accumulation, max_count: u32, weights: *gpu.Buffer, input: *gpu.Buffer, output: *gpu.Buffer) Error!SwigluRowsPipeline {
        if (aligned_words and format != .q4_1 and format != .q5_k) return error.InvalidShape;
        if (format == .f32 or format == .q8_0) return error.InvalidShape;
        if (max_count == 0 or max_count > max_rows) return error.InvalidShape;
        const device = weights.device;
        for ([_]*gpu.Buffer{ weights, input, output }) |buffer| {
            if (buffer.device != device) return error.WrongDevice;
            if (buffer.handle == null) return error.InvalidState;
        }
        const limits = device.properties.limits;
        var shared_bytes: u64 = 0;
        for (swigluRowsGroups(accumulation, format), 1..) |g, n| shared_bytes = @max(shared_bytes, 2 * @as(u64, g) * n * 64 * 4);
        if (limits.maxComputeWorkGroupInvocations < 64 or limits.maxComputeWorkGroupSize[0] < 64 or limits.maxComputeSharedMemorySize < shared_bytes) return error.UnsupportedDevice;
        var self: SwigluRowsPipeline = .{ .kernels = undefined, .max_count = 0, .format = format, .aligned_words = aligned_words, .accumulation = accumulation, .buffers = .{ weights, input, output } };
        errdefer self.deinit() catch {};
        for (0..max_count) |i| {
            if (self.supports(@intCast(i + 1))) self.kernels[i] = try gpu.Kernel.init(device, swigluRowsShader(format, aligned_words, accumulation, @intCast(i + 1)).?, &.{ weights, input, output }, @sizeOf(SwigluRowsPush));
            self.max_count += 1;
        }
        return self;
    }

    /// Whether a fused module exists for `count` input rows (1..`max_rows`).
    pub fn supports(self: *const SwigluRowsPipeline, count: u32) bool {
        return swigluRowsGroups(self.accumulation, self.format)[count - 1] != 0;
    }

    /// Driver-free validation of `count` rows: both weight regions as `RowsPipeline`
    /// validates one, the three output spans in bounds, and no output span overlapping the
    /// input span, the weights or another output span.
    pub fn projection(self: *const SwigluRowsPipeline, shape: Shape, gate_offset: u64, up_offset: u64, input_offset: u64, g_offset: u64, u_offset: u64, y_offset: u64, input_stride: u32, output_stride: u32, count: u32) Error!SwigluRowsGeometry {
        return self.projectionSpan(shape, gate_offset, up_offset, input_offset, g_offset, u_offset, y_offset, input_stride, output_stride, count, count);
    }

    /// `projection` for `span` >= `count` rows, as `RowsPipeline.projectionSpan`.
    pub fn projectionSpan(self: *const SwigluRowsPipeline, shape: Shape, gate_offset: u64, up_offset: u64, input_offset: u64, g_offset: u64, u_offset: u64, y_offset: u64, input_stride: u32, output_stride: u32, count: u32, span: u32) Error!SwigluRowsGeometry {
        if (shape.format != self.format) return error.InvalidShape;
        if (count == 0 or count > self.max_count or span < count or input_stride < shape.columns or output_stride < shape.rows) return error.InvalidShape;
        if (input_offset % 16 != 0 or input_stride % 4 != 0) return error.InvalidRange;
        if (self.aligned_words and (gate_offset % 4 != 0 or up_offset % 4 != 0)) return error.InvalidRange;
        const limits = self.buffers[0].device.properties.limits.maxComputeWorkGroupCount;
        const in_span = (@as(u64, span - 1) * input_stride + shape.columns) * 4;
        const out_span = (@as(u64, span - 1) * output_stride + shape.rows) * 4;
        const outputs = [3]u64{ g_offset, u_offset, y_offset };
        var gate: Geometry = undefined;
        for (outputs, 0..) |out, i| {
            const weight_offset = if (i == 1) up_offset else gate_offset;
            const regions = [3]Region{ .{ .offset = weight_offset, .buffer_bytes = self.buffers[0].size }, .{ .offset = input_offset, .buffer_bytes = self.buffers[1].size }, .{ .offset = out, .buffer_bytes = self.buffers[2].size } };
            const g = try geometry(shape, regions, limits);
            if (i == 0) gate = g;
            if (input_offset + in_span > self.buffers[1].size or out + out_span > self.buffers[2].size) return error.InvalidRange;
            if (self.buffers[1] == self.buffers[2] and overlaps(input_offset, in_span, out, out_span)) return error.AliasedOutput;
            if (self.buffers[0] == self.buffers[2] and overlaps(weight_offset, g.bytes[0], out, out_span)) return error.AliasedOutput;
        }
        for (outputs[0..2], 0..) |a, i| for (outputs[i + 1 ..]) |b| if (overlaps(a, out_span, b, out_span)) return error.AliasedOutput;
        var groups: [max_rows][3]u32 = @splat(.{ 0, 0, 0 });
        for (groups[0..count], swigluRowsGroups(self.accumulation, self.format)[0..count]) |*slot, g| {
            if (g == 0) continue; // no fused module for this count
            const total = (shape.rows + g - 1) / g;
            const max_x = @min(limits[0], 65535);
            const gy = (total + max_x - 1) / max_x;
            if (gy > limits[1]) return error.InvalidDispatch;
            slot.* = .{ (total + gy - 1) / gy, gy, 1 };
        }
        return .{ .push = .{ .rows = .{ .base = gate.push, .input_stride = input_stride, .output_stride = output_stride, .count = count }, .up_offset = @intCast(up_offset), .u_offset = @intCast(u_offset / 4), .y_offset = @intCast(y_offset / 4) }, .groups = groups, .span = span };
    }

    /// `record` of rows `first` .. `first + count - 1` of the validated span.
    pub fn recordAt(self: *SwigluRowsPipeline, commands: *gpu.Commands, layout: SwigluRowsGeometry, first: u32, count: u32) Error!void {
        if (first >= layout.span or count > layout.span - first) return error.InvalidShape;
        var shifted = layout; // push offsets are in words
        const out_rows = first * layout.push.rows.output_stride;
        shifted.push.rows.base.input_offset += first * layout.push.rows.input_stride;
        shifted.push.rows.base.output_offset += out_rows;
        shifted.push.u_offset += out_rows;
        shifted.push.y_offset += out_rows;
        try self.record(commands, shifted, count);
    }

    /// Record with `count` (1..the count validated in `projection`, `supports(count)`)
    /// input rows.
    pub fn record(self: *SwigluRowsPipeline, commands: *gpu.Commands, layout: SwigluRowsGeometry, count: u32) Error!void {
        if (commands.device != self.buffers[0].device) return error.WrongDevice;
        if (count == 0 or count > layout.push.rows.count or !self.supports(count)) return error.InvalidShape;
        var push = layout.push;
        push.rows.count = count;
        push.rows.base.groups_x = layout.groups[count - 1][0];
        try commands.dispatch(&self.kernels[count - 1], std.mem.asBytes(&push), layout.groups[count - 1]);
    }
    pub fn deinit(self: *SwigluRowsPipeline) Error!void {
        while (self.max_count > 0) {
            if (self.supports(self.max_count)) try self.kernels[self.max_count - 1].deinit();
            self.max_count -= 1;
        }
    }
};

/// The fused module for `count` rows, or null when the table has none for that count.
fn swigluRowsShader(format: Format, aligned_words: bool, accumulation: Accumulation, count: u32) ?[]align(4) const u8 {
    return switch (accumulation) {
        .fma => swigluRowsShaderIn("shaders/", .fma, format, aligned_words, count),
        .separate => swigluRowsShaderIn("shaders/separate/", .separate, format, aligned_words, count),
    };
}
fn swigluRowsShaderIn(comptime dir: []const u8, comptime accumulation: Accumulation, format: Format, aligned_words: bool, count: u32) ?[]align(4) const u8 {
    // (count, format) pairs without a module are decided at compile time: their files do
    // not exist and are never embedded.
    inline for (1..max_rows + 1) |n| {
        if (n == count) {
            const M = SwigluModules(std.fmt.comptimePrint("{s}rows{d}_", .{ dir, n }));
            inline for (.{ Format.q4_0, Format.q4_1, Format.q5_k, Format.q6_k }) |f| {
                if (comptime swigluRowsGroups(accumulation, f)[n - 1] != 0) {
                    if (format == f) return switch (f) {
                        .q4_0 => &M.q4_0,
                        .q4_1 => if (aligned_words) &M.q4_1_aligned else &M.q4_1,
                        .q5_k => if (aligned_words) &M.q5_k_aligned else &M.q5_k,
                        .q6_k => &M.q6_k,
                        else => comptime unreachable,
                    };
                }
            }
            return null;
        }
    }
    return null;
}

fn rowsShader(format: Format, wide_f32: bool, aligned_words: bool, accumulation: Accumulation, count: u32) []align(4) const u8 {
    return switch (accumulation) {
        .fma => rowsShaderIn("shaders/", format, wide_f32, aligned_words, count),
        .separate => rowsShaderIn("shaders/separate/", format, wide_f32, aligned_words, count),
    };
}
fn rowsShaderIn(comptime dir: []const u8, format: Format, wide_f32: bool, aligned_words: bool, count: u32) []align(4) const u8 {
    inline for (1..max_rows + 1) |n| {
        if (n == count) {
            const M = RowsModules(dir, n);
            return switch (format) {
                .f32 => if (wide_f32) &M.f32_code else &M.f32_small_code,
                .q4_0 => &M.q4_0_code,
                .q4_1 => if (aligned_words) &M.q4_1_aligned_code else &M.q4_1_code,
                .q5_k => if (aligned_words) &M.q5_k_aligned_code else &M.q5_k_code,
                .q6_k => &M.q6_k_code,
                .q8_0 => &M.q8_0_code,
            };
        }
    }
    unreachable; // 1 <= count <= max_rows, validated by init
}
fn RowsModules(comptime dir: []const u8, comptime n: u32) type {
    const prefix = std.fmt.comptimePrint("{s}rows{d}_", .{ dir, n });
    return struct {
        const f32_code align(4) = @embedFile(prefix ++ "f32.spv").*;
        const f32_small_code align(4) = @embedFile(prefix ++ "f32_small.spv").*;
        const q4_0_code align(4) = @embedFile(prefix ++ "q4_0.spv").*;
        const q4_1_code align(4) = @embedFile(prefix ++ "q4_1.spv").*;
        const q5_k_code align(4) = @embedFile(prefix ++ "q5_k.spv").*;
        const q4_1_aligned_code align(4) = @embedFile(prefix ++ "q4_1_aligned.spv").*;
        const q5_k_aligned_code align(4) = @embedFile(prefix ++ "q5_k_aligned.spv").*;
        const q6_k_code align(4) = @embedFile(prefix ++ "q6_k.spv").*;
        const q8_0_code align(4) = @embedFile(prefix ++ "q8_0.spv").*;
    };
}
