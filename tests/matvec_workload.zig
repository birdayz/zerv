//! Test/benchmark ownership and transfers only, never native model execution.
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const m = zerv.matvec;
pub const timeout_ns = 30 * std.time.ns_per_s;
pub const Timing = @import("gpu_timing.zig").Timing;
pub const Marks = @import("gpu_timing.zig").Marks;
pub const Case = struct {
    name: []const u8,
    kind: enum { explicit, half, row_ramp },
    format: m.Format,
    columns: u32,
    rows: u32,
    exact: bool,
    input_sha256: []const u8,
    packed_hex: []const u8 = "",
    input_hex: []const u8 = "",
    base_hex: []const u8 = "",
    field: usize = 0,
    ideal: []const f64 = &.{},
    sumabs: []const f64 = &.{},
    output_sha256: []const u8 = "",
};
pub const Goldens = struct { cases: []const Case };
pub fn goldens(allocator: std.mem.Allocator) !std.json.Parsed(Goldens) {
    return std.json.parseFromSlice(Goldens, allocator, @embedFile("fixtures/gpu/matvec.json"), .{ .ignore_unknown_fields = true });
}
pub fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
pub const Data = struct {
    shape: m.Shape,
    weights: []const u8,
    input: []const u8,
};
pub fn parseCase(raw: []const u8) !Data {
    if (raw.len < 32) return error.InvalidCase;
    var h: [8]u32 = undefined;
    for (&h, 0..) |*v, i| v.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
    if (h[0] != 0x38564d5a or h[1] != 1 or h[7] != 0) return error.InvalidCase;
    const format = std.enums.fromInt(m.Format, h[2]) orelse return error.InvalidCase;
    const shape: m.Shape = .{ .format = format, .columns = h[3], .rows = h[4] };
    if (h[5] != try shape.weightBytes() or h[6] != @as(u64, shape.columns) * 4 or raw.len != 32 + @as(u64, h[5]) + h[6]) return error.InvalidCase;
    const weights = raw[32 .. 32 + @as(usize, h[5])];
    const input = raw[32 + @as(usize, h[5]) ..];
    try m.validateWeights(shape, weights);
    for (0..shape.columns) |i| {
        const bits = std.mem.readInt(u32, input[i * 4 ..][0..4], .little) & 0x7fffffff;
        if (bits >= 0x7f800000 or (bits != 0 and bits < 0x00800000)) return error.InvalidCase;
    }
    return .{ .shape = shape, .weights = weights, .input = input };
}
pub fn fixtureBytes(allocator: std.mem.Allocator, case: Case) ![]u8 {
    const shape: m.Shape = .{ .format = case.format, .columns = case.columns, .rows = case.rows };
    const weight_bytes: usize = @intCast(try shape.weightBytes());
    const result = try allocator.alloc(u8, 32 + weight_bytes + @as(usize, case.columns) * 4);
    errdefer allocator.free(result);
    const header = [_]u32{ 0x38564d5a, 1, @intFromEnum(case.format), case.columns, case.rows, @intCast(weight_bytes), case.columns * 4, 0 };
    for (header, 0..) |v, i| std.mem.writeInt(u32, result[i * 4 ..][0..4], v, .little);
    const weights = result[32 .. 32 + weight_bytes];
    const input = result[32 + weight_bytes ..];
    switch (case.kind) {
        .explicit => {
            if (case.packed_hex.len != weights.len * 2 or case.input_hex.len != input.len * 2) return error.InvalidCase;
            _ = try std.fmt.hexToBytes(weights, case.packed_hex);
            _ = try std.fmt.hexToBytes(input, case.input_hex);
        },
        .half => {
            const width = case.format.blockBytes();
            var base: [210]u8 = undefined;
            if (case.base_hex.len != width * 2 or case.rows != 63488 or case.field + 2 > width or case.columns != case.format.blockElements()) return error.InvalidCase;
            _ = try std.fmt.hexToBytes(base[0..width], case.base_hex);
            var row: usize = 0;
            for (0..65536) |bits| {
                if (bits & 0x7c00 == 0x7c00) continue;
                @memcpy(weights[row * width ..][0..width], base[0..width]);
                std.mem.writeInt(u16, weights[row * width + case.field ..][0..2], @intCast(bits), .little);
                row += 1;
            }
            @memset(input, 0);
            std.mem.writeInt(u32, input[0..4], @bitCast(@as(f32, 1)), .little);
        },
        .row_ramp => {
            if (case.format != .f32 or case.columns != 1 or case.rows != 65537) return error.InvalidCase;
            for (0..case.rows) |r| {
                const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(r % 257)) - 128)) / 128;
                std.mem.writeInt(u32, weights[r * 4 ..][0..4], @bitCast(value), .little);
            }
            std.mem.writeInt(u32, input[0..4], @bitCast(@as(f32, 1.0 / 16.0)), .little);
        },
    }
    if (!std.mem.eql(u8, case.input_sha256, &hash(result))) return error.IndependentInputMismatch;
    return result;
}

/// In-place initialization: the pipeline retains resident-buffer addresses.
pub const Workload = struct {
    upload: gpu.Buffer,
    resident: gpu.Buffer,
    readback: gpu.Buffer,
    plan: m.Plan,
    commands: [3]gpu.Commands,
    shape: m.Shape,
    weight_offset: usize,
    input_offset: usize,
    output_offset: usize,

    pub fn init(self: *Workload, device: *gpu.Device, data: Data) !void {
        return self.initOffset(device, data, if (data.shape.format == .f32) 4 else 2);
    }
    pub fn initAligned(self: *Workload, device: *gpu.Device, data: Data) !void {
        return self.initOffset(device, data, 0);
    }
    fn initOffset(self: *Workload, device: *gpu.Device, data: Data, weight_offset: usize) !void {
        try m.validateWeights(data.shape, data.weights);
        if (data.input.len != @as(usize, data.shape.columns) * 4) return error.InvalidCase;
        self.shape = data.shape;
        self.weight_offset = weight_offset;
        self.input_offset = std.mem.alignForward(usize, self.weight_offset + data.weights.len, 4) + 16;
        self.output_offset = self.input_offset + data.input.len + 16;
        const output_bytes = @as(usize, data.shape.rows) * 4;
        const bytes = self.output_offset + output_bytes + 64;
        self.upload = try gpu.Buffer.init(device, bytes, .host);
        errdefer self.upload.deinit() catch @panic("upload cleanup");
        self.resident = try gpu.Buffer.init(device, bytes, .device);
        errdefer self.resident.deinit() catch @panic("resident cleanup");
        self.readback = try gpu.Buffer.init(device, output_bytes + 80, .host);
        errdefer self.readback.deinit() catch @panic("readback cleanup");
        self.plan = try m.Plan.init(data.shape, .{ .buffer = &self.resident, .offset = self.weight_offset }, .{ .buffer = &self.resident, .offset = self.input_offset }, .{ .buffer = &self.resident, .offset = self.output_offset });
        errdefer self.plan.deinit() catch @panic("plan cleanup");
        var initialized: usize = 0;
        errdefer for (self.commands[0..initialized]) |*cmd| cmd.deinit() catch @panic("command cleanup");
        for (&self.commands) |*cmd| {
            cmd.* = try gpu.Commands.init(device);
            initialized += 1;
        }
        const upload = try self.upload.mapped();
        @memset(upload, 0xcd);
        @memcpy(upload[self.weight_offset..][0..data.weights.len], data.weights);
        @memcpy(upload[self.input_offset..][0..data.input.len], data.input);
        var load = &self.commands[0];
        try load.begin();
        try load.barrier(.compute, .transfer);
        try load.barrier(.transfer, .transfer);
        try load.copy(&self.upload, 0, &self.resident, 0, bytes);
        try load.barrier(.transfer, .compute);
        try load.end();
        try self.commands[1].begin();
        try self.plan.record(&self.commands[1]);
        try self.commands[1].end();
        var download = &self.commands[2];
        try download.begin();
        try download.barrier(.compute, .transfer);
        try download.copy(&self.resident, self.output_offset - 16, &self.readback, 0, output_bytes + 80);
        try download.barrier(.transfer, .host);
        try download.end();
        try load.run(timeout_ns);
    }
    /// Explicit benchmark-only pipeline substitution, outside every timed region.
    pub fn replaceKernel(self: *Workload, code: []align(4) const u8, rows_per_group: u32) !void {
        if (rows_per_group == 0 or rows_per_group > 32) return error.InvalidCase;
        const device = self.resident.device;
        var kernel = try gpu.Kernel.init(device, code, &.{ &self.resident, &self.resident, &self.resident }, @sizeOf(m.Push));
        var transferred = false;
        errdefer if (!transferred) kernel.deinit() catch @panic("candidate kernel cleanup");
        const count = (self.shape.rows + rows_per_group - 1) / rows_per_group;
        const gx = @min(count, @min(device.properties.limits.maxComputeWorkGroupCount[0], 65535));
        const gy = (count + gx - 1) / gx;
        if (gy > device.properties.limits.maxComputeWorkGroupCount[1]) return error.InvalidDispatch;
        try self.commands[1].reset();
        try self.plan.deinit();
        self.plan.kernel = kernel;
        transferred = true;
        self.plan.layout.groups = .{ gx, gy, 1 };
        self.plan.layout.push.groups_x = gx;
        try self.commands[1].begin();
        try self.plan.record(&self.commands[1]);
        try self.commands[1].end();
    }
    pub fn execute(self: *Workload) !void {
        try self.commands[1].run(timeout_ns);
    }
    pub fn finish(self: *Workload) ![]const u8 {
        try self.commands[2].run(timeout_ns);
        const readback = try self.readback.mapped();
        const output_end = 16 + @as(usize, self.shape.rows) * 4;
        for (readback[0..16]) |b| if (b != 0xcd) return error.OutputGuardCorrupted;
        for (readback[output_end..]) |b| if (b != 0xcd) return error.OutputGuardCorrupted;
        return readback[16..output_end];
    }
    pub fn zeroInput(self: *Workload) !void {
        const upload = try self.upload.mapped();
        @memset(upload[self.input_offset..][0 .. @as(usize, self.shape.columns) * 4], 0);
        try self.commands[0].run(timeout_ns);
    }
    pub fn deinit(self: *Workload) void {
        for (&self.commands) |*cmd| cmd.deinit() catch @panic("pending matvec cleanup");
        self.plan.deinit() catch @panic("retained matvec cleanup");
        self.readback.deinit() catch @panic("readback cleanup");
        self.resident.deinit() catch @panic("resident cleanup");
        self.upload.deinit() catch @panic("upload cleanup");
    }
};

pub fn check(case: Case, output: []const u8) !void {
    if (output.len != @as(usize, case.rows) * 4) return error.WrongOutputShape;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..case.rows) |r| {
        var bits = std.mem.readInt(u32, output[r * 4 ..][0..4], .little);
        const value: f32 = @bitCast(bits);
        if (!std.math.isFinite(value)) return error.NonFiniteOutput;
        if (value == 0) bits = 0;
        var canonical: [4]u8 = undefined;
        std.mem.writeInt(u32, &canonical, bits, .little);
        hasher.update(&canonical);
        if (case.kind == .explicit) {
            if (case.ideal.len != case.rows or case.sumabs.len != case.rows) return error.InvalidCase;
            const err = @abs(@as(f64, value) - case.ideal[r]);
            const bound: f64 = if (case.exact) 0 else 2e-6 + 2e-6 * case.sumabs[r];
            if (err > bound) {
                std.debug.print("matvec mismatch {s} row={d} actual={d} ideal={d} error={d} bound={d}\n", .{ case.name, r, value, case.ideal[r], err, bound });
                return error.IndependentGoldenMismatch;
            }
        }
    }
    if (case.kind != .explicit) {
        const digest = std.fmt.bytesToHex(hasher.finalResult(), .lower);
        if (!std.mem.eql(u8, case.output_sha256, &digest)) return error.IndependentGoldenMismatch;
    }
}
