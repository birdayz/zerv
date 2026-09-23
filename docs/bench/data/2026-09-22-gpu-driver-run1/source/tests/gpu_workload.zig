//! Shared diagnostic workload for explicit GPU tests/benchmark, not production math.
const std = @import("std");
const gpu = @import("zerv").gpu;
pub const shader align(4) = @embedFile("fixtures/gpu/affine.spv").*;
pub const Kind = enum { affine, roundtrip };
pub const timeout_ns = 10 * std.time.ns_per_s;
pub const Result = struct { kind: Kind, count: usize, bytes: usize, input_sha256: [64]u8, output_sha256: [64]u8 };
pub const Case = struct { kind: Kind, count: usize, bytes: usize, input_sha256: []const u8, output_sha256: []const u8 };
pub const Goldens = struct { cases: []const Case };
pub fn goldens(allocator: std.mem.Allocator) !std.json.Parsed(Goldens) {
    return std.json.parseFromSlice(Goldens, allocator, @embedFile("fixtures/gpu/dispatch.json"), .{ .ignore_unknown_fields = true });
}
pub fn inputWord(i: usize) u32 {
    return switch (i) {
        0 => 0,
        1 => 0xffffffff,
        2 => 0x80000000,
        else => (@as(u32, @intCast(i)) *% 0x9e3779b9) ^ 0xa5a5a5a5,
    };
}
fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// Initialize in its final location; commands/pipeline retain addresses of these buffers.
pub const Workload = struct {
    kind: Kind,
    count: usize,
    bytes: usize,
    input: gpu.Buffer,
    readback: gpu.Buffer,
    a: gpu.Buffer,
    b: gpu.Buffer,
    kernel: ?gpu.Kernel,
    commands: [4]gpu.Commands,

    pub fn init(self: *Workload, device: *gpu.Device, kind: Kind, count: usize) !void {
        self.kind = kind;
        self.count = count;
        self.bytes = if (kind == .affine) try std.math.mul(usize, try std.math.add(usize, count, 64), 4) else count;
        self.input = try gpu.Buffer.init(device, self.bytes, .host);
        errdefer self.input.deinit() catch @panic("input cleanup");
        self.readback = try gpu.Buffer.init(device, self.bytes, .host);
        errdefer self.readback.deinit() catch @panic("readback cleanup");
        self.a = try gpu.Buffer.init(device, self.bytes, .device);
        errdefer self.a.deinit() catch @panic("device input cleanup");
        self.b = try gpu.Buffer.init(device, self.bytes, .device);
        errdefer self.b.deinit() catch @panic("device output cleanup");
        self.kernel = if (kind == .affine) try gpu.Kernel.init(device, &shader, &.{ &self.a, &self.b }, 4) else null;
        errdefer if (self.kernel) |*kernel| kernel.deinit() catch @panic("kernel cleanup");
        var initialized: usize = 0;
        errdefer for (self.commands[0..initialized]) |*cmd| cmd.deinit() catch @panic("command cleanup");
        for (&self.commands) |*cmd| {
            cmd.* = try gpu.Commands.init(device);
            initialized += 1;
        }
        const input = try self.input.mapped();
        const readback = try self.readback.mapped();
        @memset(readback, 0xcd);
        for (0..self.bytes / 4) |i| std.mem.writeInt(u32, input[i * 4 ..][0..4], inputWord(i), .little);
        if (self.kernel) |*kernel| {
            var load = &self.commands[0];
            try load.begin();
            try load.copy(&self.input, 0, &self.a, 0, self.bytes);
            try load.copy(&self.readback, 0, &self.b, 0, self.bytes);
            try load.barrier(.transfer, .compute);
            try load.end();
            var compute = &self.commands[1];
            try compute.begin();
            try compute.barrier(.compute, .compute);
            var push: [4]u8 = undefined;
            std.mem.writeInt(u32, &push, @intCast(count), .little);
            try compute.dispatch(kernel, &push, .{ @intCast((count + 63) / 64), 1, 1 });
            try compute.barrier(.compute, .transfer);
            try compute.end();
            var download = &self.commands[2];
            try download.begin();
            try download.barrier(.compute, .transfer);
            try download.copy(&self.b, 0, &self.readback, 0, self.bytes);
            try download.barrier(.transfer, .host);
            try download.end();
            try load.run(timeout_ns);
        } else {
            var transfer = &self.commands[3];
            try transfer.begin();
            try transfer.copy(&self.input, 0, &self.a, 0, self.bytes);
            try transfer.barrier(.transfer, .transfer);
            try transfer.copy(&self.a, 0, &self.readback, 0, self.bytes);
            try transfer.barrier(.transfer, .host);
            try transfer.end();
        }
    }

    pub fn execute(self: *Workload) !void {
        try self.commands[if (self.kind == .affine) 1 else 3].run(timeout_ns);
    }
    pub fn finish(self: *Workload) !Result {
        if (self.kind == .affine) try self.commands[2].run(timeout_ns);
        return .{ .kind = self.kind, .count = self.count, .bytes = self.bytes, .input_sha256 = hash(try self.input.mapped()), .output_sha256 = hash(try self.readback.mapped()) };
    }
    pub fn deinit(self: *Workload) void {
        for (&self.commands) |*cmd| cmd.deinit() catch @panic("command still pending at cleanup");
        if (self.kernel) |*kernel| kernel.deinit() catch @panic("kernel still referenced");
        self.b.deinit() catch @panic("buffer still referenced");
        self.a.deinit() catch @panic("buffer still referenced");
        self.readback.deinit() catch @panic("buffer still referenced");
        self.input.deinit() catch @panic("buffer still referenced");
    }
};
