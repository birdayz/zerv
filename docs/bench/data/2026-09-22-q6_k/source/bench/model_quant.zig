//! Actual-artifact diagnostic decode benchmark, not matvec or inference.
const std = @import("std");
const builtin = @import("builtin");
const zerv = @import("zerv");
const Sha256 = std.crypto.hash.sha2.Sha256;
fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
pub fn main(init: std.process.Init) !void {
    if (builtin.mode != .ReleaseFast or builtin.cpu.arch.endian() != .little) return error.UnsupportedBenchmarkBuild;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedModelAndTensor;
    const allocator = std.heap.page_allocator;
    var mapped = try zerv.artifact.MappedFile.open(init.io, args[1], 32 * 1024 * 1024 * 1024);
    defer mapped.deinit();
    var container = try zerv.artifact.gguf.Container.parse(allocator, mapped.bytes, .{});
    defer container.deinit();
    const tensor = container.findTensor(args[2]) orelse return error.MissingTensor;
    switch (tensor.kind) {
        .q4_1 => try benchmark(.q4_1, init, tensor),
        .q5_k => try benchmark(.q5_k, init, tensor),
        .q6_k => try benchmark(.q6_k, init, tensor),
        else => return error.UnsupportedTensor,
    }
}
fn benchmark(comptime format: zerv.quant.Format, init: std.process.Init, tensor: *const zerv.artifact.gguf.Tensor) !void {
    const allocator = std.heap.page_allocator;
    const columns = switch (format) {
        .q4_1 => 17408,
        .q5_k => 6144,
        .q6_k => 5120,
        else => unreachable,
    };
    const tensor_rows = if (format == .q6_k) 248320 else 5120;
    const measured_rows = if (format == .q6_k) 4096 else 5120;
    if (!std.mem.eql(u64, &tensor.dims, &.{ columns, tensor_rows, 1, 1 })) return error.UnsupportedTensor;
    var buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    for ([_]usize{ 1, 64, measured_rows }, [_]usize{ 10000, 256, 4 }) |rows, iterations| {
        const values = columns * rows;
        const encoded = tensor.data[0 .. values / zerv.quant.blockElements(format) * zerv.quant.blockBytes(format)];
        const output = try allocator.alloc(f32, values);
        defer allocator.free(output);
        const input_hash = hash(encoded);
        for (0..3) |_| {
            try zerv.quant.decode(format, encoded, output);
            std.mem.doNotOptimizeAway(output);
        }
        for (0..7) |trial| {
            const start = std.Io.Clock.awake.now(init.io);
            for (0..iterations) |_| {
                try zerv.quant.decode(format, encoded, output);
                std.mem.doNotOptimizeAway(output);
            }
            const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            if (elapsed <= 0) return error.InvalidElapsed;
            try std.json.Stringify.value(.{ .format = @tagName(format), .rows = rows, .values = values, .trial = trial, .iterations = iterations, .elapsed_ns = elapsed, .input_sha256 = input_hash, .output_sha256 = hash(std.mem.sliceAsBytes(output)) }, .{}, &stdout.interface);
            try stdout.interface.writeByte('\n');
        }
    }
    try stdout.interface.flush();
}
