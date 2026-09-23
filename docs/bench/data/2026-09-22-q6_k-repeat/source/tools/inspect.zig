//! Native artifact inspection; not a serving endpoint or inference proxy.
const std = @import("std");
const artifact = @import("zerv").artifact;
const Sha256 = std.crypto.hash.sha2.Sha256;
const MetadataReport = struct { name: []const u8, type: u32, value_sha256: [64]u8 };
const TensorReport = struct { name: []const u8, type: u32, dims: [4]u64, offset: usize, size: usize, sample_sha256: [64]u8 };

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2) {
        std.debug.print("usage: zerv-inspect MODEL.gguf\n", .{});
        return error.ExpectedModelPath;
    }
    var file = try artifact.MappedFile.open(init.io, args[1], 1024 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try artifact.gguf.Container.parse(allocator, file.bytes, .{});
    defer container.deinit();
    const metadata = try allocator.alloc(MetadataReport, container.metadata.len);
    const tensors = try allocator.alloc(TensorReport, container.tensors.len);
    for (container.metadata, metadata) |entry, *report| {
        var digest: [32]u8 = undefined;
        Sha256.hash(entry.value.encoded, &digest, .{});
        report.* = .{ .name = entry.name, .type = @intFromEnum(entry.value.kind), .value_sha256 = std.fmt.bytesToHex(digest, .lower) };
    }
    for (container.tensors, tensors) |tensor, *report| {
        var hasher = Sha256.init(.{});
        for ([_]usize{ 0, tensor.size / 2, tensor.size - @min(tensor.size, 64) }) |start| {
            hasher.update(tensor.data[start..][0..@min(64, tensor.size - start)]);
        }
        var digest: [32]u8 = undefined;
        hasher.final(&digest);
        report.* = .{ .name = tensor.name, .type = @intFromEnum(tensor.kind), .dims = tensor.dims, .offset = tensor.offset, .size = tensor.size, .sample_sha256 = std.fmt.bytesToHex(digest, .lower) };
    }
    var buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    try std.json.Stringify.value(.{ .version = container.version, .alignment = container.alignment, .data_offset = container.data_offset, .file_size = file.bytes.len, .metadata = metadata, .tensors = tensors }, .{}, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}
