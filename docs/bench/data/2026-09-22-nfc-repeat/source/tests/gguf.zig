const std = @import("std");
const t = std.testing;
const artifact = @import("zerv").artifact;
const gguf = artifact.gguf;
const Sha256 = std.crypto.hash.sha2.Sha256;
const full = @embedFile("fixtures/gguf/default.gguf");
const aligned = @embedFile("fixtures/gguf/aligned64.gguf");
const metadata_only = @embedFile("fixtures/gguf/metadata.gguf");

const Case = struct {
    path: []const u8,
    sha256: []const u8,
    version: u32,
    alignment: usize,
    data_offset: usize,
    file_size: usize,
    metadata: []const struct { name: []const u8, type: u32, value_sha256: []const u8 },
    tensors: []const struct { name: []const u8, type: u32, dims: [4]u64, offset: usize, size: usize, sample_sha256: []const u8 },
};
const Manifest = struct { generator_sha256: []const u8, cases: []const Case };

fn digest(bytes: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

fn reject(bytes: []const u8, limits: gguf.Limits, expected: ?anyerror) !void {
    if (gguf.Container.parse(t.allocator, bytes, limits)) |value| {
        var container = value;
        defer container.deinit();
        return error.AcceptedInvalidContainer;
    } else |err| {
        if (expected) |specific| try t.expectEqual(specific, err);
    }
}

fn mutation(comptime T: type, bytes: []const u8, at: usize, value: T, expected: anyerror) !void {
    const copy = try t.allocator.dupe(u8, bytes);
    defer t.allocator.free(copy);
    std.mem.writeInt(T, copy[at..][0..@sizeOf(T)], value, .little);
    try reject(copy, .{}, expected);
}

fn offset(bytes: []const u8, view: []const u8) usize {
    return @intFromPtr(view.ptr) - @intFromPtr(bytes.ptr);
}

test "GGUF reference writer/reader manifests, payloads and borrowed views" {
    const parsed = try std.json.parseFromSlice(Manifest, t.allocator, @embedFile("fixtures/gguf/manifest.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try t.expectEqualStrings(parsed.value.generator_sha256, &digest(@embedFile("reference/gguf_oracle.py")));
    try t.expectEqual(@as(usize, 3), parsed.value.cases.len);
    for (parsed.value.cases) |expected| {
        const bytes = if (std.mem.eql(u8, expected.path, "default.gguf")) full else if (std.mem.eql(u8, expected.path, "aligned64.gguf")) aligned else metadata_only;
        try t.expectEqualStrings(expected.sha256, &digest(bytes));
        var container = try gguf.Container.parse(t.allocator, bytes, .{});
        defer container.deinit();
        try t.expectEqual(expected.version, container.version);
        try t.expectEqual(expected.alignment, container.alignment);
        try t.expectEqual(expected.data_offset, container.data_offset);
        try t.expectEqual(expected.file_size, bytes.len);
        try t.expectEqual(expected.metadata.len, container.metadata.len);
        try t.expectEqual(expected.tensors.len, container.tensors.len);
        for (expected.metadata, container.metadata) |golden, actual| {
            try t.expectEqualStrings(golden.name, actual.name);
            try t.expectEqual(golden.type, @intFromEnum(actual.value.kind));
            try t.expectEqualStrings(golden.value_sha256, &digest(actual.value.encoded));
            try t.expectEqual(actual.value.encoded.ptr, container.findMetadata(golden.name).?.encoded.ptr);
        }
        for (expected.tensors, container.tensors) |golden, actual| {
            try t.expectEqualStrings(golden.name, actual.name);
            try t.expectEqual(golden.type, @intFromEnum(actual.kind));
            try t.expectEqualSlices(u64, &golden.dims, &actual.dims);
            try t.expectEqual(golden.offset, actual.offset);
            try t.expectEqual(golden.size, actual.size);
            try t.expectEqual(bytes[container.data_offset + actual.offset ..].ptr, actual.data.ptr);
            try t.expectEqual(actual.data.ptr, container.findTensor(golden.name).?.data.ptr);
            var hasher = Sha256.init(.{});
            for ([_]usize{ 0, actual.size / 2, actual.size - @min(actual.size, 64) }) |start| {
                hasher.update(actual.data[start..][0..@min(64, actual.size - start)]);
            }
            var hash: [32]u8 = undefined;
            hasher.final(&hash);
            try t.expectEqualStrings(golden.sample_sha256, &std.fmt.bytesToHex(hash, .lower));
        }
        try t.expect(container.findTensor("missing") == null);
        try t.expect(container.findMetadata("missing") == null);
        const required = if (container.tensors.len == 0) container.data_offset else bytes.len;
        for (0..required) |end| try reject(bytes[0..end], .{}, null);
    }
}

test "GGUF typed scalars and zero-allocation array iteration" {
    var container = try gguf.Container.parse(t.allocator, full, .{});
    defer container.deinit();
    const types = .{ u8, i8, u16, i16, u32, i32, f32, bool, u64, i64, f64 };
    const names = .{ "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "u64", "i64", "f64" };
    const values = .{ 241, -97, 61234, -23456, 3456789012, -123456789, -0.125, true, 12345678901234567890, -1234567890123456789, 1.125 };
    inline for (types, names, values) |T, name, expected| {
        const scalar = container.findMetadata("test." ++ name).?;
        try t.expectEqual(@as(T, expected), try scalar.scalar(T));
        const array = try container.findMetadata("test.array_" ++ name).?.array();
        try t.expectEqual(@as(usize, 3), array.count);
        var it = array.iterator();
        try t.expectEqual(@as(T, expected), try (try it.next()).?.scalar(T));
        try t.expectEqual(@as(T, if (T == bool) false else 0), try (try it.next()).?.scalar(T));
        try t.expectEqual(@as(T, expected), try (try it.next()).?.scalar(T));
        try t.expect(try it.next() == null);
    }
    const text = container.findMetadata("test.text").?;
    try t.expectEqualStrings("\"héllo\"\n世界", try text.string());
    try t.expectError(error.WrongValueType, text.scalar(u32));
    try t.expectError(error.WrongValueType, text.array());
    try t.expectError(error.WrongValueType, container.findMetadata("test.u32").?.string());
    var strings = (try container.findMetadata("test.strings").?.array()).iterator();
    for ([_][]const u8{ "", "é", "a\nb" }) |expected| try t.expectEqualStrings(expected, try (try strings.next()).?.string());
    try t.expect(try strings.next() == null);
    var empty = (try container.findMetadata("test.empty").?.array()).iterator();
    try t.expect(try empty.next() == null);
    try t.expectEqual(@as(u32, 4), container.findTensor("tensor.type_1").?.rank);
}

test "GGUF header, metadata and limits reject malformed input" {
    var container = try gguf.Container.parse(t.allocator, full, .{});
    defer container.deinit();
    try mutation(u32, full, 0, 0, error.InvalidMagic);
    try mutation(u32, full, 4, 2, error.UnsupportedVersion);
    try mutation(u32, full, 4, 0x03000000, error.UnsupportedVersion);
    try mutation(u64, full, 8, std.math.maxInt(u64), error.LimitExceeded);
    try mutation(u64, full, 16, std.math.maxInt(u64), error.LimitExceeded);
    var large_counts: [24]u8 = full[0..24].*;
    std.mem.writeInt(u64, large_counts[16..24], std.math.maxInt(u32), .little);
    try reject(&large_counts, .{ .max_metadata = std.math.maxInt(u32) }, error.LimitExceeded);
    const text = offset(full, container.findMetadata("test.text").?.encoded);
    try mutation(u64, full, text, std.math.maxInt(u64), error.LimitExceeded);
    try mutation(u8, full, text + 8, 255, error.InvalidString);
    try mutation(u32, full, text - 4, 99, error.UnsupportedValueType);
    const boolean = offset(full, container.findMetadata("test.bool").?.encoded);
    try mutation(u8, full, boolean, 2, error.InvalidBool);
    const bool_array = offset(full, container.findMetadata("test.array_bool").?.encoded);
    try mutation(u8, full, bool_array + 12, 2, error.InvalidBool);
    try mutation(u64, full, bool_array + 4, std.math.maxInt(u64), error.LimitExceeded);
    try mutation(u32, full, bool_array, 9, error.UnsupportedNestedArray);
    const first_name = offset(full, container.metadata[0].name);
    try mutation(u8, full, first_name, ' ', error.InvalidKey);
    try mutation(u8, full, first_name, 0, error.InvalidKey);
    const copy = try t.allocator.dupe(u8, full);
    defer t.allocator.free(copy);
    const duplicate = offset(full, container.metadata[2].name); // test.i8 has same length as test.u8
    @memcpy(copy[duplicate..][0..container.metadata[0].name.len], container.metadata[0].name);
    try reject(copy, .{}, error.DuplicateKey);
    for ([_]gguf.Limits{ .{ .max_metadata = 0 }, .{ .max_tensors = 0 }, .{ .max_header_bytes = 24 }, .{ .max_string_bytes = 1 }, .{ .max_array_elements = 0 }, .{ .max_alignment = 16 } }) |limits| try reject(full, limits, error.LimitExceeded);
}

test "GGUF shape, alignment, types, offsets and overflow are checked" {
    var container = try gguf.Container.parse(t.allocator, full, .{});
    defer container.deinit();
    const a = container.tensors[0];
    const after_name = offset(full, a.name) + a.name.len;
    try mutation(u32, full, after_name, 0, error.InvalidRank);
    try mutation(u32, full, after_name, 5, error.InvalidRank);
    try mutation(u64, full, after_name + 4, 0, error.InvalidDimension);
    try mutation(u64, full, after_name + 4, std.math.maxInt(u64), error.InvalidDimension);
    try mutation(u64, full, after_name + 4, std.math.maxInt(i64), error.Overflow);
    try mutation(u32, full, after_name + 4 + a.rank * 8, 42, error.UnsupportedTensorType);
    try mutation(u64, full, after_name + 8 + a.rank * 8, 32, error.InvalidTensorOffset);
    const q = container.findTensor("tensor.type_2").?;
    try mutation(u64, full, offset(full, q.name) + q.name.len + 4, 31, error.InvalidDimension);
    const copy = try t.allocator.dupe(u8, full);
    defer t.allocator.free(copy);
    const second_name = offset(full, container.tensors[1].name);
    @memcpy(copy[second_name..][0..a.name.len], a.name);
    try reject(copy, .{}, error.DuplicateTensor);
    var custom = try gguf.Container.parse(t.allocator, aligned, .{});
    defer custom.deinit();
    const alignment = offset(aligned, custom.findMetadata("general.alignment").?.encoded);
    try mutation(u32, aligned, alignment, 0, error.InvalidAlignment);
    try mutation(u32, aligned, alignment, 3, error.InvalidAlignment);
    try mutation(u32, aligned, alignment - 4, 6, error.InvalidAlignment);
}

fn allocationCase(allocator: std.mem.Allocator) !void {
    var container = try gguf.Container.parse(allocator, full, .{});
    defer container.deinit();
    try t.expectEqual(@as(usize, 8), container.tensors.len);
}

test "GGUF every allocation failure cleans up" {
    try t.checkAllAllocationFailures(t.allocator, allocationCase, .{});
}

test "mapped file owns storage independently from the container" {
    var file = try artifact.MappedFile.open(t.io, "tests/fixtures/gguf/default.gguf", 4096);
    defer file.deinit();
    try t.expectEqualSlices(u8, full, file.bytes);
    var container = try gguf.Container.parse(t.allocator, file.bytes, .{});
    defer container.deinit();
    try t.expectEqual(@as(usize, 8), container.tensors.len);
    try t.expectError(error.FileTooLarge, artifact.MappedFile.open(t.io, "tests/fixtures/gguf/default.gguf", 1));
    try t.expectError(error.NotRegularFile, artifact.MappedFile.open(t.io, "tests/fixtures/gguf", 4096));
    try t.expectError(error.FileNotFound, artifact.MappedFile.open(t.io, "tests/fixtures/gguf/nonexistent", 4096));
}
