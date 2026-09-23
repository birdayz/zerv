const std = @import("std");
const quant = @import("zerv").quant;
const t = std.testing;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Goldens = struct {
    schema_version: u32,
    generator_sha256: []const u8,
    edge_fields: []const u16,
    fingerprint: struct { blocks: usize, values: usize, output_sha256: []const u8 },
    examples: []const struct { name: []const u8, packed_hex: []const u8, output_le_hex: []const u8 },
};
fn load() !std.json.Parsed(Goldens) {
    return std.json.parseFromSlice(Goldens, t.allocator, @embedFile("fixtures/q4_1.json"), .{ .ignore_unknown_fields = true });
}
fn canonical(values: []const f32, bytes: []u8) void {
    for (values, 0..) |value, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
}
fn expectHash(expected: []const u8, hash: *Sha256) !void {
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    try t.expectEqualStrings(expected, &std.fmt.bytesToHex(digest, .lower));
}

test "Q4_1 independent explicit patterns and all eight actual model tensors" {
    const parsed = try load();
    defer parsed.deinit();
    try t.expectEqual(@as(u32, 1), parsed.value.schema_version);
    try t.expectEqual(@as(usize, 20), quant.blockBytes(.q4_1));
    try t.expectEqual(@as(usize, 32), quant.blockElements(.q4_1));
    var hash = Sha256.init(.{});
    hash.update(@embedFile("reference/generate_q4_1_goldens.py"));
    try expectHash(parsed.value.generator_sha256, &hash);
    try t.expectEqual(@as(usize, 11), parsed.value.examples.len);
    for (parsed.value.examples) |example| {
        errdefer std.debug.print("Q4_1 case {s}\n", .{example.name});
        const storage = try t.allocator.alloc(u8, example.packed_hex.len / 2 + 1);
        defer t.allocator.free(storage);
        const packed_bytes = try std.fmt.hexToBytes(storage[1..], example.packed_hex);
        const expected = try t.allocator.alloc(u8, example.output_le_hex.len / 2);
        defer t.allocator.free(expected);
        _ = try std.fmt.hexToBytes(expected, example.output_le_hex);
        const output = try t.allocator.alloc(f32, expected.len / 4);
        defer t.allocator.free(output);
        try quant.decode(.q4_1, packed_bytes, output);
        const actual = try t.allocator.alloc(u8, expected.len);
        defer t.allocator.free(actual);
        canonical(output, actual);
        try t.expectEqualSlices(u8, expected, actual);
    }
}

test "Q4_1 every finite scale/minimum against eleven partner fields and all coefficients" {
    const parsed = try load();
    defer parsed.deinit();
    const edge = [_]u16{ 0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff };
    try t.expectEqualSlices(u16, &edge, parsed.value.edge_fields);
    var encoded: [edge.len * 20]u8 = undefined;
    var output: [edge.len * 32]f32 = undefined;
    var bytes: [edge.len * 32 * 4]u8 = undefined;
    var hash = Sha256.init(.{});
    var blocks: usize = 0;
    for (0..2) |field| {
        for (0..65536) |bits| {
            if (bits & 0x7c00 == 0x7c00) continue;
            for (edge, 0..) |other, i| {
                const at = i * 20;
                std.mem.writeInt(u16, encoded[at..][0..2], if (field == 0) @intCast(bits) else other, .little);
                std.mem.writeInt(u16, encoded[at + 2 ..][0..2], if (field == 1) @intCast(bits) else other, .little);
                for (0..16) |j| encoded[at + 4 + j] = @intCast(j | ((15 - j) << 4));
            }
            try quant.decode(.q4_1, &encoded, &output);
            canonical(&output, &bytes);
            hash.update(&bytes);
            blocks += edge.len;
        }
    }
    try t.expectEqual(@as(usize, 1396736), blocks);
    try t.expectEqual(blocks, parsed.value.fingerprint.blocks);
    try t.expectEqual(blocks * 32, parsed.value.fingerprint.values);
    try expectHash(parsed.value.fingerprint.output_sha256, &hash);
}

test "Q4_1 rejects every nonfinite scale and minimum atomically in either block" {
    var encoded: [40]u8 = @splat(0);
    var output: [64]f32 = @splat(123.0);
    for ([_]usize{ 0, 2, 20, 22 }) |at| {
        var cases: usize = 0;
        for (0..65536) |value| {
            const bits: u16 = @intCast(value);
            if (bits & 0x7c00 != 0x7c00) continue;
            std.mem.writeInt(u16, encoded[at..][0..2], bits, .little);
            try t.expectError(error.NonFiniteScale, quant.decode(.q4_1, &encoded, &output));
            for (output) |x| try t.expectEqual(@as(f32, 123.0), x);
            cases += 1;
        }
        try t.expectEqual(@as(usize, 2048), cases);
        std.mem.writeInt(u16, encoded[at..][0..2], 0, .little);
    }
}
