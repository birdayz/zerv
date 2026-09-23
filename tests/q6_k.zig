const std = @import("std");
const quant = @import("zerv").quant;
const t = std.testing;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Fingerprint = struct { blocks: usize, values: usize, output_sha256: []const u8 };
const Goldens = struct {
    generator_sha256: []const u8,
    helper_sha256: []const u8,
    patterns_hex: []const []const u8,
    fingerprint: Fingerprint,
    scales_fingerprint: Fingerprint,
    examples: []const struct { name: []const u8, packed_hex: []const u8, output_le_hex: []const u8 },
};
fn load() !std.json.Parsed(Goldens) {
    return std.json.parseFromSlice(Goldens, t.allocator, @embedFile("fixtures/q6_k.json"), .{ .ignore_unknown_fields = true });
}
fn canonical(values: []const f32, bytes: []u8) void {
    for (values, 0..) |value, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
}
fn expectHash(expected: []const u8, hash: *Sha256) !void {
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    try t.expectEqualStrings(expected, &std.fmt.bytesToHex(digest, .lower));
}

test "Q6_K independent payload planes, signed scales and actual output weight" {
    const parsed = try load();
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 210), quant.blockBytes(.q6_k));
    try t.expectEqual(@as(usize, 256), quant.blockElements(.q6_k));
    var hash = Sha256.init(.{});
    hash.update(@embedFile("reference/generate_q6_k_goldens.py"));
    try expectHash(parsed.value.generator_sha256, &hash);
    hash = Sha256.init(.{});
    hash.update(@embedFile("reference/generate_q4_1_goldens.py"));
    try expectHash(parsed.value.helper_sha256, &hash);
    try t.expectEqual(@as(usize, 4), parsed.value.examples.len);
    for (parsed.value.examples) |example| {
        errdefer std.debug.print("Q6_K case {s}\n", .{example.name});
        const storage = try t.allocator.alloc(u8, example.packed_hex.len / 2 + 1);
        defer t.allocator.free(storage);
        const encoded = try std.fmt.hexToBytes(storage[1..], example.packed_hex);
        const expected = try t.allocator.alloc(u8, example.output_le_hex.len / 2);
        defer t.allocator.free(expected);
        _ = try std.fmt.hexToBytes(expected, example.output_le_hex);
        const output = try t.allocator.alloc(f32, expected.len / 4);
        defer t.allocator.free(output);
        const actual = try t.allocator.alloc(u8, expected.len);
        defer t.allocator.free(actual);
        try quant.decode(.q6_k, encoded, output);
        canonical(output, actual);
        try t.expectEqualSlices(u8, expected, actual);
    }
}

test "Q6_K every finite trailing half and every signed subgroup scale encoding" {
    const parsed = try load();
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 4), parsed.value.patterns_hex.len);
    var patterns: [4][210]u8 = undefined;
    for (&patterns, parsed.value.patterns_hex) |*pattern, hex|
        try t.expectEqual(pattern.len, (try std.fmt.hexToBytes(pattern, hex)).len);
    var encoded: [210 * 4]u8 = undefined;
    var output: [256 * 4]f32 = undefined;
    var bytes: [256 * 4 * 4]u8 = undefined;
    var hash = Sha256.init(.{});
    var blocks: usize = 0;
    for (0..65536) |bits| {
        if (bits & 0x7c00 == 0x7c00) continue;
        for (patterns, 0..) |pattern, phase| {
            const at = phase * 210;
            @memcpy(encoded[at..][0..210], &pattern);
            std.mem.writeInt(u16, encoded[at + 208 ..][0..2], @intCast(bits), .little);
        }
        try quant.decode(.q6_k, &encoded, &output);
        canonical(&output, &bytes);
        hash.update(&bytes);
        blocks += 4;
    }
    try t.expectEqual(@as(usize, 253952), blocks);
    try t.expectEqual(blocks, parsed.value.fingerprint.blocks);
    try t.expectEqual(blocks * 256, parsed.value.fingerprint.values);
    try expectHash(parsed.value.fingerprint.output_sha256, &hash);
    hash = Sha256.init(.{});
    for (192..208) |scale_at| {
        for (0..256) |byte| {
            for (patterns, 0..) |pattern, phase| {
                const at = phase * 210;
                @memcpy(encoded[at..][0..210], &pattern);
                encoded[at + scale_at] = @intCast(byte);
                std.mem.writeInt(u16, encoded[at + 208 ..][0..2], 0x3555, .little);
            }
            try quant.decode(.q6_k, &encoded, &output);
            canonical(&output, &bytes);
            hash.update(&bytes);
        }
    }
    try t.expectEqual(@as(usize, 16384), parsed.value.scales_fingerprint.blocks);
    try t.expectEqual(@as(usize, 4194304), parsed.value.scales_fingerprint.values);
    try expectHash(parsed.value.scales_fingerprint.output_sha256, &hash);
}

test "Q6_K lengths and nonfinite trailing halves are atomic, prefix is not a half" {
    var encoded: [420]u8 = @splat(0);
    var output: [512]f32 = @splat(123.0);
    try quant.decode(.q6_k, &.{}, output[0..0]);
    try t.expectError(error.InvalidBlockLength, quant.decode(.q6_k, encoded[0..209], output[0..256]));
    for ([_]usize{ 0, 32, 255, 257, 512 }) |length|
        try t.expectError(error.InvalidOutputLength, quant.decode(.q6_k, encoded[0..210], output[0..length]));
    try t.expectError(error.InvalidOutputLength, quant.decode(.q6_k, &encoded, output[0..256]));
    try t.expectError(error.InvalidOutputLength, quant.decode(.q6_k, &.{}, &output));
    for ([_]usize{ 208, 418 }) |at| {
        var cases: usize = 0;
        for (0..65536) |value| {
            const bits: u16 = @intCast(value);
            if (bits & 0x7c00 != 0x7c00) continue;
            std.mem.writeInt(u16, encoded[at..][0..2], bits, .little);
            try t.expectError(error.NonFiniteScale, quant.decode(.q6_k, &encoded, &output));
            for (output) |x| try t.expectEqual(@as(f32, 123.0), x);
            cases += 1;
        }
        try t.expectEqual(@as(usize, 2048), cases);
        std.mem.writeInt(u16, encoded[at..][0..2], 0, .little);
    }
    @memset(encoded[192..208], 1);
    std.mem.writeInt(u16, encoded[208..210], 0x3c00, .little);
    for (0..65536) |value| {
        const bits: u16 = @intCast(value);
        if (bits & 0x7c00 != 0x7c00) continue;
        std.mem.writeInt(u16, encoded[0..2], bits, .little);
        try quant.decode(.q6_k, encoded[0..210], output[0..256]);
        try t.expectEqual(@as(f32, @floatFromInt(@as(i32, bits & 15) - 32)), output[0]);
        try t.expectEqual(@as(f32, @floatFromInt(@as(i32, bits >> 4 & 15) - 32)), output[64]);
    }
}
