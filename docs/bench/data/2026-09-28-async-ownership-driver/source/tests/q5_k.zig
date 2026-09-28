const std = @import("std");
const quant = @import("quant");
const t = std.testing;
const Sha256 = @import("fast_sha256.zig").Sha256; // hashing in the ReleaseFast support object
const parallel = @import("parallel.zig");
const Fingerprint = struct { blocks: usize, values: usize, output_sha256: []const u8 };
const Goldens = struct {
    generator_sha256: []const u8,
    helper_sha256: []const u8,
    pattern_hex: []const u8,
    partners: []const u16,
    fingerprint: Fingerprint,
    scales_fingerprint: Fingerprint,
    examples: []const struct { name: []const u8, packed_hex: []const u8, output_le_hex: []const u8 },
};
/// Chunk c: field c / 64 (the scale d or the minimum dmin), halves (c % 64) * 1024 + 0..1024,
/// each block row with the three partner values in the other field.
const GlobalFields = struct {
    pattern: *const [176]u8,
    partners: []const u16,
    fn produce(self: GlobalFields, c: usize, out: *std.ArrayList(u8)) !void {
        var encoded: [176 * 3]u8 = undefined;
        var output: [256 * 3]f32 = undefined;
        var bytes: [256 * 3 * 4]u8 = undefined;
        std.debug.assert(self.partners.len == 3);
        const field = c / 64;
        for ((c % 64) * 1024..(c % 64 + 1) * 1024) |bits| {
            if (bits & 0x7c00 == 0x7c00) continue;
            for (self.partners, 0..) |other, i| {
                const at = i * 176;
                @memcpy(encoded[at..][0..176], self.pattern);
                std.mem.writeInt(u16, encoded[at..][0..2], if (field == 0) @intCast(bits) else other, .little);
                std.mem.writeInt(u16, encoded[at + 2 ..][0..2], if (field == 1) @intCast(bits) else other, .little);
            }
            try quant.decode(.q5_k, &encoded, &output);
            canonical(&output, &bytes);
            try out.appendSlice(t.allocator, &bytes);
        }
    }
};

fn load() !std.json.Parsed(Goldens) {
    return std.json.parseFromSlice(Goldens, t.allocator, @embedFile("fixtures/q5_k.json"), .{ .ignore_unknown_fields = true });
}
fn canonical(values: []const f32, bytes: []u8) void {
    for (values, 0..) |value, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
}
fn expectHash(expected: []const u8, hash: *Sha256) !void {
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    try t.expectEqualStrings(expected, &std.fmt.bytesToHex(digest, .lower));
}

test "Q5_K independent patterns, packed high planes and all actual tensors" {
    const parsed = try load();
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 176), quant.blockBytes(.q5_k));
    try t.expectEqual(@as(usize, 256), quant.blockElements(.q5_k));
    var hash = Sha256.init(.{});
    hash.update(@embedFile("reference/generate_q5_k_goldens.py"));
    try expectHash(parsed.value.generator_sha256, &hash);
    hash = Sha256.init(.{});
    hash.update(@embedFile("reference/generate_q4_1_goldens.py"));
    try expectHash(parsed.value.helper_sha256, &hash);
    try t.expectEqual(@as(usize, 51), parsed.value.examples.len);
    for (parsed.value.examples) |example| {
        errdefer std.debug.print("Q5_K case {s}\n", .{example.name});
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
        try quant.decode(.q5_k, encoded, output);
        canonical(output, actual);
        try t.expectEqualSlices(u8, expected, actual);
    }
}

test "Q5_K finite global fields and exhaustive packed subscale byte positions" {
    const parsed = try load();
    defer parsed.deinit();
    const partners = [_]u16{ 0, 0x3555, 0xfbff };
    try t.expectEqualSlices(u16, &partners, parsed.value.partners);
    var pattern: [176]u8 = undefined;
    try t.expectEqual(pattern.len, (try std.fmt.hexToBytes(&pattern, parsed.value.pattern_hex)).len);
    var encoded: [176 * partners.len]u8 = undefined;
    var output: [256 * partners.len]f32 = undefined;
    var bytes: [256 * partners.len * 4]u8 = undefined;
    // Both fields x all finite halves, in that order: chunks of 1024 halves run concurrently
    // and are hashed in order (tests/parallel.zig).
    var hash = Sha256.init(.{});
    try parallel.hashChunks(&hash, 2 * 64, GlobalFields{ .pattern = &pattern, .partners = &partners }, GlobalFields.produce);
    var blocks: usize = 0;
    for (0..2 * 65536) |i| {
        if (i & 0x7c00 != 0x7c00) blocks += partners.len;
    }
    try t.expectEqual(@as(usize, 380928), blocks);
    try t.expectEqual(blocks, parsed.value.fingerprint.blocks);
    try t.expectEqual(blocks * 256, parsed.value.fingerprint.values);
    try expectHash(parsed.value.fingerprint.output_sha256, &hash);
    hash = Sha256.init(.{});
    for (4..16) |at| {
        for (0..256) |byte| {
            @memcpy(encoded[0..176], &pattern);
            encoded[at] = @intCast(byte);
            try quant.decode(.q5_k, encoded[0..176], output[0..256]);
            canonical(output[0..256], bytes[0..1024]);
            hash.update(bytes[0..1024]);
        }
    }
    try t.expectEqual(@as(usize, 3072), parsed.value.scales_fingerprint.blocks);
    try t.expectEqual(@as(usize, 786432), parsed.value.scales_fingerprint.values);
    try expectHash(parsed.value.scales_fingerprint.output_sha256, &hash);
}

test "Q5_K 256-element block sizes, empty rows and both nonfinite fields fail atomically" {
    var encoded: [352]u8 = @splat(0);
    var output: [512]f32 = @splat(123.0);
    try quant.decode(.q5_k, &.{}, output[0..0]);
    try t.expectError(error.InvalidBlockLength, quant.decode(.q5_k, encoded[0..175], output[0..256]));
    for ([_]usize{ 0, 32, 255, 257, 512 }) |length|
        try t.expectError(error.InvalidOutputLength, quant.decode(.q5_k, encoded[0..176], output[0..length]));
    try t.expectError(error.InvalidOutputLength, quant.decode(.q5_k, &encoded, output[0..256]));
    try t.expectError(error.InvalidOutputLength, quant.decode(.q5_k, &.{}, &output));
    for ([_]usize{ 0, 2, 176, 178 }) |at| {
        var cases: usize = 0;
        for (0..65536) |value| {
            const bits: u16 = @intCast(value);
            if (bits & 0x7c00 != 0x7c00) continue;
            std.mem.writeInt(u16, encoded[at..][0..2], bits, .little);
            try t.expectError(error.NonFiniteScale, quant.decode(.q5_k, &encoded, &output));
            for (output) |x| try t.expectEqual(@as(f32, 123.0), x);
            cases += 1;
        }
        try t.expectEqual(@as(usize, 2048), cases);
        std.mem.writeInt(u16, encoded[at..][0..2], 0, .little);
    }
}
