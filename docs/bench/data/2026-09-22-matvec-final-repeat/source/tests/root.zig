const std = @import("std");
const quant = @import("zerv").quant;
const testing = std.testing;
const Sha256 = std.crypto.hash.sha2.Sha256;

test {
    _ = @import("gguf.zig");
    _ = @import("chat.zig");
    _ = @import("nfc.zig");
    _ = @import("tokenizer_split.zig");
    _ = @import("tokenizer.zig");
    _ = @import("q4_1.zig");
    _ = @import("q5_k.zig");
    _ = @import("q6_k.zig");
    _ = @import("gpu_abi.zig");
    _ = @import("matvec.zig");
}

const Fingerprint = struct {
    scale_cases: usize,
    values: usize,
    output_sha256: []const u8,
};
const Goldens = struct {
    schema_version: u32,
    generator_sha256: []const u8,
    formats: struct { q4_0: Fingerprint, q8_0: Fingerprint },
    examples: []const struct {
        format: quant.Format,
        packed_hex: []const u8,
        output_le_hex: []const u8,
    },
};

fn loadGoldens() !std.json.Parsed(Goldens) {
    return std.json.parseFromSlice(Goldens, testing.allocator, @embedFile("fixtures/quantization.json"), .{
        .ignore_unknown_fields = true, // Provenance fields are retained for independent review.
    });
}

fn encodeFloats(values: []const f32, output: []u8) void {
    std.debug.assert(output.len / 4 == values.len and output.len % 4 == 0);
    for (values, 0..) |value, i| {
        std.mem.writeInt(u32, output[i * 4 ..][0..4], @bitCast(value), .little);
    }
}

fn expectDigest(expected: []const u8, hasher: *Sha256) !void {
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    try testing.expectEqualStrings(expected, &hex);
}

test "golden schema and reference generator provenance" {
    const parsed = try loadGoldens();
    defer parsed.deinit();
    try testing.expectEqual(@as(u32, 1), parsed.value.schema_version);
    var hasher = Sha256.init(.{});
    hasher.update(@embedFile("reference/generate_quant_goldens.py"));
    try expectDigest(parsed.value.generator_sha256, &hasher);
}

test "multi-block external goldens with unaligned encoded input" {
    const parsed = try loadGoldens();
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.examples.len);
    for (parsed.value.examples) |example| {
        const storage = try testing.allocator.alloc(u8, example.packed_hex.len / 2 + 1);
        defer testing.allocator.free(storage);
        const encoded = try std.fmt.hexToBytes(storage[1..], example.packed_hex);
        const expected = try testing.allocator.alloc(u8, example.output_le_hex.len / 2);
        defer testing.allocator.free(expected);
        _ = try std.fmt.hexToBytes(expected, example.output_le_hex);
        const values = try testing.allocator.alloc(f32, expected.len / 4);
        defer testing.allocator.free(values);
        switch (example.format) {
            .q4_0 => try quant.decode(.q4_0, encoded, values),
            .q8_0 => try quant.decode(.q8_0, encoded, values),
            .q4_1 => try quant.decode(.q4_1, encoded, values),
            .q5_k => try quant.decode(.q5_k, encoded, values),
            .q6_k => try quant.decode(.q6_k, encoded, values),
        }
        const actual = try testing.allocator.alloc(u8, expected.len);
        defer testing.allocator.free(actual);
        encodeFloats(values, actual);
        try testing.expectEqualSlices(u8, expected, actual);
    }
}

test "every finite binary16 scale and Q4/Q8 coefficient matches external fingerprint" {
    const parsed = try loadGoldens();
    defer parsed.deinit();
    inline for (.{ quant.Format.q4_0, quant.Format.q8_0 }) |format| {
        const width = comptime quant.blockBytes(format);
        const blocks = if (format == .q4_0) 1 else 8;
        var encoded: [width * blocks]u8 = undefined;
        var output: [32 * blocks]f32 = undefined;
        var canonical: [32 * blocks * 4]u8 = undefined;
        var hasher = Sha256.init(.{});
        var count: usize = 0;
        for (0..65536) |scale_bits| {
            const bits: u16 = @intCast(scale_bits);
            if (bits & 0x7c00 == 0x7c00) continue;
            for (0..blocks) |block| {
                const offset = block * width;
                std.mem.writeInt(u16, encoded[offset..][0..2], bits, .little);
                for (0..width - 2) |j| {
                    encoded[offset + 2 + j] = @intCast(if (format == .q4_0)
                        j | ((15 - j) << 4)
                    else
                        block * 32 + j);
                }
            }
            try quant.decode(format, &encoded, &output);
            encodeFloats(&output, &canonical);
            hasher.update(&canonical);
            count += 1;
        }
        const expected = @field(parsed.value.formats, @tagName(format));
        try testing.expectEqual(@as(usize, 63488), count);
        try testing.expectEqual(expected.scale_cases, count);
        try testing.expectEqual(expected.values, count * output.len);
        try expectDigest(expected.output_sha256, &hasher);
    }
}

test "empty rows, incomplete blocks, and exact output size" {
    inline for (.{ quant.Format.q4_0, quant.Format.q8_0, quant.Format.q4_1 }) |format| {
        const width = comptime quant.blockBytes(format);
        const encoded: [width * 2]u8 = @splat(0);
        var output: [64]f32 = @splat(123.0);
        var empty: [0]f32 = .{};
        try quant.decode(format, &.{}, &empty);
        try testing.expectError(error.InvalidBlockLength, quant.decode(format, encoded[0 .. width - 1], output[0..32]));
        try testing.expectError(error.InvalidOutputLength, quant.decode(format, encoded[0..width], output[0..31]));
        try testing.expectError(error.InvalidOutputLength, quant.decode(format, encoded[0..width], output[0..33]));
        try testing.expectError(error.InvalidOutputLength, quant.decode(format, &encoded, output[0..32]));
        try testing.expectError(error.InvalidOutputLength, quant.decode(format, encoded[0..width], &output));
        try testing.expectError(error.InvalidOutputLength, quant.decode(format, &.{}, &output));
        try testing.expectError(error.InvalidOutputLength, quant.decode(format, encoded[0..width], &empty));
        for (output) |value| try testing.expectEqual(@as(f32, 123.0), value);
    }
}

test "all non-finite scales reject atomically, including an invalid later block" {
    inline for (.{ quant.Format.q4_0, quant.Format.q8_0 }) |format| {
        const width = comptime quant.blockBytes(format);
        var encoded: [width * 2]u8 = @splat(0);
        var output: [64]f32 = @splat(123.0);
        var count: usize = 0;
        for (0..65536) |scale_bits| {
            const bits: u16 = @intCast(scale_bits);
            if (bits & 0x7c00 != 0x7c00) continue;
            std.mem.writeInt(u16, encoded[width..][0..2], bits, .little);
            try testing.expectError(error.NonFiniteScale, quant.decode(format, &encoded, &output));
            for (output) |value| try testing.expectEqual(@as(f32, 123.0), value);
            count += 1;
        }
        try testing.expectEqual(@as(usize, 2048), count);
    }
}
