const std = @import("std");

pub const Format = enum { q4_0, q8_0, q4_1 };
pub const DecodeError = error{ InvalidBlockLength, InvalidOutputLength, NonFiniteScale };

pub fn blockBytes(format: Format) usize {
    return switch (format) {
        .q4_0 => 18,
        .q8_0 => 34,
        .q4_1 => 20,
    };
}

pub fn blockElements(format: Format) usize {
    return switch (format) {
        .q4_0, .q8_0, .q4_1 => 32,
    };
}

/// CPU diagnostic decoder. Source and destination must not overlap.
/// All validation precedes writes; no allocation or foreign code is involved.
pub fn decode(comptime format: Format, encoded: []const u8, output: []f32) DecodeError!void {
    @setFloatMode(.strict);
    const width = comptime blockBytes(format);
    const elements = comptime blockElements(format);
    if (encoded.len % width != 0) return error.InvalidBlockLength;
    if (output.len % elements != 0 or output.len / elements != encoded.len / width)
        return error.InvalidOutputLength;

    var offset: usize = 0;
    while (offset < encoded.len) : (offset += width) {
        inline for (0..if (format == .q4_1) 2 else 1) |field| {
            const bits = std.mem.readInt(u16, encoded[offset + field * 2 ..][0..2], .little);
            if (bits & 0x7c00 == 0x7c00) return error.NonFiniteScale;
        }
    }

    offset = 0;
    var output_offset: usize = 0;
    while (offset < encoded.len) : ({
        offset += width;
        output_offset += elements;
    }) {
        const bits = std.mem.readInt(u16, encoded[offset..][0..2], .little);
        const scale: f32 = @floatCast(@as(f16, @bitCast(bits)));
        const payload_offset = if (format == .q4_1) 4 else 2;
        const payload = encoded[offset + payload_offset ..][0 .. width - payload_offset];
        switch (format) {
            .q4_0, .q4_1 => {
                const bytes: @Vector(16, u8) = payload[0..16].*;
                const bias: @Vector(16, i32) = @splat(if (format == .q4_0) 8 else 0);
                const low: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(bytes & @as(@Vector(16, u8), @splat(15)))) - bias;
                const high: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(bytes >> @as(@Vector(16, u3), @splat(4)))) - bias;
                const factor: @Vector(16, f32) = @splat(scale);
                // GGUF stores low-nibble values first, then high-nibble values.
                const low_values = @as(@Vector(16, f32), @floatFromInt(low)) * factor;
                const high_values = @as(@Vector(16, f32), @floatFromInt(high)) * factor;
                if (format == .q4_1) {
                    const min_bits = std.mem.readInt(u16, encoded[offset + 2 ..][0..2], .little);
                    const minimum: @Vector(16, f32) = @splat(@floatCast(@as(f16, @bitCast(min_bits))));
                    output[output_offset..][0..16].* = low_values + minimum;
                    output[output_offset + 16 ..][0..16].* = high_values + minimum;
                } else {
                    output[output_offset..][0..16].* = low_values;
                    output[output_offset + 16 ..][0..16].* = high_values;
                }
            },
            .q8_0 => {
                const bytes: @Vector(32, u8) = payload[0..32].*;
                const signed: @Vector(32, i8) = @bitCast(bytes);
                const factor: @Vector(32, f32) = @splat(scale);
                output[output_offset..][0..32].* = @as(@Vector(32, f32), @floatFromInt(signed)) * factor;
            },
        }
    }
}
