const std = @import("std");

pub const Format = enum { q4_0, q8_0 };
pub const DecodeError = error{ InvalidBlockLength, InvalidOutputLength, NonFiniteScale };

pub fn blockBytes(format: Format) usize {
    return switch (format) {
        .q4_0 => 18,
        .q8_0 => 34,
    };
}

/// CPU diagnostic decoder. Source and destination must not overlap.
/// All validation precedes writes; no allocation or foreign code is involved.
pub fn decode(comptime format: Format, encoded: []const u8, output: []f32) DecodeError!void {
    @setFloatMode(.strict);
    const width = comptime blockBytes(format);
    if (encoded.len % width != 0) return error.InvalidBlockLength;
    if (output.len % 32 != 0 or output.len / 32 != encoded.len / width)
        return error.InvalidOutputLength;

    var offset: usize = 0;
    while (offset < encoded.len) : (offset += width) {
        const bits = std.mem.readInt(u16, encoded[offset..][0..2], .little);
        if (bits & 0x7c00 == 0x7c00) return error.NonFiniteScale;
    }

    offset = 0;
    var output_offset: usize = 0;
    while (offset < encoded.len) : ({
        offset += width;
        output_offset += 32;
    }) {
        const bits = std.mem.readInt(u16, encoded[offset..][0..2], .little);
        const scale: f32 = @floatCast(@as(f16, @bitCast(bits)));
        const payload = encoded[offset + 2 ..][0 .. width - 2];
        switch (format) {
            .q4_0 => {
                const bytes: @Vector(16, u8) = payload[0..16].*;
                const bias: @Vector(16, i32) = @splat(8);
                const low: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(bytes & @as(@Vector(16, u8), @splat(15)))) - bias;
                const high: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(bytes >> @as(@Vector(16, u3), @splat(4)))) - bias;
                const factor: @Vector(16, f32) = @splat(scale);
                // GGUF stores low-nibble values first, then high-nibble values.
                output[output_offset..][0..16].* = @as(@Vector(16, f32), @floatFromInt(low)) * factor;
                output[output_offset + 16 ..][0..16].* = @as(@Vector(16, f32), @floatFromInt(high)) * factor;
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
