const std = @import("std");

pub const Format = enum { q4_0, q8_0 };
pub const DecodeError = error{ InvalidBlockLength, InvalidOutputLength, NonFiniteScale };

pub fn blockBytes(format: Format) usize {
    return switch (format) {
        .q4_0 => 18,
        .q8_0 => 34,
    };
}

/// Scalar diagnostic decoder. Source and destination must not overlap.
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
            .q4_0 => for (payload, 0..) |byte, j| {
                const low: i8 = @as(i8, @intCast(byte & 15)) - 8;
                const high: i8 = @as(i8, @intCast(byte >> 4)) - 8;
                output[output_offset + j] = @as(f32, @floatFromInt(low)) * scale;
                output[output_offset + 16 + j] = @as(f32, @floatFromInt(high)) * scale;
            },
            .q8_0 => for (payload, 0..) |byte, j| {
                const signed: i8 = @bitCast(byte);
                output[output_offset + j] = @as(f32, @floatFromInt(signed)) * scale;
            },
        }
    }
}
