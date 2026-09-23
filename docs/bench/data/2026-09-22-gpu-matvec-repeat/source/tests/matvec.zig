const std = @import("std");
const m = @import("zerv").matvec;
const t = std.testing;

test "matvec bounded shapes, byte addressing and two-dimensional dispatch" {
    const shape: m.Shape = .{ .format = .q6_k, .columns = 5120, .rows = 248320 };
    try t.expectEqual(@as(u64, 1042944000), try shape.weightBytes());
    const g = try m.geometry(shape, .{ .{ .offset = 2, .buffer_bytes = 1042944004 }, .{ .offset = 16, .buffer_bytes = 20496 }, .{ .offset = 4, .buffer_bytes = 993284 } }, .{ 65535, 65535, 65535 });
    try t.expectEqualSlices(u32, &.{ 65535, 4, 1 }, &g.groups);
    try t.expectEqual(@as(u32, 4200), g.push.row_bytes);
    try t.expectEqual(@as(u32, 2), g.push.weight_offset);
    try t.expectEqual(@as(u32, 4), g.push.input_offset);
    try t.expectEqual(@as(u32, 1), g.push.output_offset);
    const regions: [3]m.Region = @splat(.{ .offset = 0, .buffer_bytes = 0xfffffffc });
    inline for ([_]m.Format{ .f32, .q4_0, .q4_1, .q5_k, .q6_k }) |format| {
        var s: m.Shape = .{ .format = format, .columns = format.blockElements(), .rows = 1 };
        _ = try m.geometry(s, regions, .{ 1, 1, 1 });
        s.columns = 0;
        try t.expectError(error.InvalidShape, s.weightBytes());
        s.columns = 32769;
        try t.expectError(error.InvalidShape, s.weightBytes());
        s.columns = format.blockElements();
        s.rows = 0;
        try t.expectError(error.InvalidShape, s.weightBytes());
        s.rows = 1048577;
        try t.expectError(error.InvalidShape, s.weightBytes());
        s.rows = 1048576;
        s.columns = 32768;
        try t.expectError(error.InvalidRange, s.weightBytes());
        if (format != .f32) {
            s.columns = format.blockElements() - 1;
            s.rows = 1;
            try t.expectError(error.InvalidShape, s.weightBytes());
        }
    }
    const small: m.Shape = .{ .format = .q4_0, .columns = 32, .rows = 3 };
    for (0..3) |index| {
        var invalid = regions;
        invalid[index].offset = 1;
        try t.expectError(error.InvalidRange, m.geometry(small, invalid, .{ 10, 10, 10 }));
        invalid = regions;
        invalid[index].buffer_bytes = 0xffffffff;
        try t.expectError(error.InvalidRange, m.geometry(small, invalid, .{ 10, 10, 10 }));
        invalid[index].buffer_bytes = 0x100000000;
        try t.expectError(error.InvalidRange, m.geometry(small, invalid, .{ 10, 10, 10 }));
        invalid = regions;
        invalid[index].offset = std.math.maxInt(u64) - 3;
        try t.expectError(error.InvalidRange, m.geometry(small, invalid, .{ 10, 10, 10 }));
        invalid = regions;
        invalid[index].buffer_bytes = 4;
        try t.expectError(error.InvalidRange, m.geometry(small, invalid, .{ 10, 10, 10 }));
        var limits: [3]u32 = .{ 10, 10, 10 };
        limits[index] = 0;
        try t.expectError(error.InvalidDispatch, m.geometry(small, regions, limits));
    }
    try t.expectError(error.InvalidDispatch, m.geometry(small, regions, .{ 1, 2, 1 }));
}

test "matvec immutable upload validation, all nonfinite halves and trailing Q6 scale" {
    inline for ([_]m.Format{ .q4_0, .q4_1, .q5_k, .q6_k }) |format| {
        const width = comptime format.blockBytes();
        const shape: m.Shape = .{ .format = format, .columns = format.blockElements(), .rows = 2 };
        var storage: [width * 2 + 1]u8 = @splat(0);
        const bytes = storage[1..]; // Byte-unaligned host input supported.
        try m.validateWeights(shape, bytes);
        try t.expectError(error.InvalidWeights, m.validateWeights(shape, bytes[0 .. bytes.len - 1]));
        for (0..2) |block| {
            const fields: usize = if (format == .q4_1 or format == .q5_k) 2 else 1;
            for (0..fields) |field| {
                const offset = block * width + (if (format == .q6_k) @as(usize, 208) else field * 2);
                for (0..65536) |value| {
                    const bits: u16 = @intCast(value);
                    std.mem.writeInt(u16, bytes[offset..][0..2], bits, .little);
                    if (bits & 0x7c00 == 0x7c00) {
                        try t.expectError(error.NonFiniteWeight, m.validateWeights(shape, bytes));
                    } else try m.validateWeights(shape, bytes);
                    try t.expectEqual(bits, std.mem.readInt(u16, bytes[offset..][0..2], .little));
                }
                std.mem.writeInt(u16, bytes[offset..][0..2], 0, .little);
            }
        }
        if (format == .q6_k) {
            std.mem.writeInt(u16, bytes[0..2], 0x7c00, .little);
            try m.validateWeights(shape, bytes); // Packed coefficient prefix is not a scale.
        }
    }
    const shape: m.Shape = .{ .format = .f32, .columns = 1, .rows = 1 };
    var bytes: [4]u8 = undefined;
    for ([_]u32{ 0, 0x80000000, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff }) |bits| {
        std.mem.writeInt(u32, &bytes, bits, .little);
        try m.validateWeights(shape, &bytes);
    }
    for ([_]u32{ 1, 0x80000001, 0x007fffff, 0x807fffff }) |bits| {
        std.mem.writeInt(u32, &bytes, bits, .little);
        try t.expectError(error.UnsupportedSubnormal, m.validateWeights(shape, &bytes));
    }
    for ([_]u32{ 0x7f800000, 0xff800000, 0x7fc00000, 0xffffffff }) |bits| {
        std.mem.writeInt(u32, &bytes, bits, .little);
        try t.expectError(error.NonFiniteWeight, m.validateWeights(shape, &bytes));
    }
}
