const std = @import("std");
const t = std.testing;
const split = @import("zerv").tokenizer.qwen_split;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Manifest = struct {
    generator_sha256: []const u8,
    data_sha256: []const u8,
    golden_sha256: []const u8,
    property_sha256: []const u8,
    records: usize,
};
fn manifest() !std.json.Parsed(Manifest) {
    return std.json.parseFromSlice(Manifest, t.allocator, @embedFile("fixtures/tokenizer-split/manifest.json"), .{ .ignore_unknown_fields = true });
}
fn hex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

test "Qwen split independent exhaustive short strings, Unicode folds, mixed text and prompts" {
    const m = try manifest();
    defer m.deinit();
    try t.expectEqualStrings(m.value.generator_sha256, &hex(@embedFile("reference/generate_split_goldens.py")));
    try t.expectEqualStrings(m.value.data_sha256, &split.table_sha256);
    const data = @embedFile("fixtures/tokenizer-split/cases.bin");
    try t.expectEqualStrings(m.value.golden_sha256, &hex(data));
    var cursor: usize = 0;
    var records: usize = 0;
    while (cursor < data.len) {
        const len = std.mem.readInt(u32, data[cursor..][0..4], .little);
        const count = std.mem.readInt(u32, data[cursor + 4 ..][0..4], .little);
        cursor += 8;
        const input = data[cursor..][0..len];
        cursor += len;
        var iterator = try split.Iterator.init(input, .{});
        var start: usize = 0;
        for (0..count) |_| {
            const end = std.mem.readInt(u32, data[cursor..][0..4], .little);
            cursor += 4;
            const piece = iterator.next() orelse return error.MissingPiece;
            errdefer std.debug.print("split record {d}, input={any}, start={d}, expected_end={d}, actual_end={d}\n", .{ records, input, start, end, start + piece.len });
            try t.expect(piece.len > 0);
            try t.expectEqualSlices(u8, input[start..end], piece);
            try t.expectEqual(@intFromPtr(input.ptr) + start, @intFromPtr(piece.ptr));
            start = end;
        }
        try t.expectEqual(input.len, start);
        try t.expectEqual(null, iterator.next());
        try t.expectEqual(null, iterator.next());
        records += 1;
    }
    try t.expectEqual(m.value.records, records);
}

test "Qwen splitter every Unicode scalar matches independent regex properties" {
    const m = try manifest();
    defer m.deinit();
    var digest = Sha256.init(.{});
    for (0..0x110000) |cp| {
        if (cp >= 0xd800 and cp <= 0xdfff) continue;
        const byte: u8 = @bitCast(split.properties(@intCast(cp)));
        digest.update(&.{byte});
    }
    var hash: [32]u8 = undefined;
    digest.final(&hash);
    try t.expectEqualStrings(m.value.property_sha256, &std.fmt.bytesToHex(hash, .lower));
    for ([_]u21{ 0xd800, 0xdfff, 0x110000, 0x1fffff }) |cp| {
        try t.expectEqual(@as(u8, 0), @as(u8, @bitCast(split.properties(cp))));
    }
}

test "Qwen splitter strict UTF-8, byte limits and independent iterator state" {
    for ([_][]const u8{ "\xff", "\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x80", "a\x80" }) |bad| {
        try t.expectError(error.InvalidUtf8, split.Iterator.init(bad, .{}));
    }
    try t.expectError(error.InputTooLarge, split.Iterator.init("é", .{ .max_input_bytes = 1 }));
    try t.expectError(error.InputTooLarge, split.Iterator.init("\xff", .{ .max_input_bytes = 0 }));
    var empty = try split.Iterator.init("", .{ .max_input_bytes = 0 });
    try t.expectEqual(null, empty.next());
    var a = try split.Iterator.init("é1", .{ .max_input_bytes = 3 });
    var b = try split.Iterator.init("é1", .{ .max_input_bytes = 3 });
    try t.expectEqualStrings("é", a.next().?);
    try t.expectEqualStrings("1", a.next().?);
    try t.expectEqualStrings("é", b.next().?);
    try t.expectEqual(null, a.next());
    try t.expectEqualStrings("1", b.next().?);
    try t.expectEqual(null, b.next());
}
