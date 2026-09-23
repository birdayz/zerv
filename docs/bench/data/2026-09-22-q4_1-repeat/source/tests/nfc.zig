const std = @import("std");
const t = std.testing;
const nfc = @import("zerv").text.nfc;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Manifest = struct {
    generator_sha256: []const u8,
    golden_sha256: []const u8,
    data_sha256: []const u8,
    records: usize,
    fingerprints: struct { scalar: []const u8, mark_context: []const u8 },
    workloads: []const struct { name: []const u8, text: []const u8, output: []const u8 },
};
fn manifest() !std.json.Parsed(Manifest) {
    return std.json.parseFromSlice(Manifest, t.allocator, @embedFile("fixtures/nfc9/manifest.json"), .{ .ignore_unknown_fields = true });
}
fn hex(bytes: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

test "Unicode-9 NFC all independent normative cases and long runs" {
    const m = try manifest();
    defer m.deinit();
    try t.expectEqualStrings(m.value.generator_sha256, &hex(@embedFile("reference/generate_nfc.py")));
    try t.expectEqualStrings(m.value.data_sha256, &nfc.table_sha256);
    const data = @embedFile("fixtures/nfc9/cases.bin");
    try t.expectEqualStrings(m.value.golden_sha256, &hex(data));
    var cursor: usize = 0;
    var records: usize = 0;
    var output: [4096]u8 = undefined;
    var scratch: [4096]u21 = undefined;
    while (cursor < data.len) {
        const a = std.mem.readInt(u32, data[cursor..][0..4], .little);
        const b = std.mem.readInt(u32, data[cursor + 4 ..][0..4], .little);
        cursor += 8;
        const input = data[cursor..][0..a];
        const expected = data[cursor + a ..][0..b];
        cursor += a + b;
        const n = try nfc.normalize(input, &output, &scratch);
        try t.expectEqualStrings(expected, output[0..n]);
        records += 1;
    }
    try t.expectEqual(m.value.records, records);
    for (m.value.workloads) |case| {
        const workspace = try t.allocator.alloc(u21, try nfc.scratchSize(case.text));
        defer t.allocator.free(workspace);
        const result = try t.allocator.alloc(u8, case.output.len);
        defer t.allocator.free(result);
        const n = try nfc.normalize(case.text, result, workspace);
        try t.expectEqualStrings(case.output, result[0..n]);
    }
}

test "NFC exhaustive scalar and combining-context fingerprints" {
    const m = try manifest();
    defer m.deinit();
    for ([_]bool{ false, true }) |context| {
        var digest = Sha256.init(.{});
        var input: [16]u8 = undefined;
        var output: [32]u8 = undefined;
        var scratch: [32]u21 = undefined;
        for (0..0x110000) |value| {
            if (value >= 0xd800 and value <= 0xdfff) continue;
            var len: usize = 0;
            if (context) {
                @memcpy(input[0..3], "[\xcc\x81");
                len = 3;
            }
            len += try std.unicode.utf8Encode(@intCast(value), input[len..]);
            if (context) {
                @memcpy(input[len..][0..2], "\xcc\xa3");
                len += 2;
            }
            const size = try nfc.normalize(input[0..len], &output, &scratch);
            var encoded: [4]u8 = undefined;
            std.mem.writeInt(u32, &encoded, @intCast(size), .little);
            digest.update(&encoded);
            digest.update(output[0..size]);
        }
        var hash: [32]u8 = undefined;
        digest.final(&hash);
        try t.expectEqualStrings(if (context) m.value.fingerprints.mark_context else m.value.fingerprints.scalar, &std.fmt.bytesToHex(hash, .lower));
    }
}

test "NFC strict UTF-8, bounded buffers and failure atomicity" {
    var output: [32]u8 = @splat(0xaa);
    var scratch: [32]u21 = undefined;
    for ([_][]const u8{ "\xff", "\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x80" }) |bad| {
        try t.expectError(error.InvalidUtf8, nfc.normalize(bad, &output, &scratch));
    }
    const input = "e\xcc\x81";
    try t.expectEqual(@as(usize, 4), try nfc.scratchSize(input));
    try t.expectError(error.InsufficientScratch, nfc.normalize(input, &output, scratch[0..3]));
    try t.expectError(error.InsufficientOutput, nfc.normalize(input, output[0..1], &scratch));
    try t.expectError(error.InsufficientOutput, nfc.normalize("ab", output[0..1], &scratch));
    for (output) |byte| try t.expectEqual(@as(u8, 0xaa), byte);
    try t.expectEqual(@as(usize, 0), try nfc.normalize("", &.{}, &.{}));
    try t.expectEqual(@as(usize, 2), try nfc.normalize(input, output[0..2], scratch[0..4]));
    try t.expectEqualStrings("é", output[0..2]);
    try t.expectEqual(@as(usize, 0), try nfc.scratchSize("ASCII"));
}
