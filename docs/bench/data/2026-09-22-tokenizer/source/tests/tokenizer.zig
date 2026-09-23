const std = @import("std");
const t = std.testing;
const bpe = @import("zerv").tokenizer.bpe;
const Sha256 = std.crypto.hash.sha2.Sha256;
const data = @embedFile("fixtures/tokenizer/model.bin");
const Manifest = struct {
    generator_sha256: []const u8,
    oracle_adapter_sha256: []const u8,
    model_data_sha256: []const u8,
    raw_pieces_sha256: []const u8,
    raw_piece_count: u32,
    cases: []const struct { text: []const u8, ids: []const u32, raw_hex: []const u8 },
    decode_cases: []const struct { ids: []const u32, raw_hex: []const u8 },
};
fn hex(bytes: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}
fn integer(at: usize) u32 {
    return std.mem.readInt(u32, data[at..][0..4], .little);
}
fn load(allocator: std.mem.Allocator) !bpe.Tokenizer {
    const entries = try allocator.alloc(bpe.Entry, integer(4));
    defer allocator.free(entries);
    const merges = try allocator.alloc(bpe.Merge, integer(8));
    defer allocator.free(merges);
    var at: usize = 12;
    for (entries, 0..) |*entry, id| {
        const len = integer(at);
        entry.* = .{ .id = @intCast(id), .kind = std.enums.fromInt(bpe.Kind, data[at + 4]) orelse return error.InvalidFixture, .bytes = data[at + 8 ..][0..len] };
        at += 8 + len;
    }
    for (merges) |*merge| {
        merge.* = .{ .left = integer(at), .right = integer(at + 4), .result = integer(at + 8) };
        at += 12;
    }
    try t.expectEqual(data.len, at);
    return bpe.Tokenizer.init(allocator, entries, merges, .{});
}
fn checkDecode(tokenizer: *const bpe.Tokenizer, ids: []const u32, raw_hex: []const u8) !void {
    const expected = try t.allocator.alloc(u8, raw_hex.len / 2);
    defer t.allocator.free(expected);
    _ = try std.fmt.hexToBytes(expected, raw_hex);
    const actual = try t.allocator.alloc(u8, expected.len);
    defer t.allocator.free(actual);
    var writer: std.Io.Writer = .fixed(actual);
    try tokenizer.decode(ids, &writer);
    try t.expectEqualSlices(u8, expected, writer.buffered());
}

test "Qwen full vocabulary raw-piece oracle and official end-to-end token IDs" {
    const parsed = try std.json.parseFromSlice(Manifest, t.allocator, @embedFile("fixtures/tokenizer/manifest.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    try t.expectEqualStrings(m.generator_sha256, &hex(@embedFile("reference/generate_tokenizer_goldens.py")));
    try t.expectEqualStrings(m.oracle_adapter_sha256, &hex(@embedFile("reference/tokenizer_pieces.c")));
    try t.expectEqualStrings(m.model_data_sha256, &hex(data));
    var tokenizer = try load(t.allocator);
    defer tokenizer.deinit();
    var hash = Sha256.init(.{});
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, m.raw_piece_count, .little);
    hash.update(&bytes);
    for (0..m.raw_piece_count) |id| {
        const piece = try tokenizer.piece(@intCast(id));
        std.mem.writeInt(u32, &bytes, @intCast(piece.len), .little);
        hash.update(&bytes);
        hash.update(piece);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    try t.expectEqualStrings(m.raw_pieces_sha256, &std.fmt.bytesToHex(digest, .lower));
    for (m.cases, 0..) |case, i| {
        const ids = try tokenizer.encode(t.allocator, case.text, .{});
        defer t.allocator.free(ids);
        errdefer std.debug.print("encode case {d}: {s}\n", .{ i, case.text });
        try t.expectEqualSlices(u32, case.ids, ids);
        try checkDecode(&tokenizer, ids, case.raw_hex);
    }
    for (m.decode_cases) |case| try checkDecode(&tokenizer, case.ids, case.raw_hex);
}

fn roots(bytes: *[256]u8) [256]bpe.Entry {
    var result: [256]bpe.Entry = undefined;
    for (&result, 0..) |*entry, id| {
        bytes[id] = @intCast(id);
        entry.* = .{ .id = @intCast(id), .kind = .normal, .bytes = bytes[id..][0..1] };
    }
    return result;
}
fn allocationScenario(allocator: std.mem.Allocator) !void {
    var bytes: [256]u8 = undefined;
    const entries = roots(&bytes) ++ [_]bpe.Entry{
        .{ .id = 256, .kind = .normal, .bytes = "aa" },
        .{ .id = 257, .kind = .normal, .bytes = "aaaa" },
        .{ .id = 258, .kind = .added, .bytes = "<x>" },
        .{ .id = 259, .kind = .added, .bytes = "<x>x" },
        .{ .id = 261, .kind = .unused, .bytes = "" },
    };
    const merges = [_]bpe.Merge{
        .{ .left = 'a', .right = 'a', .result = 256 },
        .{ .left = 256, .right = 256, .result = 257 },
    };
    var tokenizer = try bpe.Tokenizer.init(allocator, &entries, &merges, .{});
    defer tokenizer.deinit();
    @memset(&bytes, 0); // Table ownership is independent of its inputs.
    const ids = try tokenizer.encode(allocator, "aaaa e\u{301} <x>x", .{});
    defer allocator.free(ids);
    var buffer: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try tokenizer.decode(ids, &writer);
    try t.expectEqualStrings("aaaa é <x>x", writer.buffered());
    try t.expectEqual(@as(u32, 257), ids[0]);
    try t.expectEqual(@as(u32, 259), ids[ids.len - 1]);
    try t.expectEqualStrings("", try tokenizer.piece(261));
    try t.expectError(error.InvalidTokenId, tokenizer.piece(260));
}

test "BPE allocation failure cleanup, copied ownership and request scratch" {
    try t.checkAllAllocationFailures(t.allocator, allocationScenario, .{});
}

test "BPE malformed tables, input/normalized/token limits and error-atomic decode" {
    var bytes: [256]u8 = undefined;
    var entries = roots(&bytes);
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, entries[0..255], &.{}, .{}));
    try t.expectError(error.LimitExceeded, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{ .max_vocabulary = 255 }));
    try t.expectError(error.LimitExceeded, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{ .max_total_bytes = 255 }));
    try t.expectError(error.InvalidMerge, bpe.Tokenizer.init(t.allocator, &entries, &.{.{ .left = 'a', .right = 'b', .result = 'c' }}, .{}));
    try t.expectError(error.InvalidMerge, bpe.Tokenizer.init(t.allocator, &entries, &.{.{ .left = 'a', .right = 999, .result = 'c' }}, .{}));
    entries[1].id = 0;
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{}));
    entries = roots(&bytes);
    var tokenizer = try bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{});
    defer tokenizer.deinit();
    try t.expectError(error.InvalidUtf8, tokenizer.encode(t.allocator, "\xff", .{}));
    try t.expectError(error.InputTooLarge, tokenizer.encode(t.allocator, "ab", .{ .max_input_bytes = 1 }));
    try t.expectError(error.NormalizedTooLarge, tokenizer.encode(t.allocator, "é", .{ .max_normalized_bytes = 1 }));
    try t.expectError(error.TooManyTokens, tokenizer.encode(t.allocator, "ab", .{ .max_tokens = 1 }));
    const empty = try tokenizer.encode(t.allocator, "", .{ .max_input_bytes = 0, .max_normalized_bytes = 0, .max_tokens = 0 });
    defer t.allocator.free(empty);
    try t.expectEqual(@as(usize, 0), empty.len);
    var buffer: [8]u8 = @splat(0xaa);
    var writer: std.Io.Writer = .fixed(&buffer);
    try t.expectError(error.InvalidTokenId, tokenizer.decode(&.{ 'a', 256 }, &writer));
    try t.expectEqual(@as(usize, 0), writer.end);
    for (buffer) |b| try t.expectEqual(@as(u8, 0xaa), b);
    try t.expectError(error.InvalidTokenId, tokenizer.piece(256));
    var too_small: std.Io.Writer = .fixed(buffer[0..0]);
    try t.expectError(error.WriteFailed, tokenizer.decode(&.{'a'}, &too_small));
}

test "BPE rejects duplicate roots/additions/pairs and enforces limits across spans" {
    var bytes: [256]u8 = undefined;
    const base = roots(&bytes);
    var entries = base ++ [_]bpe.Entry{.{ .id = 256, .kind = .normal, .bytes = "aa" }};
    const rule: bpe.Merge = .{ .left = 'a', .right = 'a', .result = 256 };
    try t.expectError(error.InvalidMerge, bpe.Tokenizer.init(t.allocator, &entries, &.{ rule, rule }, .{}));
    try t.expectError(error.LimitExceeded, bpe.Tokenizer.init(t.allocator, &entries, &.{rule}, .{ .max_merges = 0 }));
    entries[256].bytes = "a";
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{}));
    entries[256] = .{ .id = 256, .kind = .unused, .bytes = "x" };
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{}));
    entries[256] = .{ .id = 256, .kind = .added, .bytes = "\xff" };
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{}));
    entries[256].bytes = "";
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{}));
    entries[256].bytes = "<x>";
    try t.expectError(error.LimitExceeded, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{ .max_added = 0 }));
    try t.expectError(error.LimitExceeded, bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{ .max_piece_bytes = 2 }));
    const duplicates = entries ++ [_]bpe.Entry{.{ .id = 257, .kind = .added, .bytes = "<x>" }};
    try t.expectError(error.InvalidVocabulary, bpe.Tokenizer.init(t.allocator, &duplicates, &.{}, .{}));
    var tokenizer = try bpe.Tokenizer.init(t.allocator, &entries, &.{}, .{});
    defer tokenizer.deinit();
    try t.expectError(error.NormalizedTooLarge, tokenizer.encode(t.allocator, "a<x>b", .{ .max_normalized_bytes = 4 }));
    try t.expectError(error.TooManyTokens, tokenizer.encode(t.allocator, "<x><x>", .{ .max_tokens = 1 }));
    const ids = try tokenizer.encode(t.allocator, "a<x>b", .{ .max_normalized_bytes = 5, .max_tokens = 3 });
    defer t.allocator.free(ids);
    try t.expectEqualSlices(u32, &.{ 'a', 256, 'b' }, ids);
    const artifact = @import("zerv").artifact;
    var container = try artifact.gguf.Container.parse(t.allocator, @embedFile("fixtures/gguf/default.gguf"), .{});
    defer container.deinit();
    try t.expectError(error.UnsupportedProfile, @import("zerv").tokenizer.fromGGUF(t.allocator, &container, .{}));
}
