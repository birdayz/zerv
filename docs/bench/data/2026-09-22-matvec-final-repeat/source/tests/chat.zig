const std = @import("std");
const t = std.testing;
const chat = @import("zerv").chat.qwen38;
const Golden = struct {
    generator_sha256: []const u8,
    cases: []const struct {
        name: []const u8,
        messages: []const chat.Message,
        options: chat.Options,
        official: struct { output: ?[]const u8 = null, @"error": ?[]const u8 = null },
    },
};

test "official Qwen3.8 template matches independent Jinja goldens" {
    const parsed = try std.json.parseFromSlice(Golden, t.allocator, @embedFile("fixtures/chat-template.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("reference/generate_chat_goldens.py"), &hash, .{});
    try t.expectEqualStrings(parsed.value.generator_sha256, &std.fmt.bytesToHex(hash, .lower));
    try t.expectEqual(@as(usize, 100), parsed.value.cases.len);
    var buffer: [8192]u8 = undefined;
    for (parsed.value.cases) |case| {
        var writer: std.Io.Writer = .fixed(&buffer);
        if (case.official.output) |expected| {
            try chat.render(case.messages, case.options, &writer);
            try t.expectEqualStrings(expected, writer.buffered());
        } else {
            const expected: anyerror = if (std.mem.eql(u8, case.name, "empty-messages")) error.EmptyMessages else if (std.mem.eql(u8, case.name, "misplaced-system")) error.MisplacedSystem else if (std.mem.eql(u8, case.name, "developer-prefix")) error.UnsupportedRole else error.MissingUserQuery;
            try t.expect(case.official.@"error" != null);
            try t.expectError(expected, chat.render(case.messages, case.options, &writer));
            try t.expectEqual(@as(usize, 0), writer.buffered().len);
        }
    }
}

test "chat rendering validation is bounded and failure-atomic" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    const messages = [_]chat.Message{.{ .role = .user, .content = "Hi" }};
    try t.expectError(error.LimitExceeded, chat.render(&messages, .{ .max_messages = 0 }, &writer));
    try t.expectError(error.LimitExceeded, chat.render(&messages, .{ .max_content_bytes = 1 }, &writer));
    try t.expectError(error.InvalidUtf8, chat.render(&.{.{ .role = .user, .content = "\xff" }}, .{}, &writer));
    try t.expectError(error.InvalidUtf8, chat.render(&.{ messages[0], .{ .role = .assistant, .content = "ok", .reasoning_content = "\xc0\x80" } }, .{}, &writer));
    try t.expectError(error.MisplacedSystem, chat.render(&.{ messages[0], .{ .role = .system, .content = "late" } }, .{}, &writer));
    try t.expectEqual(@as(usize, 0), writer.buffered().len);
    try chat.render(&messages, .{ .max_content_bytes = 2, .max_messages = 1 }, &writer);
    try t.expect(writer.buffered().len > 0);
    var tiny: [1]u8 = undefined;
    var small: std.Io.Writer = .fixed(&tiny);
    try t.expectError(error.WriteFailed, chat.render(&messages, .{}, &small));
}
