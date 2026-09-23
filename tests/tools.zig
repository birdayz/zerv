//! Tool calling (docs/specs/tool-calling.md): prompt rendering against the independent
//! Jinja oracle, Python json.dumps numbers, the output call parser, and the
//! between-call sampling constraint.
const std = @import("std");
const zerv = @import("zerv");
const api = zerv.serve.api;
const chat = zerv.chat;
const session = zerv.session;
const text = session.text;
const tools = session.tools;
const t = std.testing;

const ids = [_][]const u8{"qwen3.8-27b"};

test "tool prompts match the independent Jinja oracle through the request parser" {
    var arena_state: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fixture = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/chat-tools.json"), .{ .parse_numbers = false });
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("reference/render_tools.py"), &hash, .{});
    try t.expectEqualStrings(fixture.object.get("generator_sha256").?.string, &std.fmt.bytesToHex(hash, .lower));
    const cases = fixture.object.get("cases").?.array.items;
    try t.expectEqual(@as(usize, 27), cases.len);
    var rendered: usize = 0;
    for (cases) |case| {
        const c = case.object;
        // The request body: model, messages, tools, then the case's request options.
        var body: std.json.ObjectMap = .empty;
        try body.put(a, "model", .{ .string = "qwen3.8-27b" });
        try body.put(a, "messages", c.get("messages").?);
        try body.put(a, "tools", c.get("tools").?);
        var options = c.get("options").?.object.iterator();
        while (options.next()) |o| try body.put(a, o.key_ptr.*, o.value_ptr.*);
        const json = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = body }, .{});
        const expected = c.get("expected").?.object;
        const parsed = try api.parseChat(a, json, &ids, .{}, .{});
        if (expected.get("error")) |_| {
            try t.expect(parsed == .err);
            continue;
        }
        const request = switch (parsed) {
            .ok => |r| r,
            .err => |e| {
                std.debug.print("{s}: rejected: {s}\n", .{ c.get("name").?.string, e.message });
                return error.TestUnexpectedResult;
            },
        };
        var out: std.Io.Writer.Allocating = .init(a);
        try chat.qwen38.render(request.messages, request.template, &out.writer);
        t.expectEqualStrings(expected.get("output").?.string, out.written()) catch |e| {
            std.debug.print("case {s}\n", .{c.get("name").?.string});
            return e;
        };
        rendered += 1;
    }
    try t.expectEqual(@as(usize, 26), rendered);
}

test "JSON numbers are printed as Python's json.dumps prints them" {
    var arena_state: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fixture = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/chat-tools.json"), .{});
    const numbers = fixture.object.get("numbers").?.array.items;
    try t.expect(numbers.len > 800);
    var buf: [128]u8 = undefined;
    for (numbers) |n| {
        var w: std.Io.Writer = .fixed(&buf);
        try chat.pyjson.writeNumber(&w, n.object.get("literal").?.string);
        t.expectEqualStrings(n.object.get("python").?.string, w.buffered()) catch |e| {
            std.debug.print("literal {s}\n", .{n.object.get("literal").?.string});
            return e;
        };
    }
    var w: std.Io.Writer = .fixed(&buf);
    try t.expectError(error.InvalidNumber, chat.pyjson.writeNumber(&w, "01"));
    try t.expectError(error.InvalidNumber, chat.pyjson.writeNumber(&w, "1."));
    try chat.pyjson.writeString(&w, "a\"\\\n\r\t\x08\x0c\x01\x1f\x7f\u{2028}é");
    try t.expectEqualStrings("\"a\\\"\\\\\\n\\r\\t\\b\\f\\u0001\\u001f\x7f\u{2028}é\"", w.buffered());
}

// ---- output parser ----

const Call = struct { name: []const u8, arguments: std.ArrayList(u8) = .empty };
const Collect = struct {
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,
    events: usize = 0,
    pub fn emit(self: *Collect, event: text.Event) !void {
        self.events += 1;
        switch (event) {
            .reasoning => |b| try self.reasoning.appendSlice(t.allocator, b),
            .content => |b| try self.content.appendSlice(t.allocator, b),
            .call_begin => |c| {
                try t.expectEqual(self.calls.items.len, c.index);
                var call: Call = .{ .name = try t.allocator.dupe(u8, c.name) };
                try call.arguments.append(t.allocator, '{'); // implied by call_begin
                try self.calls.append(t.allocator, call);
            },
            .call_arguments => |c| {
                try t.expect(c.text.len > 0);
                try t.expectEqual(self.calls.items.len - 1, c.index);
                try self.calls.items[c.index].arguments.appendSlice(t.allocator, c.text);
            },
        }
    }
    fn deinit(self: *Collect) void {
        self.reasoning.deinit(t.allocator);
        self.content.deinit(t.allocator);
        for (self.calls.items) |*c| {
            t.allocator.free(c.name);
            c.arguments.deinit(t.allocator);
        }
        self.calls.deinit(t.allocator);
    }
};

const weather = [_]tools.Tool{.{ .name = "get_weather", .params = &.{
    .{ .name = "city", .types = .{ .string = true } },
    .{ .name = "days", .types = .{ .integer = true } },
    .{ .name = "hourly", .types = .{ .boolean = true } },
    .{ .name = "ratio", .types = .{ .number = true, .integer = true } },
    .{ .name = "tags", .types = .{ .array = true, .null = true } },
    .{ .name = "note", .types = .{ .string = true, .integer = true } },
} }};

const Outcome = struct { collect: Collect, count: u32, done: bool, failed: bool, after_call: bool };

fn parse(thinking: bool, parallel: bool, chunks: []const []const u8) !Outcome {
    const value_buf = try t.allocator.alloc(u8, tools.max_value);
    defer t.allocator.free(value_buf);
    var calls = tools.Calls.init(.{ .tools = &weather, .parallel = parallel }, value_buf);
    var splitter = text.Splitter.init(thinking, &calls);
    var collect: Collect = .{};
    errdefer collect.deinit();
    var out: [512]u8 = undefined;
    for (chunks) |chunk| {
        if (splitter.done()) break;
        try splitter.push(chunk, &out, &collect);
    }
    const after_call = splitter.afterCall() != null;
    try splitter.finish(&collect);
    return .{ .collect = collect, .count = calls.count, .done = splitter.done(), .failed = calls.failed, .after_call = after_call };
}

/// One-shot parse equals the parse of every split into three chunks.
fn expectChunkingInvariant(thinking: bool, stream: []const u8) !void {
    var one = try parse(thinking, true, &.{stream});
    defer one.collect.deinit();
    var a: usize = 0;
    while (a < stream.len) : (a += 1) {
        var b = a;
        while (b < stream.len) : (b += 3) {
            var got = try parse(thinking, true, &.{ stream[0..a], stream[a..b], stream[b..] });
            defer got.collect.deinit();
            try t.expectEqualStrings(one.collect.reasoning.items, got.collect.reasoning.items);
            try t.expectEqualStrings(one.collect.content.items, got.collect.content.items);
            try t.expectEqual(one.collect.calls.items.len, got.collect.calls.items.len);
            for (one.collect.calls.items, got.collect.calls.items) |x, y| {
                try t.expectEqualStrings(x.name, y.name);
                try t.expectEqualStrings(x.arguments.items, y.arguments.items);
            }
        }
    }
}

test "tool calls: reference shapes, typing, escaping and parallel calls" {
    // Observed llama-server outputs (docs/research/tool-calling.md).
    {
        var r = try parse(true, true, &.{"The user wants the date.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\ndate\n</parameter>\n</function>\n</tool_call>"});
        defer r.collect.deinit();
        try t.expectEqualStrings("The user wants the date.\n", r.collect.reasoning.items);
        try t.expectEqualStrings("", r.collect.content.items);
        try t.expectEqual(@as(usize, 1), r.collect.calls.items.len);
        try t.expectEqualStrings("bash", r.collect.calls.items[0].name);
        try t.expectEqualStrings("{\"command\":\"date\"}", r.collect.calls.items[0].arguments.items);
        try t.expect(r.after_call and !r.failed);
    }
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=hourly>\ntrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTo\"k\\yo\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=ratio>\n2.5\n</parameter>\n<parameter=note>\n42\n</parameter>\n</function>\n</tool_call>"});
        defer r.collect.deinit();
        try t.expectEqual(@as(usize, 2), r.collect.calls.items.len);
        try t.expectEqualStrings("{\"city\":\"Paris\",\"days\":3,\"hourly\":true}", r.collect.calls.items[0].arguments.items);
        try t.expectEqualStrings("{\"city\":\"To\\\"k\\\\yo\",\"tags\":[\"a\", \"b\"],\"ratio\":2.5,\"note\":42}", r.collect.calls.items[1].arguments.items);
    }
    // Content before a call keeps its trailing whitespace; multi-line string values.
    {
        var r = try parse(false, true, &.{"Let me write it.\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nline 1\n\tline 2\n\n</parameter>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"});
        defer r.collect.deinit();
        try t.expectEqualStrings("Let me write it.\n\n", r.collect.content.items);
        try t.expectEqualStrings("write_file", r.collect.calls.items[0].name);
        // Unknown tool: every parameter may be JSON; these are not, so they are strings.
        try t.expectEqualStrings("{\"content\":\"line 1\\n\\tline 2\\n\",\"path\":\"a.py\"}", r.collect.calls.items[0].arguments.items);
    }
    // Values that do not fit the declared kinds fall back to strings; JSON string
    // literals are never taken as JSON; integers reject fractions.
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=get_weather>\n<parameter=days>\n3 days\n</parameter>\n<parameter=note>\n\"quoted\"\n</parameter>\n<parameter=days>\n2.0\n</parameter>\n<parameter=tags>\n null \n</parameter>\n</function>\n</tool_call>"});
        defer r.collect.deinit();
        try t.expectEqualStrings("{\"days\":\"3 days\",\"note\":\"\\\"quoted\\\"\",\"days\":\"2.0\",\"tags\":null}", r.collect.calls.items[0].arguments.items);
    }
    // Every chunking of these streams gives the same result.
    try expectChunkingInvariant(true, "think <b>\n</think>\n\nOK \n<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</param\n</parameter>\n<parameter=days>\n 7 \n</parameter>\n</function>\n</tool_call>\n\n  <tool_call>\n<function=x>\n</function>\n</tool_call>");
    try expectChunkingInvariant(false, "<tool_call>\n<function=get_weather>\n<parameter=city>\n\n\n</parameter>\n</parameter>\n</function>\n</tool_call>");
}

test "tool calls: incomplete, malformed and non-parallel streams" {
    // Length cut inside a value: the value is closed, the object is not (llama-server
    // reports `{"path":"hello.py"` the same way); a partial terminator is dropped.
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=get_weather>\n<parameter=city>\nPar"});
        defer r.collect.deinit();
        try t.expectEqualStrings("{\"city\":\"Par\"", r.collect.calls.items[0].arguments.items);
        try t.expect(!r.failed);
    }
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=write_file>\n<parameter=path>\nhello.py\n</par"});
        defer r.collect.deinit();
        try t.expectEqualStrings("{\"path\":\"hello.py\"", r.collect.calls.items[0].arguments.items);
    }
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=get_weather>\n<parameter=days>\n1"});
        defer r.collect.deinit();
        try t.expectEqualStrings("{\"days\":1", r.collect.calls.items[0].arguments.items);
    }
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=write_file>\n<parameter=path>\nhello.py\n</parameter>\n"});
        defer r.collect.deinit();
        try t.expectEqualStrings("{\"path\":\"hello.py\"", r.collect.calls.items[0].arguments.items);
    }
    // Cut before NAME is complete: no call, nothing reported.
    {
        var r = try parse(false, true, &.{"Sure.<tool_call>\n<function=get_wea"});
        defer r.collect.deinit();
        try t.expectEqual(@as(usize, 0), r.collect.calls.items.len);
        try t.expectEqualStrings("Sure.", r.collect.content.items);
    }
    // Malformed before NAME: raw text becomes content, generation stops.
    {
        var r = try parse(false, true, &.{ "<tool_call>\n{\"name\": \"x\"}", " more" });
        defer r.collect.deinit();
        try t.expectEqualStrings("<tool_call>\n{\"name\": \"x\"}", r.collect.content.items);
        try t.expect(r.done and r.failed and r.count == 0);
    }
    // Malformed after NAME: the call is closed as incomplete, generation stops.
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<oops>"});
        defer r.collect.deinit();
        try t.expectEqualStrings("{\"city\":\"Paris\"", r.collect.calls.items[0].arguments.items);
        try t.expect(r.done and r.failed and r.count == 1);
    }
    // Text after a call ends generation (the grammar allows only whitespace, a call or
    // the end); without parallel calls a second call also ends it.
    {
        var r = try parse(false, true, &.{"<tool_call>\n<function=a>\n</function>\n</tool_call>\n Done."});
        defer r.collect.deinit();
        try t.expect(r.done and !r.failed and r.count == 1);
        try t.expectEqualStrings("{}", r.collect.calls.items[0].arguments.items);
    }
    {
        var r = try parse(false, false, &.{"<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>"});
        defer r.collect.deinit();
        try t.expect(r.done and r.count == 1);
    }
    // Without tool mode `<tool_call>` stays literal content (tool_choice none).
    {
        var collect: Collect = .{};
        defer collect.deinit();
        var splitter = text.Splitter.init(false, null);
        var out: [128]u8 = undefined;
        try splitter.push("<tool_call>\n<function=a>", &out, &collect);
        try splitter.finish(&collect);
        try t.expectEqualStrings("<tool_call>\n<function=a>", collect.content.items);
    }
}

test "SPACE_RULE prefixes" {
    for ([_][]const u8{ "", " ", "\n", "\n\n", "\n \t", "\n\n" ++ " " ** 20 }) |s| try t.expect(tools.spaceRuleExtends("", s));
    for ([_][]const u8{ "  ", " \n", "\n\n\n", "\n \n", "\n" ++ " " ** 21, "x" }) |s| try t.expect(!tools.spaceRuleExtends("", s));
    try t.expect(tools.spaceRuleExtends("\n", "\n  "));
    try t.expect(!tools.spaceRuleExtends(" ", "\n"));
}

// ---- constrained sampling after a call ----

// 0 EOS, 1 "Hi", 2 "\n", 3 "<tool_call>", 4 "\n<function=a>\n</function>\n</tool_call>", 5 "  ", 6 " "
const pieces = [_][]const u8{ "<|im_end|>", "Hi", "\n", "<tool_call>", "\n<function=a>\n</function>\n</tool_call>", "  ", " " };
const Scripted = struct {
    /// Logits favour these tokens in order, one per generated step (after the prompt).
    favourite: []const u32,
    /// And these get the second-highest logit.
    second: []const u32,
    at: usize = 0,
    logits: [pieces.len]f32 = undefined,
    pub fn context(_: *Scripted) u32 {
        return 64;
    }
    pub fn vocab(_: *Scripted) usize {
        return pieces.len;
    }
    pub fn reset(self: *Scripted) !void {
        self.at = 0;
    }
    pub fn prefill(self: *Scripted, _: []const u32) ![]const f32 {
        return self.next();
    }
    pub fn step(self: *Scripted, _: u32) ![]const f32 {
        self.at += 1;
        return self.next();
    }
    fn next(self: *Scripted) []const f32 {
        @memset(&self.logits, 0);
        const i = @min(self.at, self.favourite.len - 1);
        self.logits[self.second[i]] = 5;
        self.logits[self.favourite[i]] = 10;
        return &self.logits;
    }
};
const Tok = struct {
    pub fn outputPiece(_: Tok, id: u32) ![]const u8 {
        return pieces[id];
    }
};
const Gen = session.Generation(*Scripted, Tok, *Collect);
const tool_special: session.Special = .{ .eos = &.{0}, .whitespace = &.{ 2, 5, 6 }, .tool_call = 3 };

test "after a call only whitespace, EOS or another call can be sampled" {
    // The model prefers "Hi" right after the call; the constraint takes the best
    // allowed token instead ("\n", then EOS).
    {
        var backend: Scripted = .{ .favourite = &.{ 3, 4, 1, 1, 0 }, .second = &.{ 0, 0, 2, 0, 0 } };
        var collect: Collect = .{};
        defer collect.deinit();
        const r = try Gen.run(t.io, t.allocator, &backend, Tok{}, tool_special, .{ .prompt = &.{1}, .max_tokens = 16, .tools = .{ .tools = &.{} } }, &collect);
        try t.expectEqual(session.Finish.stop, r.finish);
        try t.expectEqual(@as(u32, 1), r.tool_calls);
        try t.expectEqual(@as(u32, 4), r.completion_tokens); // call (2 tokens), "\n", EOS
        try t.expectEqualStrings("", collect.content.items);
    }
    // "  " would break SPACE_RULE after "\n"? No: "\n" + "  " is valid; " " first then
    // "\n" is not, so after " " the best allowed token is EOS.
    {
        var backend: Scripted = .{ .favourite = &.{ 3, 4, 6, 2, 0 }, .second = &.{ 0, 0, 0, 1, 0 } };
        var collect: Collect = .{};
        defer collect.deinit();
        const r = try Gen.run(t.io, t.allocator, &backend, Tok{}, tool_special, .{ .prompt = &.{1}, .max_tokens = 16, .tools = .{ .tools = &.{} } }, &collect);
        try t.expectEqual(@as(u32, 4), r.completion_tokens); // call, " ", EOS
    }
    // Parallel calls: `<tool_call>` is allowed after a call; without them it is not.
    {
        var backend: Scripted = .{ .favourite = &.{ 3, 4, 2, 3, 4, 0 }, .second = &.{ 0, 0, 0, 0, 0, 0 } };
        var collect: Collect = .{};
        defer collect.deinit();
        const r = try Gen.run(t.io, t.allocator, &backend, Tok{}, tool_special, .{ .prompt = &.{1}, .max_tokens = 16, .tools = .{ .tools = &.{} } }, &collect);
        try t.expectEqual(@as(u32, 2), r.tool_calls);
    }
    {
        var backend: Scripted = .{ .favourite = &.{ 3, 4, 2, 3, 4, 0 }, .second = &.{ 0, 0, 0, 0, 0, 0 } };
        var collect: Collect = .{};
        defer collect.deinit();
        const r = try Gen.run(t.io, t.allocator, &backend, Tok{}, tool_special, .{ .prompt = &.{1}, .max_tokens = 16, .tools = .{ .tools = &.{}, .parallel = false } }, &collect);
        try t.expectEqual(@as(u32, 1), r.tool_calls);
        try t.expectEqual(@as(u32, 4), r.completion_tokens); // call, "\n", EOS
    }
}
