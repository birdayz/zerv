const std = @import("std");
const serve = @import("serve");
const api = serve.api;
const chat_mod = @import("chat").qwen38;
const http = serve.http;
const t = std.testing;

const ids = [_][]const u8{"qwen3.8-27b"};

fn parse(arena: std.mem.Allocator, body: []const u8) !api.Parsed {
    return api.parseChat(arena, body, &ids, .{}, .{});
}

fn expectError(body: []const u8, status: std.http.Status, param: ?[]const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena_state.deinit();
    const result = try parse(arena_state.allocator(), body);
    switch (result) {
        .ok => return error.ExpectedRejection,
        .err => |e| {
            try t.expectEqual(status, e.status);
            if (param) |p| try t.expectEqualStrings(p, e.param.?);
        },
    }
}

test "chat request parsing: fields, defaults, content parts and template controls" {
    var arena_state: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const r = (try parse(a,
        \\{"model":"qwen3.8-27b","messages":[{"role":"system","content":"S"},
        \\ {"role":"user","content":[{"type":"text","text":"Hel"},{"type":"text","text":"lo"}]},
        \\ {"role":"assistant","content":null,"reasoning_content":"r"},{"role":"user","content":"x"}],
        \\ "stream":true,"stream_options":{"include_usage":true},"max_tokens":50,"max_completion_tokens":40,
        \\ "temperature":0,"top_p":0.5,"top_k":3,"min_p":0.1,"presence_penalty":1.0,"seed":-1,
        \\ "stop":["a","b"],"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"low",
        \\ "n":1,"logprobs":false,"tools":[],"tool_choice":"none","response_format":{"type":"text"},"user":"u"}
    )).ok;
    try t.expectEqual(@as(usize, 4), r.messages.len);
    try t.expectEqualStrings("Hello", r.messages[1].content);
    try t.expectEqualStrings("", r.messages[2].content);
    try t.expectEqualStrings("r", r.messages[2].reasoning_content);
    try t.expect(r.stream and r.include_usage and r.seed_given);
    try t.expectEqual(@as(?u32, 40), r.max_tokens);
    try t.expectEqual(@as(f32, 0), r.params.temperature);
    try t.expectEqual(@as(u32, 3), r.params.top_k);
    try t.expectEqual(std.math.maxInt(u64), r.params.seed);
    try t.expect(!r.template.enable_thinking);
    try t.expectEqual(.low, r.template.reasoning_effort);
    try t.expectEqual(@as(usize, 2), r.stops.len);
    const d = (try parse(a, "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stop\":\"x\",\"reasoning_effort\":\"high\"}")).ok;
    try t.expectEqual(@as(f32, 1.0), d.params.temperature);
    try t.expectEqual(@as(u32, 20), d.params.top_k);
    try t.expect(d.template.enable_thinking and !d.stream and d.max_tokens == null);
    try t.expectEqual(.xhigh, d.template.reasoning_effort);
    try t.expectEqual(@as(usize, 1), d.stops.len);
    const tr = (try parse(a,
        \\{"model":"qwen3.8-27b","tool_choice":"auto","parallel_tool_calls":false,
        \\ "tools":[{"type":"function","function":{"name":"f","description":"d","strict":true,"parameters":{"type":"object",
        \\   "properties":{"s":{"type":"string"},"n":{"type":["integer","null"]},"e":{"enum":["a",1.5]},"r":{"$ref":"#/$defs/x"}},
        \\   "$defs":{"x":{"type":"boolean"}}}}}],
        \\ "messages":[{"role":"user","content":"q"},
        \\ {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"s\": \"x\", \"n\": 1.0}"}}]},
        \\ {"role":"tool","tool_call_id":"c1","content":null}]}
    )).ok;
    try t.expectEqual(@as(usize, 1), tr.template.tools.len);
    // Normalized as llama-server does (`strict` dropped), rendered as Python json.dumps.
    try t.expect(std.mem.startsWith(u8, tr.template.tools[0], "{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"d\", \"parameters\": {\"type\": \"object\""));
    try t.expect(std.mem.indexOf(u8, tr.template.tools[0], "strict") == null);
    const params = tr.tools[0].params;
    try t.expectEqual(@as(usize, 4), params.len);
    try t.expect(params[0].types.onlyString());
    try t.expect(params[1].types.integer and params[1].types.null and !params[1].types.string);
    try t.expect(params[2].types.string and params[2].types.number and !params[2].types.integer);
    try t.expect(params[3].types.boolean and !params[3].types.string);
    const mode = tr.toolMode().?;
    try t.expect(!mode.parallel);
    try t.expectEqual(chat_mod.Role.tool, tr.messages[2].role);
    try t.expectEqualStrings("x", tr.messages[1].tool_calls[0].arguments[0].value);
    try t.expectEqualStrings("1.0", tr.messages[1].tool_calls[0].arguments[1].value);
}

test "chat request rejections are explicit OpenAI errors" {
    const ok_messages = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
    try expectError("not json", .bad_request, null);
    try expectError("[]", .bad_request, null);
    try expectError("{" ++ ok_messages ++ "}", .bad_request, "model");
    try expectError("{\"model\":\"gpt-4o\"," ++ ok_messages ++ "}", .not_found, "model");
    try expectError("{\"model\":\"qwen3.8-27b\"}", .bad_request, "messages");
    try expectError("{\"model\":\"qwen3.8-27b\",\"messages\":[]}", .bad_request, "messages");
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "\"n\":2", "n" },
        .{ "\"logprobs\":true", "logprobs" },
        .{ "\"top_logprobs\":2", "top_logprobs" },
        .{ "\"tools\":[{\"type\":\"function\"}]", "tools" },
        .{ "\"tools\":{}", "tools" },
        .{ "\"tools\":[{\"type\":\"code\",\"function\":{\"name\":\"f\"}}]", "tools" },
        .{ "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"\"}}]", "tools" },
        .{ "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"description\":1}}]", "tools" },
        .{ "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":[]}}]", "tools" },
        .{ "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"properties\":{\"a\":{\"type\":\"text\"}}}}}]", "tools" },
        .{ "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"properties\":{\"a\":{\"$ref\":\"http://x\"}}}}}]", "tools" },
        .{ "\"tool_choice\":\"required\"", "tool_choice" },
        .{ "\"tool_choice\":\"sometimes\"", "tool_choice" },
        .{ "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}", "tool_choice" },
        .{ "\"parallel_tool_calls\":1", "parallel_tool_calls" },
        .{ "\"functions\":[]", "functions" },
        .{ "\"logit_bias\":{\"1\":5}", "logit_bias" },
        .{ "\"response_format\":{\"type\":\"json_object\"}", "response_format" },
        .{ "\"temperature\":3", "temperature" },
        .{ "\"temperature\":\"hot\"", "temperature" },
        .{ "\"top_p\":0", "top_p" },
        .{ "\"top_k\":-1", "top_k" },
        .{ "\"presence_penalty\":2.5", "presence_penalty" },
        .{ "\"seed\":1.5", "seed" },
        .{ "\"max_tokens\":0", "max_tokens" },
        .{ "\"stop\":[\"a\",\"b\",\"c\",\"d\",\"e\"]", "stop" },
        .{ "\"stop\":[1]", "stop" },
        .{ "\"stream\":1", "stream" },
        .{ "\"reasoning_effort\":\"minimal\"", "reasoning_effort" },
        .{ "\"chat_template_kwargs\":{\"unknown\":true}", "chat_template_kwargs" },
    };
    for (cases) |c| {
        var buf: [512]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf, "{{\"model\":\"qwen3.8-27b\",{s},{s}}}", .{ ok_messages, c[0] });
        try expectError(body, .bad_request, c[1]);
    }
    for ([_][]const u8{
        "[{\"role\":\"function\",\"content\":\"x\"}]",
        "[{\"role\":\"tool\",\"content\":\"x\",\"tool_call_id\":5}]",
        "[{\"role\":\"user\",\"content\":\"x\",\"tool_calls\":[]}]",
        "[{\"role\":\"assistant\",\"content\":\"x\",\"tool_calls\":{}}]",
        "[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}]",
        "[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"[1]\"}}]}]",
        "[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1,\\\"a\\\":2}\"}}]}]",
        "[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{bad\"}}]}]",
        "[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":7}}]}]",
        "[{\"role\":\"wizard\",\"content\":\"x\"}]",
        "[{\"role\":\"user\"}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\"}]}]",
        "[{\"role\":\"user\",\"content\":\"x\",\"reasoning_content\":\"y\"}]",
        "[{\"role\":\"assistant\",\"content\":\"x\",\"tool_calls\":[{}]}]",
        "[\"hi\"]",
    }) |messages| {
        var buf: [512]u8 = undefined;
        try expectError(try std.fmt.bufPrint(&buf, "{{\"model\":\"qwen3.8-27b\",\"messages\":{s}}}", .{messages}), .bad_request, "messages");
    }
    try expectError("{\"model\":\"qwen3.8-27b\",\"model\":\"x\"," ++ ok_messages ++ "}", .bad_request, null);
    var arena_state: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena_state.deinit();
    const big = try arena_state.allocator().alloc(u8, 5 * 1024 * 1024);
    @memset(big, ' ');
    try t.expectEqual(std.http.Status.payload_too_large, (try parse(arena_state.allocator(), big)).err.status);
}

test "response encoding: completion, chunks, usage, errors and models" {
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    const meta: api.Meta = .{ .id = "chatcmpl-1", .created = 7, .model = "qwen3.8-27b" };
    try api.writeCompletion(&out.writer, meta, "think \"q\"", "Paris\n", null, "stop", .{ .prompt_tokens = 3, .completion_tokens = 4 });
    try t.expectEqualStrings(
        \\{"id":"chatcmpl-1","object":"chat.completion","created":7,"model":"qwen3.8-27b","choices":[{"index":0,"message":{"role":"assistant","content":"Paris\n","reasoning_content":"think \"q\""},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7,"prompt_tokens_details":{"cached_tokens":0}}}
    , out.written());
    out.clearRetainingCapacity();
    try api.writeCompletion(&out.writer, meta, null, "x", null, "length", .{ .prompt_tokens = 1, .completion_tokens = 1 });
    try t.expect(std.mem.indexOf(u8, out.written(), "reasoning_content") == null);
    out.clearRetainingCapacity();
    try api.writeChunk(&out.writer, meta, .role);
    try api.writeChunk(&out.writer, meta, .{ .reasoning = "a" });
    try api.writeChunk(&out.writer, meta, .{ .content = "b" });
    try api.writeChunk(&out.writer, meta, .{ .finish = "stop" });
    try api.writeUsageChunk(&out.writer, meta, .{ .prompt_tokens = 2, .completion_tokens = 5, .cached_tokens = 1 });
    try t.expectEqualStrings(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"qwen3.8-27b\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n" ++
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"qwen3.8-27b\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"a\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n" ++
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"qwen3.8-27b\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n" ++
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"qwen3.8-27b\",\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"stop\"}]}\n\n" ++
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"qwen3.8-27b\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":5,\"total_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n",
        out.written(),
    );
    out.clearRetainingCapacity();
    try api.writeError(&out.writer, .{ .message = "bad", .param = "n", .code = "unsupported_parameter" });
    try t.expectEqualStrings("{\"error\":{\"message\":\"bad\",\"type\":\"invalid_request_error\",\"param\":\"n\",\"code\":\"unsupported_parameter\"}}", out.written());
    out.clearRetainingCapacity();
    try api.writeModels(&out.writer, &ids, 5);
    try t.expectEqualStrings("{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.8-27b\",\"object\":\"model\",\"created\":5,\"owned_by\":\"zerv\"}]}", out.written());
}

const Fake = struct {
    fn prepare(_: *anyopaque, arena: std.mem.Allocator, request: *const api.ChatRequest, rejection: *api.ApiError) anyerror!http.Prepared {
        const last = request.messages[request.messages.len - 1].content;
        if (std.mem.eql(u8, last, "reject")) {
            rejection.* = .{ .message = "No user query found in messages.", .param = "messages" };
            return error.Rejected;
        }
        const tokens = try arena.alloc(u32, last.len);
        @memset(tokens, 1);
        return .{ .tokens = tokens, .thinking = request.template.enable_thinking };
    }
    fn generate(_: *anyopaque, _: std.mem.Allocator, request: *const api.ChatRequest, prepared: http.Prepared, sink: http.Sink) anyerror!http.Completion {
        if (request.toolMode() != null) {
            try sink.emit(.{ .content = "Checking." });
            try sink.emit(.{ .call_begin = .{ .index = 0, .name = "get_weather" } });
            try sink.emit(.{ .call_arguments = .{ .index = 0, .text = "\"city\":\"" } });
            try sink.emit(.{ .call_arguments = .{ .index = 0, .text = "Paris\"}" } });
            try sink.emit(.{ .call_begin = .{ .index = 1, .name = "get_weather" } });
            try sink.emit(.{ .call_arguments = .{ .index = 1, .text = "}" } });
            return .{ .finish = .stop, .prompt_tokens = 1, .completion_tokens = 9, .prefill_ns = 1, .decode_ns = 1, .tool_calls = 2 };
        }
        if (prepared.thinking) try sink.emit(.{ .reasoning = "hmm \"ok\"" });
        try sink.emit(.{ .content = "Hello" });
        try sink.emit(.{ .content = " w\u{f6}rld" });
        const limited = request.max_tokens != null and request.max_tokens.? < 3;
        return .{ .finish = if (limited) .length else .stop, .prompt_tokens = @intCast(prepared.tokens.len), .completion_tokens = 3, .prefill_ns = 1, .decode_ns = 1, .spec = .{ .verifies = 1, .drafted = 3, .verified = 2, .accepted = 1 } };
    }
};

fn serveTask(server: *http.Server, listener: *std.Io.net.Server) std.Io.Cancelable!void {
    server.serve(listener) catch |e| switch (e) {
        error.Canceled => return error.Canceled,
        else => std.debug.panic("server failed: {s}", .{@errorName(e)}),
    };
}

/// A minimal HTTP/1.1 client for the server tests: one kept-alive connection (reopened when
/// the server answers `connection: close`), Content-Length and chunked bodies.
/// `std.http.Client` would compile the TLS stack: ~30 s of LLVM for the ReleaseFast test
/// binary (docs/bench/2026-09-26-test-parallelism.md).
const Client = struct {
    io: std.Io,
    port: u16,
    stream: ?std.Io.net.Stream = null,
    in_buf: [16384]u8 = undefined,
    out_buf: [4096]u8 = undefined,
    reader: std.Io.net.Stream.Reader = undefined,
    writer: std.Io.net.Stream.Writer = undefined,
    /// Connections opened (1 while the server keeps the connection alive).
    connects: u32 = 0,

    fn deinit(self: *Client) void {
        if (self.stream) |st| st.close(self.io);
        self.stream = null;
    }

    fn request(self: *Client, method: []const u8, path: []const u8, body: ?[]const u8, out: *std.Io.Writer.Allocating) !std.http.Status {
        out.clearRetainingCapacity();
        if (self.stream == null) {
            const target = try std.Io.net.IpAddress.parse("127.0.0.1", self.port);
            self.stream = try target.connect(self.io, .{ .mode = .stream });
            self.reader = self.stream.?.reader(self.io, &self.in_buf);
            self.writer = self.stream.?.writer(self.io, &self.out_buf);
            self.connects += 1;
        }
        const w = &self.writer.interface;
        try w.print("{s} {s} HTTP/1.1\r\nhost: 127.0.0.1\r\n", .{ method, path });
        if (body) |b| try w.print("content-type: application/json\r\ncontent-length: {d}\r\n", .{b.len});
        try w.writeAll("\r\n");
        if (body) |b| try w.writeAll(b);
        try w.flush();
        const r = &self.reader.interface;
        const status_line = std.mem.trimEnd(u8, try r.takeDelimiterExclusive('\n'), "\r");
        r.toss(1);
        if (!std.mem.startsWith(u8, status_line, "HTTP/1.1 ") or status_line.len < 12) return error.BadResponse;
        const code = try std.fmt.parseInt(u10, status_line[9..12], 10);
        var length: ?usize = null;
        var chunked = false;
        var close = false;
        while (true) {
            const line = std.mem.trimEnd(u8, try r.takeDelimiterExclusive('\n'), "\r");
            r.toss(1);
            if (line.len == 0) break;
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadResponse;
            const name = line[0..colon];
            const value = std.mem.trim(u8, line[colon + 1 ..], " ");
            if (std.ascii.eqlIgnoreCase(name, "content-length")) length = try std.fmt.parseInt(usize, value, 10);
            if (std.ascii.eqlIgnoreCase(name, "transfer-encoding") and std.ascii.eqlIgnoreCase(value, "chunked")) chunked = true;
            if (std.ascii.eqlIgnoreCase(name, "connection") and std.ascii.eqlIgnoreCase(value, "close")) close = true;
        }
        if (chunked) {
            while (true) {
                const line = std.mem.trimEnd(u8, try r.takeDelimiterExclusive('\n'), "\r");
                r.toss(1);
                const hex = if (std.mem.indexOfScalar(u8, line, ';')) |i| line[0..i] else line;
                const size = try std.fmt.parseInt(usize, hex, 16);
                if (size == 0) {
                    // Trailer section, then the empty line.
                    while (std.mem.trimEnd(u8, try r.takeDelimiterExclusive('\n'), "\r").len > 0) r.toss(1);
                    r.toss(1);
                    break;
                }
                try r.streamExact(&out.writer, size);
                if (!std.mem.eql(u8, try r.take(2), "\r\n")) return error.BadResponse;
            }
        } else try r.streamExact(&out.writer, length orelse return error.BadResponse);
        if (close) self.deinit();
        return @enumFromInt(code);
    }
};

fn post(client: *Client, path: []const u8, body: []const u8, out: *std.Io.Writer.Allocating) !std.http.Status {
    return client.request("POST", path, body, out);
}

test "HTTP server: real sockets, JSON and SSE framing, errors, keep-alive" {
    const io = t.io;
    var dummy: u8 = 0;
    const engine: http.Engine = .{ .context = &dummy, .prepareFn = Fake.prepare, .generateFn = Fake.generate, .ids = &ids, .defaults = .{} };
    var server = http.Server.init(io, t.allocator, engine, .{});
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    const port = listener.socket.address.getPort();
    var group: std.Io.Group = .init;
    try group.concurrent(io, serveTask, .{ &server, &listener });
    defer group.cancel(io);

    var client: Client = .{ .io = io, .port = port };
    defer client.deinit();
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    const chat = "/v1/chat/completions";

    // Non-streaming with reasoning.
    try t.expectEqual(std.http.Status.ok, try post(&client, chat, "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"four\"}]}", &out));
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, out.written(), .{});
        defer parsed.deinit();
        const choice = parsed.value.object.get("choices").?.array.items[0].object;
        try t.expectEqualStrings("Hello w\u{f6}rld", choice.get("message").?.object.get("content").?.string);
        try t.expectEqualStrings("hmm \"ok\"", choice.get("message").?.object.get("reasoning_content").?.string);
        try t.expectEqualStrings("stop", choice.get("finish_reason").?.string);
        try t.expectEqual(@as(i64, 7), parsed.value.object.get("usage").?.object.get("total_tokens").?.integer);
    }
    // Streaming, thinking disabled, usage requested, length finish.
    try t.expectEqual(std.http.Status.ok, try post(&client, chat, "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"stream_options\":{\"include_usage\":true},\"max_tokens\":2,\"chat_template_kwargs\":{\"enable_thinking\":false}}", &out));
    {
        var events: std.ArrayList([]const u8) = .empty;
        defer events.deinit(t.allocator);
        var it = std.mem.splitSequence(u8, out.written(), "\n\n");
        while (it.next()) |e| if (e.len > 0) try events.append(t.allocator, e);
        try t.expectEqual(@as(usize, 6), events.items.len);
        for (events.items) |e| try t.expect(std.mem.startsWith(u8, e, "data: "));
        try t.expect(std.mem.indexOf(u8, events.items[0], "\"role\":\"assistant\"") != null);
        try t.expect(std.mem.indexOf(u8, events.items[1], "\"content\":\"Hello\"") != null);
        try t.expect(std.mem.indexOf(u8, events.items[3], "\"finish_reason\":\"length\"") != null);
        try t.expect(std.mem.indexOf(u8, events.items[4], "\"total_tokens\":5") != null);
        try t.expectEqualStrings("data: [DONE]", events.items[5]);
        for (events.items) |e| try t.expect(std.mem.indexOf(u8, e, "reasoning_content") == null);
    }
    // Tool calls: JSON message.tool_calls and SSE tool_call deltas, finish tool_calls.
    const tool_body = "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"w\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]";
    try t.expectEqual(std.http.Status.ok, try post(&client, chat, tool_body ++ "}", &out));
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, out.written(), .{});
        defer parsed.deinit();
        const choice = parsed.value.object.get("choices").?.array.items[0].object;
        try t.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
        const message = choice.get("message").?.object;
        try t.expectEqualStrings("Checking.", message.get("content").?.string);
        const calls = message.get("tool_calls").?.array.items;
        try t.expectEqual(@as(usize, 2), calls.len);
        try t.expectEqualStrings("function", calls[0].object.get("type").?.string);
        try t.expectEqualStrings("get_weather", calls[0].object.get("function").?.object.get("name").?.string);
        try t.expectEqualStrings("{\"city\":\"Paris\"}", calls[0].object.get("function").?.object.get("arguments").?.string);
        try t.expectEqualStrings("{}", calls[1].object.get("function").?.object.get("arguments").?.string);
        const id0 = calls[0].object.get("id").?.string;
        try t.expect(std.mem.startsWith(u8, id0, "call_") and !std.mem.eql(u8, id0, calls[1].object.get("id").?.string));
    }
    try t.expectEqual(std.http.Status.ok, try post(&client, chat, tool_body ++ ",\"stream\":true}", &out));
    {
        const s = out.written();
        try t.expect(std.mem.indexOf(u8, s, "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_") != null);
        try t.expect(std.mem.indexOf(u8, s, "\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\"}}]}") != null);
        try t.expect(std.mem.indexOf(u8, s, "\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"Paris\\\"}\"}}]}") != null);
        try t.expect(std.mem.indexOf(u8, s, "{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\"}}]}") != null);
        try t.expect(std.mem.indexOf(u8, s, "\"finish_reason\":\"tool_calls\"") != null);
    }
    // tool_choice none: no tool mode, ordinary content.
    try t.expectEqual(std.http.Status.ok, try post(&client, chat, tool_body ++ ",\"tool_choice\":\"none\"}", &out));
    try t.expect(std.mem.indexOf(u8, out.written(), "tool_calls") == null);
    // Rejections before generation keep OpenAI error bodies and statuses.
    try t.expectEqual(std.http.Status.bad_request, try post(&client, chat, "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"reject\"}]}", &out));
    try t.expect(std.mem.indexOf(u8, out.written(), "No user query found in messages.") != null);
    try t.expectEqual(std.http.Status.not_found, try post(&client, chat, "{\"model\":\"other\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", &out));
    try t.expectEqual(std.http.Status.bad_request, try post(&client, chat, "{", &out));
    try t.expectEqual(std.http.Status.ok, try client.request("GET", "/v1/models", null, &out));
    try t.expect(std.mem.indexOf(u8, out.written(), "\"id\":\"qwen3.8-27b\"") != null);
    try t.expectEqual(std.http.Status.not_found, try post(&client, "/v1/completions", "{}", &out));
    try t.expectEqual(std.http.Status.method_not_allowed, try client.request("GET", chat, null, &out));
    try t.expectEqual(std.http.Status.ok, try client.request("GET", "/metrics", null, &out));
    // 2 plain + 3 tool requests generated; 3 rejected.
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_requests_total 8\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_rejected_total 3\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_completion_tokens_total 27\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_tool_call_parse_failures_total 0\n") != null);
    // Speculative counters are summed over the 3 plain (non-tool) requests.
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_spec_verifies_total 3\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_spec_draft_tokens_total{stage=\"drafted\"} 9\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_spec_draft_tokens_total{stage=\"verified\"} 6\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "zerv_spec_draft_tokens_total{stage=\"accepted\"} 3\n") != null);
    // Keep-alive: every request above went over one connection (none answered `close`).
    try t.expectEqual(@as(u32, 1), client.connects);
}

const Gated = struct {
    released: std.atomic.Value(bool) = .init(false),
    timed_out: std.atomic.Value(bool) = .init(false),
    io: std.Io,
    fn prepare(_: *anyopaque, arena: std.mem.Allocator, _: *const api.ChatRequest, _: *api.ApiError) anyerror!http.Prepared {
        return .{ .tokens = try arena.dupe(u32, &.{1}), .thinking = false };
    }
    fn generate(ctx: *anyopaque, _: std.mem.Allocator, _: *const api.ChatRequest, _: http.Prepared, sink: http.Sink) anyerror!http.Completion {
        const self: *Gated = @ptrCast(@alignCast(ctx));
        try sink.emit(.{ .content = "first" });
        // The client only releases us after it has received "first" on the socket.
        var waited: u32 = 0;
        while (!self.released.load(.acquire)) : (waited += 1) {
            if (waited == 500) {
                self.timed_out.store(true, .release);
                break;
            }
            try self.io.sleep(.fromMilliseconds(10), .awake);
        }
        try sink.emit(.{ .content = "second" });
        return .{ .finish = .stop, .prompt_tokens = 1, .completion_tokens = 2, .prefill_ns = 1, .decode_ns = 1 };
    }
};

test "HTTP server: every streamed delta reaches the socket before generation continues" {
    const io = t.io;
    var gated: Gated = .{ .io = io };
    const engine: http.Engine = .{ .context = &gated, .prepareFn = Gated.prepare, .generateFn = Gated.generate, .ids = &ids, .defaults = .{} };
    var server = http.Server.init(io, t.allocator, engine, .{});
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    var group: std.Io.Group = .init;
    try group.concurrent(io, serveTask, .{ &server, &listener });
    defer group.cancel(io);
    const target = try std.Io.net.IpAddress.parse("127.0.0.1", listener.socket.address.getPort());
    const stream = try target.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    const body = "{\"model\":\"qwen3.8-27b\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    var out_buf: [512]u8 = undefined;
    var writer = stream.writer(io, &out_buf);
    try writer.interface.print("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ body.len, body });
    try writer.interface.flush();
    var in_buf: [4096]u8 = undefined;
    var reader = stream.reader(io, &in_buf);
    var seen: std.ArrayList(u8) = .empty;
    defer seen.deinit(t.allocator);
    while (std.mem.indexOf(u8, seen.items, "\"content\":\"first\"") == null) {
        try t.expect(!gated.timed_out.load(.acquire)); // buffered: the first delta never arrived
        const chunk = reader.interface.peekGreedy(1) catch break;
        try seen.appendSlice(t.allocator, chunk);
        reader.interface.toss(chunk.len);
    }
    try t.expect(!gated.timed_out.load(.acquire));
    gated.released.store(true, .release);
    while (true) {
        const chunk = reader.interface.peekGreedy(1) catch break;
        try seen.appendSlice(t.allocator, chunk);
        reader.interface.toss(chunk.len);
    }
    try t.expect(std.mem.indexOf(u8, seen.items, "\"content\":\"second\"") != null);
    try t.expect(std.mem.indexOf(u8, seen.items, "data: [DONE]") != null);
    try t.expect(!gated.timed_out.load(.acquire));
}

/// Blocks in generate until released (or ~5 s), after emitting one delta.
const Hold = struct {
    io: std.Io,
    started: std.atomic.Value(u32) = .init(0),
    released: std.atomic.Value(bool) = .init(false),
    fn prepare(_: *anyopaque, arena: std.mem.Allocator, _: *const api.ChatRequest, _: *api.ApiError) anyerror!http.Prepared {
        return .{ .tokens = try arena.dupe(u32, &.{1}), .thinking = false };
    }
    fn generate(ctx: *anyopaque, _: std.mem.Allocator, _: *const api.ChatRequest, _: http.Prepared, sink: http.Sink) anyerror!http.Completion {
        const self: *Hold = @ptrCast(@alignCast(ctx));
        _ = self.started.fetchAdd(1, .acq_rel);
        try sink.emit(.{ .content = "x" });
        var waited: u32 = 0;
        while (!self.released.load(.acquire) and waited < 500) : (waited += 1) try self.io.sleep(.fromMilliseconds(10), .awake);
        return .{ .finish = .stop, .prompt_tokens = 1, .completion_tokens = 1, .prefill_ns = 1, .decode_ns = 1 };
    }
};

fn rawPost(io: std.Io, port: u16, stream_flag: bool) !std.Io.net.Stream {
    const target = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    const stream = try target.connect(io, .{ .mode = .stream });
    const body = if (stream_flag)
        "{\"model\":\"qwen3.8-27b\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}"
    else
        "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    var out_buf: [512]u8 = undefined;
    var writer = stream.writer(io, &out_buf);
    try writer.interface.print("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ body.len, body });
    try writer.interface.flush();
    return stream;
}

/// Reads until the peer closes; returns everything received.
fn readAll(io: std.Io, stream: std.Io.net.Stream, out: *std.ArrayList(u8)) !void {
    var in_buf: [4096]u8 = undefined;
    var reader = stream.reader(io, &in_buf);
    while (true) {
        const chunk = reader.interface.peekGreedy(1) catch break;
        try out.appendSlice(t.allocator, chunk);
        reader.interface.toss(chunk.len);
    }
}

test "HTTP server: real sockets, bounded queue answers 503 overloaded, queued requests complete" {
    const io = t.io;
    var hold: Hold = .{ .io = io };
    const engine: http.Engine = .{ .context = &hold, .prepareFn = Hold.prepare, .generateFn = Hold.generate, .ids = &ids, .defaults = .{} };
    var server = http.Server.init(io, t.allocator, engine, .{ .max_waiting = 1 });
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    const port = listener.socket.address.getPort();
    var group: std.Io.Group = .init;
    try group.concurrent(io, serveTask, .{ &server, &listener });
    defer group.cancel(io);

    // 1: generating (holds the single sequence slot).
    const first = try rawPost(io, port, false);
    defer first.close(io);
    var spins: u32 = 0;
    while (hold.started.load(.acquire) == 0 and spins < 500) : (spins += 1) try io.sleep(.fromMilliseconds(10), .awake);
    try t.expectEqual(@as(u32, 1), hold.started.load(.acquire));
    // 2: queued behind it (fills max_waiting = 1).
    const second = try rawPost(io, port, false);
    defer second.close(io);
    spins = 0;
    while (server.waiting.load(.acquire) == 0 and spins < 500) : (spins += 1) try io.sleep(.fromMilliseconds(10), .awake);
    try t.expectEqual(@as(u32, 1), server.waiting.load(.acquire));
    // 3: over capacity -> immediate 503 with the overloaded code, while 1 still generates.
    const third = try rawPost(io, port, false);
    defer third.close(io);
    var rejected: std.ArrayList(u8) = .empty;
    defer rejected.deinit(t.allocator);
    try readAll(io, third, &rejected);
    try t.expect(std.mem.startsWith(u8, rejected.items, "HTTP/1.1 503"));
    try t.expect(std.mem.indexOf(u8, rejected.items, "\"code\":\"overloaded\"") != null);
    try t.expectEqual(@as(u32, 1), hold.started.load(.acquire));
    // Release: both admitted requests complete with 200.
    hold.released.store(true, .release);
    for ([_]std.Io.net.Stream{ first, second }) |s| {
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(t.allocator);
        try readAll(io, s, &got);
        try t.expect(std.mem.startsWith(u8, got.items, "HTTP/1.1 200"));
        try t.expect(std.mem.indexOf(u8, got.items, "\"content\":\"x\"") != null);
    }
    try t.expectEqual(@as(u32, 2), hold.started.load(.acquire));
    try t.expectEqual(@as(u64, 1), server.metrics.overloaded.load(.acquire));
}

/// Streams deltas until a sink write fails (client gone) or a cap is reached.
const Endless = struct {
    io: std.Io,
    emitted: std.atomic.Value(u32) = .init(0),
    cancelled: std.atomic.Value(bool) = .init(false),
    const cap = 5000;
    fn prepare(_: *anyopaque, arena: std.mem.Allocator, _: *const api.ChatRequest, _: *api.ApiError) anyerror!http.Prepared {
        return .{ .tokens = try arena.dupe(u32, &.{1}), .thinking = false };
    }
    fn generate(ctx: *anyopaque, _: std.mem.Allocator, _: *const api.ChatRequest, _: http.Prepared, sink: http.Sink) anyerror!http.Completion {
        const self: *Endless = @ptrCast(@alignCast(ctx));
        while (self.emitted.load(.acquire) < cap) {
            sink.emit(.{ .content = "tok " }) catch |e| {
                self.cancelled.store(true, .release);
                return e;
            };
            _ = self.emitted.fetchAdd(1, .acq_rel);
            try self.io.sleep(.fromMilliseconds(1), .awake);
        }
        return .{ .finish = .length, .prompt_tokens = 1, .completion_tokens = cap, .prefill_ns = 1, .decode_ns = 1 };
    }
};

test "HTTP server: client disconnect mid-stream cancels generation and frees the slot" {
    const io = t.io;
    var endless: Endless = .{ .io = io };
    const engine: http.Engine = .{ .context = &endless, .prepareFn = Endless.prepare, .generateFn = Endless.generate, .ids = &ids, .defaults = .{} };
    var server = http.Server.init(io, t.allocator, engine, .{});
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    const port = listener.socket.address.getPort();
    var group: std.Io.Group = .init;
    try group.concurrent(io, serveTask, .{ &server, &listener });
    defer group.cancel(io);

    const stream = try rawPost(io, port, true);
    {
        var in_buf: [4096]u8 = undefined;
        var reader = stream.reader(io, &in_buf);
        var seen: std.ArrayList(u8) = .empty;
        defer seen.deinit(t.allocator);
        while (std.mem.indexOf(u8, seen.items, "\"content\":\"tok \"") == null) {
            const chunk = try reader.interface.peekGreedy(1);
            try seen.appendSlice(t.allocator, chunk);
            reader.interface.toss(chunk.len);
        }
    }
    stream.close(io); // client goes away mid-generation
    var spins: u32 = 0;
    while (!endless.cancelled.load(.acquire) and spins < 500) : (spins += 1) try io.sleep(.fromMilliseconds(10), .awake);
    try t.expect(endless.cancelled.load(.acquire));
    try t.expect(endless.emitted.load(.acquire) < Endless.cap);
    try t.expectEqual(@as(u64, 1), server.metrics.failed.load(.acquire));
    // The sequence slot is free again: a new request is served (it runs to the cap).
    endless.cancelled.store(false, .release);
    endless.emitted.store(Endless.cap - 3, .release);
    const next = try rawPost(io, port, false);
    defer next.close(io);
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(t.allocator);
    try readAll(io, next, &got);
    try t.expect(std.mem.startsWith(u8, got.items, "HTTP/1.1 200"));
    try t.expect(std.mem.indexOf(u8, got.items, "\"finish_reason\":\"length\"") != null);
}

fn runTask(server: *http.Server, listener: *std.Io.net.Server, stop: *const std.atomic.Value(bool), result: *?anyerror) std.Io.Cancelable!void {
    server.run(listener, stop) catch |e| {
        result.* = e;
    };
}

fn connectIdle(io: std.Io, port: u16) !std.Io.net.Stream {
    const target = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    return target.connect(io, .{ .mode = .stream });
}

fn sendRaw(io: std.Io, stream: std.Io.net.Stream, bytes: []const u8) !void {
    var out_buf: [512]u8 = undefined;
    var writer = stream.writer(io, &out_buf);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

test "HTTP server: stop drains admitted requests, rejects new ones, closes idle connections" {
    const io = t.io;
    var hold: Hold = .{ .io = io };
    const engine: http.Engine = .{ .context = &hold, .prepareFn = Hold.prepare, .generateFn = Hold.generate, .ids = &ids, .defaults = .{} };
    var server = http.Server.init(io, t.allocator, engine, .{ .drain_timeout = .fromSeconds(10) });
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{ .reuse_address = true }); // owned by `run`
    const port = listener.socket.address.getPort();
    var stop: std.atomic.Value(bool) = .init(false);
    var result: ?anyerror = null;
    var group: std.Io.Group = .init;
    try group.concurrent(io, runTask, .{ &server, &listener, &stop, &result });
    defer group.cancel(io);

    const idle = try connectIdle(io, port); // never sends a request
    defer idle.close(io);
    const late_chat = try connectIdle(io, port); // sends a chat request after stop
    defer late_chat.close(io);
    const late_ready = try connectIdle(io, port); // asks /ready after stop
    defer late_ready.close(io);
    const running = try rawPost(io, port, false);
    defer running.close(io);
    var spins: u32 = 0;
    while ((hold.started.load(.acquire) == 0 or server.connections.load(.acquire) < 4) and spins < 500) : (spins += 1) try io.sleep(.fromMilliseconds(10), .awake);
    try t.expectEqual(@as(u32, 4), server.connections.load(.acquire));

    stop.store(true, .release);
    spins = 0;
    while (!server.draining.load(.acquire) and spins < 500) : (spins += 1) try io.sleep(.fromMilliseconds(10), .awake);
    try t.expect(server.draining.load(.acquire));
    // The listener closes when draining starts: new connections are refused.
    spins = 0;
    while (spins < 100) : (spins += 1) {
        const probe = connectIdle(io, port) catch |e| switch (e) {
            error.ConnectionRefused => break,
            error.ConnectionResetByPeer => continue, // raced with the close
            else => return e,
        };
        probe.close(io);
        try io.sleep(.fromMilliseconds(10), .awake);
    } else return error.ListenerStillOpen;

    const body = "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    var request_buf: [256]u8 = undefined;
    try sendRaw(io, late_chat, try std.fmt.bufPrint(&request_buf, "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {d}\r\n\r\n{s}", .{ body.len, body }));
    var rejected: std.ArrayList(u8) = .empty;
    defer rejected.deinit(t.allocator);
    try readAll(io, late_chat, &rejected); // the server closes after answering
    try t.expect(std.mem.startsWith(u8, rejected.items, "HTTP/1.1 503"));
    try t.expect(std.mem.indexOf(u8, rejected.items, "connection: close\r\n") != null);
    try t.expect(std.mem.indexOf(u8, rejected.items, "\"code\":\"shutting_down\"") != null);

    try sendRaw(io, late_ready, "GET /ready HTTP/1.1\r\nhost: x\r\n\r\n");
    var ready: std.ArrayList(u8) = .empty;
    defer ready.deinit(t.allocator);
    try readAll(io, late_ready, &ready);
    try t.expect(std.mem.startsWith(u8, ready.items, "HTTP/1.1 503"));
    try t.expect(std.mem.indexOf(u8, ready.items, "\"status\":\"draining\"") != null);

    // The admitted request still completes normally, then serving ends.
    try t.expectEqual(@as(u32, 1), server.admitted.load(.acquire));
    hold.released.store(true, .release);
    var done: std.ArrayList(u8) = .empty;
    defer done.deinit(t.allocator);
    try readAll(io, running, &done);
    try t.expect(std.mem.startsWith(u8, done.items, "HTTP/1.1 200"));
    try t.expect(std.mem.indexOf(u8, done.items, "\"content\":\"x\"") != null);
    try group.await(io);
    try t.expectEqual(@as(?anyerror, null), result);
    try t.expectEqual(@as(u32, 0), server.connections.load(.acquire));
    var idle_rest: std.ArrayList(u8) = .empty;
    defer idle_rest.deinit(t.allocator);
    try readAll(io, idle, &idle_rest); // closed by the server without a response
    try t.expectEqual(@as(usize, 0), idle_rest.items.len);
    try t.expectEqual(@as(u32, 1), hold.started.load(.acquire));
    try t.expectEqual(@as(u64, 0), server.metrics.failed.load(.acquire));
}

test "HTTP server: drain deadline cancels a generation that is still running" {
    const io = t.io;
    var endless: Endless = .{ .io = io };
    const engine: http.Engine = .{ .context = &endless, .prepareFn = Endless.prepare, .generateFn = Endless.generate, .ids = &ids, .defaults = .{} };
    var server = http.Server.init(io, t.allocator, engine, .{ .drain_timeout = .fromMilliseconds(100) });
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{ .reuse_address = true }); // owned by `run`
    const port = listener.socket.address.getPort();
    var stop: std.atomic.Value(bool) = .init(false);
    var result: ?anyerror = null;
    var group: std.Io.Group = .init;
    try group.concurrent(io, runTask, .{ &server, &listener, &stop, &result });
    defer group.cancel(io);

    const running = try rawPost(io, port, false);
    defer running.close(io);
    var spins: u32 = 0;
    while (endless.emitted.load(.acquire) == 0 and spins < 500) : (spins += 1) try io.sleep(.fromMilliseconds(10), .awake);
    const start = std.Io.Clock.awake.now(io);
    stop.store(true, .release);
    try group.await(io);
    const elapsed_ms = @divTrunc(std.Io.Clock.awake.now(io).nanoseconds - start.nanoseconds, std.time.ns_per_ms);
    try t.expect(elapsed_ms < 2000);
    try t.expectEqual(@as(?anyerror, null), result);
    try t.expect(endless.emitted.load(.acquire) < Endless.cap);
    try t.expectEqual(@as(u64, 1), server.metrics.failed.load(.acquire));
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(t.allocator);
    try readAll(io, running, &got); // closed without a response
    try t.expectEqual(@as(usize, 0), got.items.len);
}

/// Fails every generation; afterwards reports itself unusable (as after a lost device).
const Broken = struct {
    lost: std.atomic.Value(bool) = .init(false),
    fn prepare(_: *anyopaque, arena: std.mem.Allocator, _: *const api.ChatRequest, _: *api.ApiError) anyerror!http.Prepared {
        return .{ .tokens = try arena.dupe(u32, &.{1}), .thinking = false };
    }
    fn generate(ctx: *anyopaque, _: std.mem.Allocator, _: *const api.ChatRequest, _: http.Prepared, _: http.Sink) anyerror!http.Completion {
        const self: *Broken = @ptrCast(@alignCast(ctx));
        self.lost.store(true, .release);
        return error.DeviceLost;
    }
    fn usable(ctx: *anyopaque) bool {
        const self: *Broken = @ptrCast(@alignCast(ctx));
        return !self.lost.load(.acquire);
    }
};

fn rawGet(io: std.Io, port: u16, path: []const u8, out: *std.ArrayList(u8)) !void {
    const stream = try connectIdle(io, port);
    defer stream.close(io);
    var request_buf: [128]u8 = undefined;
    try sendRaw(io, stream, try std.fmt.bufPrint(&request_buf, "GET {s} HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n", .{path}));
    try readAll(io, stream, out);
}

test "HTTP server: an unusable engine fails health checks and requests, and ends run" {
    const io = t.io;
    // With `serve` (no shutdown), the failed state is observable on new connections.
    {
        var broken: Broken = .{};
        const engine: http.Engine = .{ .context = &broken, .prepareFn = Broken.prepare, .generateFn = Broken.generate, .usableFn = Broken.usable, .ids = &ids, .defaults = .{} };
        var server = http.Server.init(io, t.allocator, engine, .{});
        const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
        var listener = try address.listen(io, .{});
        defer listener.deinit(io);
        const port = listener.socket.address.getPort();
        var group: std.Io.Group = .init;
        try group.concurrent(io, serveTask, .{ &server, &listener });
        defer group.cancel(io);
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(t.allocator);
        try rawGet(io, port, "/health", &got);
        try t.expect(std.mem.startsWith(u8, got.items, "HTTP/1.1 200"));
        const failing = try rawPost(io, port, false);
        got.clearRetainingCapacity();
        try readAll(io, failing, &got);
        failing.close(io);
        try t.expect(std.mem.startsWith(u8, got.items, "HTTP/1.1 500"));
        try t.expect(server.failed.load(.acquire));
        for ([_][]const u8{ "/health", "/ready" }) |path| {
            got.clearRetainingCapacity();
            try rawGet(io, port, path, &got);
            try t.expect(std.mem.startsWith(u8, got.items, "HTTP/1.1 503"));
            try t.expect(std.mem.indexOf(u8, got.items, "\"status\":\"failed\"") != null);
        }
        const next = try rawPost(io, port, true);
        got.clearRetainingCapacity();
        try readAll(io, next, &got);
        next.close(io);
        try t.expect(std.mem.startsWith(u8, got.items, "HTTP/1.1 503"));
        try t.expect(std.mem.indexOf(u8, got.items, "\"code\":\"engine_failed\"") != null);
    }
    // With `run`, the failure ends serving without a stop request.
    {
        var broken: Broken = .{};
        const engine: http.Engine = .{ .context = &broken, .prepareFn = Broken.prepare, .generateFn = Broken.generate, .usableFn = Broken.usable, .ids = &ids, .defaults = .{} };
        var server = http.Server.init(io, t.allocator, engine, .{ .drain_timeout = .fromSeconds(5) });
        const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
        var listener = try address.listen(io, .{}); // owned by `run`
        const port = listener.socket.address.getPort();
        const stop: std.atomic.Value(bool) = .init(false);
        var result: ?anyerror = null;
        var group: std.Io.Group = .init;
        try group.concurrent(io, runTask, .{ &server, &listener, &stop, &result });
        defer group.cancel(io);
        const failing = try rawPost(io, port, true); // streaming: headers already sent
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(t.allocator);
        try readAll(io, failing, &got);
        failing.close(io);
        try t.expect(std.mem.indexOf(u8, got.items, "\"message\":\"generation failed\"") != null);
        try group.await(io);
        try t.expectEqual(@as(?anyerror, error.EngineFailed), result);
        try t.expectError(error.ConnectionRefused, connectIdle(io, port));
    }
}

test "listener: exclusive port, immediate rebind after connections close" {
    const io = t.io;
    const any = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var first = try serve.listen.listen(any, 16);
    const port = first.socket.address.getPort();
    const same = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    // A second listener on the same address is refused, whether it asks for port
    // sharing (std's reuse_address sets SO_REUSEPORT) or not.
    try t.expectError(error.AddressInUse, serve.listen.listen(same, 16));
    try t.expectError(error.AddressInUse, same.listen(io, .{ .reuse_address = true }));
    // Server-side close first leaves the port's connection in TIME_WAIT; closing the
    // listener then re-listening at once must still work (SO_REUSEADDR).
    const client = try connectIdle(io, port);
    const accepted = try first.accept(io);
    accepted.close(io);
    var buf: [16]u8 = undefined;
    var reader = client.reader(io, &buf);
    _ = reader.interface.peekGreedy(1) catch {}; // wait for the server's FIN
    client.close(io);
    first.deinit(io);
    var again = try serve.listen.listen(same, 16);
    defer again.deinit(io);
    try t.expectEqual(port, again.socket.address.getPort());
}
