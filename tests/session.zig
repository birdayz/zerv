const std = @import("std");
const zerv = @import("zerv");
const session = zerv.session;
const text = session.text;
const sampler = session.sampler;
const t = std.testing;

fn decodeChunked(input: []const u8, cut: []const usize, out: []u8) []const u8 {
    var d: text.Utf8 = .{};
    var n: usize = 0;
    var start: usize = 0;
    var scratch: [256]u8 = undefined;
    for (cut) |c| {
        const s = d.push(input[start..c], &scratch);
        @memcpy(out[n..][0..s.len], s);
        n += s.len;
        start = c;
    }
    const s = d.push(input[start..], &scratch);
    @memcpy(out[n..][0..s.len], s);
    n += s.len;
    d.finish();
    return out[0..n];
}

test "streaming UTF-8 matches Python errors='replace' for every chunking" {
    // Expected values were produced by CPython 3.14 bytes.decode("utf-8", "replace"),
    // except that an incomplete (valid-prefix) tail at the end of the stream is
    // dropped, as llama.cpp's output parser drops it (docs/research/output-parsing.md);
    // CPython would end "a\xc3", "\xe2\x82" and "\xf0\x9f\x98" with one U+FFFD.
    const R = "\xef\xbf\xbd";
    const cases = [_][2][]const u8{
        .{ "a\xc3", "a" },                                             .{ "\xc3(", R ++ "(" },
        .{ "\xe2\x82", "" },                                           .{ "\xe2\x82(", R ++ "(" },
        .{ "\xf0\x9f\x98\x80", "\xf0\x9f\x98\x80" },                   .{ "\xed\xa0\x80", R ++ R ++ R },
        .{ "\xc0\x80", R ++ R },                                       .{ "\xf4\x90\x80\x80", R ++ R ++ R ++ R },
        .{ "\xe0\x80\xaf", R ++ R ++ R },                              .{ "\xff", R },
        .{ "\x80abc", R ++ "abc" },                                    .{ "\xf0\x9f\x98", "" },
        .{ "h\xc3\xa9llo \xe4\xb8\x96", "h\xc3\xa9llo \xe4\xb8\x96" },
    };
    var out: [128]u8 = undefined;
    for (cases) |c| {
        // Every single split point and the all-bytes split.
        try t.expectEqualStrings(c[1], decodeChunked(c[0], &.{}, &out));
        for (0..c[0].len + 1) |cut| try t.expectEqualStrings(c[1], decodeChunked(c[0], &.{cut}, &out));
        var all: [16]usize = undefined;
        for (0..c[0].len) |i| all[i] = i;
        try t.expectEqualStrings(c[1], decodeChunked(c[0], all[0..c[0].len], &out));
    }
}

test "stop strings: earliest match across chunks, holdback and release" {
    var out: [256]u8 = undefined;
    var s = try text.Stops.init(&.{ "END", "\n\n" });
    try t.expectEqualStrings("hello ", s.push("hello E", &out));
    try t.expectEqualStrings("", s.push("N", &out));
    var s2 = try text.Stops.init(&.{ "END", "\n\n" });
    try t.expectEqualStrings("hello ", s2.push("hello E", &out));
    try t.expectEqualStrings("EN", s2.push("Nx", &out)[0..2]);
    var s3 = try text.Stops.init(&.{ "END", "\n\n" });
    try t.expectEqualStrings("a", s3.push("a\n", &out));
    try t.expectEqualStrings("", s3.push("\nEND", &out));
    try t.expect(s3.stopped);
    try t.expectEqualStrings("", s3.push("more", &out));
    var s4 = try text.Stops.init(&.{"xyz"});
    try t.expectEqualStrings("ab", s4.push("abxy", &out));
    try t.expectEqualStrings("xy", s4.finish(&out));
    var s5 = try text.Stops.init(&.{ "bc", "abcd" });
    try t.expectEqualStrings("", s5.push("abcd", &out)); // earliest start within one chunk
    try t.expectError(error.InvalidStop, text.Stops.init(&.{""}));
    try t.expectError(error.InvalidStop, text.Stops.init(&.{ "a", "b", "c", "d", "e" }));
    try t.expectError(error.InvalidStop, text.Stops.init(&.{"x" ** 65}));
}

test "sampler: greedy ties, validation, top-k selection and distributions" {
    const a = t.allocator;
    try t.expectEqual(@as(u32, 1), try sampler.greedy(&.{ 0, 3, 3, -1 }));
    try t.expectError(error.NonFiniteLogits, sampler.greedy(&.{ 0, std.math.nan(f32) }));
    for ([_]sampler.Params{ .{ .temperature = -1 }, .{ .top_p = 0 }, .{ .top_p = 1.5 }, .{ .min_p = 2 }, .{ .repetition_penalty = 0 }, .{ .presence_penalty = 3 }, .{ .temperature = std.math.inf(f32) } }) |p|
        try t.expectError(error.InvalidSampling, sampler.Sampler.init(a, 4, p));
    // top-k = 1 at any temperature is greedy.
    var prng = std.Random.DefaultPrng.init(7);
    var logits: [5000]f32 = undefined;
    for (0..20) |round| {
        for (&logits) |*l| l.* = prng.random().floatNorm(f32) * 3;
        var s = try sampler.Sampler.init(a, logits.len, .{ .temperature = 0.7, .top_k = 1, .seed = round });
        defer s.deinit(a);
        try t.expectEqual(try sampler.greedy(&logits), try s.sample(&logits));
    }
    // Empirical distribution with top-k 3 / temperature 1 matches softmax of the top three.
    const small = [_]f32{ 1.0, 0.0, 2.0, -5.0, 0.5 };
    var s = try sampler.Sampler.init(a, small.len, .{ .temperature = 1, .top_k = 3, .seed = 42 });
    defer s.deinit(a);
    var counts: [5]u32 = @splat(0);
    const draws = 200000;
    for (0..draws) |_| counts[try s.sample(&small)] += 1;
    try t.expectEqual(@as(u32, 0), counts[1] + counts[3]);
    const z = @exp(2.0) + @exp(1.0) + @exp(0.5);
    for ([_]usize{ 2, 0, 4 }, [_]f64{ @exp(2.0) / z, @exp(1.0) / z, @exp(0.5) / z }) |id, expected| {
        const observed = @as(f64, @floatFromInt(counts[id])) / draws;
        try t.expect(@abs(observed - expected) < 0.005);
    }
    // top-p keeps the smallest prefix reaching p; min-p drops relatively unlikely ids.
    var s2 = try sampler.Sampler.init(a, small.len, .{ .temperature = 1, .top_p = 0.5, .seed = 1 });
    defer s2.deinit(a);
    for (0..1000) |_| try t.expectEqual(@as(u32, 2), try s2.sample(&small));
    var s3 = try sampler.Sampler.init(a, small.len, .{ .temperature = 1, .min_p = 0.5, .seed = 1 });
    defer s3.deinit(a);
    for (0..2000) |_| {
        const id = try s3.sample(&small);
        try t.expect(id == 2 or id == 0); // p(0)/p(2) = e^-1 > 0.5 > p(4)/p(2)
    }
    // Presence penalty removes a repeated winner at temperature 0.
    var s4 = try sampler.Sampler.init(a, small.len, .{ .temperature = 0, .presence_penalty = 1.5 });
    defer s4.deinit(a);
    try t.expectEqual(@as(u32, 2), try s4.sample(&small));
    try s4.accept(a, 2);
    try t.expectEqual(@as(u32, 0), try s4.sample(&small));
    try t.expectError(error.InvalidLogits, s4.sample(small[0..3]));
}

test "sampler top-k partial selection equals full sort for random inputs" {
    const a = t.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    for (0..200) |round| {
        const n = 1 + prng.random().uintLessThan(usize, 300);
        const logits = try a.alloc(f32, n);
        defer a.free(logits);
        for (logits) |*l| l.* = @floatFromInt(prng.random().intRangeAtMost(i32, -8, 8)); // many ties
        const k: u32 = @intCast(1 + prng.random().uintLessThan(usize, n));
        // With a tiny temperature the draw is the maximum of the kept set, which must be
        // the global argmax; with top-p 1e-6 the kept set is exactly that argmax.
        var s = try sampler.Sampler.init(a, n, .{ .temperature = 1, .top_k = k, .top_p = 1e-6, .seed = round });
        defer s.deinit(a);
        try t.expectEqual(try sampler.greedy(logits), try s.sample(logits));
    }
}

const FakeBackend = struct {
    script: []const u32,
    vocab_size: usize,
    ctx: u32,
    at: usize = 0,
    resets: u32 = 0,
    logits: [16]f32 = undefined,
    steps: u32 = 0,
    pub fn context(self: *FakeBackend) u32 {
        return self.ctx;
    }
    pub fn vocab(self: *FakeBackend) usize {
        return self.vocab_size;
    }
    pub fn reset(self: *FakeBackend) !void {
        self.resets += 1;
        self.at = 0;
    }
    pub fn step(self: *FakeBackend, token: u32) ![]const f32 {
        _ = token;
        self.steps += 1;
        @memset(self.logits[0..self.vocab_size], 0);
        // After the prompt, prefer the next scripted token.
        const next = self.script[@min(self.at, self.script.len - 1)];
        self.logits[next] = 10;
        return self.logits[0..self.vocab_size];
    }
};
const FakeTokenizer = struct {
    pieces: []const []const u8,
    pub fn outputPiece(self: FakeTokenizer, id: u32) ![]const u8 {
        return self.pieces[id];
    }
};
const Sink = struct {
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    fail_after: usize = std.math.maxInt(usize),
    calls: usize = 0,
    pub fn emit(self: *Sink, event: session.Event) !void {
        self.calls += 1;
        if (self.calls > self.fail_after) return error.ClientGone;
        switch (event) {
            .reasoning => |bytes| try self.reasoning.appendSlice(t.allocator, bytes),
            .content => |bytes| try self.content.appendSlice(t.allocator, bytes),
            else => return error.UnexpectedEvent,
        }
    }
    fn deinit(self: *Sink) void {
        self.reasoning.deinit(t.allocator);
        self.content.deinit(t.allocator);
    }
};
// ids: 0 EOS, 1 </think>, 2 "\n", 3 "Hi", 4 " there", 5 "\xe4\xb8", 6 "\x96!", 7 "STOP", 8 "  ",
// 9 "</thi", 10 "nk>", 11 "<tool_call>", 12 "" (a control token renders as nothing)
const pieces = [_][]const u8{ "<|im_end|>", "</think>", "\n", "Hi", " there", "\xe4\xb8", "\x96!", "STOP", "  ", "</thi", "nk>", "<tool_call>", "" };
const special: session.Special = .{ .eos = &.{0} };

test "generation: reasoning split, whitespace trimming, UTF-8 across tokens, EOS" {
    var backend: FakeBackend = undefined;
    var sink: Sink = .{};
    defer sink.deinit();
    const Script = struct {
        var order = [_]u32{ 2, 3, 8, 4, 2, 1, 2, 2, 5, 6, 0 };
    };
    backend = .{ .script = &Script.order, .vocab_size = pieces.len, .ctx = 64 };
    var adv: Advancing = .{ .inner = &backend, .prompt = 2 };
    const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{ 3, 3 }, .max_tokens = 32, .thinking = true }, &sink);
    try t.expectEqual(session.Finish.stop, r.finish);
    try t.expectEqual(@as(u32, 11), r.completion_tokens);
    // Leading whitespace is skipped; trailing reasoning whitespace is kept (llama.cpp).
    try t.expectEqualStrings("Hi   there\n", sink.reasoning.items);
    try t.expectEqualStrings("\xe4\xb8\x96!", sink.content.items);
    try t.expectEqual(@as(u32, 1), backend.resets);
}

/// Wraps the fake so that it advances one scripted token per generated step.
const Advancing = struct {
    inner: *FakeBackend,
    prompt: usize,
    seen: usize = 0,
    pub fn context(self: *Advancing) u32 {
        return self.inner.ctx;
    }
    pub fn vocab(self: *Advancing) usize {
        return self.inner.vocab_size;
    }
    pub fn reset(self: *Advancing) !void {
        self.seen = 0;
        try self.inner.reset();
    }
    pub fn prefill(self: *Advancing, tokens: []const u32) ![]const f32 {
        var logits: []const f32 = undefined;
        for (tokens) |token| logits = try self.step(token);
        return logits;
    }
    pub fn step(self: *Advancing, token: u32) ![]const f32 {
        self.seen += 1;
        if (self.seen > self.prompt) self.inner.at += 1;
        return self.inner.step(token);
    }
};
const GenA = session.Generation(*Advancing, FakeTokenizer, *Sink);

test "generation: length limit, context cap, stop strings and sink cancellation" {
    var backend: FakeBackend = undefined;
    const Script = struct {
        var order = [_]u32{ 3, 4, 3, 4, 3, 4, 3, 4 };
        var stop_order = [_]u32{ 3, 4, 7, 3 };
    };
    {
        backend = .{ .script = &Script.order, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 3 }, &sink);
        try t.expectEqual(session.Finish.length, r.finish);
        try t.expectEqual(@as(u32, 3), r.completion_tokens);
        try t.expectEqualStrings("Hi thereHi", sink.content.items);
        try t.expectEqual(@as(u32, 1 + 2), backend.steps); // the last token is never stepped
    }
    {
        backend = .{ .script = &Script.order, .vocab_size = pieces.len, .ctx = 4 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 2 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{ 3, 3 }, .max_tokens = 100 }, &sink);
        try t.expectEqual(@as(u32, 2), r.completion_tokens);
        try t.expectError(error.ContextExceeded, GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{ 3, 3, 3, 3 }, .max_tokens = 1 }, &sink));
        try t.expectError(error.EmptyPrompt, GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{}, .max_tokens = 1 }, &sink));
    }
    {
        backend = .{ .script = &Script.stop_order, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 10, .stops = &.{"reST"} }, &sink);
        try t.expectEqual(session.Finish.stop, r.finish);
        try t.expectEqualStrings("Hi the", sink.content.items);
        try t.expectEqual(@as(u32, 3), r.completion_tokens);
    }
    {
        backend = .{ .script = &Script.order, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{ .fail_after = 2 };
        defer sink.deinit();
        try t.expectError(error.ClientGone, GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 10 }, &sink));
    }
}

const SplitSink = struct {
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    pub fn emit(self: *SplitSink, event: text.Event) !void {
        switch (event) {
            .reasoning => |bytes| {
                try t.expect(bytes.len > 0);
                try self.reasoning.appendSlice(t.allocator, bytes);
            },
            .content => |bytes| {
                try t.expect(bytes.len > 0);
                try self.content.appendSlice(t.allocator, bytes);
            },
            else => return error.UnexpectedEvent,
        }
    }
    fn deinit(self: *SplitSink) void {
        self.reasoning.deinit(t.allocator);
        self.content.deinit(t.allocator);
    }
};

/// Feeds `chunks` through a splitter and checks both channels.
fn expectSplit(thinking: bool, chunks: []const []const u8, reasoning: []const u8, content: []const u8) !void {
    var sink: SplitSink = .{};
    defer sink.deinit();
    var splitter = text.Splitter.init(thinking, null);
    var out: [256]u8 = undefined;
    for (chunks) |c| try splitter.push(c, &out, &sink);
    try splitter.finish(&sink);
    try t.expectEqualStrings(reasoning, sink.reasoning.items);
    try t.expectEqualStrings(content, sink.content.items);
}

test "splitter: reasoning/content split follows the reference parser" {
    // Leading isspace skipped, trailing reasoning whitespace kept, delimiter across chunks.
    try expectSplit(true, &.{ "\n\n  Hello", " world\n", "</th", "ink>\n\nAnswer\n" }, "Hello world\n", "Answer\n");
    // A held partial delimiter that turns out to be text is released.
    try expectSplit(true, &.{ "a <", "b" }, "a <b", "");
    try expectSplit(true, &.{ "a </thin", "g" }, "a </thing", "");
    // The stream ends inside a partial delimiter: it is dropped.
    try expectSplit(true, &.{"abc</thi"}, "abc", "");
    try expectSplit(true, &.{ "abc", "<" }, "abc", "");
    // `<tool_call>` ends reasoning and starts content (not consumed).
    try expectSplit(true, &.{ "r <tool_", "call>x" }, "r ", "<tool_call>x");
    // Whitespace-only reasoning (all six isspace bytes) is empty.
    try expectSplit(true, &.{ " \n\t\x0b\x0c\r", "</think>  c" }, "", "c");
    // After the split, later delimiters are literal content.
    try expectSplit(true, &.{"x</think>\ny</think> <tool_call> "}, "x", "y</think> <tool_call> ");
    // Thinking off: everything after leading isspace is content, delimiters literal.
    try expectSplit(false, &.{ "\n\nx</think> y", "<tool_call>" }, "", "x</think> y<tool_call>");
    // Streaming: text that cannot begin a delimiter is released immediately.
    {
        var sink: SplitSink = .{};
        defer sink.deinit();
        var splitter = text.Splitter.init(true, null);
        var out: [64]u8 = undefined;
        try splitter.push("abc<", &out, &sink);
        try t.expectEqualStrings("abc", sink.reasoning.items);
        try splitter.push("/", &out, &sink);
        try t.expectEqualStrings("abc", sink.reasoning.items);
        try splitter.push("x", &out, &sink);
        try t.expectEqualStrings("abc</x", sink.reasoning.items);
    }
}

test "splitter: every chunking of a stream gives the one-shot result" {
    const stream = " \nThink <b> </thin </think>\n\n Answer </think> end\n";
    var one: SplitSink = .{};
    defer one.deinit();
    var s0 = text.Splitter.init(true, null);
    var out: [128]u8 = undefined;
    try s0.push(stream, &out, &one);
    try s0.finish(&one);
    try t.expectEqualStrings("Think <b> </thin ", one.reasoning.items);
    try t.expectEqualStrings("Answer </think> end\n", one.content.items);
    for (1..stream.len) |a| for (a..stream.len) |b| {
        try expectSplit(true, &.{ stream[0..a], stream[a..b], stream[b..] }, one.reasoning.items, one.content.items);
    };
}

test "generation: stops on raw text, text-level delimiters, control tokens, dropped tails" {
    var backend: FakeBackend = undefined;
    const Script = struct {
        var cross = [_]u32{ 3, 2, 1, 2, 3, 0 }; // "Hi" "\n" "</think>" "\n" "Hi"
        var spelled = [_]u32{ 3, 9, 10, 12, 4, 0 }; // "Hi" "</thi" "nk>" <control> " there"
        var partial = [_]u32{ 3, 9, 3 }; // "Hi" "</thi" ...
        var tail = [_]u32{ 3, 5, 6 }; // "Hi" "\xe4\xb8" ...
    };
    {
        // The stop string spans the reasoning/content boundary in the raw text; the
        // truncated text then ends inside a partial `</think>`, which is dropped.
        backend = .{ .script = &Script.cross, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 10, .stops = &.{"k>\nH"}, .thinking = true }, &sink);
        try t.expectEqual(session.Finish.stop, r.finish);
        try t.expectEqual(@as(u32, 5), r.completion_tokens);
        try t.expectEqualStrings("Hi\n", sink.reasoning.items);
        try t.expectEqualStrings("", sink.content.items);
    }
    {
        // A stop equal to the delimiter ends generation with complete reasoning.
        backend = .{ .script = &Script.cross, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 10, .stops = &.{"</think>"}, .thinking = true }, &sink);
        try t.expectEqual(session.Finish.stop, r.finish);
        try t.expectEqual(@as(u32, 3), r.completion_tokens);
        try t.expectEqualStrings("Hi\n", sink.reasoning.items);
        try t.expectEqualStrings("", sink.content.items);
    }
    {
        // `</think>` spelled by two ordinary tokens still splits; a control token adds nothing.
        backend = .{ .script = &Script.spelled, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 10, .thinking = true }, &sink);
        try t.expectEqual(session.Finish.stop, r.finish);
        try t.expectEqualStrings("Hi", sink.reasoning.items);
        try t.expectEqualStrings("there", sink.content.items);
    }
    {
        // Length ends inside a partial delimiter: dropped.
        backend = .{ .script = &Script.partial, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 2, .thinking = true }, &sink);
        try t.expectEqual(session.Finish.length, r.finish);
        try t.expectEqualStrings("Hi", sink.reasoning.items);
        try t.expectEqualStrings("", sink.content.items);
    }
    {
        // Length ends inside a UTF-8 sequence: the incomplete tail is dropped.
        backend = .{ .script = &Script.tail, .vocab_size = pieces.len, .ctx = 64 };
        var adv: Advancing = .{ .inner = &backend, .prompt = 1 };
        var sink: Sink = .{};
        defer sink.deinit();
        const r = try GenA.run(t.io, t.allocator, &adv, .{ .pieces = &pieces }, special, .{ .prompt = &.{3}, .max_tokens = 2 }, &sink);
        try t.expectEqual(session.Finish.length, r.finish);
        try t.expectEqualStrings("Hi", sink.content.items);
    }
}
