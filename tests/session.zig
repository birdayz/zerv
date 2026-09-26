const std = @import("std");
const session = @import("session");
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

/// A deterministic CPU "model" for the speculative loop: the logits after a history are a
/// hash of the whole history over the harmless pieces {2, 3, 4, 8}, with EOS (0) likely
/// once the history is long. `drafts` is non-null for the speculative variant: it drafts
/// the argmax continuation but corrupts every `wrong_every`-th draft.
const HashModel = struct {
    history: [512]u32 = undefined,
    len: usize = 0,
    logits: [8 * pieces.len]f32 = undefined,
    drafts: [8]u32 = undefined,
    probs: [8]f32 = undefined,
    draft_n: u32 = 0,
    wrong_every: u32 = 3,
    pending: u32 = 0,
    verify_tokens: [8]u32 = undefined,
    steps: u32 = 0,
    verifies: u32 = 0,
    ctx: u32 = 200,

    fn row(history: []const u32, out: []f32) void {
        var h: u64 = 0x9e3779b97f4a7c15;
        for (history) |x| h = (h ^ x) *% 0x100000001b3;
        @memset(out, -30);
        for ([_]u32{ 2, 3, 4, 8 }, 0..) |v, i| out[v] = @as(f32, @floatFromInt((h >> @intCast(8 * i)) & 255)) / 32.0;
        out[0] = if (history.len > 60) 9 else -30;
    }
    fn argmax(x: []const f32) u32 {
        var best: u32 = 0;
        for (x, 0..) |v, i| if (v > x[best]) {
            best = @intCast(i);
        };
        return best;
    }
    pub fn context(self: *HashModel) u32 {
        return self.ctx;
    }
    pub fn vocab(_: *HashModel) usize {
        return pieces.len;
    }
    pub fn reset(self: *HashModel) !void {
        self.len = 0;
        self.pending = 0;
    }
    pub fn prefill(self: *HashModel, tokens: []const u32) ![]const f32 {
        var logits: []const f32 = undefined;
        for (tokens) |token| logits = try self.step(token);
        return logits;
    }
    pub fn step(self: *HashModel, token: u32) ![]const f32 {
        if (self.pending != 0) return error.VerifyPending;
        self.steps += 1;
        self.history[self.len] = token;
        self.len += 1;
        row(self.history[0..self.len], self.logits[0..pieces.len]);
        return self.logits[0..pieces.len];
    }
};
/// The speculative interface over a HashModel (the non-speculative runs use the model
/// directly, so the same history function drives both).
const SpecModel = struct {
    m: *HashModel,
    n: u32,
    pub fn context(self: SpecModel) u32 {
        return self.m.context();
    }
    pub fn vocab(self: SpecModel) usize {
        return self.m.vocab();
    }
    pub fn reset(self: SpecModel) !void {
        try self.m.reset();
    }
    pub fn prefill(self: SpecModel, tokens: []const u32) ![]const f32 {
        return self.m.prefill(tokens);
    }
    pub fn step(self: SpecModel, token: u32) ![]const f32 {
        return self.m.step(token);
    }
    pub fn speculative(self: SpecModel) u32 {
        return self.n;
    }
    pub fn draft(self: SpecModel, token: u32, k: u32) ![]const u32 {
        const m = self.m;
        if (m.pending != 0) return error.VerifyPending;
        if (k == 0 or k > self.n) return error.InvalidDraft;
        var h: [512]u32 = undefined;
        @memcpy(h[0..m.len], m.history[0..m.len]);
        h[m.len] = token;
        var scratch: [pieces.len]f32 = undefined;
        for (0..k) |i| {
            HashModel.row(h[0 .. m.len + 1 + i], &scratch);
            m.draft_n += 1;
            var d = HashModel.argmax(&scratch);
            if (m.draft_n % m.wrong_every == 0) d = if (d == 3) 4 else 3;
            m.drafts[i] = d;
            h[m.len + 1 + i] = d;
        }
        return m.drafts[0..k];
    }
    pub fn draftProbs(self: SpecModel, k: u32) []const f32 {
        // Pseudo-probabilities: high for every draft but the corrupted ones.
        for (self.m.probs[0..k], 0..) |*p, i| p.* = if ((self.m.draft_n - k + 1 + @as(u32, @intCast(i))) % self.m.wrong_every == 0) 0.15 else 0.85;
        return self.m.probs[0..k];
    }
    pub fn verify(self: SpecModel, tokens: []const u32) ![]const f32 {
        const m = self.m;
        if (m.pending != 0) return error.VerifyPending;
        if (m.len + tokens.len > m.ctx) return error.ContextFull;
        m.verifies += 1;
        var h: [512]u32 = undefined;
        @memcpy(h[0..m.len], m.history[0..m.len]);
        for (tokens, 0..) |token, i| {
            h[m.len + i] = token;
            HashModel.row(h[0 .. m.len + i + 1], m.logits[i * pieces.len ..][0..pieces.len]);
        }
        @memcpy(m.verify_tokens[0..tokens.len], tokens);
        m.pending = @intCast(tokens.len);
        return m.logits[0 .. tokens.len * pieces.len];
    }
    pub fn commit(self: SpecModel, rows: u32) !void {
        const m = self.m;
        if (rows == 0 or rows > m.pending) return error.InvalidCommit;
        @memcpy(m.history[m.len..][0..rows], m.verify_tokens[0..rows]);
        m.len += rows;
        m.pending = 0;
    }
};
const GenHash = session.Generation(*HashModel, FakeTokenizer, *Sink);
const GenSpec = session.Generation(SpecModel, FakeTokenizer, *Sink);

// Block 17b gate 3 (the control flow): with sample-matching acceptance the speculative
// loop emits exactly the non-speculative output, for greedy and seeded sampling, every
// draft count, wrong drafts, EOS or the length limit inside verified rows, and context
// clamping; the model's processed history equals the non-speculative one at the end.
test "generation: speculative decoding output equals non-speculative output" {
    const Case = struct { params: sampler.Params, max_tokens: u32, ctx: u32 };
    const cases = [_]Case{
        .{ .params = .{ .temperature = 0 }, .max_tokens = 100, .ctx = 200 },
        .{ .params = .{ .temperature = 1.0, .seed = 7 }, .max_tokens = 100, .ctx = 200 },
        .{ .params = .{ .temperature = 0.7, .top_k = 3, .seed = 11, .repetition_penalty = 1.3 }, .max_tokens = 37, .ctx = 200 },
        .{ .params = .{ .temperature = 0 }, .max_tokens = 100, .ctx = 23 },
        .{ .params = .{ .temperature = 1.0, .seed = 3 }, .max_tokens = 1, .ctx = 200 },
        .{ .params = .{ .temperature = 1.0, .seed = 5 }, .max_tokens = 2, .ctx = 200 },
    };
    const prompt = [_]u32{ 3, 4, 2, 3 };
    for (cases) |c| {
        var plain: HashModel = .{ .ctx = c.ctx };
        var plain_sink: Sink = .{};
        defer plain_sink.deinit();
        const want = try GenHash.run(t.io, t.allocator, &plain, .{ .pieces = &pieces }, special, .{ .prompt = &prompt, .max_tokens = c.max_tokens, .params = c.params }, &plain_sink);
        for (1..5) |n| for ([_]u32{ 2, 3, 1000 }) |wrong| for ([_]bool{ false, true }) |adaptive| {
            var m: HashModel = .{ .ctx = c.ctx, .wrong_every = wrong };
            var sink: Sink = .{};
            defer sink.deinit();
            var policy: session.spec.Policy = .init();
            const got = try GenSpec.run(t.io, t.allocator, .{ .m = &m, .n = @intCast(n) }, .{ .pieces = &pieces }, special, .{ .prompt = &prompt, .max_tokens = c.max_tokens, .params = c.params, .spec_policy = if (adaptive) &policy else null }, &sink);
            try t.expectEqual(want.finish, got.finish);
            try t.expectEqual(want.completion_tokens, got.completion_tokens);
            try t.expectEqualStrings(plain_sink.content.items, sink.content.items);
            try t.expectEqualSlices(u32, plain.history[0..plain.len], m.history[0..m.len]);
            try t.expectEqual(@as(u32, 0), m.pending);
            if (c.max_tokens > 2 and c.ctx > 100 and !adaptive) try t.expect(m.verifies > 0 and m.verifies < want.completion_tokens);
            // Counters: every verify is counted; acceptance never exceeds what was verified
            // or drafted; correct greedy drafts are all accepted.
            try t.expectEqual(m.verifies, got.spec.verifies);
            try t.expect(got.spec.accepted <= got.spec.verified and got.spec.verified <= got.spec.drafted);
            try t.expect(got.spec.accepted + got.spec.verifies <= got.completion_tokens);
            if (wrong == 1000 and c.params.temperature == 0) try t.expectEqual(got.spec.verified, got.spec.accepted);
            if (wrong == 2 and c.max_tokens > 2 and c.ctx > 100 and !adaptive and n > 1) try t.expect(got.spec.accepted < got.spec.verified);
        };
    }
}

test "speculative verify-count policy: acceptance bins, cost scaling and choice" {
    const Policy = session.spec.Policy;
    var p: Policy = .init();
    // Priors: the bin center.
    try t.expectApproxEqAbs(@as(f64, 0.95), p.acceptance(0.99), 1e-12);
    try t.expectApproxEqAbs(@as(f64, 0.05), p.acceptance(0.01), 1e-12);
    try t.expectApproxEqAbs(@as(f64, 0.05), p.acceptance(std.math.nan(f32)), 1e-12);
    // A step verifying 3 drafts with 1 accepted: bins of drafts 1 (accepted) and 2 (rejected).
    p.observe(&.{ 0.95, 0.95, 0.95 }, 1);
    try t.expectApproxEqAbs((2 * 0.95 + 1) / 4.0, p.acceptance(0.95), 1e-12);
    // Costs: verify of n rows grows; confident drafts are all verified, doubtful ones not.
    var q: Policy = .init();
    q.timeDraft(4_000_000);
    q.timeCommit(500_000);
    for (1..6) |n| q.timeVerify(n, 20_000_000 + 3_000_000 * (n - 1));
    try t.expectEqual(@as(u32, 4), q.choose(&.{ 0.99, 0.99, 0.99, 0.99 }));
    try t.expectEqual(@as(u32, 0), q.choose(&.{ 0.01, 0.99, 0.99, 0.99 }));
    try t.expectEqual(@as(u32, 1), q.choose(&.{ 0.99, 0.05, 0.99, 0.99 }));
    // Unmeasured counts scale from the nearest measured one by the relative prior.
    var r: Policy = .init();
    r.timeVerify(5, 32_000_000);
    try t.expectEqual(@as(u32, 3), r.choose(&.{ 0.99, 0.99, 0.99, 0.2 }));
}
