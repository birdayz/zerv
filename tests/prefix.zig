//! Prefix-cache policy (docs/specs/prefix-cache.md): unit cases, and a randomized
//! differential test against uncached runs with a fake backend whose logits depend on
//! the whole KV content and the recurrent state.
const std = @import("std");
const zerv = @import("zerv");
const session = zerv.session;
const prefix = session.prefix;
const t = std.testing;

const ctx = 512;
const vocab_size = 8;
const eos = 0;
const boundary = 1; // `<|im_start|>`
const pieces = [vocab_size][]const u8{ "", "B", "a", "b", "c", "d", "e", "f" };

fn mix(h: u64, x: u64) u64 {
    return std.hash.Wyhash.hash(h, std.mem.asBytes(&x));
}

/// KV = the token at each position; recurrent state = a hash of the processed sequence.
/// Logits are a function of both, so any reuse of the wrong state changes the output.
const Fake = struct {
    kv: [ctx]u32 = @splat(0xdead),
    rec: u64 = 0,
    pos: u32 = 0,
    snaps: [8]struct { rec: u64, pos: u32, kv: u64 } = undefined,
    logits: [vocab_size]f32 = undefined,
    resets: u32 = 0,
    restores: u32 = 0,
    saves: u32 = 0,
    processed: u64 = 0,
    /// Fail the processing of the n-th token from now (backend error injection).
    fail_after: ?u64 = null,

    fn kvHash(self: *const Fake, n: u32) u64 {
        var h: u64 = 17;
        for (self.kv[0..n]) |x| h = mix(h, x);
        return h;
    }
    pub fn context(_: *Fake) u32 {
        return ctx;
    }
    pub fn vocab(_: *Fake) usize {
        return pieces.len;
    }
    pub fn reset(self: *Fake) !void {
        self.rec = 0;
        self.pos = 0;
        self.resets += 1;
    }
    pub fn step(self: *Fake, token: u32) ![]const f32 {
        if (self.fail_after) |*n| {
            if (n.* == 0) return error.DeviceLost;
            n.* -= 1;
        }
        if (self.pos >= ctx) return error.ContextFull;
        self.kv[self.pos] = token;
        self.pos += 1;
        self.rec = mix(self.rec, token);
        self.processed += 1;
        const h = mix(self.rec, self.kvHash(self.pos));
        for (&self.logits, 0..) |*l, i| l.* = @floatFromInt(mix(h, i) % 1000);
        // End sometimes, so generations have varied lengths.
        if (h % 5 == 0) self.logits[eos] = 2000;
        return &self.logits;
    }
    pub fn prefill(self: *Fake, tokens: []const u32) ![]const f32 {
        if (tokens.len == 0) return error.InvalidToken;
        var l: []const f32 = undefined;
        for (tokens) |x| l = try self.step(x);
        return l;
    }
    pub fn saveSnapshot(self: *Fake, slot: u32) !void {
        self.snaps[slot] = .{ .rec = self.rec, .pos = self.pos, .kv = self.kvHash(self.pos) };
        self.saves += 1;
    }
    pub fn loadSnapshot(self: *Fake, slot: u32, position: u32) !void {
        const s = self.snaps[slot];
        // The caller's guarantee: the slot was saved at `position` and the KV below it is
        // unchanged since.
        if (s.pos != position) return error.SnapshotPositionMismatch;
        if (self.kvHash(position) != s.kv) return error.KvChangedUnderSnapshot;
        self.rec = s.rec;
        self.pos = position;
        self.restores += 1;
    }
};

const Tok = struct {
    pub fn outputPiece(_: Tok, id: u32) ![]const u8 {
        return pieces[id];
    }
};

const Sink = struct {
    out: std.ArrayList(u8) = .empty,
    fail_after: ?usize = null,
    pub fn emit(self: *Sink, event: session.Event) !void {
        switch (event) {
            .content, .reasoning => |b| {
                if (self.fail_after) |n| if (self.out.items.len >= n) return error.ClientGone;
                try self.out.appendSlice(t.allocator, b);
            },
            else => return error.UnexpectedEvent,
        }
    }
};

const Gen = session.Generation(*Fake, Tok, *Sink);
const special: session.Special = .{ .eos = &.{eos} };

const Outcome = struct { text: []u8, result: session.Result };

fn run(backend: *Fake, cache: ?*prefix.Cache, prompt: []const u32, max_tokens: u32) !Outcome {
    var sink: Sink = .{};
    errdefer sink.out.deinit(t.allocator);
    const r = try Gen.run(t.io, t.allocator, backend, Tok{}, special, .{ .prompt = prompt, .max_tokens = max_tokens, .params = .{ .temperature = 0 }, .cache = cache }, &sink);
    return .{ .text = try sink.out.toOwnedSlice(t.allocator), .result = r };
}

fn initCache(slots: u32, spacing: u32) !prefix.Cache {
    return prefix.Cache.init(t.allocator, ctx, .{ .slots = slots, .boundary = boundary, .spacing = spacing });
}

test "begin: reset, keep, restore and the one-token minimum" {
    var cache = try initCache(4, 100);
    defer cache.deinit(t.allocator);
    // Empty: reset.
    const p1 = [_]u32{ 1, 2, 3, 1, 4, 5, 1, 6 };
    var b = cache.begin(&p1);
    try t.expectEqual(prefix.Outcome.reset, b.outcome);
    try t.expectEqual(@as(u32, 0), b.start);
    // Prefill with snapshots before each boundary after the start: 3 and 6.
    var buf: [prefix.max_points]u32 = undefined;
    const pts = cache.points(&p1, 0, &buf);
    try t.expectEqualSlices(u32, &.{ 3, 6 }, pts);
    cache.record(p1[0..3]);
    _ = cache.claim();
    cache.record(p1[3..6]);
    _ = cache.claim();
    cache.record(p1[6..]);
    cache.record(&.{ 7, 7 }); // two generated tokens were processed
    // Extends the history exactly: keep.
    b = cache.begin(&(p1 ++ [_]u32{ 7, 7, 0, 1, 2 }));
    try t.expectEqual(prefix.Outcome.keep, b.outcome);
    try t.expectEqual(@as(u32, 10), b.start);
    cache.record(&.{ 0, 1, 2 });
    // Diverges at 7 (after the last snapshot at 6): restore 6.
    b = cache.begin(&(p1[0..7].* ++ [_]u32{ 5, 5 }));
    try t.expectEqual(prefix.Outcome.restore, b.outcome);
    try t.expectEqual(@as(u32, 6), b.start);
    try t.expectEqual(@as(u32, 6), cache.len);
    // The identical prompt again: at least one token is processed, so restore <= n - 1.
    cache.record(&.{6});
    b = cache.begin(p1[0..7]);
    try t.expectEqual(prefix.Outcome.restore, b.outcome);
    try t.expectEqual(@as(u32, 6), b.start);
    // A prompt that ends exactly at a snapshot position (6): the snapshot at 6 would leave
    // nothing to process, so the one before (3) is used.
    b = cache.begin(p1[0..6]);
    try t.expectEqual(prefix.Outcome.restore, b.outcome);
    try t.expectEqual(@as(u32, 3), b.start);
    cache.record(p1[3..6]);
    _ = cache.claim(); // at 6 again
    // Diverges before every snapshot position except 3: restore 3; later ones dropped.
    b = cache.begin(&.{ 1, 2, 3, 1, 9 });
    try t.expectEqual(@as(u32, 3), b.start);
    var held: [8]u32 = undefined;
    try t.expectEqualSlices(u32, &.{3}, cache.positions(&held));
    // Diverges at 0: reset, everything dropped.
    b = cache.begin(&.{ 5, 5 });
    try t.expectEqual(prefix.Outcome.reset, b.outcome);
    try t.expectEqual(@as(usize, 0), cache.positions(&held).len);
}

test "points: final boundary always, optional ones spaced" {
    var cache = try initCache(8, 10);
    defer cache.deinit(t.allocator);
    var buf: [prefix.max_points]u32 = undefined;
    // Boundaries at 0, 4, 12, 15, 30, 33. From start 0 with no snapshot: the first
    // boundary after the start (4) is taken, then those >= 10 after the last taken
    // (15, not 12: 12 - 4 < 10; 30 is optional too but 33 is final).
    var p: [40]u32 = @splat(5);
    for ([_]usize{ 0, 4, 12, 15, 30, 33 }) |i| p[i] = boundary;
    try t.expectEqualSlices(u32, &.{ 4, 15, 30, 33 }, cache.points(&p, 0, &buf));
    // Starting at 15 with an existing snapshot at 15: 30 is 15 after it, 33 is final.
    cache.record(p[0..15]);
    _ = cache.claim();
    try t.expectEqualSlices(u32, &.{ 30, 33 }, cache.points(&p, 15, &buf));
    // No boundary after the start: no points; a boundary at the start is not a point.
    try t.expectEqual(@as(usize, 0), cache.points(&p, 33, &buf).len);
}

test "claim: slots are reused, eviction thins evenly and keeps the newest" {
    var cache = try initCache(3, 1);
    defer cache.deinit(t.allocator);
    var held: [8]u32 = undefined;
    const tokens: [100]u32 = @splat(4);
    var used: u64 = 0;
    for ([_]u32{ 10, 12, 50 }) |pos| {
        cache.record(tokens[0 .. pos - cache.len]);
        used |= @as(u64, 1) << @intCast(cache.claim());
    }
    try t.expectEqual(@as(u64, 0b111), used);
    // Full: adding 90. Removing 10 leaves the gap 0..12 = 12, removing 12 leaves 10..50 =
    // 40, removing 50 leaves 12..90 = 78: 10 goes.
    cache.record(tokens[0 .. 90 - cache.len]);
    _ = cache.claim();
    try t.expectEqualSlices(u32, &.{ 12, 50, 90 }, cache.positions(&held));
    // Adding 95: removing 12 leaves 50, 50 leaves 78, 90 leaves 50..95 = 45: 90 goes.
    cache.record(tokens[0 .. 95 - cache.len]);
    _ = cache.claim();
    try t.expectEqualSlices(u32, &.{ 12, 50, 95 }, cache.positions(&held));
    // A far-away early snapshot (the system prompt) survives a run of close ones.
    var agent = try initCache(3, 1);
    defer agent.deinit(t.allocator);
    const long: [400]u32 = @splat(4);
    for ([_]u32{ 200, 220, 240, 260, 280 }) |pos| {
        agent.record(long[0 .. pos - agent.len]);
        _ = agent.claim();
    }
    try t.expectEqualSlices(u32, &.{ 200, 240, 280 }, agent.positions(&held));
}

test "backend failures invalidate the cache; client disconnects keep it consistent" {
    var backend: Fake = .{};
    var cache = try initCache(4, 100);
    defer cache.deinit(t.allocator);
    const prompt = [_]u32{ 1, 2, 3, 1, 4, 5, 1, 6 };
    var ok = try run(&backend, &cache, &prompt, 5);
    t.allocator.free(ok.text);
    try t.expect(cache.len > prompt.len);
    // Device failure during prefill: error, cache forgotten, next request resets.
    backend.fail_after = 2;
    try t.expectError(error.DeviceLost, run(&backend, &cache, &(prompt ++ [_]u32{ 2, 2 }), 5));
    try t.expectEqual(@as(u32, 0), cache.len);
    backend.fail_after = null;
    const resets = backend.resets;
    ok = try run(&backend, &cache, &prompt, 5);
    t.allocator.free(ok.text);
    try t.expectEqual(resets + 1, backend.resets);
    // Device failure during decode: same.
    backend.fail_after = 3;
    try t.expectError(error.DeviceLost, run(&backend, &cache, &(prompt ++ [_]u32{ 3, 1, 4 }), 20));
    try t.expectEqual(@as(u32, 0), cache.len);
    backend.fail_after = null;
    // A client disconnect: the history is exactly what the backend processed.
    ok = try run(&backend, &cache, &prompt, 5);
    t.allocator.free(ok.text);
    var sink: Sink = .{ .fail_after = 0 };
    defer sink.out.deinit(t.allocator);
    const p2 = prompt ++ [_]u32{ 4, 4, 1, 2 };
    try t.expectError(error.ClientGone, Gen.run(t.io, t.allocator, &backend, Tok{}, special, .{ .prompt = &p2, .max_tokens = 20, .params = .{ .temperature = 0 }, .cache = &cache }, &sink));
    try t.expectEqual(backend.pos, cache.len);
    try t.expectEqualSlices(u32, backend.kv[0..backend.pos], cache.history[0..cache.len]);
}

test "cached request sequences produce exactly the uncached outputs (randomized)" {
    var rng: std.Random.DefaultPrng = .init(0x5eed);
    const r = rng.random();
    for ([_][2]u32{ .{ 1, 4 }, .{ 2, 16 }, .{ 3, 6 }, .{ 8, 30 } }) |cfg| {
        var backend: Fake = .{};
        var cache = try initCache(cfg[0], cfg[1]);
        defer cache.deinit(t.allocator);
        var convo: std.ArrayList(u32) = .empty;
        defer convo.deinit(t.allocator);
        var reused: u64 = 0;
        var outcomes: [3]u32 = @splat(0);
        for (0..300) |_| {
            // Build the next prompt from the conversation so far, as clients do.
            const choice = r.uintLessThan(u32, 10);
            if (choice == 0 or convo.items.len > 300) {
                convo.clearRetainingCapacity(); // new conversation, sometimes same "system"
                if (r.boolean()) try convo.appendSlice(t.allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 2, 3, 4 });
            } else if (choice == 1 and convo.items.len > 4) {
                convo.shrinkRetainingCapacity(r.uintLessThan(usize, convo.items.len)); // edit history
            }
            // Sometimes the prompt ends right before a message boundary (a snapshot
            // position), with no generation prompt.
            if (choice == 2 and convo.items.len > 0) {
                var cut = convo.items.len - 1;
                while (cut > 0 and convo.items[cut] != boundary) cut -= 1;
                if (cut > 0) {
                    var cold: Fake = .{};
                    const want = try run(&cold, null, convo.items[0..cut], 3);
                    defer t.allocator.free(want.text);
                    const got = try run(&backend, &cache, convo.items[0..cut], 3);
                    defer t.allocator.free(got.text);
                    try t.expectEqualStrings(want.text, got.text);
                    try t.expect(got.result.cached_tokens < cut);
                }
            }
            // A new message, then the generation prompt.
            try convo.append(t.allocator, boundary);
            for (0..r.uintLessThan(u32, 12)) |_| try convo.append(t.allocator, 2 + r.uintLessThan(u32, 6));
            const gen_start = convo.items.len;
            try convo.appendSlice(t.allocator, &.{ boundary, 2, 3 });
            const max_tokens = 1 + r.uintLessThan(u32, 8);

            var cold: Fake = .{};
            const want = try run(&cold, null, convo.items, max_tokens);
            defer t.allocator.free(want.text);
            const got = try run(&backend, &cache, convo.items, max_tokens);
            defer t.allocator.free(got.text);
            try t.expectEqualStrings(want.text, got.text);
            try t.expectEqual(want.result.completion_tokens, got.result.completion_tokens);
            try t.expectEqual(want.result.finish, got.result.finish);
            try t.expect(got.result.cached_tokens < convo.items.len);
            reused += got.result.cached_tokens;
            outcomes[@intFromEnum(got.result.cache_outcome)] += 1;
            // The client sends the answer back, either as generated (the history can
            // extend the live state) or altered (restore at the generation prompt).
            if (r.boolean()) {
                for (got.text) |c| try convo.append(t.allocator, if (c == 'B') boundary else 2 + @as(u32, c - 'a'));
                try convo.append(t.allocator, eos); // the template's end-of-message token
            } else {
                convo.shrinkRetainingCapacity(gen_start);
                try convo.appendSlice(t.allocator, &.{ boundary, 2, 3, 7 });
            }
        }
        // Every path was exercised, and the cache actually saved work.
        try t.expect(outcomes[0] > 0 and outcomes[1] > 0 and outcomes[2] > 0);
        try t.expect(reused > 1000);
        try t.expect(backend.restores > 0 and backend.saves > 0);
    }
}
