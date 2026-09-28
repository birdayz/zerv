//! Continuous batching (src/serve/batcher.zig, docs/specs/concurrent.md "18c design") with a
//! fake backend: every returned logits row must be exactly what the sequence computes
//! alone, while generations join and leave, hold rows, pause, get canceled, see errors
//! and prefill in packed chunks.
const std = @import("std");
const serve = @import("serve");
const batcher = serve.batcher;
const t = std.testing;

const V = 16;
const max_slots = 8;

fn mix(h: u64, token: u32) u64 {
    var z = h +% 0x9e3779b97f4a7c15 +% token;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
fn logitsFor(h: u64, out: []f32) void {
    for (out, 0..) |*v, k| v.* = @floatFromInt(mix(h, @intCast(k)) & 0xffff);
}
fn pick(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits, 0..) |v, k| if (v > logits[best]) {
        best = k;
    };
    return @intCast(best);
}

/// Deterministic model: a slot's logits are a function of its token history. Prefill takes
/// 3 tokens per chunk, each chunk in `segments` calls (the tokens apply at the chunk's last
/// one), as the segmented model does. The output buffers are shared and overwritten by every
/// call, as the real model's are. It enforces the batcher's contract: while a chunk is in
/// flight no other slot's prefill or reset runs and no batch contains that slot (and, with
/// `no_batch_in_chunk`, no batch runs at all).
const Fake = struct {
    io: std.Io,
    hist: [max_slots]u64 = @splat(0),
    rows_out: [max_slots * V]f32 = @splat(0),
    prefill_out: [batcher.max_pack * V]f32 = @splat(0),
    delay_ns: u64 = 0,
    unit_delay_ns: u64 = 0,
    fail_batches: u32 = 0,
    sizes: [max_slots + 1]u32 = @splat(0),
    calls: u32 = 0,
    segments: u32 = 1,
    /// The chunk in flight: its slots, their chunk tokens, and their remaining lengths.
    flight: ?usize = null,
    /// `flight` for other threads (0: none).
    members: std.atomic.Value(usize) = .init(0),
    fslots: [batcher.max_pack]u32 = undefined,
    fcount: [batcher.max_pack]usize = undefined,
    ftotal: [batcher.max_pack]usize = undefined,
    ftokens: [batcher.max_pack][3]u32 = undefined,
    pack_cap: usize = 1,
    pool: usize = 0,
    page: usize = 4,
    held: [max_slots]usize = @splat(0),
    /// Positions processed per slot; host swap store (pages; 0: none) and what each
    /// swapped-out slot holds there.
    pos: [max_slots]usize = @splat(0),
    host_pool: usize = 0,
    host_held: [max_slots]usize = @splat(0),
    swapped: [max_slots]bool = @splat(false),
    swap_outs: u32 = 0,
    /// Checkpoints (`checkpoint`/`begin`): prefix tokens and the history value after them.
    ck_tokens: [8][16]u32 = undefined,
    ck_len: [8]usize = @splat(0),
    ck_hist: [8]u64 = undefined,
    ck_count: usize = 0,
    restores: u32 = 0,
    admits: u32 = 0,
    packs: u32 = 0,
    seg: u32 = 0,
    no_batch_in_chunk: bool = false,
    aborts: u32 = 0,
    violations: u32 = 0,
    /// Slots in the order their prefill completed.
    done_order: [64]u32 = undefined,
    done_count: usize = 0,
    /// A row whose token is `bad_token` cannot run (as a slot the model refuses).
    bad_token: u32 = std.math.maxInt(u32),
    /// The reset of this slot waits for `resume_reset` (it signals `in_reset` first): holds
    /// the scheduler inside a backend call while a test queues operations.
    block_reset: ?u32 = null,
    in_reset: std.atomic.Value(bool) = .init(false),
    resume_reset: std.Io.Event = .unset,

    io_pause_poll: bool = false,
    io_in_poll: std.Io.Event = .unset,
    io_resume_poll: std.Io.Event = .unset,
    release_calls: [max_slots]u32 = @splat(0),
    reset_release_seen: [max_slots]u32 = @splat(0),
    io_kind: enum { off, begin, checkpoint } = .off,
    io_live: ?u32 = null,
    io_tokens: []const u32 = &.{},
    io_hash: u64 = 0,
    io_cancel_polls: u32 = 0,
    io_entered: std.atomic.Value(bool) = .init(false),
    io_ready: std.atomic.Value(bool) = .init(false),
    fn pending(self: *Fake, slot: u32, tokens: []const u32) anyerror!void {
        self.io_live = slot;
        self.io_tokens = tokens;
        self.io_hash = 7;
        for (tokens) |token| self.io_hash = mix(self.io_hash, token);
        self.io_entered.store(true, .release);
        return error.PendingIo;
    }
    pub fn pollCache(self: *Fake, slot: u32, cancel: bool) !batcher.CachePoll {
        if (self.io_pause_poll) {
            self.io_pause_poll = false;
            self.io_in_poll.set(self.io);
            self.io_resume_poll.waitUncancelable(self.io);
        }
        if (self.flight != null or self.io_live != slot) return error.InvalidState;
        var h: u64 = 7;
        for (self.io_tokens) |token| h = mix(h, token);
        if (h != self.io_hash) return error.BorrowedTokensChanged;
        if (cancel) self.io_cancel_polls += 1;
        if (self.io_cancel_polls < 3 and !self.io_ready.load(.acquire)) return .{};
        self.io_live = null;
        self.io_tokens = &.{};
        if (cancel) return error.Canceled;
        return .{ .done = true, .progressed = true, .position = 2 };
    }

    pub fn reset(self: *Fake, slot: u32) !void {
        self.reset_release_seen[slot] = self.release_calls[slot];
        if (self.flight != null) self.violations += 1;
        if (self.block_reset != null and self.block_reset.? == slot) {
            self.in_reset.store(true, .release);
            self.resume_reset.waitUncancelable(self.io);
        }
        self.held[slot] = 0;
        self.host_held[slot] = 0;
        self.swapped[slot] = false;
        self.pos[slot] = 0;
        self.hist[slot] = 7;
    }
    pub fn begin(self: *Fake, slot: u32, prompt: []const u32) !u32 {
        try self.reset(slot);
        if (self.io_kind == .begin and slot == 0) try self.pending(slot, prompt);
        var best: ?usize = null;
        for (0..self.ck_count) |i| {
            const n = self.ck_len[i];
            if (n >= prompt.len or !std.mem.eql(u32, self.ck_tokens[i][0..n], prompt[0..n])) continue;
            if (best == null or n > self.ck_len[best.?]) best = i;
        }
        const i = best orelse return 0;
        const want = (self.ck_len[i] + self.page - 1) / self.page;
        if (self.pool != 0 and self.usedPages() + want > self.pool) return 0; // no room: cold
        self.held[slot] = if (self.pool != 0) want else 0;
        self.hist[slot] = self.ck_hist[i];
        self.pos[slot] = self.ck_len[i];
        self.restores += 1;
        return @intCast(self.ck_len[i]);
    }
    pub fn checkpoint(self: *Fake, slot: u32, prefix: []const u32) !void {
        var h: u64 = 7;
        for (prefix) |token| h = mix(h, token);
        if (h != self.hist[slot] or prefix.len != self.pos[slot] or prefix.len > 16) {
            self.violations += 1; // the slot's state is not the state after `prefix`
            return;
        }
        if (self.ck_count == self.ck_len.len) return;
        @memcpy(self.ck_tokens[self.ck_count][0..prefix.len], prefix);
        self.ck_len[self.ck_count] = prefix.len;
        self.ck_hist[self.ck_count] = h;
        self.ck_count += 1;
        if (self.io_kind == .checkpoint and slot == 0) try self.pending(slot, prefix);
    }
    pub fn packFits(self: *Fake, remaining: []const usize) bool {
        return remaining.len <= self.pack_cap;
    }
    /// Starting a chunk copies each item's next 3 tokens (the model copies them into its io
    /// block); later units never read the prompts.
    pub fn prefillUnit(self: *Fake, items: []const batcher.Item) !batcher.Unit {
        if (self.flight == null) {
            if (items.len == 0 or items.len > self.pack_cap) {
                self.violations += 1;
                return error.FakeBadPack;
            }
            self.flight = items.len;
            self.members.store(items.len, .release);
            for (items) |item| if (self.pool != 0 and self.held[item.slot] * self.page < self.pos[item.slot] + @min(3, item.tokens.len)) {
                self.violations += 1; // a chunk without memory for its positions
            };
            for (items, 0..) |item, i| {
                for (item.tokens) |token| if (token == poison) {
                    self.violations += 1;
                };
                const n = @min(3, item.tokens.len);
                self.fslots[i] = item.slot;
                self.fcount[i] = n;
                self.ftotal[i] = item.tokens.len;
                @memcpy(self.ftokens[i][0..n], item.tokens[0..n]);
            }
            if (items.len > 1) self.packs += 1;
        } else if (items.len != 0) self.violations += 1;
        if (self.unit_delay_ns > 0) try self.io.sleep(.fromNanoseconds(self.unit_delay_ns), .awake);
        self.seg += 1;
        if (self.seg < self.segments) return .{ .done = false };
        self.seg = 0;
        const k = self.flight.?;
        self.flight = null;
        self.members.store(0, .release);
        var unit: batcher.Unit = .{ .done = true, .logits = self.prefill_out[0 .. k * V] };
        for (0..k) |i| {
            const slot = self.fslots[i];
            for (self.ftokens[i][0..self.fcount[i]]) |token| self.hist[slot] = mix(self.hist[slot], token);
            self.pos[slot] += self.fcount[i];
            unit.consumed[i] = self.fcount[i];
            logitsFor(self.hist[slot], self.prefill_out[i * V ..][0..V]);
            if (self.fcount[i] == self.ftotal[i] and self.done_count < self.done_order.len) {
                self.done_order[self.done_count] = slot;
                self.done_count += 1;
            }
        }
        return unit;
    }
    fn inFlight(self: *const Fake, slot: u32) bool {
        const k = self.flight orelse return false;
        for (self.fslots[0..k]) |s| if (s == slot) return true;
        return false;
    }
    /// Memory: `pool` pages of `page` positions (0 pool = unlimited); a slot holds the pages
    /// it was admitted for until `release` or its reset.
    pub fn admit(self: *Fake, slot: u32, tokens: usize) bool {
        if (self.pool == 0) return true;
        const want = (tokens + self.page - 1) / self.page;
        if (want <= self.held[slot]) return true;
        var used: usize = 0;
        for (self.held) |h| used += h;
        if (used + want - self.held[slot] > self.pool) return false;
        self.held[slot] = want;
        self.admits += 1;
        return true;
    }
    pub fn release(self: *Fake, slot: u32) void {
        self.release_calls[slot] += 1;
        if (self.io_live == slot) self.violations += 1;
        self.held[slot] = 0;
        self.host_held[slot] = 0;
        self.swapped[slot] = false;
    }
    fn usedPages(self: *const Fake) usize {
        var n: usize = 0;
        for (self.held) |h| n += h;
        return n;
    }
    pub fn grow(self: *Fake, slot: u32) !bool {
        if (self.pool == 0) return true;
        if (self.swapped[slot]) return error.FakeSwapped;
        const want = (self.pos[slot] + 1 + self.page - 1) / self.page;
        if (want <= self.held[slot]) return true;
        if (self.usedPages() + want - self.held[slot] > self.pool) return false;
        self.held[slot] = want;
        return true;
    }
    pub fn swapOut(self: *Fake, slot: u32) !bool {
        if (self.swapped[slot] or self.held[slot] == 0) self.violations += 1;
        var host: usize = 0;
        for (self.host_held) |h| host += h;
        if (host + self.held[slot] > self.host_pool) return false;
        self.host_held[slot] = self.held[slot];
        self.held[slot] = 0;
        self.swapped[slot] = true;
        self.swap_outs += 1;
        return true;
    }
    pub fn swapIn(self: *Fake, slot: u32, spare: u32) !bool {
        if (!self.swapped[slot]) return error.FakeNotSwapped;
        if (self.usedPages() + self.host_held[slot] + spare > self.pool) return false;
        self.held[slot] = self.host_held[slot];
        self.host_held[slot] = 0;
        self.swapped[slot] = false;
        return true;
    }
    pub fn checkRow(self: *Fake, row: batcher.Row) !void {
        if (row.token == self.bad_token) return error.FakeBadRow;
    }
    pub fn abortChunk(self: *Fake) void {
        self.aborts += 1;
        self.flight = null;
        self.members.store(0, .release);
        self.seg = 0;
    }
    pub fn decodeBatch(self: *Fake, rows: []const batcher.Row) ![]const f32 {
        self.calls += 1;
        if (self.flight != null) {
            if (self.no_batch_in_chunk) self.violations += 1;
            for (rows) |row| if (self.inFlight(row.slot)) {
                self.violations += 1;
            };
        }
        if (self.delay_ns > 0) try self.io.sleep(.fromNanoseconds(self.delay_ns), .awake);
        if (self.fail_batches > 0) {
            self.fail_batches -= 1;
            return error.FakeFailure;
        }
        self.sizes[rows.len] += 1;
        for (rows) |row| if (self.pool != 0 and (self.swapped[row.slot] or self.held[row.slot] * self.page < self.pos[row.slot] + 1)) {
            self.violations += 1; // a row without its memory
        };
        for (rows, 0..) |row, r| {
            self.pos[row.slot] += 1;
            self.hist[row.slot] = mix(self.hist[row.slot], row.token);
            logitsFor(self.hist[row.slot], self.rows_out[r * V ..][0..V]);
        }
        return self.rows_out[0 .. rows.len * V];
    }
};
const B = batcher.Batcher(*Fake);
/// Written over a canceled generation's prompt: the scheduler must never read it.
const poison: u32 = 0xaaaa_aaaa;

const Plan = struct {
    prompt: []const u32,
    steps: u32,
    /// Keep the logits this long before sampling (other batches must not overwrite them).
    hold_ns: u64 = 0,
    /// Pause after sampling (a slow client).
    pause_ns: u64 = 0,
};

/// One generation against the batcher; checks every logits row against its own solo
/// computation and returns the tokens.
fn generation(b: *B, gate: *std.Io.Semaphore, plan: Plan, out: []u32) !void {
    const io = b.io;
    try gate.wait(io);
    defer gate.post(io);
    const slot = try b.join();
    defer b.leave(slot);
    var h: u64 = 7;
    try b.reset(slot);
    for (plan.prompt) |token| h = mix(h, token);
    var logits = try b.prefill(slot, plan.prompt);
    var want: [V]f32 = undefined;
    for (0..plan.steps) |i| {
        if (plan.hold_ns > 0) try io.sleep(.fromNanoseconds(plan.hold_ns), .awake);
        logitsFor(h, &want);
        if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
        const token = pick(logits);
        b.sampled(slot);
        out[i] = token;
        if (plan.pause_ns > 0) try io.sleep(.fromNanoseconds(plan.pause_ns), .awake);
        h = mix(h, token);
        logits = try b.step(slot, token);
    }
    logitsFor(h, &want);
    if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
}

fn soloTokens(prompt: []const u32, steps: u32, out: []u32) void {
    var h: u64 = 7;
    for (prompt) |token| h = mix(h, token);
    var logits: [V]f32 = undefined;
    for (0..steps) |i| {
        logitsFor(h, &logits);
        out[i] = pick(&logits);
        h = mix(h, out[i]);
    }
}

fn runAll(b: *B, gate: *std.Io.Semaphore, plans: []const Plan, outs: [][64]u32, results: []anyerror!void) !void {
    const io = b.io;
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (plans, outs, results) |plan, *out, *result| try group.concurrent(io, store, .{ result, b, gate, plan, out });
    try group.await(io);
}
fn store(result: *anyerror!void, b: *B, gate: *std.Io.Semaphore, plan: Plan, out: *[64]u32) void {
    result.* = generation(b, gate, plan, out);
}

test "batcher: concurrent generations get exactly their solo logits, with joins, leaves and holds" {
    // Whole chunks (the previous behaviour), then segmented chunks under each stall policy
    // and order (docs/specs/concurrent.md, "18c.2 design").
    try exactness(1, .chunk, .fifo);
    try exactness(4, .chunk, .fifo);
    try exactness(4, .{ .ns = 0 }, .shortest);
    try exactness(4, .{ .ns = 300_000 }, .shortest);
}

fn exactness(segments: u32, stall: batcher.Stall, order: batcher.Order) !void {
    const io = t.io;
    var fake: Fake = .{ .io = io, .delay_ns = 200_000, .segments = segments, .unit_delay_ns = if (segments > 1) 100_000 else 0, .no_batch_in_chunk = stall == .chunk };
    var b = try B.init(io, &fake, .{ .slots = 4, .vocab = V, .stall = stall, .order = order });
    var task = try io.concurrent(B.run, .{&b});
    var prompts: [12][10]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x18c);
    for (&prompts) |*p| for (p) |*x| {
        x.* = prng.random().intRangeLessThan(u32, 0, 1000);
    };
    // 12 generations over 4 slots (a semaphore of 4, as the HTTP server gates them):
    // different prompt and output lengths, some holding their rows, some pausing.
    var plans: [12]Plan = undefined;
    for (&plans, 0..) |*plan, i| plan.* = .{
        .prompt = prompts[i][0 .. 1 + i % 10],
        .steps = @intCast(5 + (i * 7) % 40),
        .hold_ns = if (i % 4 == 1) 1_000_000 else 0,
        .pause_ns = if (i % 5 == 2) 500_000 else 0,
    };
    var outs: [12][64]u32 = undefined;
    var results: [12]anyerror!void = undefined;
    var gate: std.Io.Semaphore = .{ .permits = 4 };
    try runAll(&b, &gate, &plans, &outs, &results);
    b.stop();
    task.await(io);
    var total: u64 = 0;
    for (plans, outs, results) |plan, out, result| {
        try result;
        var want: [64]u32 = undefined;
        soloTokens(plan.prompt, plan.steps, &want);
        try t.expectEqualSlices(u32, want[0..plan.steps], out[0..plan.steps]);
        total += plan.steps;
    }
    // Every step ran exactly once, and steps were batched.
    try t.expectEqual(total, b.stats.batch_rows);
    try t.expect(b.stats.batches < total);
    var multi: u32 = 0;
    for (fake.sizes[2..5]) |n| multi += n;
    try t.expect(multi > 0);
    try t.expect(b.stats.prefill_chunks >= 12);
    for (b.slot[0..4]) |s| try t.expect(!s.used);
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expectEqual(@as(u64, 0), b.stats.aborted_chunks);
    if (stall == .chunk) try t.expectEqual(@as(u64, 0), b.stats.batches_in_chunk);
    // Per segment the waiting steps preempt: batches run inside chunks.
    if (segments > 1 and stall == .ns and stall.ns == 0) try t.expect(b.stats.batches_in_chunk > 0);
}

/// Prefill and release the logits at once, as a generation samples them right away.
fn longPrefill(b: *B, slot: u32, prompt: []const u32) anyerror!void {
    _ = try b.prefill(slot, prompt);
    b.sampled(slot);
}

test "batcher: shortest remaining prompt goes first between chunks; fifo keeps arrival order" {
    for ([_]batcher.Order{ .shortest, .fifo }) |order| {
        const io = t.io;
        var fake: Fake = .{ .io = io, .segments = 4, .unit_delay_ns = 500_000 };
        var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V, .stall = .{ .ns = 0 }, .order = order });
        var task = try io.concurrent(B.run, .{&b});
        var long: [60]u32 = undefined;
        for (&long, 0..) |*x, i| x.* = @intCast(i + 1);
        const a = try b.join();
        const s = try b.join();
        try b.reset(a);
        try b.reset(s);
        // The long prompt (20 chunks of 4 units) starts; the short one arrives mid-way.
        var fut = try io.concurrent(longPrefill, .{ &b, a, @as([]const u32, &long) });
        while (fake.done_count == 0 and b.stats.prefill_units < 6) try io.sleep(.fromMilliseconds(1), .awake);
        _ = try b.prefill(s, &.{ 5, 6 });
        b.sampled(s);
        try fut.await(io);
        b.leave(a);
        b.leave(s);
        b.stop();
        task.await(io);
        try t.expectEqual(@as(usize, 2), fake.done_count);
        try t.expectEqual(@as(u32, 0), fake.violations);
        // shortest: the short prompt completes first; fifo: the long one does.
        try t.expectEqual(if (order == .shortest) s else a, fake.done_order[0]);
    }
}

test "batcher: a generation canceled mid-chunk aborts its chunk; the slot is reset and reused" {
    const io = t.io;
    var fake: Fake = .{ .io = io, .segments = 4, .unit_delay_ns = 500_000 };
    var b = try B.init(io, &fake, .{ .slots = 1, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest });
    var task = try io.concurrent(B.run, .{&b});
    var stopped = false;
    defer if (!stopped) {
        b.stop();
        task.await(io);
    };
    var long: [60]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast(i + 1);
    const a = try b.join();
    try b.reset(a);
    var fut = try io.concurrent(longPrefill, .{ &b, a, @as([]const u32, &long) });
    while (b.stats.prefill_units < 6) try io.sleep(.fromMilliseconds(1), .awake);
    try t.expectError(error.Canceled, fut.cancel(io));
    b.leave(a);
    // The next generation in the same slot (free once a running unit completed) resets
    // and prefills normally.
    const again = while (true) break b.join() catch |e| switch (e) {
        error.NoSlot => {
            try io.sleep(.fromMilliseconds(1), .awake);
            continue;
        },
        else => return e,
    };
    try t.expectEqual(a, again);
    try b.reset(again);
    const logits = try b.prefill(again, &.{9});
    var want: [V]f32 = undefined;
    logitsFor(mix(7, 9), &want);
    try t.expectEqualSlices(f32, &want, logits);
    b.leave(again);
    b.stop();
    task.await(io);
    stopped = true;
    // The cancel lands inside a chunk (aborted) or on its last unit (nothing in flight).
    try t.expect(fake.aborts <= 1);
    try t.expectEqual(@as(u64, fake.aborts), b.stats.aborted_chunks);
    try t.expectEqual(@as(u32, 0), fake.violations);
}

test "batcher: a slow client does not hold the others (gather expires)" {
    const io = t.io;
    var fake: Fake = .{ .io = io, .delay_ns = 100_000 };
    var b = try B.init(io, &fake, .{ .slots = 3, .vocab = V });
    var task = try io.concurrent(B.run, .{&b});
    const prompt = [_]u32{ 1, 2, 3, 4 };
    const plans = [_]Plan{
        .{ .prompt = &prompt, .steps = 30 },
        .{ .prompt = prompt[1..], .steps = 30 },
        .{ .prompt = prompt[2..], .steps = 6, .pause_ns = 20_000_000 },
    };
    var outs: [3][64]u32 = undefined;
    var results: [3]anyerror!void = undefined;
    var gate: std.Io.Semaphore = .{ .permits = 3 };
    try runAll(&b, &gate, &plans, &outs, &results);
    b.stop();
    task.await(io);
    for (plans, outs, results) |plan, out, result| {
        try result;
        var want: [64]u32 = undefined;
        soloTokens(plan.prompt, plan.steps, &want);
        try t.expectEqualSlices(u32, want[0..plan.steps], out[0..plan.steps]);
    }
    // The fast generations ran without the paused one in many batches.
    try t.expect(b.stats.partial_batches > 0);
    try t.expectEqual(@as(u64, 66), b.stats.batch_rows);
}

fn blockedStep(b: *B, slot: u32) anyerror![]const f32 {
    return b.step(slot, 5);
}

test "batcher: a canceled waiter withdraws; a backend error fails only its batch; stop" {
    const io = t.io;
    var fake: Fake = .{ .io = io };
    var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V });
    // Without the scheduler running, a step waits forever: cancel it.
    const slot = try b.join();
    var waiter = try io.concurrent(blockedStep, .{ &b, slot });
    while (b.slot[slot].op != .step) try io.sleep(.fromMilliseconds(1), .awake);
    try t.expectError(error.Canceled, waiter.cancel(io));
    try t.expectEqual(.none, b.slot[slot].op);
    b.leave(slot);
    try t.expect(!b.slot[slot].used);
    // Operations on a slot that is not held are refused.
    try t.expectError(error.InvalidState, b.step(slot, 1));

    var task = try io.concurrent(B.run, .{&b});
    // The first batch fails: its row gets the error, and the slot stays usable.
    fake.fail_batches = 1;
    const s = try b.join();
    try b.reset(s);
    _ = try b.prefill(s, &.{ 1, 2, 3, 4, 5 });
    b.sampled(s);
    try t.expectError(error.FakeFailure, b.step(s, 3));
    try b.reset(s);
    const logits = try b.prefill(s, &.{9});
    var want: [V]f32 = undefined;
    logitsFor(mix(7, 9), &want);
    try t.expectEqualSlices(f32, &want, logits);
    b.leave(s);
    // Slots run out; an empty prompt is refused.
    const s0 = try b.join();
    const s1 = try b.join();
    try t.expectError(error.NoSlot, b.join());
    try t.expectError(error.InvalidToken, b.prefill(s0, &.{}));
    b.leave(s0);
    b.leave(s1);
    b.stop();
    task.await(io);
}

fn prefillOwned(b: *B, slot: u32, prompt: []const u32) anyerror!void {
    _ = try b.prefill(slot, prompt);
    b.sampled(slot);
}

test "batcher: a canceled generation's prompt is never read after its submit returns" {
    const io = t.io;
    // Units long enough that the cancel lands while one runs; several chunks, so a dropped
    // operation would otherwise be continued from the freed prompt.
    for ([_]u32{ 1, 4 }) |segments| {
        var fake: Fake = .{ .io = io, .segments = segments, .unit_delay_ns = 5_000_000 };
        var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest });
        var task = try io.concurrent(B.run, .{&b});
        var stopped = false;
        defer if (!stopped) {
            b.stop();
            task.await(io);
        };
        const prompt = try t.allocator.alloc(u32, 30);
        defer t.allocator.free(prompt);
        for (prompt, 0..) |*x, i| x.* = @intCast(i + 1);
        const a = try b.join();
        try b.reset(a);
        var fut = try io.concurrent(prefillOwned, .{ &b, a, @as([]const u32, prompt) });
        while (chunkMembers(&fake) == 0) try io.sleep(.fromMicroseconds(200), .awake);
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        try t.expectError(error.Canceled, fut.cancel(io));
        const waited: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);
        // The generation owns the prompt again: free it (poison stands in for reuse).
        @memset(prompt, poison);
        b.leave(a);
        try io.sleep(.fromMilliseconds(40), .awake); // units that would read the poison
        // Another generation runs normally in the other slot.
        const s = try b.join();
        try b.reset(s);
        const logits = try b.prefill(s, &.{9});
        var want: [V]f32 = undefined;
        logitsFor(mix(7, 9), &want);
        try t.expectEqualSlices(f32, &want, logits);
        b.leave(s);
        b.stop();
        task.await(io);
        stopped = true;
        try t.expectEqual(@as(u32, 0), fake.violations);
        // The cancel waited at most for the running unit, not for the whole prompt: of its 10
        // chunks at most 2 ran (the running one, and one begun as the cancel landed), plus the
        // other generation's chunk. Load-independent; before 2026-09-26 a canceled prompt
        // with one-unit chunks ran all 10 (the check ran only for a chunk in flight).
        try t.expect(b.stats.prefill_chunks <= 3);
        // The same in time (5 ms units; a generous bound for a loaded machine).
        try t.expect(waited < 50 * std.time.ns_per_ms);
        if (segments > 1) try t.expect(b.stats.aborted_chunks <= 1);
    }
}

/// One step; the logits are copied and released at once, as a generation samples them.
fn stepOnce(b: *B, slot: u32, token: u32, out: *anyerror![V]f32) void {
    const logits = b.step(slot, token) catch |e| {
        out.* = e;
        return;
    };
    out.* = logits[0..V].*;
    b.sampled(slot);
}

test "batcher: a row that cannot run fails alone; the other rows of its batch run" {
    const io = t.io;
    var fake: Fake = .{ .io = io, .bad_token = 666 };
    var b = try B.init(io, &fake, .{ .slots = 3, .vocab = V });
    var task = try io.concurrent(B.run, .{&b});
    defer {
        b.stop();
        task.await(io);
    }
    var slots: [3]u32 = undefined;
    for (&slots, 0..) |*sl, i| {
        sl.* = try b.join();
        try b.reset(sl.*);
        _ = try b.prefill(sl.*, &.{@as(u32, @intCast(10 + i))});
        b.sampled(sl.*);
    }
    var outs: [3]anyerror![V]f32 = undefined;
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (slots, &outs, [_]u32{ 1, 666, 2 }) |sl, *o, token| try group.concurrent(io, stepOnce, .{ &b, sl, token, o });
    try group.await(io);
    try t.expectError(error.FakeBadRow, outs[1]);
    for ([_]usize{ 0, 2 }, [_]u32{ 1, 2 }) |i, token| {
        var want: [V]f32 = undefined;
        logitsFor(mix(mix(7, @as(u32, @intCast(10 + i))), token), &want);
        const got = try outs[i];
        try t.expectEqualSlices(f32, &want, &got);
    }
    for (slots) |sl| b.leave(sl);
}

test "batcher: pending prompts pack into one chunk; each gets its solo logits; a canceled member does not stop the others" {
    const io = t.io;
    // 1. Six generations with prompts of 2..12 tokens start together over 6 slots, packs of
    // up to 4: packed chunks happen, and every logits row is the solo one.
    {
        var fake: Fake = .{ .io = io, .segments = 3, .unit_delay_ns = 200_000, .pack_cap = 4 };
        var b = try B.init(io, &fake, .{ .slots = 6, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest, .pack = 4 });
        var task = try io.concurrent(B.run, .{&b});
        var prompts: [6][12]u32 = undefined;
        var plans: [6]Plan = undefined;
        for (&prompts, &plans, 0..) |*pr, *pl, i| {
            for (pr, 0..) |*x, j| x.* = @intCast(100 * i + j + 1);
            pl.* = .{ .prompt = pr[0 .. 2 + 2 * i], .steps = 6 };
        }
        var gate: std.Io.Semaphore = .{ .permits = 6 };
        var outs: [6][64]u32 = undefined;
        var results: [6]anyerror!void = undefined;
        try runAll(&b, &gate, &plans, &outs, &results);
        b.stop();
        task.await(io);
        for (results) |r| try r;
        for (plans, outs) |plan, out| {
            var want: [64]u32 = undefined;
            soloTokens(plan.prompt, plan.steps, &want);
            try t.expectEqualSlices(u32, want[0..plan.steps], out[0..plan.steps]);
        }
        try t.expect(fake.packs > 0);
        try t.expect(b.stats.packed_chunks > 0);
        try t.expectEqual(@as(u32, 0), fake.violations);
    }
    // 2. Two prompts in one pack; one generation is canceled mid-chunk: the other completes
    // with its solo logits, the chunk is not aborted, and the canceled slot is reused.
    {
        var fake: Fake = .{ .io = io, .segments = 8, .unit_delay_ns = 2_000_000, .pack_cap = 2 };
        var b = try B.init(io, &fake, .{ .slots = 3, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest, .pack = 2 });
        var task = try io.concurrent(B.run, .{&b});
        const a = try b.join();
        const c = try b.join();
        try b.reset(a);
        try b.reset(c);
        var long: [9]u32 = undefined;
        for (&long, 0..) |*x, i| x.* = @intCast(i + 1);
        const short = [_]u32{ 40, 41, 42 };
        // Both prompts must be queued when the scheduler forms the chunk, whatever the thread
        // timing (before 2026-09-26 the short one could run alone first, and the wait below
        // never ended): a third slot's reset holds the scheduler until both are queued.
        const x = try b.join();
        fake.block_reset = x;
        var fx = try io.concurrent(B.reset, .{ &b, x });
        while (!fake.in_reset.load(.acquire)) try io.sleep(.fromMicroseconds(200), .awake);
        var fa = try io.concurrent(longPrefill, .{ &b, a, @as([]const u32, &long) });
        var fc = try io.concurrent(shortPrefill, .{ &b, c, @as([]const u32, &short) });
        while (queued(&b, .prefill) < 2) try io.sleep(.fromMicroseconds(200), .awake);
        fake.resume_reset.set(io);
        try fx.await(io); // `x` stays joined, so the next join must reuse the canceled slot
        while (chunkMembers(&fake) < 2) try io.sleep(.fromMicroseconds(200), .awake);
        try t.expectError(error.Canceled, fa.cancel(io));
        b.leave(a);
        const logits = try fc.await(io);
        var want: [V]f32 = undefined;
        logitsFor(mix(mix(mix(7, 40), 41), 42), &want);
        try t.expectEqualSlices(f32, &want, &logits);
        b.sampled(c);
        const again = while (true) break b.join() catch |e| switch (e) {
            error.NoSlot => {
                try io.sleep(.fromMilliseconds(1), .awake);
                continue;
            },
            else => return e,
        };
        try b.reset(again);
        const l2 = try b.prefill(again, &.{9});
        logitsFor(mix(7, 9), &want);
        try t.expectEqualSlices(f32, &want, l2);
        try t.expectEqual(a, again);
        b.leave(again);
        b.leave(c);
        b.leave(x);
        b.stop();
        task.await(io);
        try t.expectEqual(@as(u64, 0), b.stats.aborted_chunks);
        try t.expectEqual(@as(u32, 0), fake.violations);
    }
}

fn shortPrefill(b: *B, slot: u32, prompt: []const u32) anyerror![V]f32 {
    const logits = try b.prefill(slot, prompt);
    return logits[0..V].*;
}

/// Slots whose operation `op` is queued or running (read under the batcher's lock).
fn queued(b: *B, op: anytype) usize {
    b.mutex.lockUncancelable(b.io);
    defer b.mutex.unlock(b.io);
    var n: usize = 0;
    for (b.slot[0..b.options.slots]) |s| {
        if (s.used and s.op == op) n += 1;
    }
    return n;
}

/// Members of the chunk in flight (0: none), for a test thread (units run on the scheduler).
fn chunkMembers(fake: *const Fake) usize {
    return fake.members.load(.acquire);
}

fn pooledGeneration(b: *B, prompt: []const u32, steps: u32, out: *anyerror!void) void {
    out.* = pooled(b, prompt, steps, true);
}
fn growingGeneration(b: *B, prompt: []const u32, steps: u32, out: *anyerror!void) void {
    out.* = pooled(b, prompt, steps, false);
}
fn pooled(b: *B, prompt: []const u32, steps: u32, reserve: bool) !void {
    const slot = while (true) break b.join() catch |e| switch (e) {
        error.NoSlot => {
            try b.io.sleep(.fromMilliseconds(1), .awake);
            continue;
        },
        else => return e,
    };
    defer b.leave(slot);
    if (reserve) b.reserve(slot, prompt.len + steps);
    try b.reset(slot);
    var h: u64 = 7;
    for (prompt) |token| h = mix(h, token);
    var logits = try b.prefill(slot, prompt);
    var want: [V]f32 = undefined;
    for (0..steps) |_| {
        logitsFor(h, &want);
        if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
        const token = pick(logits);
        b.sampled(slot);
        h = mix(h, token);
        logits = try b.step(slot, token);
    }
    b.sampled(slot);
}

test "batcher: shared memory pool: prompts wait for memory, a large prompt is not starved, memory comes back" {
    const io = t.io;
    // 6 slots over a pool of 10 pages of 4 positions (40): each request reserves prompt +
    // steps. The large one (6 + 20 = 26 positions, 7 pages) arrives second; with shortest-
    // first ordering the small ones (3 pages each) would keep overtaking it without the
    // oldest-waiter rule.
    var fake: Fake = .{ .io = io, .segments = 2, .unit_delay_ns = 100_000, .pool = 10, .page = 4, .pack_cap = 3 };
    var b = try B.init(io, &fake, .{ .slots = 6, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest, .pack = 3 });
    var prompts: [12][6]u32 = undefined;
    for (&prompts, 0..) |*pr, i| for (pr, 0..) |*x, j| {
        x.* = @intCast(1000 * i + j + 1);
    };
    var results: [12]anyerror!void = undefined;
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (0..12) |i| {
        const big = i == 1;
        try group.concurrent(io, pooledGeneration, .{ &b, if (big) prompts[i][0..6] else prompts[i][0..2], if (big) @as(u32, 20) else 8, &results[i] });
        try io.sleep(.fromMicroseconds(300), .awake);
    }
    // The pool must really be a constraint, independent of timing: the scheduler starts only
    // when every slot's generation (the large one among them) is waiting in its reset, so 6
    // sequences (5 x 3 + 7 pages) compete for 10 pages.
    while (queued(&b, .reset) < 6) try io.sleep(.fromMicroseconds(200), .awake);
    var task = try io.concurrent(B.run, .{&b});
    try group.await(io);
    b.stop();
    task.await(io);
    for (results) |r| try r;
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expect(b.stats.admission_waits > 0); // the pool was really a constraint
    for (fake.held) |h| try t.expectEqual(@as(usize, 0), h); // every page came back
}

fn growingRun(fake: *Fake, slots: u32, n: usize, steps: u32, results: []anyerror!void) !B {
    return growingRunSliced(fake, slots, n, steps, results, null);
}
fn growingRunSliced(fake: *Fake, slots: u32, n: usize, steps: u32, results: []anyerror!void, slice: ?std.Io.Duration) !B {
    const io = t.io;
    var b = try B.init(io, fake, .{ .slots = slots, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest, .pack = 3, .swap_slice = slice });
    var prompts: [12][6]u32 = undefined;
    for (&prompts, 0..) |*pr, i| for (pr, 0..) |*x, j| {
        x.* = @intCast(1000 * i + j + 1);
    };
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (0..n) |i| try group.concurrent(io, growingGeneration, .{ &b, prompts[i][0 .. 2 + i % 5], steps, &results[i] });
    while (queued(&b, .reset) < slots) try io.sleep(.fromMicroseconds(200), .awake);
    var task = try io.concurrent(B.run, .{&b});
    try group.await(io);
    b.stop();
    task.await(io);
    return b;
}

test "batcher: prompt admission grows page by page; the youngest sequence swaps out and back, outputs unchanged" {
    // 6 slots over 10 pages of 4 positions: every prompt fits (1-2 pages), but 6 sequences
    // of up to 26 positions (7 pages) cannot all grow; swapped sequences come back.
    var fake: Fake = .{ .io = t.io, .segments = 2, .unit_delay_ns = 50_000, .pool = 10, .page = 4, .pack_cap = 3, .host_pool = 100 };
    var results: [12]anyerror!void = undefined;
    const b = try growingRun(&fake, 6, 12, 20, &results);
    for (results) |r| try r;
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expect(b.stats.swap_outs > 0);
    try t.expectEqual(b.stats.swap_outs, b.stats.swap_ins);
    try t.expectEqual(@as(u64, fake.swap_outs), b.stats.swap_outs);
    try t.expectEqual(@as(u64, 0), b.stats.swap_failures);
    for (fake.held, fake.host_held) |h, x| {
        try t.expectEqual(@as(usize, 0), h);
        try t.expectEqual(@as(usize, 0), x);
    }
}

test "batcher: prompt admission without a host store: a row that finds no page fails alone" {
    var fake: Fake = .{ .io = t.io, .segments = 1, .pool = 10, .page = 4, .pack_cap = 3, .host_pool = 0 };
    var results: [6]anyerror!void = undefined;
    const b = try growingRun(&fake, 6, 6, 20, &results);
    var failed: u32 = 0;
    for (results) |r| r catch |e| {
        try t.expectEqual(error.PoolExhausted, e);
        failed += 1;
    };
    try t.expect(failed > 0 and failed < 6);
    try t.expectEqual(@as(u64, failed), b.stats.swap_failures);
    try t.expectEqual(@as(u32, 0), fake.violations);
    for (fake.held) |h| try t.expectEqual(@as(usize, 0), h);
}

test "batcher: time slice: a swapped sequence waiting a slice swaps out the longest-running one; outputs unchanged" {
    // 4 slots over 10 pages of 4: two 60-step sequences fill the pool (16 pages each would be
    // needed), so without the slice the swapped ones wait for a finish. With a 2 ms slice
    // (batches take 0.3 ms) running sequences are rotated out.
    var fake: Fake = .{ .io = t.io, .segments = 1, .delay_ns = 300_000, .pool = 10, .page = 4, .pack_cap = 3, .host_pool = 100 };
    var results: [4]anyerror!void = undefined;
    const b = try growingRunSliced(&fake, 4, 4, 30, &results, .fromMilliseconds(2));
    for (results) |r| try r;
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expect(b.stats.slice_swaps > 0);
    try t.expectEqual(b.stats.swap_outs, b.stats.swap_ins);
    for (fake.held, fake.host_held) |h, x| {
        try t.expectEqual(@as(usize, 0), h);
        try t.expectEqual(@as(usize, 0), x);
    }
}

const Timed = struct { prompt_done_ns: i96 = 0, end_ns: i96 = 0, result: anyerror!void = {} };
fn timedGeneration(b: *B, prompt: []const u32, steps: u32, start_after_ns: u64, out: *Timed) void {
    out.result = timed(b, prompt, steps, start_after_ns, out);
}
fn timed(b: *B, prompt: []const u32, steps: u32, start_after_ns: u64, out: *Timed) !void {
    const io = b.io;
    if (start_after_ns > 0) try io.sleep(.fromNanoseconds(start_after_ns), .awake);
    const slot = try b.join();
    defer b.leave(slot);
    try b.reset(slot);
    var h: u64 = 7;
    for (prompt) |token| h = mix(h, token);
    var logits = try b.prefill(slot, prompt);
    out.prompt_done_ns = std.Io.Clock.awake.now(io).nanoseconds;
    var want: [V]f32 = undefined;
    for (0..steps) |_| {
        logitsFor(h, &want);
        if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
        const token = pick(logits);
        b.sampled(slot);
        h = mix(h, token);
        logits = try b.step(slot, token);
    }
    b.sampled(slot);
    out.end_ns = std.Io.Clock.awake.now(io).nanoseconds;
}

test "batcher: time slice: a new prompt is not starved while long sequences rotate through the pool" {
    // 3 long sequences (120 steps: 31 pages each, each fits alone) share 40 pages of 4,
    // rotating by the 2 ms slice. A prompt arriving 20 ms later must start before any long
    // one ends.
    const io = t.io;
    var fake: Fake = .{ .io = io, .segments = 1, .delay_ns = 200_000, .pool = 40, .page = 4, .pack_cap = 1, .host_pool = 200 };
    var b = try B.init(io, &fake, .{ .slots = 4, .vocab = V, .stall = .{ .ns = 0 }, .order = .fifo, .pack = 1, .swap_slice = .fromMilliseconds(2) });
    var task = try io.concurrent(B.run, .{&b});
    const prompts = [4][2]u32{ .{ 1, 2 }, .{ 3, 4 }, .{ 5, 6 }, .{ 7, 8 } };
    var outs: [4]Timed = @splat(.{});
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (0..3) |i| try group.concurrent(io, timedGeneration, .{ &b, &prompts[i], @as(u32, 120), @as(u64, 0), &outs[i] });
    try group.concurrent(io, timedGeneration, .{ &b, &prompts[3], @as(u32, 4), @as(u64, 20_000_000), &outs[3] });
    try group.await(io);
    b.stop();
    task.await(io);
    for (outs) |o| try o.result;
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expect(b.stats.slice_swaps > 0);
    var first_end: i96 = std.math.maxInt(i96);
    for (outs[0..3]) |o| first_end = @min(first_end, o.end_ns);
    try t.expect(outs[3].prompt_done_ns < first_end);
    for (fake.held, fake.host_held) |h, x| {
        try t.expectEqual(@as(usize, 0), h);
        try t.expectEqual(@as(usize, 0), x);
    }
}

fn cachedGeneration(b: *B, prompt: []const u32, point: usize, steps: u32, out: *anyerror!void, restored: *u32) void {
    out.* = cached(b, prompt, point, steps, restored);
}
/// A generation through `begin` and one `checkpoint` at `point` (when it lies after the start).
fn cached(b: *B, prompt: []const u32, point: usize, steps: u32, restored: *u32) !void {
    const slot = while (true) break b.join() catch |e| switch (e) {
        error.NoSlot => {
            try b.io.sleep(.fromMilliseconds(1), .awake);
            continue;
        },
        else => return e,
    };
    defer b.leave(slot);
    const start = try b.begin(slot, prompt);
    if (start > 0) restored.* += 1;
    var at: usize = start;
    if (point > at and point < prompt.len) {
        _ = try b.prefill(slot, prompt[at..point]);
        b.sampled(slot);
        try b.checkpoint(slot, prompt[0..point]);
        at = point;
    }
    var logits = try b.prefill(slot, prompt[at..]);
    var h: u64 = 7;
    for (prompt) |token| h = mix(h, token);
    var want: [V]f32 = undefined;
    for (0..steps) |_| {
        logitsFor(h, &want);
        if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
        const token = pick(logits);
        b.sampled(slot);
        h = mix(h, token);
        logits = try b.step(slot, token);
    }
    b.sampled(slot);
}

test "batcher: begin restores the longest checkpoint; checkpoints are taken mid-prompt; outputs unchanged" {
    const io = t.io;
    var fake: Fake = .{ .io = io, .segments = 2, .pool = 40, .page = 4, .pack_cap = 3, .host_pool = 100 };
    var b = try B.init(io, &fake, .{ .slots = 4, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest, .pack = 3 });
    var task = try io.concurrent(B.run, .{&b});
    // A shared 5-token "system prompt", then each request's own part.
    const prompts = [_][8]u32{ .{ 1, 2, 3, 4, 5, 10, 11, 12 }, .{ 1, 2, 3, 4, 5, 20, 21, 22 }, .{ 1, 2, 3, 4, 5, 30, 31, 32 }, .{ 1, 2, 3, 4, 5, 10, 11, 13 } };
    var restored: u32 = 0;
    // The first request alone (it takes the checkpoint), then three at once.
    var r0: anyerror!void = undefined;
    cachedGeneration(&b, &prompts[0], 5, 6, &r0, &restored);
    try r0;
    var results: [3]anyerror!void = undefined;
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (1..4) |i| try group.concurrent(io, cachedGeneration, .{ &b, &prompts[i], @as(usize, 5), @as(u32, 6), &results[i - 1], &restored });
    try group.await(io);
    b.stop();
    task.await(io);
    for (results) |r| try r;
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expectEqual(@as(u32, 3), fake.restores);
    try t.expectEqual(@as(u32, 3), restored);
    try t.expectEqual(@as(usize, 1), fake.ck_count); // restored requests start at the point: no new checkpoint
    for (fake.held) |h| try t.expectEqual(@as(usize, 0), h);
}

test "batcher: every prompt segment is admitted (checkpoint points split a prompt; tight pool)" {
    const io = t.io;
    // Pages of 2 positions: a 12-token prompt split at 5 needs its later segment admitted
    // too (the first admission covers 5 positions only).
    var fake: Fake = .{ .io = io, .segments = 1, .pool = 30, .page = 2, .pack_cap = 1, .host_pool = 100 };
    var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V, .stall = .{ .ns = 0 }, .order = .fifo, .pack = 1 });
    var task = try io.concurrent(B.run, .{&b});
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var restored: u32 = 0;
    var r: anyerror!void = undefined;
    cachedGeneration(&b, &prompt, 5, 4, &r, &restored);
    b.stop();
    task.await(io);
    try r;
    try t.expectEqual(@as(u32, 0), fake.violations);
}

fn pendingCacheOp(b: *B, slot: u32, tokens: []const u32, checkpoint: bool) anyerror!u32 {
    if (checkpoint) {
        try b.checkpoint(slot, tokens);
        return 0;
    }
    return b.begin(slot, tokens);
}

test "batcher: pending cache keeps tokens and permits decode; cancel leave stop drain" {
    const io = t.io;
    for ([_]bool{ false, true }) |checkpoint| for (0..4) |action| {
        var fake: Fake = .{ .io = io, .io_kind = if (checkpoint) .checkpoint else .begin };
        var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V });
        var task = try io.concurrent(B.run, .{&b});
        var stopped = false;
        defer if (!stopped) {
            b.stop();
            task.await(io);
        };
        const a = try b.join();
        const other = try b.join();
        var tokens = [_]u32{ 9, 4, 8 };
        if (checkpoint) {
            try b.reset(a);
            _ = try b.prefill(a, &tokens);
            b.sampled(a);
        }
        var pending = try io.concurrent(pendingCacheOp, .{ &b, a, @as([]const u32, &tokens), checkpoint });
        while (!fake.io_entered.load(.acquire)) try io.sleep(.fromMicroseconds(100), .awake);
        // No disk completion yet. Another sequence must still reach a decode result.
        try b.reset(other);
        _ = try b.prefill(other, &.{7});
        b.sampled(other);
        const row = try b.step(other, 5);
        var want: [V]f32 = undefined;
        logitsFor(mix(mix(7, 7), 5), &want);
        try t.expectEqualSlices(f32, &want, row);
        b.sampled(other);
        switch (action) {
            0 => {
                fake.io_ready.store(true, .release);
                try t.expectEqual(@as(u32, if (checkpoint) 0 else 2), try pending.await(io));
                b.leave(a);
            },
            1 => {
                try t.expectError(error.Canceled, pending.cancel(io));
                b.leave(a);
            },
            2 => {
                b.leave(a);
                try t.expectError(error.Canceled, pending.await(io));
            },
            3 => {
                b.stop();
                task.await(io);
                stopped = true;
                try t.expectError(error.Canceled, pending.await(io));
                b.leave(a);
            },
            else => unreachable,
        }
        // Native cancellation returns only when the backend has stopped borrowing this.
        @memset(&tokens, 0xdead);
        b.leave(other);
        if (!stopped) {
            b.stop();
            task.await(io);
            stopped = true;
        }
        try t.expectEqual(null, fake.io_live);
        if (action > 0) try t.expect(fake.io_cancel_polls >= 3);
        try t.expectEqual(@as(u32, 0), fake.violations);
    };
}

test "batcher: leave and reuse during an I/O poll releases old pages before new begin" {
    const io = t.io;
    var fake: Fake = .{ .io = io, .io_kind = .begin, .io_pause_poll = true };
    var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V });
    const a = try b.join();
    const old = try b.join();
    var scheduler = try io.concurrent(B.run, .{&b});
    var pending = try io.concurrent(pendingCacheOp, .{ &b, a, @as([]const u32, &.{ 1, 2, 3 }), false });
    fake.io_in_poll.waitUncancelable(io);
    // The scheduler is unlocked in the callback, after its release-queue check.
    b.leave(old);
    const reused = try b.join();
    var next = try io.concurrent(pendingCacheOp, .{ &b, reused, @as([]const u32, &.{ 4, 5, 6 }), false });
    while (true) {
        b.mutex.lockUncancelable(io);
        const is_queued = b.slot[reused].op == .begin;
        b.mutex.unlock(io);
        if (is_queued) break;
        try io.sleep(.fromMicroseconds(100), .awake);
    }
    fake.io_resume_poll.set(io);
    _ = try next.await(io);
    b.stop();
    scheduler.await(io);
    try t.expectError(error.Canceled, pending.await(io));
    try t.expectEqual(@as(u32, 1), fake.release_calls[reused]);
    try t.expectEqual(fake.release_calls[reused], fake.reset_release_seen[reused]);
    b.leave(a);
    b.leave(reused);
}

test "batcher: stop aborts a packed chunk before draining pending cache I/O" {
    const io = t.io;
    var fake: Fake = .{ .io = io, .io_kind = .begin, .segments = 4, .unit_delay_ns = 5_000_000 };
    var b = try B.init(io, &fake, .{ .slots = 2, .vocab = V });
    const a = try b.join();
    const other = try b.join();
    var scheduler = try io.concurrent(B.run, .{&b});
    var pending = try io.concurrent(pendingCacheOp, .{ &b, a, @as([]const u32, &.{ 1, 2, 3 }), false });
    while (!fake.io_entered.load(.acquire)) try io.sleep(.fromMicroseconds(100), .awake);
    const tokens: [60]u32 = @splat(7);
    var prefill = try io.concurrent(prefillOwned, .{ &b, other, @as([]const u32, &tokens) });
    while (fake.members.load(.acquire) == 0) try io.sleep(.fromMicroseconds(100), .awake);
    b.stop();
    scheduler.await(io);
    try t.expectError(error.Canceled, pending.await(io));
    try t.expectError(error.Canceled, prefill.await(io));
    try t.expectEqual(@as(u32, 1), fake.aborts);
    try t.expect(fake.io_cancel_polls >= 3);
    try t.expectEqual(@as(u32, 0), fake.violations);
    b.leave(a);
    b.leave(other);
}
