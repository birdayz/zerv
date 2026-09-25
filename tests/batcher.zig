//! Continuous batching (src/serve/batcher.zig, docs/specs/concurrent.md "18c design") with a
//! fake backend: every returned logits row must be exactly what the sequence computes
//! alone, while generations join and leave, hold rows, pause, get canceled and see errors.
const std = @import("std");
const zerv = @import("zerv");
const batcher = zerv.serve.batcher;
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
    prefill_out: [V]f32 = @splat(0),
    delay_ns: u64 = 0,
    unit_delay_ns: u64 = 0,
    fail_batches: u32 = 0,
    sizes: [max_slots + 1]u32 = @splat(0),
    calls: u32 = 0,
    segments: u32 = 1,
    flight: ?u32 = null,
    seg: u32 = 0,
    no_batch_in_chunk: bool = false,
    aborts: u32 = 0,
    violations: u32 = 0,
    /// Slots in the order their prefill completed.
    done_order: [64]u32 = undefined,
    done_count: usize = 0,
    /// A row whose token is `bad_token` cannot run (as a slot the model refuses).
    bad_token: u32 = std.math.maxInt(u32),

    pub fn reset(self: *Fake, slot: u32) !void {
        if (self.flight != null) self.violations += 1;
        self.hist[slot] = 7;
    }
    pub fn prefillChunk(self: *Fake, slot: u32, tokens: []const u32) !batcher.Chunk {
        for (tokens) |token| if (token == poison) {
            self.violations += 1;
        };
        if (self.flight) |f| {
            if (f != slot) self.violations += 1;
        } else self.flight = slot;
        if (self.unit_delay_ns > 0) try self.io.sleep(.fromNanoseconds(self.unit_delay_ns), .awake);
        self.seg += 1;
        if (self.seg < self.segments) return .{ .consumed = 0, .logits = null };
        self.seg = 0;
        self.flight = null;
        const n = @min(3, tokens.len);
        for (tokens[0..n]) |token| if (token == poison) {
            self.violations += 1;
        };
        for (tokens[0..n]) |token| self.hist[slot] = mix(self.hist[slot], token);
        if (n < tokens.len) return .{ .consumed = n, .logits = null };
        logitsFor(self.hist[slot], &self.prefill_out);
        if (self.done_count < self.done_order.len) {
            self.done_order[self.done_count] = slot;
            self.done_count += 1;
        }
        return .{ .consumed = n, .logits = &self.prefill_out };
    }
    pub fn checkRow(self: *Fake, row: batcher.Row) !void {
        if (row.token == self.bad_token) return error.FakeBadRow;
    }
    pub fn abortChunk(self: *Fake) void {
        self.aborts += 1;
        self.flight = null;
        self.seg = 0;
    }
    pub fn decodeBatch(self: *Fake, rows: []const batcher.Row) ![]const f32 {
        self.calls += 1;
        if (self.flight) |f| {
            if (self.no_batch_in_chunk) self.violations += 1;
            for (rows) |row| if (row.slot == f) {
                self.violations += 1;
            };
        }
        if (self.delay_ns > 0) try self.io.sleep(.fromNanoseconds(self.delay_ns), .awake);
        if (self.fail_batches > 0) {
            self.fail_batches -= 1;
            return error.FakeFailure;
        }
        self.sizes[rows.len] += 1;
        for (rows, 0..) |row, r| {
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
        while (fake.flight == null) try io.sleep(.fromMicroseconds(200), .awake);
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
        // The cancel waited at most for the running unit (5 ms), not for the whole prompt.
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
