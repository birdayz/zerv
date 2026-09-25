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
/// 3 tokens per chunk. The output buffers are shared and overwritten by every call, as the
/// real model's are.
const Fake = struct {
    io: std.Io,
    hist: [max_slots]u64 = @splat(0),
    rows_out: [max_slots * V]f32 = @splat(0),
    prefill_out: [V]f32 = @splat(0),
    delay_ns: u64 = 0,
    fail_batches: u32 = 0,
    sizes: [max_slots + 1]u32 = @splat(0),
    calls: u32 = 0,

    pub fn reset(self: *Fake, slot: u32) !void {
        self.hist[slot] = 7;
    }
    pub fn prefillChunk(self: *Fake, slot: u32, tokens: []const u32) !batcher.Chunk {
        const n = @min(3, tokens.len);
        for (tokens[0..n]) |token| self.hist[slot] = mix(self.hist[slot], token);
        if (n < tokens.len) return .{ .consumed = n, .logits = null };
        logitsFor(self.hist[slot], &self.prefill_out);
        return .{ .consumed = n, .logits = &self.prefill_out };
    }
    pub fn decodeBatch(self: *Fake, rows: []const batcher.Row) ![]const f32 {
        self.calls += 1;
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
    const io = t.io;
    var fake: Fake = .{ .io = io, .delay_ns = 200_000 };
    var b = try B.init(io, &fake, .{ .slots = 4, .vocab = V });
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
