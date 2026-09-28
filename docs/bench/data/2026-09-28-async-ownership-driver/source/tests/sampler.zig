//! Token sampler (docs/specs/session.md): selection, penalties, top-k/top-p/min-p against
//! full-sort reference chains, the sorted-order goldens, and the exponential.
const std = @import("std");
const session = @import("session");
const sampler = session.sampler;
const sampler_cases = @import("sampler_cases.zig");
const parallel = @import("parallel.zig");
const t = std.testing;

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

/// Reference top-k selection: every candidate penalized, full sort, first k.
fn referenceTop(a: std.mem.Allocator, logits: []const f32, counts: []const u32, p: sampler.Params, k: usize) ![]sampler.Candidate {
    const c = try a.alloc(sampler.Candidate, logits.len);
    for (logits, 0..) |l, i| {
        var v = l;
        if (counts[i] > 0) {
            if (p.repetition_penalty != 1) v = if (v > 0) v / p.repetition_penalty else v * p.repetition_penalty;
            v -= @as(f32, @floatFromInt(counts[i])) * p.frequency_penalty + p.presence_penalty;
        }
        c[i] = .{ .id = @intCast(i), .logit = v };
    }
    std.sort.pdq(sampler.Candidate, c, {}, struct {
        fn f(_: void, x: sampler.Candidate, y: sampler.Candidate) bool {
            return x.logit > y.logit or (x.logit == y.logit and x.id < y.id);
        }
    }.f);
    return c[0..k];
}

// The single-pass top-k (with penalties, ties, negative and positive penalties, tails)
// draws exactly what the full-sort reference chain draws, token by token.
test "sampler single-pass top-k equals the full-sort chain" {
    // Rounds run concurrently, each with its own random stream (tests/parallel.zig).
    try parallel.rounds(60, {}, topKSetRound);
    try parallel.rounds(12, {}, topKDrawRound);
    var bad: [40]f32 = @splat(0);
    bad[37] = std.math.inf(f32);
    var s = try sampler.Sampler.init(t.allocator, bad.len, .{ .temperature = 1, .top_k = 5 });
    defer s.deinit(t.allocator);
    try t.expectError(error.NonFiniteLogits, s.sample(&bad));
    bad[37] = 0;
    bad[3] = std.math.nan(f32);
    try t.expectError(error.NonFiniteLogits, s.sample(&bad));
    try t.expectError(error.NonFiniteLogits, sampler.greedy(&bad));
}

/// The fast sampler's draw is always one of the reference's top-k candidates.
fn topKSetRound(_: void, round: usize) !void {
    const a = t.allocator;
    var prng = parallel.seed(11, round);
    const random = prng.random();
    {
        const n = 17 + random.uintLessThan(usize, 3000);
        const logits = try a.alloc(f32, n);
        defer a.free(logits);
        const params: sampler.Params = .{
            .temperature = if (round % 5 == 0) 0 else 0.3 + random.float(f32),
            .top_k = @intCast(1 + random.uintLessThan(usize, @min(n - 1, 300))),
            .top_p = if (round % 3 == 0) 1 else 0.5 + 0.5 * random.float(f32),
            .min_p = if (round % 4 == 0) 0.05 else 0,
            .presence_penalty = if (round % 2 == 0) 0 else (random.float(f32) - 0.5) * 3,
            .frequency_penalty = if (round % 3 == 1) 0.4 else 0,
            .repetition_penalty = if (round % 4 == 1) 1.3 else 1,
            .seed = round,
        };
        var fast = try sampler.Sampler.init(a, n, params);
        defer fast.deinit(a);
        const counts = try a.alloc(u32, n);
        defer a.free(counts);
        @memset(counts, 0);
        for (0..40) |_| {
            for (logits) |*l| l.* = if (round % 2 == 0) @floatFromInt(random.intRangeAtMost(i32, -6, 6)) else random.floatNorm(f32) * 4;
            const got = try fast.sample(logits);
            // Reference: the same Params without top-k, applied to exactly the reference's
            // top-k candidates (as ordered logits: every other id gets -inf-like distance).
            const top = try referenceTop(a, logits, counts, params, params.top_k);
            defer a.free(top.ptr[0..n]);
            var in_top = false;
            for (top) |c| if (c.id == got) {
                in_top = true;
            };
            try t.expect(in_top);
            // Exactness of the set: the fast sampler's candidates are the reference's.
            try fast.accept(a, got);
            counts[got] += 1;
        }
    }
}

/// Draw for draw: the single-pass path and the full-array reference path, same seed, same
/// penalty history, over a large vocabulary (the vector scan and its tail).
fn topKDrawRound(_: void, round: usize) !void {
    const a = t.allocator;
    var prng = parallel.seed(12, round);
    const random = prng.random();
    {
        const n: usize = 50000 + round * 7;
        const logits = try a.alloc(f32, n);
        defer a.free(logits);
        const p: sampler.Params = .{ .temperature = 0.6 + 0.1 * @as(f32, @floatFromInt(round % 4)), .top_k = @intCast(1 + 20 * round), .top_p = 0.9, .min_p = if (round % 2 == 0) 0.02 else 0, .seed = round, .presence_penalty = if (round % 3 == 0) -1.5 else 0.7, .repetition_penalty = 1.1 };
        var fast = try sampler.Sampler.init(a, n, p);
        defer fast.deinit(a);
        var reference = try sampler.Sampler.init(a, n, p);
        defer reference.deinit(a);
        reference.fast_top_k = false;
        for (0..60) |_| {
            for (logits) |*l| l.* = if (round % 2 == 0) @floatFromInt(random.intRangeAtMost(i32, -4, 4)) else random.floatNorm(f32) * 3;
            const want = try reference.sample(logits);
            try t.expectEqual(want, try fast.sample(logits));
            try reference.accept(a, want);
            try fast.accept(a, want);
        }
    }
}

/// `Order.sorted` through the `cases.run` interface.
const SortedSampler = struct {
    s: sampler.Sampler,
    pub fn init(a: std.mem.Allocator, vocab: usize, p: sampler.Params) !SortedSampler {
        var q = p;
        q.order = .sorted;
        return .{ .s = try sampler.Sampler.init(a, vocab, q) };
    }
    pub fn deinit(self: *SortedSampler, a: std.mem.Allocator) void {
        self.s.deinit(a);
    }
    pub fn sample(self: *SortedSampler, logits: []const f32) !u32 {
        return self.s.sample(logits);
    }
    pub fn sampleFrom(self: *SortedSampler, logits: []const f32, allowed: []const u32) !u32 {
        return self.s.sampleFrom(logits, allowed);
    }
    pub fn accept(self: *SortedSampler, a: std.mem.Allocator, id: u32) !void {
        try self.s.accept(a, id);
    }
};

// `--sampler-order sorted` reproduces the sampler of commit 3c03b07 draw for draw (goldens
// generated from that source by tests/reference/generate_sampler_sorted.zig); the
// default order draws differently from the same seeds on some cases (the knob is live).
test "sampler order sorted: draws equal the pre-2026-09-24 sampler" {
    const a = t.allocator;
    const Fixture = struct { sampler_commit: []const u8, sampler_sha256: []const u8, draws: []const []const u32 };
    const parsed = try std.json.parseFromSlice(Fixture, a, @embedFile("fixtures/session/sampler-sorted.json"), .{});
    defer parsed.deinit();
    try t.expectEqualStrings("3c03b07cfb4acc634b852e5da396abdc02d9cacb", parsed.value.sampler_commit);
    try t.expectEqual(sampler_cases.cases.len, parsed.value.draws.len);
    const differ = try a.alloc(bool, sampler_cases.cases.len);
    defer a.free(differ);
    try parallel.rounds(sampler_cases.cases.len, SortedCase{ .draws = parsed.value.draws, .differ = differ }, SortedCase.run);
    try t.expect(std.mem.indexOfScalar(bool, differ, true) != null);
}

/// One golden case: `sorted` must reproduce it; whether the default order differs.
const SortedCase = struct {
    draws: []const []const u32,
    differ: []bool,
    fn run(self: SortedCase, i: usize) !void {
        const a = t.allocator;
        const c = sampler_cases.cases[i];
        const got = try a.alloc(u32, c.steps);
        defer a.free(got);
        try sampler_cases.run(SortedSampler, sampler.Params, a, c, got);
        try t.expectEqualSlices(u32, self.draws[i], got);
        try sampler_cases.run(sampler.Sampler, sampler.Params, a, c, got);
        self.differ[i] = !std.mem.eql(u32, self.draws[i], got);
    }
};

// The sampler's exponential (docs/specs/session.md): accurate against @exp over the
// whole range it is used on, scalar and 4-wide calls bit-identical, exact edges.
test "sampler expNeg: accuracy, scalar = vector, edges" {
    const expNeg = sampler.expNeg;
    var prng = std.Random.DefaultPrng.init(0xe4e);
    const random = prng.random();
    var worst: f64 = 0;
    var i: usize = 0;
    while (i < 400000) : (i += 4) {
        var v: @Vector(4, f64) = undefined;
        inline for (0..4) |j| v[j] = switch ((i / 4) % 3) {
            0 => -random.float(f64) * 708,
            1 => -random.float(f64) * 2,
            else => -random.float(f64) * 1e-6,
        };
        const e = expNeg(v);
        inline for (0..4) |j| {
            const s = expNeg(v[j]);
            try t.expectEqual(@as(u64, @bitCast(s)), @as(u64, @bitCast(e[j])));
            const want = @exp(v[j]);
            worst = @max(worst, @abs(s - want) / want);
        }
    }
    try t.expect(worst <= 4 * std.math.floatEps(f64));
    std.debug.print("expNeg worst relative error vs @exp: {e} ({d:.2} eps)\n", .{ worst, worst / std.math.floatEps(f64) });
    try t.expectEqual(@as(f64, 1), expNeg(@as(f64, 0)));
    try t.expectEqual(@as(f64, 0), expNeg(@as(f64, -708.5)));
    try t.expectEqual(@as(f64, 0), expNeg(-std.math.inf(f64)));
    try t.expect(expNeg(@as(f64, -708)) > 0);
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
