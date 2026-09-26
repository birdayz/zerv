//! Sampler top-k-off fast path against the full-sort reference chain (docs/specs/session.md).
//! Its own test binary: ~2 s in Debug, run in parallel with the other files.
const std = @import("std");
const sampler = @import("session").sampler;
const parallel = @import("parallel.zig");
const t = std.testing;

// The top-k-off fast path (nucleus prefix only) against the full-sort reference, draw
// for draw: top-p alone (peaked, flat and far-tailed logits: the prefix grows, or is
// everything), min-p alone, both, with penalties, and no truncation at all.
test "sampler nucleus prefix equals the full-sort chain with top-k off" {
    // 48 rounds of 20 draws (960 in all): a round's draws are sequential (penalty history),
    // rounds run concurrently, so the longest round bounds the test.
    try parallel.rounds(48, {}, nucleusRound);
}

fn nucleusRound(_: void, round: usize) !void {
    const a = t.allocator;
    var prng = parallel.seed(0x70b9, round);
    const random = prng.random();
    const Shape = enum { peaked, flat, tail };
    {
        const n: usize = 30000 + (round % 24) * 13;
        const shape: Shape = @enumFromInt(round % 3);
        const logits = try a.alloc(f32, n);
        defer a.free(logits);
        const top_p: f32 = switch (round % 4) {
            0 => 0.95,
            1 => 1.0,
            2 => 0.5,
            else => 0.999,
        };
        const p: sampler.Params = .{ .temperature = 0.5 + 0.25 * @as(f32, @floatFromInt(round % 3)), .top_k = if (round % 5 == 0) @intCast(n) else 0, .top_p = top_p, .min_p = if (round % 6 < 2) 0.05 else 0, .seed = round, .presence_penalty = if (round % 2 == 0) 0.9 else 0, .repetition_penalty = if (round % 3 == 0) 1.2 else 1 };
        var fast = try sampler.Sampler.init(a, n, p);
        defer fast.deinit(a);
        var reference = try sampler.Sampler.init(a, n, p);
        defer reference.deinit(a);
        reference.fast_top_k = false;
        for (0..20) |_| {
            for (logits, 0..) |*l, i| l.* = switch (shape) {
                .peaked => random.floatNorm(f32) * 4 + if (i % 997 == 0) @as(f32, 12) else 0,
                .flat => random.floatNorm(f32) * 0.3,
                // Integer logits spread over 0..-80: ties and many empty buckets.
                .tail => -@as(f32, @floatFromInt(random.intRangeAtMost(u32, 0, 80))),
            };
            const want = try reference.sample(logits);
            try t.expectEqual(want, try fast.sample(logits));
            try reference.accept(a, want);
            try fast.accept(a, want);
        }
    }
}
