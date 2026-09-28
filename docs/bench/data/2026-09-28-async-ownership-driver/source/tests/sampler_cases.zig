//! Sampler draw-sequence cases shared by the `Order.sorted` golden generator
//! (tests/reference/generate_sampler_sorted.zig, run against the pre-2026-09-24 sampler
//! source) and tests/session.zig. Standard library only: both sides must build the same
//! logits from the same seeds.
const std = @import("std");

pub const Shape = enum { normal, peaked, flat, ties };

pub const Case = struct {
    vocab: u32,
    steps: u32,
    shape: Shape,
    temperature: f32,
    top_k: u32 = 0,
    top_p: f32 = 1,
    min_p: f32 = 0,
    presence_penalty: f32 = 0,
    frequency_penalty: f32 = 0,
    repetition_penalty: f32 = 1,
    seed: u64,
    /// > 0: draw with `sampleFrom` over this many distinct ids in a scrambled order.
    allowed: u32 = 0,
};

pub const cases = [_]Case{
    .{ .vocab = 4099, .steps = 48, .shape = .normal, .temperature = 1, .seed = 1 },
    .{ .vocab = 65537, .steps = 24, .shape = .normal, .temperature = 1, .seed = 2 },
    .{ .vocab = 20011, .steps = 32, .shape = .peaked, .temperature = 0.7, .top_p = 0.95, .seed = 3 },
    .{ .vocab = 20011, .steps = 32, .shape = .flat, .temperature = 0.8, .top_k = 40, .top_p = 0.9, .min_p = 0.05, .seed = 4 },
    .{ .vocab = 20011, .steps = 32, .shape = .normal, .temperature = 0.6, .top_k = 20, .top_p = 0.95, .seed = 5 },
    .{ .vocab = 4099, .steps = 48, .shape = .ties, .temperature = 1.2, .min_p = 0.1, .seed = 6 },
    .{ .vocab = 4099, .steps = 48, .shape = .ties, .temperature = 0.6, .top_p = 0.8, .presence_penalty = 1.5, .frequency_penalty = 0.5, .repetition_penalty = 1.1, .seed = 7 },
    .{ .vocab = 4099, .steps = 24, .shape = .normal, .temperature = 0, .presence_penalty = 1.5, .seed = 8 },
    .{ .vocab = 4099, .steps = 32, .shape = .normal, .temperature = 1, .allowed = 50, .seed = 9 },
    .{ .vocab = 4099, .steps = 32, .shape = .flat, .temperature = 0.9, .top_p = 0.7, .allowed = 300, .seed = 10 },
};

/// Sampler parameters of `c` as `P` (the fields of `sampler.Params` at both revisions).
pub fn params(comptime P: type, c: Case) P {
    var p: P = .{};
    p.temperature = c.temperature;
    p.top_k = c.top_k;
    p.top_p = c.top_p;
    p.min_p = c.min_p;
    p.presence_penalty = c.presence_penalty;
    p.frequency_penalty = c.frequency_penalty;
    p.repetition_penalty = c.repetition_penalty;
    p.seed = c.seed;
    return p;
}

/// The logits of one step.
pub fn fill(random: std.Random, shape: Shape, logits: []f32) void {
    for (logits, 0..) |*l, i| l.* = switch (shape) {
        .normal => random.floatNorm(f32) * 3,
        .peaked => random.floatNorm(f32) * 4 + if (i % 997 == 0) @as(f32, 12) else 0,
        .flat => random.floatNorm(f32) * 0.3,
        .ties => -@as(f32, @floatFromInt(random.intRangeAtMost(u32, 0, 40))) / 4,
    };
}

/// `n` distinct ids below `vocab` in a scrambled order (the same for every step).
pub fn allowedIds(n: u32, vocab: u32, out: []u32) []u32 {
    for (out[0..n], 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % vocab);
    return out[0..n];
}

/// Runs case `c` with a sampler of type `S` (init/sample/sampleFrom/accept/deinit),
/// writing the drawn ids to `draws[0..c.steps]`.
pub fn run(comptime S: type, comptime P: type, allocator: std.mem.Allocator, c: Case, draws: []u32) !void {
    var s = try S.init(allocator, c.vocab, params(P, c));
    defer s.deinit(allocator);
    const logits = try allocator.alloc(f32, c.vocab);
    defer allocator.free(logits);
    const ids = try allocator.alloc(u32, c.vocab);
    defer allocator.free(ids);
    const allowed = allowedIds(c.allowed, c.vocab, ids);
    var prng = std.Random.DefaultPrng.init(c.seed *% 0x9e3779b97f4a7c15 +% 1);
    for (draws[0..c.steps]) |*draw| {
        fill(prng.random(), c.shape, logits);
        draw.* = if (c.allowed > 0) try s.sampleFrom(logits, allowed) else try s.sample(logits);
        try s.accept(allocator, draw.*);
    }
}
