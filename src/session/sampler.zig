//! Token selection from FP32 logits. Driver-free; bounded scratch owned by the caller.
//! Order (llama.cpp-compatible chain): penalties -> top-k -> top-p -> min-p ->
//! temperature -> categorical draw. Temperature 0 is greedy (lowest index on ties).
//! The PRNG is Xoshiro256++ seeded explicitly; draws are not bit-compatible with
//! llama.cpp's mt19937, so only greedy output is comparable token-for-token.
const std = @import("std");

pub const Params = struct {
    temperature: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    top_p: f32 = 1.0, // 1 = disabled
    min_p: f32 = 0.0, // 0 = disabled
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    repetition_penalty: f32 = 1.0, // 1 = disabled
    seed: u64 = 0,

    pub fn validate(p: Params) error{InvalidSampling}!void {
        const ok = std.math.isFinite(p.temperature) and p.temperature >= 0 and p.temperature <= 100 and
            std.math.isFinite(p.top_p) and p.top_p > 0 and p.top_p <= 1 and
            std.math.isFinite(p.min_p) and p.min_p >= 0 and p.min_p <= 1 and
            std.math.isFinite(p.presence_penalty) and @abs(p.presence_penalty) <= 2 and
            std.math.isFinite(p.frequency_penalty) and @abs(p.frequency_penalty) <= 2 and
            std.math.isFinite(p.repetition_penalty) and p.repetition_penalty > 0 and p.repetition_penalty <= 10;
        if (!ok) return error.InvalidSampling;
    }
};

pub const Candidate = struct { id: u32, logit: f32 };
pub const Error = error{ InvalidSampling, NonFiniteLogits, InvalidLogits };

/// Owns per-sequence penalty counts and candidate scratch sized to the vocabulary.
pub const Sampler = struct {
    params: Params,
    prng: std.Random.Xoshiro256,
    counts: []u32,
    touched: std.ArrayList(u32),
    scratch: []Candidate,

    pub fn init(allocator: std.mem.Allocator, vocab: usize, params: Params) (std.mem.Allocator.Error || Error)!Sampler {
        try params.validate();
        const counts = try allocator.alloc(u32, vocab);
        errdefer allocator.free(counts);
        @memset(counts, 0);
        const scratch = try allocator.alloc(Candidate, vocab);
        errdefer allocator.free(scratch);
        var touched: std.ArrayList(u32) = .empty;
        try touched.ensureTotalCapacity(allocator, 4096);
        return .{ .params = params, .prng = .init(params.seed), .counts = counts, .touched = touched, .scratch = scratch };
    }
    pub fn deinit(self: *Sampler, allocator: std.mem.Allocator) void {
        allocator.free(self.counts);
        allocator.free(self.scratch);
        self.touched.deinit(allocator);
    }

    /// Record a generated token for presence/frequency/repetition penalties.
    pub fn accept(self: *Sampler, allocator: std.mem.Allocator, id: u32) std.mem.Allocator.Error!void {
        if (self.counts[id] == 0) try self.touched.append(allocator, id);
        self.counts[id] += 1;
    }

    pub fn sample(self: *Sampler, logits: []const f32) Error!u32 {
        if (logits.len != self.counts.len or logits.len == 0) return error.InvalidLogits;
        const p = self.params;
        const penalized = p.presence_penalty != 0 or p.frequency_penalty != 0 or p.repetition_penalty != 1;
        if (p.temperature == 0 and !penalized) return greedy(logits);
        // Candidates with penalties applied (only touched ids change).
        const c = self.scratch;
        for (logits, 0..) |l, i| {
            if (!std.math.isFinite(l)) return error.NonFiniteLogits;
            c[i] = .{ .id = @intCast(i), .logit = l };
        }
        for (self.touched.items) |id| c[id].logit = self.penalize(c[id].logit, id);
        return self.choose(c);
    }

    /// Samples among `allowed` (distinct token ids) only, with the same chain: the
    /// logits of every other token are treated as -inf (a grammar mask).
    pub fn sampleFrom(self: *Sampler, logits: []const f32, allowed: []const u32) Error!u32 {
        if (logits.len != self.counts.len or allowed.len == 0 or allowed.len > logits.len) return error.InvalidLogits;
        const c = self.scratch[0..allowed.len];
        for (allowed, c) |id, *slot| {
            if (id >= logits.len) return error.InvalidLogits;
            const l = logits[id];
            if (!std.math.isFinite(l)) return error.NonFiniteLogits;
            slot.* = .{ .id = id, .logit = if (self.counts[id] > 0) self.penalize(l, id) else l };
        }
        return self.choose(c);
    }

    fn penalize(self: *const Sampler, logit: f32, id: u32) f32 {
        const p = self.params;
        var l = logit;
        if (p.repetition_penalty != 1) l = if (l > 0) l / p.repetition_penalty else l * p.repetition_penalty;
        return l - (@as(f32, @floatFromInt(self.counts[id])) * p.frequency_penalty + p.presence_penalty);
    }

    /// top-k -> top-p -> min-p -> temperature -> draw over penalized candidates.
    fn choose(self: *Sampler, c: []Candidate) Error!u32 {
        const p = self.params;
        var n: usize = c.len;
        if (p.temperature == 0) return argmaxCandidates(c[0..n]);
        if (p.top_k > 0 and p.top_k < n) {
            selectTop(c[0..n], p.top_k);
            n = p.top_k;
        }
        // Sorted descending (stable by id) from here on.
        std.sort.pdq(Candidate, c[0..n], {}, greater);
        if (p.top_p < 1 or p.min_p > 0) {
            const max = c[0].logit;
            var total: f64 = 0;
            for (c[0..n]) |x| total += @exp(@as(f64, x.logit) - max);
            if (p.top_p < 1) {
                var cumulative: f64 = 0;
                var keep: usize = n;
                for (c[0..n], 0..) |x, i| {
                    cumulative += @exp(@as(f64, x.logit) - max) / total;
                    if (cumulative >= p.top_p) {
                        keep = i + 1;
                        break;
                    }
                }
                n = keep;
            }
            if (p.min_p > 0) {
                // Relative to the (renormalized) top probability: p_i >= min_p * p_max.
                const threshold = @log(@as(f64, p.min_p));
                var keep: usize = 1;
                while (keep < n and @as(f64, c[keep].logit) - max >= threshold) keep += 1;
                n = keep;
            }
        }
        const max = c[0].logit;
        var total: f64 = 0;
        for (c[0..n]) |x| total += @exp((@as(f64, x.logit) - max) / p.temperature);
        const r = self.prng.random().float(f64) * total;
        var cumulative: f64 = 0;
        for (c[0..n]) |x| {
            cumulative += @exp((@as(f64, x.logit) - max) / p.temperature);
            if (r < cumulative) return x.id;
        }
        return c[n - 1].id;
    }
};

fn greater(_: void, a: Candidate, b: Candidate) bool {
    return a.logit > b.logit or (a.logit == b.logit and a.id < b.id);
}

pub fn greedy(logits: []const f32) Error!u32 {
    var best: usize = 0;
    for (logits, 0..) |l, i| {
        if (!std.math.isFinite(l)) return error.NonFiniteLogits;
        if (l > logits[best]) best = i;
    }
    return @intCast(best);
}

fn argmaxCandidates(c: []const Candidate) u32 {
    var best = c[0];
    for (c[1..]) |x| if (greater({}, x, best)) {
        best = x;
    };
    return best.id;
}

/// Partial selection: the k greatest candidates (in any order) end up in c[0..k].
fn selectTop(c: []Candidate, k: usize) void {
    var lo: usize = 0;
    var hi: usize = c.len;
    while (hi - lo > 1) {
        const pivot = c[lo + (hi - 1 - lo) / 2]; // never the last element (Hoare)
        var i = lo;
        var j = hi - 1;
        while (true) {
            while (greater({}, c[i], pivot)) i += 1;
            while (greater({}, pivot, c[j])) j -= 1;
            if (i >= j) break;
            std.mem.swap(Candidate, &c[i], &c[j]);
            i += 1;
            j -= 1;
        }
        // c[lo..j+1] >= pivot >= c[j+1..hi] (Hoare partition)
        if (k <= j + 1) hi = j + 1 else lo = j + 1;
        if (lo >= k or hi <= k) break;
    }
}
