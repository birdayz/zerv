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
    /// Summation and draw order (a server option, `--sampler-order`; not an API field).
    order: Order = .id,

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

/// The summation and draw orders (docs/specs/session.md, "Sampling"). Both sample the
/// same distribution; a seed draws different tokens under each.
pub const Order = enum {
    /// Default (2026-09-24): top-p's denominator and the untruncated draw in the given (id)
    /// order, `expNeg`; the vocabulary is never sorted.
    id,
    /// The previous definition: every candidate sorted descending (after top-k), every sum
    /// and the draw in that order, `@exp`. Seeded draws equal builds before 2026-09-24.
    sorted,
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
    /// Penalized logits of the vocabulary (the top-k-off fast path).
    plain: []f32,
    /// Fast paths (default): single-pass top-k selection, and with top-k off, sorting
    /// only the nucleus prefix. false: the full candidate array with partial selection
    /// or a full sort, the reference the fast paths are tested against (same draws).
    fast_top_k: bool = true,

    pub fn init(allocator: std.mem.Allocator, vocab: usize, params: Params) (std.mem.Allocator.Error || Error)!Sampler {
        try params.validate();
        const counts = try allocator.alloc(u32, vocab);
        errdefer allocator.free(counts);
        @memset(counts, 0);
        const scratch = try allocator.alloc(Candidate, vocab);
        errdefer allocator.free(scratch);
        const plain = try allocator.alloc(f32, vocab);
        errdefer allocator.free(plain);
        var touched: std.ArrayList(u32) = .empty;
        try touched.ensureTotalCapacity(allocator, 4096);
        return .{ .params = params, .prng = .init(params.seed), .counts = counts, .touched = touched, .scratch = scratch, .plain = plain };
    }
    pub fn deinit(self: *Sampler, allocator: std.mem.Allocator) void {
        allocator.free(self.counts);
        allocator.free(self.scratch);
        allocator.free(self.plain);
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
        if (p.order == .sorted) return self.chooseSorted(try self.candidates(logits));
        // top-k: one exact pass keeps the k greatest candidates (the order is total, so
        // the set is the one the full-array selection finds), then the same chain.
        if (self.fast_top_k and p.top_k > 0 and p.top_k <= max_fast_k and p.top_k < logits.len) return self.choose(try self.topK(logits, p.top_k));
        // top-k off with top-p / min-p: sort only the nucleus prefix.
        if (self.fast_top_k and (p.top_k == 0 or p.top_k >= logits.len) and (p.top_p < 1 or p.min_p > 0)) return self.nucleus(logits);
        return self.choose(try self.candidates(logits));
    }

    /// Candidates with penalties applied (only touched ids change), in id order.
    fn candidates(self: *Sampler, logits: []const f32) Error![]Candidate {
        const c = self.scratch;
        for (logits, 0..) |l, i| {
            if (!std.math.isFinite(l)) return error.NonFiniteLogits;
            c[i] = .{ .id = @intCast(i), .logit = l };
        }
        for (self.touched.items) |id| c[id].logit = self.penalize(c[id].logit, id);
        return c;
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
        return if (self.params.order == .sorted) self.chooseSorted(c) else self.choose(c);
    }

    /// Greatest `k` penalized candidates of `logits` (any order) in `scratch[0..k]`.
    /// Untouched ids are scanned with a running threshold (vectorized; most blocks are
    /// rejected by one compare); touched ids are inserted afterwards with their penalized
    /// logit. Checks every logit is finite.
    fn topK(self: *Sampler, logits: []const f32, k: u32) Error![]Candidate {
        const heap = self.scratch[0..k]; // ascending by `greater`: heap[0] is the least
        var n: usize = 0;
        const V = 16;
        const inf: @Vector(V, f32) = @splat(std.math.inf(f32));
        var i: usize = 0;
        while (i + V <= logits.len) : (i += V) {
            const v: @Vector(V, f32) = logits[i..][0..V].*;
            if (!@reduce(.And, @abs(v) < inf)) return error.NonFiniteLogits;
            if (n == k and !@reduce(.Or, v >= @as(@Vector(V, f32), @splat(heap[0].logit)))) continue;
            for (i..i + V) |id| if (self.counts[id] == 0) insert(heap, &n, .{ .id = @intCast(id), .logit = logits[id] });
        }
        for (i..logits.len) |id| {
            if (!std.math.isFinite(logits[id])) return error.NonFiniteLogits;
            if (self.counts[id] == 0) insert(heap, &n, .{ .id = @intCast(id), .logit = logits[id] });
        }
        for (self.touched.items) |id| insert(heap, &n, .{ .id = id, .logit = self.penalize(logits[id], id) });
        return heap[0..n];
    }

    fn penalize(self: *const Sampler, logit: f32, id: u32) f32 {
        const p = self.params;
        var l = logit;
        if (p.repetition_penalty != 1) l = if (l > 0) l / p.repetition_penalty else l * p.repetition_penalty;
        return l - (@as(f32, @floatFromInt(self.counts[id])) * p.frequency_penalty + p.presence_penalty);
    }

    /// top-k -> top-p -> min-p -> temperature -> draw over penalized candidates
    /// (summation orders: docs/specs/session.md, "Sampling").
    fn choose(self: *Sampler, c: []Candidate) Error!u32 {
        const p = self.params;
        if (p.temperature == 0) return argmaxCandidates(c);
        if (p.top_k > 0 and p.top_k < c.len) {
            selectTop(c, p.top_k);
            const top = c[0..p.top_k];
            std.sort.pdq(Candidate, top, {}, greater);
            return self.truncated(top, null);
        }
        // No truncation: draw over the candidates in their given order.
        if (p.top_p >= 1 and p.min_p <= 0) return self.draw(c, maxOf(c));
        // top-k off: top-p's denominator in the given order, then the sorted chain.
        const total = if (p.top_p < 1) denominator(c, maxOf(c)) else 0;
        std.sort.pdq(Candidate, c, {}, greater);
        return self.truncated(c, total);
    }

    /// `Order.sorted`: the chain over all candidates sorted descending (after top-k's
    /// selection), every sum and the draw in that order, with `@exp`. The sampler of
    /// builds before 2026-09-24, kept operation for operation (`tests/session.zig` checks
    /// its draws against goldens generated with that source).
    fn chooseSorted(self: *Sampler, c: []Candidate) Error!u32 {
        const p = self.params;
        var n: usize = c.len;
        if (p.temperature == 0) return argmaxCandidates(c[0..n]);
        if (p.top_k > 0 and p.top_k < n) {
            selectTop(c[0..n], p.top_k);
            n = p.top_k;
        }
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

    /// Candidates sorted descending: top-p (denominator `total`, or the sum over `c` in
    /// sorted order when null), min-p, then the draw. When `c` is a prefix of a longer
    /// sorted set, `total` covers that set and top-p's cutoff must lie inside `c`.
    fn truncated(self: *Sampler, c: []Candidate, total: ?f64) Error!u32 {
        const p = self.params;
        var n: usize = c.len;
        const max = c[0].logit;
        if (p.top_p < 1) {
            const sum = total orelse denominator(c, max);
            n = cutoff(c, max, sum, p.top_p) orelse c.len;
        }
        if (p.min_p > 0) {
            // Relative to the (renormalized) top probability: p_i >= min_p * p_max.
            const threshold = @log(@as(f64, p.min_p));
            var keep: usize = 1;
            while (keep < n and @as(f64, c[keep].logit) - max >= threshold) keep += 1;
            n = keep;
        }
        return self.draw(c[0..n], max);
    }

    /// Categorical draw at the temperature over `c` in its order; `max` = the greatest
    /// logit. The weights are computed 4 at a time (`expNeg`, the same values as one at a
    /// time) and summed in order.
    fn draw(self: *Sampler, c: []const Candidate, max: f32) u32 {
        const p = self.params;
        const W = struct {
            fn at(cs: []const Candidate, i: usize, m: f32, temperature: f32) @Vector(4, f64) {
                var v: @Vector(4, f64) = undefined;
                inline for (0..4) |j| v[j] = @as(f64, cs[i + j].logit);
                return expNeg((v - @as(@Vector(4, f64), @splat(m))) / @as(@Vector(4, f64), @splat(temperature)));
            }
        };
        var total: f64 = 0;
        var i: usize = 0;
        while (i + 4 <= c.len) : (i += 4) {
            const e = W.at(c, i, max, p.temperature);
            inline for (0..4) |j| total += e[j];
        }
        for (c[i..]) |x| total += expNeg((@as(f64, x.logit) - max) / p.temperature);
        const r = self.prng.random().float(f64) * total;
        var cumulative: f64 = 0;
        i = 0;
        while (i + 4 <= c.len) : (i += 4) {
            const e = W.at(c, i, max, p.temperature);
            inline for (0..4) |j| {
                cumulative += e[j];
                if (r < cumulative) return c[i + j].id;
            }
        }
        for (c[i..]) |x| {
            cumulative += expNeg((@as(f64, x.logit) - max) / p.temperature);
            if (r < cumulative) return x.id;
        }
        return c[c.len - 1].id;
    }

    /// top-k off, top-p and/or min-p on (the fast path of `choose`'s top-k-off branch,
    /// draw for draw equal): the penalized logits, their maximum and top-p's denominator
    /// in id order, with the mass histogram by distance below the maximum; then only the
    /// candidates of the smallest bucket prefix holding the nucleus are sorted. Buckets
    /// are strict logit ranges (the bucket index is monotone in the logit), so the sorted
    /// prefix is exactly the start of the full descending order; if top-p's cutoff falls
    /// outside it (rounding), the prefix grows until it contains it or is everything.
    fn nucleus(self: *Sampler, logits: []const f32) Error!u32 {
        const p = self.params;
        const l = self.plain;
        const V = 16;
        const inf: @Vector(V, f32) = @splat(std.math.inf(f32));
        var vmax: @Vector(V, f32) = @splat(-std.math.inf(f32));
        var i: usize = 0;
        while (i + V <= logits.len) : (i += V) {
            const v: @Vector(V, f32) = logits[i..][0..V].*;
            if (!@reduce(.And, @abs(v) < inf)) return error.NonFiniteLogits;
            l[i..][0..V].* = v;
        }
        for (i..logits.len) |id| {
            if (!std.math.isFinite(logits[id])) return error.NonFiniteLogits;
            l[id] = logits[id];
        }
        for (self.touched.items) |id| l[id] = self.penalize(logits[id], id);
        i = 0;
        while (i + V <= l.len) : (i += V) vmax = @max(vmax, @as(@Vector(V, f32), l[i..][0..V].*));
        var max = @reduce(.Max, vmax);
        for (l[i..]) |x| max = @max(max, x);
        const c = self.scratch;
        if (p.top_p >= 1) {
            // min-p only: exactly the candidates min-p keeps (the test is monotone).
            const threshold = @log(@as(f64, p.min_p));
            var n: usize = 0;
            for (l, 0..) |x, id| if (@as(f64, x) - max >= threshold) {
                c[n] = .{ .id = @intCast(id), .logit = x };
                n += 1;
            };
            std.sort.pdq(Candidate, c[0..n], {}, greater);
            return self.truncated(c[0..n], 0);
        }
        var total: f64 = 0;
        var mass: [buckets]f64 = @splat(0);
        var k: usize = 0;
        while (k + 4 <= l.len) : (k += 4) {
            var v: @Vector(4, f64) = undefined;
            inline for (0..4) |j| v[j] = @as(f64, l[k + j]);
            const e = expNeg(v - @as(@Vector(4, f64), @splat(max)));
            inline for (0..4) |j| {
                total += e[j];
                mass[bucket(l[k + j], max)] += e[j];
            }
        }
        for (l[k..]) |x| {
            const e = expNeg(@as(f64, x) - max);
            total += e;
            mass[bucket(x, max)] += e;
        }
        var last: usize = 0; // the prefix: buckets 0..last
        var acc: f64 = mass[0];
        while (last + 1 < buckets and acc < p.top_p * total) {
            last += 1;
            acc += mass[last];
        }
        while (true) {
            var n: usize = 0;
            for (l, 0..) |x, id| if (bucket(x, max) <= last) {
                c[n] = .{ .id = @intCast(id), .logit = x };
                n += 1;
            };
            std.sort.pdq(Candidate, c[0..n], {}, greater);
            if (last + 1 == buckets or cutoff(c[0..n], max, total, p.top_p) != null) return self.truncated(c[0..n], total);
            last += 1;
            while (last + 1 < buckets and mass[last] == 0) last += 1;
        }
    }
};

/// Quarter-logit buckets below the maximum; the last holds everything further.
const buckets = 256;
fn bucket(x: f32, max: f32) usize {
    const d = (max - x) * 4;
    return if (d >= buckets - 1) buckets - 1 else @intFromFloat(d);
}

fn maxOf(c: []const Candidate) f32 {
    var m = c[0].logit;
    for (c[1..]) |x| m = @max(m, x.logit);
    return m;
}

/// Top-p's denominator: sum of exp(l - max) over `c` in its order (f64 terms).
fn denominator(c: []const Candidate, max: f32) f64 {
    var total: f64 = 0;
    for (c) |x| total += expNeg(@as(f64, x.logit) - max);
    return total;
}

/// The sampler's exponential for x <= 0 (f64 or a vector of f64; docs/specs/session.md):
/// 0 below -708; otherwise k = round(x / ln 2), r = (x - k ln2_hi) - k ln2_lo (Cody-Waite,
/// ln2_hi with 32 trailing zero bits, so k ln2_hi is exact), e^r by the degree-13 Taylor
/// polynomial in Horner form with fused multiply-adds, times 2^k built from exponent bits.
/// Every operation is elementwise and correctly rounded, so scalar and vector calls return
/// the same values. Relative error below 2^-52 (tests/session.zig checks against @exp).
pub fn expNeg(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const vector = @typeInfo(T) == .vector;
    const S = struct {
        fn c(v: f64) T {
            return if (vector) @splat(v) else v;
        }
    };
    const xc = @max(x, S.c(-708.0));
    const k = @round(xc * S.c(1.4426950408889634));
    const r = @mulAdd(T, k, S.c(-1.9082149292705877e-10), @mulAdd(T, k, S.c(-6.93147180369123816490e-01), xc));
    const coefficients = [_]f64{ 1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0 };
    var poly = S.c(coefficients[0]);
    inline for (coefficients[1..]) |coefficient| poly = @mulAdd(T, poly, r, S.c(coefficient));
    if (vector) {
        const n = @typeInfo(T).vector.len;
        const bits: @Vector(n, u64) = @bitCast((@as(@Vector(n, i64), @intFromFloat(k)) + @as(@Vector(n, i64), @splat(1023))) << @as(@Vector(n, u6), @splat(52)));
        const scaled = poly * @as(T, @bitCast(bits));
        return @select(f64, x < S.c(-708.0), S.c(0), scaled);
    } else {
        if (x < -708.0) return 0;
        const bits: u64 = @bitCast((@as(i64, @intFromFloat(k)) + 1023) << 52);
        return poly * @as(f64, @bitCast(bits));
    }
}

/// Top-p's cutoff over sorted `c`: the kept count, or null if the cumulative
/// probability never reaches `top_p` inside `c`.
fn cutoff(c: []const Candidate, max: f32, total: f64, top_p: f32) ?usize {
    var cumulative: f64 = 0;
    for (c, 0..) |x, i| {
        cumulative += expNeg(@as(f64, x.logit) - max) / total;
        if (cumulative >= top_p) return i + 1;
    }
    return null;
}

/// Largest top-k served by the single-pass selection (insertion into a sorted array).
const max_fast_k = 256;

/// Inserts `c` into `heap[0..n]` (ascending by `greater`, capacity heap.len) if it is
/// among the heap.len greatest seen.
fn insert(heap: []Candidate, n: *usize, c: Candidate) void {
    var at: usize = n.*;
    if (n.* == heap.len) {
        if (!greater({}, c, heap[0])) return;
        at = 0;
        // Drop the least: shift up while c beats the next.
        while (at + 1 < heap.len and greater({}, c, heap[at + 1])) : (at += 1) heap[at] = heap[at + 1];
        heap[at] = c;
        return;
    }
    // Not full: insertion from the top.
    while (at > 0 and greater({}, heap[at - 1], c)) : (at -= 1) heap[at] = heap[at - 1];
    heap[at] = c;
    n.* += 1;
}

fn greater(_: void, a: Candidate, b: Candidate) bool {
    return a.logit > b.logit or (a.logit == b.logit and a.id < b.id);
}

/// First index of the maximum (vectorized maximum, then the first position holding it).
pub fn greedy(logits: []const f32) Error!u32 {
    const V = 16;
    const inf: @Vector(V, f32) = @splat(std.math.inf(f32));
    var vmax: @Vector(V, f32) = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + V <= logits.len) : (i += V) {
        const v: @Vector(V, f32) = logits[i..][0..V].*;
        if (!@reduce(.And, @abs(v) < inf)) return error.NonFiniteLogits;
        vmax = @max(vmax, v);
    }
    var max = @reduce(.Max, vmax);
    for (logits[i..]) |l| {
        if (!std.math.isFinite(l)) return error.NonFiniteLogits;
        max = @max(max, l);
    }
    for (logits, 0..) |l, j| if (l == max) return @intCast(j);
    unreachable;
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
