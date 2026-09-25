//! Speculative decoding verify-count policy (block 17b, docs/specs/speculative.md).
//! Drafts change only speed, never output (sample matching), so every choice here is
//! lossless. After drafting N tokens with probabilities p_1..p_N (the drafter's softmax
//! probability of each draft), verify the k drafts (k + 1 rows) that maximize expected
//! tokens per second:
//!   E(k) = 1 + sum_{i<=k} prod_{j<=i} a(p_j),  T(k) = draft + verify(k + 1) + commit,
//! where a(p) is the measured acceptance rate of drafts in p's probability bin and the
//! costs are measured wall times (exponential moving averages). Pure CPU; no allocation.
const std = @import("std");

pub const max_drafts = 8;
pub const bins = 10;

pub const Policy = struct {
    /// Acceptance counts per probability bin (a draft is tried when every earlier draft of
    /// its step was accepted), seeded with `prior` pseudo-trials at the bin center.
    accepted: [bins]f64,
    tried: [bins]f64,
    /// Wall-time averages (ns): verify of n rows at [n - 1] (NaN until measured), drafting,
    /// commit.
    verify_ns: [max_drafts + 1]f64,
    draft_ns: f64,
    commit_ns: f64,

    pub const prior = 2.0;
    /// Moving-average weight of a new timing.
    pub const weight = 0.1;
    /// Relative verify cost of n rows when no time for n is measured yet (a prior, scaled
    /// by the measured counts): measured on the RX 7900 XTX (docs/bench/2026-09-24-spec-verify.md).
    pub const relative_cost = [max_drafts + 1]f64{ 1.03, 1.07, 1.2, 1.35, 1.62, 1.9, 2.2, 2.5, 2.8 };

    pub fn init() Policy {
        var p: Policy = .{ .accepted = undefined, .tried = undefined, .verify_ns = @splat(std.math.nan(f64)), .draft_ns = 0, .commit_ns = 0 };
        for (&p.accepted, &p.tried, 0..) |*a, *t, b| {
            t.* = prior;
            a.* = prior * (@as(f64, @floatFromInt(b)) + 0.5) / bins;
        }
        return p;
    }

    fn bin(p: f32) usize {
        if (!(p > 0)) return 0; // also NaN
        return @min(@as(usize, @intFromFloat(p * bins)), bins - 1);
    }

    /// Estimated probability that a draft with drafter probability `p` is accepted.
    pub fn acceptance(self: *const Policy, p: f32) f64 {
        const b = bin(p);
        return self.accepted[b] / self.tried[b];
    }

    /// Verify cost estimate of n rows: measured, else the prior scaled from the nearest
    /// measured count (or 1 when nothing is measured: only the ratios matter then).
    fn verifyCost(self: *const Policy, n: usize) f64 {
        if (!std.math.isNan(self.verify_ns[n - 1])) return self.verify_ns[n - 1];
        var best: ?usize = null;
        for (self.verify_ns, 1..) |v, m| {
            if (std.math.isNan(v)) continue;
            if (best == null or @abs(@as(isize, @intCast(m)) - @as(isize, @intCast(n))) < @abs(@as(isize, @intCast(best.?)) - @as(isize, @intCast(n)))) best = m;
        }
        const m = best orelse return relative_cost[n - 1];
        return self.verify_ns[m - 1] * relative_cost[n - 1] / relative_cost[m - 1];
    }

    /// Drafts to verify, 0..probs.len (probs.len <= max_drafts).
    pub fn choose(self: *const Policy, probs: []const f32) u32 {
        std.debug.assert(probs.len <= max_drafts);
        var best_k: u32 = 0;
        var best_rate: f64 = -1;
        var expected: f64 = 1;
        var chain: f64 = 1;
        for (0..probs.len + 1) |k| {
            if (k > 0) {
                chain *= self.acceptance(probs[k - 1]);
                expected += chain;
            }
            const cost = self.draft_ns + self.commit_ns + self.verifyCost(k + 1);
            const rate = expected / cost;
            if (rate > best_rate) {
                best_rate = rate;
                best_k = @intCast(k);
            }
        }
        return best_k;
    }

    /// Outcome of a step that verified `probs.len` drafts: the first `accepted` matched.
    pub fn observe(self: *Policy, probs: []const f32, accepted: usize) void {
        std.debug.assert(accepted <= probs.len);
        for (probs[0..@min(accepted + 1, probs.len)], 0..) |p, i| {
            const b = bin(p);
            self.tried[b] += 1;
            if (i < accepted) self.accepted[b] += 1;
        }
    }

    fn average(old: f64, new: f64) f64 {
        return if (std.math.isNan(old) or old == 0) new else old + weight * (new - old);
    }
    pub fn timeVerify(self: *Policy, rows: usize, ns: u64) void {
        self.verify_ns[rows - 1] = average(self.verify_ns[rows - 1], @floatFromInt(ns));
    }
    pub fn timeDraft(self: *Policy, ns: u64) void {
        self.draft_ns = average(self.draft_ns, @floatFromInt(ns));
    }
    pub fn timeCommit(self: *Policy, ns: u64) void {
        self.commit_ns = average(self.commit_ns, @floatFromInt(ns));
    }
};
