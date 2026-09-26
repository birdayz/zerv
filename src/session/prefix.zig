//! Prefix-cache policy (docs/specs/prefix-cache.md): which processed tokens a new prompt
//! can reuse, where to restore from, and where to take recurrent-state snapshots.
//! Driver-free bookkeeping; the backend performs the resets, restores and saves this
//! decides. Not thread-safe: one sequence, serialized by the caller.
//! Media spans (rows fed by an image's encoder, not their placeholder tokens) are compared
//! by identity, not by token (spec, "Media spans").
const std = @import("std");
const media = @import("media.zig");
const Media = media.Media;

pub const max_slots = 64;
/// Most snapshot points taken while prefilling one prompt.
pub const max_points = max_slots;

pub const Options = struct {
    /// Backend snapshot slots (1..max_slots).
    slots: u32,
    /// Token that starts every chat message (`<|im_start|>`); snapshots go right before it.
    boundary: u32,
    /// Minimum distance between optional (non-final) snapshot points.
    spacing: u32 = 4096,
};

pub const Outcome = enum { reset, restore, keep };
pub const Begin = struct {
    outcome: Outcome,
    /// First prompt token to process; tokens before it are reused.
    start: u32,
    /// Slot to load when `outcome == .restore`.
    slot: u32 = 0,
};
const Snapshot = struct { pos: u32, slot: u32 };

pub const Cache = struct {
    options: Options,
    /// Processed tokens (capacity = context). history[0..len) is live in the backend.
    history: []u32,
    len: u32 = 0,
    /// The media spans of history[0..len), whole, in row order (capacity = context: a span
    /// has at least one row).
    spans: []Media,
    nspans: u32 = 0,
    snaps: [max_slots]Snapshot = undefined,
    count: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, context: u32, options: Options) (std.mem.Allocator.Error || error{InvalidOptions})!Cache {
        if (options.slots == 0 or options.slots > max_slots or options.spacing == 0 or context == 0) return error.InvalidOptions;
        const history = try allocator.alloc(u32, context);
        errdefer allocator.free(history);
        return .{ .options = options, .history = history, .spans = try allocator.alloc(Media, context) };
    }

    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        allocator.free(self.history);
        allocator.free(self.spans);
    }

    /// Forget everything; the next prompt resets the backend.
    pub fn invalidate(self: *Cache) void {
        self.len = 0;
        self.count = 0;
        self.nspans = 0;
    }

    /// Chooses how to start `prompt` (non-empty, at most the context; `spans` valid for it,
    /// `media.validate`) and updates the bookkeeping to that start: later snapshots are
    /// dropped and the history is truncated. The caller must then reset or restore the
    /// backend as returned, or `invalidate` on failure.
    pub fn begin(self: *Cache, prompt: []const u32, spans: []const Media) Begin {
        std.debug.assert(prompt.len > 0 and prompt.len <= self.history.len);
        const n: u32 = @intCast(prompt.len);
        const d = mediaLcp(self.spans[0..self.nspans], spans, @intCast(lcp(self.history[0..self.len], prompt)));
        var result: Begin = .{ .outcome = .reset, .start = 0 };
        if (self.len > 0 and d == self.len and self.len < n) {
            result = .{ .outcome = .keep, .start = self.len };
        } else {
            const limit = @min(d, n - 1);
            var i = self.count;
            while (i > 0) : (i -= 1) {
                const s = self.snaps[i - 1];
                if (s.pos <= limit and s.pos > 0) {
                    result = .{ .outcome = .restore, .start = s.pos, .slot = s.slot };
                    break;
                }
            }
        }
        while (self.count > 0 and self.snaps[self.count - 1].pos > result.start) self.count -= 1;
        self.len = result.start;
        // Every start (0, a snapshot position, the whole history) is at a span boundary.
        while (self.nspans > 0 and self.spans[self.nspans - 1].start >= result.start) self.nspans -= 1;
        std.debug.assert(self.nspans == 0 or self.spans[self.nspans - 1].end() <= result.start);
        return result;
    }

    /// Snapshot positions for prefilling `prompt[start..]`, ascending, in `out`. Never
    /// inside a media span, so every segment holds whole spans.
    pub fn points(self: *const Cache, prompt: []const u32, spans: []const Media, start: u32, out: *[max_points]u32) []const u32 {
        var n: usize = 0;
        var last: ?u32 = if (self.count > 0) self.snaps[self.count - 1].pos else null;
        var final: ?u32 = null;
        var p: u32 = start + 1;
        while (p < prompt.len) : (p += 1) {
            if (prompt[p] != self.options.boundary or media.inside(spans, p)) continue;
            if (final) |f| {
                // `f` was the latest boundary; take it if it qualifies as an optional point.
                if (last == null or f - last.? >= self.options.spacing) {
                    if (n < out.len - 1) {
                        out[n] = f;
                        n += 1;
                        last = f;
                    }
                }
            }
            final = p;
        }
        if (final) |f| {
            out[n] = f;
            n += 1;
        }
        return out[0..n];
    }

    /// Record tokens the backend processed successfully (appended to the history), with
    /// the media spans among them (rows relative to the prompt, whole, inside the tokens).
    pub fn record(self: *Cache, tokens: []const u32, spans: []const Media) void {
        std.debug.assert(self.len + tokens.len <= self.history.len);
        for (spans) |s| {
            std.debug.assert(s.start >= self.len and s.end() <= self.len + tokens.len);
            std.debug.assert(self.nspans == 0 or self.spans[self.nspans - 1].end() <= s.start);
            self.spans[self.nspans] = s;
            self.nspans += 1;
        }
        @memcpy(self.history[self.len..][0..tokens.len], tokens);
        self.len += @intCast(tokens.len);
    }

    /// Registers a snapshot at the current history length and returns the slot the
    /// backend must save into (a free one, else one freed by thinning).
    pub fn claim(self: *Cache) u32 {
        const pos = self.len;
        std.debug.assert(self.count == 0 or self.snaps[self.count - 1].pos < pos);
        var slot: u32 = 0;
        if (self.count < self.options.slots) {
            // Lowest slot not in use.
            var used: u64 = 0;
            for (self.snaps[0..self.count]) |s| used |= @as(u64, 1) << @intCast(s.slot);
            slot = @ctz(~used);
        } else {
            // Drop the snapshot whose removal leaves the smallest gap.
            var best: u32 = 0;
            var best_gap: u64 = std.math.maxInt(u64);
            for (0..self.count) |i| {
                const prev: u64 = if (i == 0) 0 else self.snaps[i - 1].pos;
                const next: u64 = if (i + 1 < self.count) self.snaps[i + 1].pos else pos;
                if (next - prev < best_gap) {
                    best_gap = next - prev;
                    best = @intCast(i);
                }
            }
            slot = self.snaps[best].slot;
            std.mem.copyForwards(Snapshot, self.snaps[best .. self.count - 1], self.snaps[best + 1 .. self.count]);
            self.count -= 1;
        }
        self.snaps[self.count] = .{ .pos = pos, .slot = slot };
        self.count += 1;
        return slot;
    }

    /// Snapshot positions currently held, ascending (for tests and metrics).
    pub fn positions(self: *const Cache, out: []u32) []const u32 {
        for (self.snaps[0..self.count], 0..) |s, i| out[i] = s.pos;
        return out[0..self.count];
    }
};

/// Cuts the token LCP `d` at the first span that is not in both `a` and `b` identically.
fn mediaLcp(a: []const Media, b: []const Media, d: u32) u32 {
    var cut = d;
    var i: usize = 0;
    var j: usize = 0;
    while (i < a.len or j < b.len) {
        const sa: ?Media = if (i < a.len) a[i] else null;
        const sb: ?Media = if (j < b.len) b[j] else null;
        const first = @min(if (sa) |s| s.start else std.math.maxInt(u32), if (sb) |s| s.start else std.math.maxInt(u32));
        if (first >= cut) break;
        if (sa != null and sb != null and sa.?.same(sb.?)) {
            i += 1;
            j += 1;
            continue;
        }
        cut = first;
        break;
    }
    return cut;
}

fn lcp(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) i += 1;
    return i;
}
