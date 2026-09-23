//! Prefix-cache policy (docs/specs/prefix-cache.md): which processed tokens a new prompt
//! can reuse, where to restore from, and where to take recurrent-state snapshots.
//! Driver-free bookkeeping; the backend performs the resets, restores and saves this
//! decides. Not thread-safe: one sequence, serialized by the caller.
const std = @import("std");

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
    snaps: [max_slots]Snapshot = undefined,
    count: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, context: u32, options: Options) (std.mem.Allocator.Error || error{InvalidOptions})!Cache {
        if (options.slots == 0 or options.slots > max_slots or options.spacing == 0 or context == 0) return error.InvalidOptions;
        return .{ .options = options, .history = try allocator.alloc(u32, context) };
    }

    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        allocator.free(self.history);
    }

    /// Forget everything; the next prompt resets the backend.
    pub fn invalidate(self: *Cache) void {
        self.len = 0;
        self.count = 0;
    }

    /// Chooses how to start `prompt` (non-empty, at most the context) and updates the
    /// bookkeeping to that start: later snapshots are dropped and the history is
    /// truncated. The caller must then reset or restore the backend as returned, or
    /// `invalidate` on failure.
    pub fn begin(self: *Cache, prompt: []const u32) Begin {
        std.debug.assert(prompt.len > 0 and prompt.len <= self.history.len);
        const n: u32 = @intCast(prompt.len);
        const d: u32 = @intCast(lcp(self.history[0..self.len], prompt));
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
        return result;
    }

    /// Snapshot positions for prefilling `prompt[start..]`, ascending, in `out`.
    pub fn points(self: *const Cache, prompt: []const u32, start: u32, out: *[max_points]u32) []const u32 {
        var n: usize = 0;
        var last: ?u32 = if (self.count > 0) self.snaps[self.count - 1].pos else null;
        var final: ?u32 = null;
        var p: u32 = start + 1;
        while (p < prompt.len) : (p += 1) {
            if (prompt[p] != self.options.boundary) continue;
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

    /// Record tokens the backend processed successfully (appended to the history).
    pub fn record(self: *Cache, tokens: []const u32) void {
        std.debug.assert(self.len + tokens.len <= self.history.len);
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

fn lcp(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) i += 1;
    return i;
}
