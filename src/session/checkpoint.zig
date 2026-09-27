//! Prefix checkpoints of the shared KV pool (docs/specs/concurrent.md, "18d.4 design"):
//! which token prefixes have a recurrent-state snapshot and pinned KV pages, and where a
//! prompt should take new ones. Pure bookkeeping, owned by the scheduler side; the backend
//! does the device work (snapshot save/load, pin/unpin, attach).
const std = @import("std");

pub const Entry = struct {
    /// Prefix length (0: unused).
    len: u32 = 0,
    /// Least-recently-used order (larger: used later).
    used: u64 = 0,
    /// The pool pages holding positions below `len` (logical order), pinned by the backend.
    npages: u32 = 0,
};

/// Most checkpoints taken while prefilling one prompt.
pub const max_points = 2;

/// The checkpoint positions of `prompt` after `start` without the store's knowledge (a
/// generation's side; the backend skips prefixes it holds): the first message boundary
/// after position 0 and the last one, each after `start` and before the last token.
pub fn candidatePoints(prompt: []const u32, start: u32, boundary: u32, out: *[max_points]u32) []const u32 {
    var first: ?u32 = null;
    var last: ?u32 = null;
    if (prompt.len > 1) for (prompt[1..], 1..) |token, p| {
        if (token != boundary) continue;
        if (first == null) first = @intCast(p);
        last = @intCast(p);
    };
    var n: usize = 0;
    for ([_]?u32{ first, if (last != null and first != null and last.? == first.?) null else last }) |candidate| {
        const p = candidate orelse continue;
        if (p <= start or p >= prompt.len) continue;
        out[n] = p;
        n += 1;
    }
    return out[0..n];
}

pub const Store = struct {
    /// One entry per snapshot slot: entry i's snapshot is backend snapshot slot i.
    entries: []Entry,
    /// Entry i's tokens at `tokens[i * context ..][0..len]` and pages at
    /// `pages[i * max_pages ..][0..npages]`.
    tokens: []u32,
    pages: []u32,
    context: u32,
    max_pages: u32,
    /// Token that starts every chat message (`<|im_start|>`).
    boundary: u32,
    tick: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, count: u32, context: u32, max_pages: u32, boundary: u32) (std.mem.Allocator.Error || error{InvalidOptions})!Store {
        if (count == 0 or context == 0 or max_pages == 0) return error.InvalidOptions;
        const entries = try allocator.alloc(Entry, count);
        errdefer allocator.free(entries);
        @memset(entries, .{});
        const tokens = try allocator.alloc(u32, @as(usize, count) * context);
        errdefer allocator.free(tokens);
        const pages = try allocator.alloc(u32, @as(usize, count) * max_pages);
        return .{ .entries = entries, .tokens = tokens, .pages = pages, .context = context, .max_pages = max_pages, .boundary = boundary };
    }

    pub fn deinit(self: *Store, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.free(self.tokens);
        allocator.free(self.pages);
        self.* = undefined;
    }

    pub fn entryTokens(self: *const Store, i: usize) []const u32 {
        return self.tokens[i * self.context ..][0..self.entries[i].len];
    }
    pub fn entryPages(self: *const Store, i: usize) []const u32 {
        return self.pages[i * self.max_pages ..][0..self.entries[i].npages];
    }

    /// The live entry with the longest prefix of `prompt` that leaves at least one prompt
    /// token to process (its logits are needed), or null.
    pub fn lookup(self: *const Store, prompt: []const u32) ?usize {
        var best: ?usize = null;
        for (self.entries, 0..) |e, i| {
            if (e.len == 0 or e.len >= prompt.len) continue;
            if (best != null and e.len <= self.entries[best.?].len) continue;
            if (std.mem.eql(u32, self.entryTokens(i), prompt[0..e.len])) best = i;
        }
        return best;
    }

    /// Whether a live entry holds exactly `prefix`.
    pub fn has(self: *const Store, prefix: []const u32) bool {
        for (self.entries, 0..) |e, i| {
            if (e.len == prefix.len and std.mem.eql(u32, self.entryTokens(i), prefix)) return true;
        }
        return false;
    }

    pub fn touch(self: *Store, i: usize) void {
        self.tick += 1;
        self.entries[i].used = self.tick;
    }

    /// Checkpoint positions for prefilling `prompt[start..]`, ascending: the first message
    /// boundary after position 0 (the end of the system prompt) and the last one (the start
    /// of the generation prompt), each when it lies after `start`, before the last token,
    /// and no entry holds that prefix yet.
    pub fn points(self: *const Store, prompt: []const u32, start: u32, out: *[max_points]u32) []const u32 {
        var candidates: [max_points]u32 = undefined;
        var n: usize = 0;
        for (candidatePoints(prompt, start, self.boundary, &candidates)) |p| {
            if (p > self.context or self.has(prompt[0..p])) continue;
            out[n] = p;
            n += 1;
        }
        return out[0..n];
    }

    /// The entry to fill next: a free one, else the eviction victim (`victim`; the caller
    /// must release its snapshot and pages first: `entryPages` before `fill`).
    pub fn claim(self: *const Store) usize {
        for (self.entries, 0..) |e, i| if (e.len == 0) return i;
        return self.victim(null).?;
    }

    /// Whether live entry `i` is a proper prefix of another live entry (a parent: others
    /// build on it, so it is worth more than a leaf and frees little memory).
    pub fn isParent(self: *const Store, i: usize) bool {
        const mine = self.entryTokens(i);
        for (self.entries, 0..) |e, j| {
            if (j == i or e.len <= mine.len) continue;
            if (std.mem.eql(u32, self.entryTokens(j)[0..mine.len], mine)) return true;
        }
        return false;
    }

    /// Eviction choice (docs/specs/concurrent.md, "18d.4 design"): leaves first, least
    /// recently used among them; a parent only when every live entry is one (cannot happen
    /// with a finite chain, kept for safety). `keep` is never chosen; null: no live entry.
    pub fn victim(self: *const Store, keep: ?usize) ?usize {
        var best: ?usize = null;
        var best_parent = true;
        for (self.entries, 0..) |e, i| {
            if (e.len == 0 or (keep != null and i == keep.?)) continue;
            const parent = self.isParent(i);
            const better = best == null or (best_parent and !parent) or (parent == best_parent and e.used < self.entries[best.?].used);
            if (better) {
                best = i;
                best_parent = parent;
            }
        }
        return best;
    }

    /// Entry `i` now holds `prefix` with `pages` (at most `max_pages`); it is the most recent.
    pub fn fill(self: *Store, i: usize, prefix: []const u32, pages: []const u32) error{InvalidOptions}!void {
        if (prefix.len == 0 or prefix.len > self.context or pages.len > self.max_pages) return error.InvalidOptions;
        @memcpy(self.tokens[i * self.context ..][0..prefix.len], prefix);
        @memcpy(self.pages[i * self.max_pages ..][0..pages.len], pages);
        self.entries[i].len = @intCast(prefix.len);
        self.entries[i].npages = @intCast(pages.len);
        self.touch(i);
    }

    /// The entry memory pressure drops first (`victim`), or null.
    pub fn oldest(self: *const Store) ?usize {
        return self.victim(null);
    }

    /// Entry `i` becomes free (its pages already released by the caller).
    pub fn drop(self: *Store, i: usize) void {
        self.entries[i] = .{};
    }

    pub fn live(self: *const Store) u32 {
        var n: u32 = 0;
        for (self.entries) |e| n += @intFromBool(e.len != 0);
        return n;
    }
};
