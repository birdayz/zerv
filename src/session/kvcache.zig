//! Prefix cache of the shared KV pool behind an interface (docs/specs/concurrent.md,
//! "18d.4 design"): the policy (`Cache`: which prefixes to keep, restore, drop, share)
//! over the mechanism (`Device`: pages, pins, snapshots of one model). Implementations:
//! `Flat` (a list of checkpoints) and `Radix` (a prefix tree with deduplication on insert).
//! Pure host bookkeeping, called on the scheduler thread only; the device does the copies.
const std = @import("std");
const checkpoint = @import("checkpoint.zig");

/// The mechanism a cache policy drives, for one model. Every operation keeps the model
/// exact by construction; a policy can only choose among them.
pub const Device = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        /// Tokens per pool page.
        pageTokens: *const fn (ctx: *anyopaque) u32,
        /// Copy `slot`'s recurrent state into snapshot `snapshot`.
        save: *const fn (ctx: *anyopaque, slot: u32, snapshot: u32) anyerror!void,
        /// Restore snapshot `snapshot` into `slot`, whose next position becomes `position`.
        load: *const fn (ctx: *anyopaque, slot: u32, snapshot: u32, position: u32) anyerror!void,
        /// Pin the pages holding `slot`'s positions below `tokens` (logical order, into `out`).
        pin: *const fn (ctx: *anyopaque, slot: u32, tokens: u32, out: []u32) anyerror![]const u32,
        unpin: *const fn (ctx: *anyopaque, pages: []const u32) anyerror!void,
        /// Map `full` shared and a copy of `partial` into empty `slot`
        /// (`error.PoolExhausted`: no free page for the copy; nothing changed).
        attach: *const fn (ctx: *anyopaque, slot: u32, full: []const u32, partial: ?u32) anyerror!void,
        /// Replace `slot`'s logical pages 0..pages.len with byte-identical `pages`; the number
        /// of its old pages that became free.
        rebind: *const fn (ctx: *anyopaque, slot: u32, pages: []const u32) anyerror!u32,
    };
    pub fn pageTokens(d: Device) u32 {
        return d.vtable.pageTokens(d.ctx);
    }
    pub fn save(d: Device, slot: u32, snapshot: u32) !void {
        return d.vtable.save(d.ctx, slot, snapshot);
    }
    pub fn load(d: Device, slot: u32, snapshot: u32, position: u32) !void {
        return d.vtable.load(d.ctx, slot, snapshot, position);
    }
    pub fn pin(d: Device, slot: u32, tokens: u32, out: []u32) ![]const u32 {
        return d.vtable.pin(d.ctx, slot, tokens, out);
    }
    pub fn unpin(d: Device, pages: []const u32) !void {
        return d.vtable.unpin(d.ctx, pages);
    }
    pub fn attach(d: Device, slot: u32, full: []const u32, partial: ?u32) !void {
        return d.vtable.attach(d.ctx, slot, full, partial);
    }
    pub fn rebind(d: Device, slot: u32, pages: []const u32) !u32 {
        return d.vtable.rebind(d.ctx, slot, pages);
    }
};

pub const Stats = struct {
    lookups: u64 = 0,
    /// Restores, and the prompt tokens they skipped.
    restores: u64 = 0,
    restored_tokens: u64 = 0,
    /// Checkpoints taken; dropped to make room for a new one; dropped for pool memory.
    inserts: u64 = 0,
    capacity_drops: u64 = 0,
    pressure_drops: u64 = 0,
    /// Pages freed by deduplication on insert (radix).
    dedup_pages: u64 = 0,
};

/// A prefix-cache policy.
pub const Cache = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        /// `slot` is reset and holds no pages: restore the best checkpoint for `prompt` (at
        /// least one prompt token left to process); its start position (0: cold).
        restore: *const fn (ctx: *anyopaque, dev: Device, slot: u32, prompt: []const u32) anyerror!u32,
        /// `slot` has processed exactly `prefix` (a prompt prefix): keep it if worthwhile.
        checkpoint: *const fn (ctx: *anyopaque, dev: Device, slot: u32, prefix: []const u32) anyerror!void,
        /// Memory pressure: drop one checkpoint (never entry `keep`); false: nothing left.
        evict: *const fn (ctx: *anyopaque, dev: Device) bool,
        stats: *const fn (ctx: *anyopaque) Stats,
        deinit: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
    };
    pub fn restore(c: Cache, dev: Device, slot: u32, prompt: []const u32) !u32 {
        return c.vtable.restore(c.ctx, dev, slot, prompt);
    }
    pub fn checkpoint(c: Cache, dev: Device, slot: u32, prefix: []const u32) !void {
        return c.vtable.checkpoint(c.ctx, dev, slot, prefix);
    }
    pub fn evict(c: Cache, dev: Device) bool {
        return c.vtable.evict(c.ctx, dev);
    }
    pub fn stats(c: Cache) Stats {
        return c.vtable.stats(c.ctx);
    }
    pub fn deinit(c: Cache, allocator: std.mem.Allocator) void {
        c.vtable.deinit(c.ctx, allocator);
    }
};

pub const Kind = enum { flat, radix };

/// A cache of `kind` over `snapshots` snapshot slots (entry i uses snapshot i), for
/// sequences of at most `context` tokens and `max_pages` pages. Owned by the caller
/// (`Cache.deinit` frees it).
pub fn create(allocator: std.mem.Allocator, kind: Kind, snapshots: u32, context: u32, max_pages: u32, boundary: u32) !Cache {
    return switch (kind) {
        .flat => (try Flat.create(allocator, snapshots, context, max_pages, boundary)).cache(),
        .radix => (try Radix.create(allocator, snapshots, context, max_pages, boundary)).cache(),
    };
}

/// Restore entry `i` of `store` into `slot` (shared full pages, a copied partial page, the
/// snapshot). False when no free page was left for the partial copy.
fn restoreEntry(store: *checkpoint.Store, dev: Device, slot: u32, i: usize) !bool {
    const len = store.entries[i].len;
    const pages = store.entryPages(i);
    const page = dev.pageTokens();
    const nfull = len / page;
    dev.attach(slot, pages[0..nfull], if (len % page != 0) pages[nfull] else null) catch |e| {
        if (e == error.PoolExhausted) return false;
        return e;
    };
    try dev.load(slot, @intCast(i), len);
    return true;
}

/// Checkpoint `slot`'s `prefix` into entry `i` (free): snapshot, pins.
fn fillEntry(store: *checkpoint.Store, dev: Device, slot: u32, i: usize, prefix: []const u32, pin_buf: []u32) !void {
    try dev.save(slot, @intCast(i));
    const pins = try dev.pin(slot, @intCast(prefix.len), pin_buf);
    store.fill(i, prefix, pins) catch |e| {
        dev.unpin(pins) catch {};
        return e;
    };
}

fn dropEntry(store: *checkpoint.Store, dev: Device, i: usize) !void {
    try dev.unpin(store.entryPages(i));
    store.drop(i);
}

/// The first policy: a list of checkpoints; longest-prefix lookup by scanning; leaf-first,
/// then least-recently-used eviction (`checkpoint.Store.victim`).
pub const Flat = struct {
    store: checkpoint.Store,
    pin_buf: []u32,
    st: Stats = .{},

    pub fn create(allocator: std.mem.Allocator, snapshots: u32, context: u32, max_pages: u32, boundary: u32) !*Flat {
        const self = try allocator.create(Flat);
        errdefer allocator.destroy(self);
        self.* = .{ .store = try checkpoint.Store.init(allocator, snapshots, context, max_pages, boundary), .pin_buf = undefined };
        errdefer self.store.deinit(allocator);
        self.pin_buf = try allocator.alloc(u32, max_pages);
        return self;
    }
    pub fn cache(self: *Flat) Cache {
        return .{ .ctx = self, .vtable = &vtable };
    }
    const vtable: Cache.VTable = .{ .restore = restore, .checkpoint = take, .evict = evict, .stats = stats, .deinit = deinit };

    fn restore(ctx: *anyopaque, dev: Device, slot: u32, prompt: []const u32) anyerror!u32 {
        const self: *Flat = @ptrCast(@alignCast(ctx));
        self.st.lookups += 1;
        const i = self.store.lookup(prompt) orelse return 0;
        while (!try restoreEntry(&self.store, dev, slot, i)) {
            // A free page for the copy: drop another checkpoint, else start cold.
            const v = self.store.victim(i) orelse return 0;
            try dropEntry(&self.store, dev, v);
            self.st.pressure_drops += 1;
        }
        self.store.touch(i);
        self.st.restores += 1;
        self.st.restored_tokens += self.store.entries[i].len;
        return self.store.entries[i].len;
    }
    fn take(ctx: *anyopaque, dev: Device, slot: u32, prefix: []const u32) anyerror!void {
        const self: *Flat = @ptrCast(@alignCast(ctx));
        if (self.store.has(prefix)) return;
        const i = self.store.claim();
        if (self.store.entries[i].len != 0) {
            try dropEntry(&self.store, dev, i);
            self.st.capacity_drops += 1;
        }
        try fillEntry(&self.store, dev, slot, i, prefix, self.pin_buf);
        self.st.inserts += 1;
    }
    fn evict(ctx: *anyopaque, dev: Device) bool {
        const self: *Flat = @ptrCast(@alignCast(ctx));
        const i = self.store.victim(null) orelse return false;
        dropEntry(&self.store, dev, i) catch return false;
        self.st.pressure_drops += 1;
        return true;
    }
    fn stats(ctx: *anyopaque) Stats {
        const self: *Flat = @ptrCast(@alignCast(ctx));
        return self.st;
    }
    fn deinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Flat = @ptrCast(@alignCast(ctx));
        self.store.deinit(allocator);
        allocator.free(self.pin_buf);
        allocator.destroy(self);
    }
};

/// A prefix tree of checkpoints: each node's parent is its longest live proper prefix.
/// Lookup descends the tree, eviction takes leaves (least recently used first), and an
/// inserted checkpoint's slot is rebound to pages the tree already holds for the same
/// tokens (deduplication of prefixes prefilled cold at the same time).
pub const Radix = struct {
    store: checkpoint.Store,
    parent: []?u16,
    children: []u16,
    pin_buf: []u32,
    st: Stats = .{},

    pub fn create(allocator: std.mem.Allocator, snapshots: u32, context: u32, max_pages: u32, boundary: u32) !*Radix {
        if (snapshots > std.math.maxInt(u16)) return error.InvalidOptions;
        const self = try allocator.create(Radix);
        errdefer allocator.destroy(self);
        self.* = .{ .store = try checkpoint.Store.init(allocator, snapshots, context, max_pages, boundary), .parent = undefined, .children = undefined, .pin_buf = undefined };
        errdefer self.store.deinit(allocator);
        self.parent = try allocator.alloc(?u16, snapshots);
        errdefer allocator.free(self.parent);
        @memset(self.parent, null);
        self.children = try allocator.alloc(u16, snapshots);
        errdefer allocator.free(self.children);
        @memset(self.children, 0);
        self.pin_buf = try allocator.alloc(u32, max_pages);
        return self;
    }
    pub fn cache(self: *Radix) Cache {
        return .{ .ctx = self, .vtable = &vtable };
    }
    const vtable: Cache.VTable = .{ .restore = restore, .checkpoint = take, .evict = evict, .stats = stats, .deinit = deinit };

    fn live(self: *const Radix, i: usize) bool {
        return self.store.entries[i].len != 0;
    }
    fn tokens(self: *const Radix, i: usize) []const u32 {
        return self.store.entryTokens(i);
    }
    /// Whether live node `i` is a prefix of `seq` (a proper one when `proper`).
    fn prefixOf(self: *const Radix, i: usize, seq: []const u32, proper: bool) bool {
        const t = self.tokens(i);
        if (t.len > seq.len or (proper and t.len == seq.len)) return false;
        return std.mem.eql(u32, t, seq[0..t.len]);
    }

    /// The deepest live node that is a proper prefix of `seq` and leaves at least one token
    /// (by descent from the roots: at each level at most one child is a prefix of `seq`).
    fn deepest(self: *const Radix, seq: []const u32, proper: bool) ?usize {
        var at: ?usize = null;
        while (true) {
            var next: ?usize = null;
            for (self.parent, 0..) |p, i| {
                if (!self.live(i) or !eqOpt(p, at)) continue;
                if (self.prefixOf(i, seq, proper)) {
                    next = i;
                    break;
                }
            }
            at = next orelse return at;
        }
    }
    fn eqOpt(p: ?u16, at: ?usize) bool {
        if (p == null) return at == null;
        return at != null and p.? == at.?;
    }

    /// The least recently used leaf (never `keep`).
    fn leaf(self: *const Radix, keep: ?usize) ?usize {
        var best: ?usize = null;
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0 or self.children[i] != 0 or (keep != null and i == keep.?)) continue;
            if (best == null or e.used < self.store.entries[best.?].used) best = i;
        }
        return best;
    }

    /// Remove node `i`: its children move to its parent.
    fn remove(self: *Radix, dev: Device, i: usize) !void {
        try dev.unpin(self.store.entryPages(i));
        const up = self.parent[i];
        for (self.parent, 0..) |*p, j| {
            if (self.live(j) and p.* != null and p.*.? == i) {
                p.* = up;
                if (up) |u| self.children[u] += 1;
            }
        }
        if (up) |u| self.children[u] -= 1;
        self.children[i] = 0;
        self.parent[i] = null;
        self.store.drop(i);
    }

    fn restore(ctx: *anyopaque, dev: Device, slot: u32, prompt: []const u32) anyerror!u32 {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        self.st.lookups += 1;
        const i = self.deepest(prompt, true) orelse return 0;
        while (!try restoreEntry(&self.store, dev, slot, i)) {
            const v = self.leaf(i) orelse return 0;
            try self.remove(dev, v);
            self.st.pressure_drops += 1;
        }
        // Using a node counts as using its ancestors (they stay parents anyway).
        self.store.touch(i);
        self.st.restores += 1;
        self.st.restored_tokens += self.store.entries[i].len;
        return self.store.entries[i].len;
    }

    fn take(ctx: *anyopaque, dev: Device, slot: u32, prefix: []const u32) anyerror!void {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        // Deduplicate first (also when `prefix` is held already: a cold duplicate): the live node sharing the most tokens with `prefix` holds pages
        // for them; the slot's full pages below that point are byte-identical copies.
        const page = dev.pageTokens();
        var best: ?usize = null;
        var best_common: usize = 0;
        for (self.store.entries, 0..) |e, j| {
            if (e.len == 0) continue;
            const common = std.mem.indexOfDiff(u32, self.tokens(j), prefix) orelse @min(e.len, prefix.len);
            if (common > best_common) {
                best = j;
                best_common = common;
            }
        }
        if (best) |j| {
            const n = best_common / page;
            if (n > 0) self.st.dedup_pages += try dev.rebind(slot, self.store.entryPages(j)[0..n]);
        }
        if (self.store.has(prefix)) return;
        // A slot: a free one, else the least recently used leaf.
        var i: usize = 0;
        while (i < self.store.entries.len and self.live(i)) i += 1;
        if (i == self.store.entries.len) {
            i = self.leaf(null) orelse return;
            try self.remove(dev, i);
            self.st.capacity_drops += 1;
        }
        try fillEntry(&self.store, dev, slot, i, prefix, self.pin_buf);
        // Link: parent = deepest proper prefix; nodes that extend `prefix` under that parent
        // become its children.
        const up = self.deepest(prefix, true);
        var up_found = up;
        if (up_found != null and up_found.? == i) up_found = null;
        self.parent[i] = if (up_found) |u| @intCast(u) else null;
        if (up_found) |u| self.children[u] += 1;
        for (self.parent, 0..) |*p, j| {
            if (j == i or !self.live(j) or !eqOpt(p.*, up_found)) continue;
            if (self.store.entries[j].len > prefix.len and std.mem.eql(u32, self.tokens(j)[0..prefix.len], prefix)) {
                p.* = @intCast(i);
                self.children[i] += 1;
                if (up_found) |u| self.children[u] -= 1;
            }
        }
        self.st.inserts += 1;
    }

    fn evict(ctx: *anyopaque, dev: Device) bool {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        const i = self.leaf(null) orelse return false;
        self.remove(dev, i) catch return false;
        self.st.pressure_drops += 1;
        return true;
    }
    fn stats(ctx: *anyopaque) Stats {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        return self.st;
    }
    fn deinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        self.store.deinit(allocator);
        allocator.free(self.parent);
        allocator.free(self.children);
        allocator.free(self.pin_buf);
        allocator.destroy(self);
    }

    /// Tree invariants (tests): every live node's parent is its deepest live proper prefix,
    /// and child counts match.
    pub fn checkTree(self: *const Radix) bool {
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0) continue;
            var want: ?usize = null;
            for (self.store.entries, 0..) |f, j| {
                if (j == i or f.len == 0 or f.len >= e.len) continue;
                if (!self.prefixOf(j, self.tokens(i), true)) continue;
                if (want == null or f.len > self.store.entries[want.?].len) want = j;
            }
            if (!eqOpt(self.parent[i], want)) return false;
            var n: u16 = 0;
            for (self.parent, 0..) |p, j| n += @intFromBool(self.live(j) and p != null and p.? == i);
            if (n != self.children[i]) return false;
        }
        return true;
    }
};
