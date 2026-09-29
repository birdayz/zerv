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
        /// Move a checkpoint's pages that only it holds to the host store (tiering,
        /// docs/specs/concurrent.md "18d.5 design"); `pages` is rewritten (host pages carry
        /// `host_flag`). Pages moved (0: none exclusive or no host room).
        demote: *const fn (ctx: *anyopaque, pages: []u32) anyerror!u32,
        /// Bring a checkpoint's host pages back into the pool (pinned); false: too few free
        /// pages (nothing changed).
        promote: *const fn (ctx: *anyopaque, pages: []u32) anyerror!bool,
        /// Pool pages of a checkpoint list that no running sequence maps (releasing
        /// checkpoints can free them): dropping a checkpoint whose pages running sequences
        /// map frees nothing.
        reclaimable: *const fn (ctx: *anyopaque, pages: []const u32) u32,
    };
    /// A page id in a checkpoint's list that names a host-store page.
    pub const host_flag: u32 = 1 << 31;
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
    pub fn demote(d: Device, pages: []u32) !u32 {
        return d.vtable.demote(d.ctx, pages);
    }
    pub fn promote(d: Device, pages: []u32) !bool {
        return d.vtable.promote(d.ctx, pages);
    }
    pub fn reclaimable(d: Device, pages: []const u32) u32 {
        return d.vtable.reclaimable(d.ctx, pages);
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
    /// Tiering (radix): checkpoints demoted to host memory under pressure and the pages
    /// moved; restores that promoted pages back first, and those pages.
    demotions: u64 = 0,
    demoted_pages: u64 = 0,
    promotions: u64 = 0,
    promoted_pages: u64 = 0,
    /// Demoted checkpoints dropped to make host room for a sequence swap.
    host_drops: u64 = 0,
};

/// Logical source incarnation, independent of the physical snapshot/disk namespaces.
pub const Handle = struct { index: u32, generation: u64 };
pub const Lease = struct { handle: Handle, serial: u64 };
/// Borrowed metadata until releaseSource. Only valid-prefix KV bytes are immutable:
/// unused partial-page tails require the byte adapter's canonicalization/dependencies.
pub const Source = struct { lease: Lease, snapshot: u32, tokens: []const u32, pages: []const u32 };
pub const SourceState = struct { generation: u64 = 0, serial: u64 = 0, active: bool = false, refs: u32 = 0 };
pub const Capacity = struct { slots: u32, free_slots: u32 };
pub const Demand = struct { handle: Handle, tokens: u32, has_host: bool, source_held: bool };
pub const PreparationCandidate = struct { handle: Handle, used: u64, first: u32 };
pub const PreparedResult = struct { copied: u32, freed: u32 };
/// Non-yielding, non-reentrant commit of an acknowledged page-pool transaction.
/// Errors must leave every page ID/owner unchanged; no borrowed mutable slice escapes.
pub const PreparationCommit = struct {
    ctx: *anyopaque,
    apply: *const fn (ctx: *anyopaque, pages: []u32, first: u32) anyerror!PreparedResult,
};
/// Borrowed until the next cache mutation; retaining bytes requires acquireSource.
pub const Candidate = struct { handle: Handle, tokens: []const u32, used: u64, host_pages: u32, has_host: bool };

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
        /// Memory pressure: free pool pages by demoting or dropping one checkpoint; false:
        /// nothing left that would free any.
        evict: *const fn (ctx: *anyopaque, dev: Device) bool,
        /// Host-store pressure: drop one demoted checkpoint; false: none.
        evictHost: *const fn (ctx: *anyopaque, dev: Device) bool,
        setDemand: ?*const fn (ctx: *anyopaque, prompt: []const u32) ?Demand = null,
        preparationCandidate: ?*const fn (ctx: *anyopaque, index: u32) ?PreparationCandidate = null,
        finishPreparation: ?*const fn (ctx: *anyopaque, lease: Lease, commit: PreparationCommit) anyerror!PreparedResult = null,
        sourceCapacity: ?*const fn (ctx: *anyopaque) Capacity = null,
        sourceCandidate: ?*const fn (ctx: *anyopaque, index: u32) ?Candidate = null,
        discardSource: ?*const fn (ctx: *anyopaque, dev: Device, handle: Handle) anyerror!void = null,
        coldSource: ?*const fn (ctx: *anyopaque) ?Handle = null,
        acquireSource: ?*const fn (ctx: *anyopaque, handle: Handle) anyerror!Source = null,
        releaseSource: ?*const fn (ctx: *anyopaque, lease: Lease) anyerror!void = null,
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
    pub fn evictHost(c: Cache, dev: Device) bool {
        return c.vtable.evictHost(c.ctx, dev);
    }
    /// Resolves tokens synchronously; retains only a generation-qualified policy preference.
    pub fn setDemand(c: Cache, prompt: []const u32) ?Demand {
        return if (c.vtable.setDemand) |f| f(c.ctx, prompt) else null;
    }
    pub fn clearDemand(c: Cache) void {
        _ = c.setDemand(&.{});
    }
    pub fn preparationCandidate(c: Cache, index: u32) ?PreparationCandidate {
        return if (c.vtable.preparationCandidate) |f| f(c.ctx, index) else null;
    }
    /// Success consumes the source lease. Every error retains it for drain/abort/release.
    pub fn finishPreparation(c: Cache, lease: Lease, commit: PreparationCommit) !PreparedResult {
        return (c.vtable.finishPreparation orelse return error.UnsupportedSource)(c.ctx, lease, commit);
    }
    pub fn sourceCapacity(c: Cache) !Capacity {
        return (c.vtable.sourceCapacity orelse return error.UnsupportedSource)(c.ctx);
    }
    pub fn sourceCandidate(c: Cache, index: u32) ?Candidate {
        return if (c.vtable.sourceCandidate) |f| f(c.ctx, index) else null;
    }
    pub fn discardSource(c: Cache, dev: Device, handle: Handle) !void {
        return (c.vtable.discardSource orelse return error.UnsupportedSource)(c.ctx, dev, handle);
    }
    pub fn coldSource(c: Cache) ?Handle {
        return if (c.vtable.coldSource) |f| f(c.ctx) else null;
    }
    pub fn acquireSource(c: Cache, handle: Handle) !Source {
        return (c.vtable.acquireSource orelse return error.UnsupportedSource)(c.ctx, handle);
    }
    /// Caller must drain all references to the borrowed source before release.
    pub fn releaseSource(c: Cache, lease: Lease) !void {
        return (c.vtable.releaseSource orelse return error.UnsupportedSource)(c.ctx, lease);
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
    return createTiered(allocator, kind, snapshots, context, max_pages, boundary, true);
}
/// `tier`: radix demotes leaves to host memory under pressure (false: drops them).
pub fn createTiered(allocator: std.mem.Allocator, kind: Kind, snapshots: u32, context: u32, max_pages: u32, boundary: u32, tier: bool) !Cache {
    return switch (kind) {
        .flat => (try Flat.create(allocator, snapshots, context, max_pages, boundary)).cache(),
        .radix => blk: {
            const r = try Radix.create(allocator, snapshots, context, max_pages, boundary);
            r.tier = tier;
            break :blk r.cache();
        },
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
    const vtable: Cache.VTable = .{ .restore = restore, .checkpoint = take, .evict = evict, .evictHost = evictHost, .stats = stats, .deinit = deinit };

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
    fn evictHost(_: *anyopaque, _: Device) bool {
        return false; // flat keeps nothing on the host
    }
    fn evict(ctx: *anyopaque, dev: Device) bool {
        const self: *Flat = @ptrCast(@alignCast(ctx));
        // Leaf-first LRU among checkpoints whose release frees pool pages.
        var i_opt: ?usize = null;
        var i_parent = true;
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0 or dev.reclaimable(self.store.entryPages(i)) == 0) continue;
            const parent = self.store.isParent(i);
            if (i_opt == null or (i_parent and !parent) or (parent == i_parent and e.used < self.store.entries[i_opt.?].used)) {
                i_opt = i;
                i_parent = parent;
            }
        }
        const i = i_opt orelse return false;
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
    sources: []SourceState,
    /// Segment ownership (docs/specs/concurrent.md, "18d.5 design"): node i holds a pin (or
    /// the host page) only for its pages from index `own[i]` on; the leading pages equal to
    /// its parent's are its ancestors'. Every node's list still names all its pages (for
    /// attach); a page moved by a demotion or promotion is renamed in every list.
    own: []u32,
    seg_buf: []u32,
    /// A promoted path's page ids before the promotion (`restore`).
    ren_buf: []u32,
    pin_buf: []u32,
    st: Stats = .{},
    /// Demote node segments to the host store under memory pressure (tiering) instead of
    /// dropping.
    tier: bool = true,
    demand: ?Handle = null,

    pub fn create(allocator: std.mem.Allocator, snapshots: u32, context: u32, max_pages: u32, boundary: u32) !*Radix {
        if (snapshots > std.math.maxInt(u16)) return error.InvalidOptions;
        const self = try allocator.create(Radix);
        errdefer allocator.destroy(self);
        self.* = .{ .store = try checkpoint.Store.init(allocator, snapshots, context, max_pages, boundary), .parent = undefined, .children = undefined, .sources = undefined, .own = undefined, .seg_buf = undefined, .ren_buf = undefined, .pin_buf = undefined };
        errdefer self.store.deinit(allocator);
        self.parent = try allocator.alloc(?u16, snapshots);
        errdefer allocator.free(self.parent);
        @memset(self.parent, null);
        self.children = try allocator.alloc(u16, snapshots);
        errdefer allocator.free(self.children);
        @memset(self.children, 0);
        self.sources = try allocator.alloc(SourceState, snapshots);
        errdefer allocator.free(self.sources);
        @memset(self.sources, .{});
        self.own = try allocator.alloc(u32, snapshots);
        errdefer allocator.free(self.own);
        @memset(self.own, 0);
        self.seg_buf = try allocator.alloc(u32, max_pages);
        errdefer allocator.free(self.seg_buf);
        self.ren_buf = try allocator.alloc(u32, max_pages);
        errdefer allocator.free(self.ren_buf);
        self.pin_buf = try allocator.alloc(u32, max_pages);
        return self;
    }
    pub fn cache(self: *Radix) Cache {
        return .{ .ctx = self, .vtable = &vtable };
    }
    const vtable: Cache.VTable = .{ .restore = restore, .checkpoint = take, .evict = evict, .evictHost = evictHost, .stats = stats, .deinit = deinit, .coldSource = coldSource, .acquireSource = acquireSource, .releaseSource = releaseSource, .sourceCapacity = sourceCapacity, .sourceCandidate = sourceCandidate, .discardSource = discardSource, .preparationCandidate = preparationCandidate, .finishPreparation = finishPreparation, .setDemand = setDemand };

    fn setDemand(ctx: *anyopaque, prompt: []const u32) ?Demand {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        self.demand = null;
        const i = self.deepest(prompt, true) orelse return null;
        const handle: Handle = .{ .index = @intCast(i), .generation = self.sources[i].generation };
        self.demand = handle;
        return .{ .handle = handle, .tokens = self.store.entries[i].len, .has_host = self.anyHost(self.store.entryPages(i)), .source_held = self.pathHeld(i) };
    }
    pub fn demandProtected(self: *const Radix, index: usize) bool {
        const handle = self.demand orelse return false;
        if (handle.index >= self.sources.len or !self.live(handle.index) or self.sources[handle.index].generation != handle.generation) return false;
        var at: ?usize = handle.index;
        while (at) |i| : (at = if (self.parent[i]) |p| p else null) if (i == index) return true;
        return false;
    }

    fn preparationCandidate(ctx: *anyopaque, index: u32) ?PreparationCandidate {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        if (!self.tier or index >= self.sources.len or !self.live(index) or self.sources[index].refs != 0 or self.demandProtected(index) or !self.anyPool(self.segment(index))) return null;
        return .{ .handle = .{ .index = index, .generation = self.sources[index].generation }, .used = self.store.entries[index].used, .first = self.own[index] };
    }
    fn finishPreparation(ctx: *anyopaque, lease: Lease, commit: PreparationCommit) anyerror!PreparedResult {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        const i = try self.validateSource(lease.handle);
        const s = &self.sources[i];
        if (!s.active or s.serial != lease.serial) return error.InvalidSource;
        if (s.refs != 1 or self.demandProtected(i)) return error.Busy;
        const seg = self.segment(i);
        const before = self.seg_buf[0..seg.len];
        @memcpy(before, seg);
        const result = try commit.apply(commit.ctx, self.store.entryPagesMut(i), self.own[i]);
        self.rename(i, self.own[i], before);
        self.st.demotions += 1;
        self.st.demoted_pages += result.copied;
        self.releaseValidated(i);
        return result;
    }
    fn sourceCapacity(ctx: *anyopaque) Capacity {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        var free: u32 = 0;
        for (self.store.entries, self.sources) |e, s| free += @intFromBool(e.len == 0 and s.generation != std.math.maxInt(u64));
        return .{ .slots = @intCast(self.store.entries.len), .free_slots = free };
    }
    fn sourceCandidate(ctx: *anyopaque, index: u32) ?Candidate {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        if (index >= self.sources.len or !self.live(index) or self.children[index] != 0 or self.sources[index].refs != 0 or self.demandProtected(index)) return null;
        var host: u32 = 0;
        for (self.segment(index)) |q| host += @intFromBool(q & Device.host_flag != 0);
        return .{ .handle = .{ .index = index, .generation = self.sources[index].generation }, .tokens = self.tokens(index), .used = self.store.entries[index].used, .host_pages = host, .has_host = host != 0 or self.ancestorHost(index) };
    }
    fn discardSource(ctx: *anyopaque, dev: Device, h: Handle) anyerror!void {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        const i = try self.validateSource(h);
        if (self.children[i] != 0 or self.sources[i].refs != 0) return error.Busy;
        try self.remove(dev, i);
        self.st.pressure_drops += 1;
    }
    fn coldSource(ctx: *anyopaque) ?Handle {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        const i = self.leaf(null) orelse return null;
        return .{ .index = @intCast(i), .generation = self.sources[i].generation };
    }
    fn validateSource(self: *const Radix, h: Handle) !usize {
        if (h.index >= self.sources.len or !self.live(h.index) or self.sources[h.index].generation != h.generation) return error.InvalidSource;
        return h.index;
    }
    fn acquireSource(ctx: *anyopaque, h: Handle) anyerror!Source {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        const i = try self.validateSource(h);
        const s = &self.sources[i];
        if (s.active) return error.Busy;
        if (s.serial == std.math.maxInt(u64)) return error.SourceExhausted;
        var at: ?usize = i;
        while (at) |u| : (at = if (self.parent[u]) |p| p else null) {
            if (self.sources[u].refs == std.math.maxInt(u32)) return error.SourceExhausted;
        }
        at = i;
        while (at) |u| : (at = if (self.parent[u]) |p| p else null) self.sources[u].refs += 1;
        s.serial += 1;
        s.active = true;
        return .{ .lease = .{ .handle = h, .serial = s.serial }, .snapshot = @intCast(i), .tokens = self.tokens(i), .pages = self.store.entryPages(i) };
    }
    fn releaseSource(ctx: *anyopaque, lease: Lease) anyerror!void {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        const i = try self.validateSource(lease.handle);
        const s = &self.sources[i];
        if (!s.active or s.serial != lease.serial) return error.InvalidSource;
        self.releaseValidated(i);
    }
    fn releaseValidated(self: *Radix, i: usize) void {
        var at: ?usize = i;
        while (at) |u| : (at = if (self.parent[u]) |p| p else null) self.sources[u].refs -= 1;
        self.sources[i].active = false;
    }
    fn pathHeld(self: *const Radix, i: usize) bool {
        var at: ?usize = i;
        while (at) |u| : (at = if (self.parent[u]) |p| p else null) if (self.sources[u].refs != 0) return true;
        return false;
    }

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
            if (e.len == 0 or self.children[i] != 0 or self.sources[i].refs != 0 or self.demandProtected(i) or (keep != null and i == keep.?)) continue;
            if (best == null or e.used < self.store.entries[best.?].used) best = i;
        }
        return best;
    }

    /// Remove leaf `i`: its own segment's pins and host pages are released.
    fn remove(self: *Radix, dev: Device, i: usize) !void {
        if (self.sources[i].refs != 0 or self.demandProtected(i)) return error.Busy;
        if (self.children[i] != 0) return error.InvalidState; // only leaves are removed
        try dev.unpin(self.segment(i));
        if (self.parent[i]) |u| self.children[u] -= 1;
        self.parent[i] = null;
        self.own[i] = 0;
        self.store.drop(i);
    }
    /// Node `i`'s own pages (its segment of its list).
    fn segment(self: *Radix, i: usize) []u32 {
        const list = self.store.entryPagesMut(i);
        return list[@min(self.own[i], list.len)..];
    }
    /// Leading pages of `i`'s list equal to `j`'s.
    fn commonPages(self: *const Radix, i: usize, j: usize) u32 {
        const a = self.store.entryPages(i);
        const b = self.store.entryPages(j);
        var n: u32 = 0;
        while (n < a.len and n < b.len and a[n] == b[n]) n += 1;
        return n;
    }
    /// After `list[first..]` of some node changed from `before`: rename the pages in every
    /// other list (a page id names one page; ids at the same index are the same page).
    fn rename(self: *Radix, from: usize, first: u32, before: []const u32) void {
        const after = self.store.entryPages(from)[first..][0..before.len];
        for (self.store.entries, 0..) |e, j| {
            if (j == from or e.len == 0) continue;
            const list = self.store.entryPagesMut(j);
            for (before, after, 0..) |b, a, k| {
                const idx = first + k;
                if (idx < list.len and list[idx] == b) list[idx] = a;
            }
        }
    }
    /// Using a node counts as using its ancestors.
    fn touchPath(self: *Radix, i: usize) void {
        self.store.touch(i);
        const t = self.store.entries[i].used;
        var at = self.parent[i];
        while (at) |u| : (at = self.parent[u]) self.store.entries[u].used = t;
    }

    fn restore(ctx: *anyopaque, dev: Device, slot: u32, prompt: []const u32) anyerror!u32 {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        self.st.lookups += 1;
        const i = self.deepest(prompt, true) orelse return 0;
        // A demoted checkpoint (or an ancestor's pages it shares, which are its own list's
        // entries too) comes back first; room is made by demoting or dropping others.
        if (self.anyHost(self.store.entryPages(i))) {
            if (self.pathHeld(i)) return 0; // promotion would rename a leased source
            const list = self.store.entryPagesMut(i);
            var promoted: u32 = 0;
            for (list) |q| promoted += @intFromBool(q & Device.host_flag != 0);
            // The ids before the promotion, for the renaming (not `seg_buf`: `makeRoom` uses
            // it), taken again before every attempt.
            const before = self.ren_buf[0..list.len];
            while (true) {
                @memcpy(before, list);
                if (try dev.promote(list)) break;
                if (!self.makeRoom(dev, i)) return 0;
            }
            self.rename(i, 0, before);
            self.st.promotions += 1;
            self.st.promoted_pages += promoted;
        }
        while (!try restoreEntry(&self.store, dev, slot, i)) {
            if (!self.makeRoom(dev, i)) return 0;
        }
        self.touchPath(i);
        self.st.restores += 1;
        self.st.restored_tokens += self.store.entries[i].len;
        return self.store.entries[i].len;
    }

    fn take(ctx: *anyopaque, dev: Device, slot: u32, prefix: []const u32) anyerror!void {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        // Reparenting a held node would change its release path and segment ownership.
        for (self.store.entries, 0..) |e, j| {
            if (self.sources[j].refs != 0 and e.len > prefix.len and std.mem.eql(u32, self.tokens(j)[0..prefix.len], prefix)) return;
        }
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
            // Only pool-resident pages (a demoted suffix stays on the host).
            const list = self.store.entryPages(j);
            var n = best_common / page;
            for (list[0..@min(n, list.len)], 0..) |q, k| if (q & Device.host_flag != 0) {
                n = k;
                break;
            };
            if (n > 0) self.st.dedup_pages += try dev.rebind(slot, list[0..n]);
        }
        if (self.store.has(prefix)) return;
        // A slot: a free one, else the least recently used leaf.
        var i: usize = 0;
        while (i < self.store.entries.len and (self.live(i) or self.sources[i].generation == std.math.maxInt(u64))) i += 1;
        if (i == self.store.entries.len) {
            i = self.leaf(null) orelse return;
            if (self.sources[i].generation == std.math.maxInt(u64)) return;
            try self.remove(dev, i);
            self.st.capacity_drops += 1;
        }
        try fillEntry(&self.store, dev, slot, i, prefix, self.pin_buf);
        self.sources[i] = .{ .generation = self.sources[i].generation + 1 };
        // Link: parent = deepest proper prefix; nodes that extend `prefix` under that parent
        // become its children.
        const up = self.deepest(prefix, true);
        var up_found = up;
        if (up_found != null and up_found.? == i) up_found = null;
        self.parent[i] = if (up_found) |u| @intCast(u) else null;
        if (up_found) |u| self.children[u] += 1;
        // Ownership: the leading pages equal to the parent's are the ancestors' (unpin ours).
        self.own[i] = if (up_found) |u| self.commonPages(i, u) else 0;
        if (self.own[i] > 0) try dev.unpin(self.store.entryPages(i)[0..self.own[i]]);
        for (self.parent, 0..) |*p, j| {
            if (j == i or !self.live(j) or !eqOpt(p.*, up_found)) continue;
            if (self.store.entries[j].len > prefix.len and std.mem.eql(u32, self.tokens(j)[0..prefix.len], prefix)) {
                p.* = @intCast(i);
                self.children[i] += 1;
                if (up_found) |u| self.children[u] -= 1;
                // The child's pages now also held by the new node are the new node's.
                const new_own = self.commonPages(j, i);
                if (new_own > self.own[j]) {
                    try dev.unpin(self.store.entryPages(j)[self.own[j]..new_own]);
                    self.own[j] = new_own;
                }
            }
        }
        self.touchPath(i);
        self.st.inserts += 1;
    }

    /// Memory pressure: the least recently used node (any, internal ones too) whose own
    /// segment still has pool pages nobody maps is demoted to host memory (kept, restored
    /// later by a copy); when nothing can be demoted, a leaf is dropped.
    fn evict(ctx: *anyopaque, dev: Device) bool {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        return self.makeRoom(dev, null);
    }
    /// Free pool pages (demote first, drop as the fallback), never touching `keep` or its
    /// ancestors (the path being restored).
    fn makeRoom(self: *Radix, dev: Device, keep: ?usize) bool {
        var tried: [256]bool = @splat(false);
        var path: [256]bool = @splat(false);
        if (keep) |k| {
            var at: ?usize = k;
            while (at) |u| {
                if (u < tried.len) {
                    tried[u] = true;
                    path[u] = true;
                }
                at = if (self.parent[u]) |pu| pu else null;
            }
        }
        while (self.tier) {
            var best: ?usize = null;
            for (self.store.entries, 0..) |e, i| {
                if (e.len == 0 or i >= tried.len or tried[i] or self.sources[i].refs != 0 or self.demandProtected(i) or !self.anyPool(self.segment(i))) continue;
                if (best == null or e.used < self.store.entries[best.?].used) best = i;
            }
            const i = best orelse break;
            tried[i] = true;
            const seg = self.segment(i);
            const before = self.seg_buf[0..seg.len];
            @memcpy(before, seg);
            const moved = dev.demote(seg) catch return false;
            if (moved > 0) {
                self.rename(i, self.own[i], before);
                self.st.demotions += 1;
                self.st.demoted_pages += moved;
                return true;
            }
            // Reclaimable pages but none moved: the host store is full. Make host room by
            // dropping the least recently used host-held leaf (off the kept path) and retry.
            if (dev.reclaimable(seg) > 0) {
                var victim: ?usize = null;
                for (self.store.entries, 0..) |e, j| {
                    if (e.len == 0 or self.children[j] != 0 or self.sources[j].refs != 0 or self.demandProtected(j) or j == i or (j < path.len and path[j]) or !self.anyHost(self.segment(j))) continue;
                    if (victim == null or e.used < self.store.entries[victim.?].used) victim = j;
                }
                if (victim) |v| {
                    self.remove(dev, v) catch return false;
                    self.st.host_drops += 1;
                    tried[i] = false;
                }
            }
        }
        // Drop the least recently used leaf that still holds pool pages in its segment; only
        // when none does, the least recently used leaf (its release may make an ancestor's
        // pages demotable or free).
        // Drop a leaf whose own segment holds pool pages no running sequence maps (the host
        // store is full, or tiering is off). A fully demoted leaf frees no pool page (with
        // segment ownership its ancestors' pages are theirs), and a leaf whose pages running
        // sequences map frees nothing: neither is dropped here (the host tier is trimmed by
        // host pressure, `evictHost`).
        // Second choice: a leaf that frees nothing itself (an empty or demoted segment) but
        // whose removal lets an ancestor with reclaimable pages become a leaf; with an empty
        // segment it costs no pages at all.
        var victim: ?usize = null;
        var victim_frees = false;
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0 or self.children[i] != 0 or self.sources[i].refs != 0 or self.demandProtected(i) or (keep != null and i == keep.?)) continue;
            const seg = self.segment(i);
            const frees = dev.reclaimable(seg) > 0;
            if (!frees and (self.anyPool(seg) or !self.ancestorReclaimable(dev, i, if (keep) |k| k else null))) continue;
            if (victim == null or (frees and !victim_frees) or (frees == victim_frees and e.used < self.store.entries[victim.?].used)) {
                victim = i;
                victim_frees = frees;
            }
        }
        const i = victim orelse return false;
        self.remove(dev, i) catch return false;
        self.st.pressure_drops += 1;
        return true;
    }
    /// Host-store pressure (a sequence must swap out): drop the least recently used leaf that
    /// holds host pages; second choice, a leaf whose ancestor holds some (its removal lets
    /// that ancestor become a leaf). False: no node holds host pages.
    fn evictHost(ctx: *anyopaque, dev: Device) bool {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        var victim: ?usize = null;
        var victim_own = false;
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0 or self.children[i] != 0 or self.sources[i].refs != 0 or self.demandProtected(i)) continue;
            const own = self.anyHost(self.segment(i));
            if (!own and !self.ancestorHost(i)) continue;
            if (victim == null or (own and !victim_own) or (own == victim_own and e.used < self.store.entries[victim.?].used)) {
                victim = i;
                victim_own = own;
            }
        }
        const i = victim orelse return false;
        self.remove(dev, i) catch return false;
        self.st.host_drops += 1;
        return true;
    }
    /// Whether an ancestor of `i` (not on `keep`'s path) owns pool pages releasing would free.
    fn ancestorReclaimable(self: *Radix, dev: Device, i: usize, keep: ?usize) bool {
        var at = self.parent[i];
        while (at) |u16_| {
            const u: usize = u16_;
            if (keep) |k| {
                var on_path = false;
                var q: ?usize = k;
                while (q) |x| {
                    if (x == u) on_path = true;
                    q = if (self.parent[x]) |px| px else null;
                }
                if (on_path) return false;
            }
            if (dev.reclaimable(self.segment(u)) > 0) return true;
            at = self.parent[u];
        }
        return false;
    }
    fn ancestorHost(self: *Radix, i: usize) bool {
        var at = self.parent[i];
        while (at) |u| : (at = self.parent[u]) {
            if (self.anyHost(self.segment(u))) return true;
        }
        return false;
    }
    fn anyPool(_: *const Radix, list: []const u32) bool {
        for (list) |q| if (q & Device.host_flag == 0) return true;
        return false;
    }
    fn anyHost(_: *const Radix, list: []const u32) bool {
        for (list) |q| if (q & Device.host_flag != 0) return true;
        return false;
    }
    /// The least recently used leaf that is neither `i` nor one of its ancestors.
    fn leafOffPath(self: *const Radix, i: usize) ?usize {
        return self.leaf(i); // ancestors of i have a child (the path to i): never leaves
    }
    fn stats(ctx: *anyopaque) Stats {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        return self.st;
    }
    fn deinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Radix = @ptrCast(@alignCast(ctx));
        for (self.sources) |s| if (s.active or s.refs != 0) @panic("cache source destroyed before drain");
        self.store.deinit(allocator);
        allocator.free(self.sources);
        allocator.free(self.parent);
        allocator.free(self.children);
        allocator.free(self.own);
        allocator.free(self.seg_buf);
        allocator.free(self.ren_buf);
        allocator.free(self.pin_buf);
        allocator.destroy(self);
    }

    /// Pins the tree holds on pool page `q` (tests): live segments naming it.
    pub fn segmentPins(self: *Radix, q: u32) u32 {
        var n: u32 = 0;
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0) continue;
            for (self.segment(i)) |x| n += @intFromBool(x == q);
        }
        return n;
    }
    /// Ownership invariant (tests): each page a node does not own (index below `own`) is
    /// owned by an ancestor at the same index. Returns the first violating node.
    pub fn ownViolation(self: *Radix) ?usize {
        for (self.store.entries, 0..) |e, i| {
            if (e.len == 0) continue;
            const list = self.store.entryPages(i);
            if (self.own[i] > list.len) return i;
            for (list[0..self.own[i]], 0..) |q, k| {
                var ok = false;
                var at = self.parent[i];
                while (at) |u| : (at = self.parent[u]) {
                    const l = self.store.entryPages(u);
                    if (k < l.len and l[k] == q and k >= self.own[u]) ok = true;
                }
                if (!ok) return i;
            }
        }
        return null;
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
