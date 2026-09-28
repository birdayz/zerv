//! Prefix-cache policies behind `session.kvcache.Cache` (docs/specs/concurrent.md,
//! "18d.4 design"), driven through a fake device that models what the model guarantees:
//! a page's content is determined by the tokens up to its last written position, the
//! recurrent state by the whole history. Every policy must leave each slot exactly as a cold
//! run would, return every page, and (radix) deduplicate identical pages.
const std = @import("std");
const session = @import("session");
const kvcache = session.kvcache;
const t = std.testing;

const P = 4; // tokens per page
const pool = 256;
const slots = 4;
const ctx_max = 64;
const B = 9; // message boundary token

fn hashOf(tokens: []const u32) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (tokens) |x| h = (h ^ x) *% 0x100000001b3;
    return h;
}

const Fake = struct {
    mask: [pool]u8 = @splat(0),
    pins: [pool]u8 = @splat(0),
    content: [pool]u64 = @splat(0),
    logical: [pool]u32 = @splat(0),
    table: [slots][ctx_max / P + 1]u32 = undefined,
    mapped: [slots]u32 = @splat(0),
    hist: [slots][ctx_max]u32 = undefined,
    pos: [slots]u32 = @splat(0),
    snaps: [64]u64 = undefined,
    /// Host store for demoted checkpoint pages (0 pages: none).
    host_n: u32 = 0,
    host_used: [pool]bool = @splat(false),
    host_content: [pool]u64 = undefined,
    violations: u32 = 0,

    fn free(self: *const Fake, q: usize) bool {
        return self.mask[q] == 0 and self.pins[q] == 0;
    }
    fn freeCount(self: *const Fake) u32 {
        var n: u32 = 0;
        for (0..pool) |q| n += @intFromBool(self.free(q));
        return n;
    }
    fn map(self: *Fake, slot: u32, q: u32) void {
        self.mask[q] |= @as(u8, 1) << @intCast(slot);
        self.logical[q] = self.mapped[slot];
        self.table[slot][self.mapped[slot]] = q;
        self.mapped[slot] += 1;
    }
    fn release(self: *Fake, slot: u32) void {
        for (&self.mask) |*m| m.* &= ~(@as(u8, 1) << @intCast(slot));
        self.mapped[slot] = 0;
        self.pos[slot] = 0;
    }
    /// Process `tokens` in `slot` (prefill): pages grow and get their content.
    fn run(self: *Fake, slot: u32, tokens: []const u32) !void {
        for (tokens) |x| {
            const at = self.pos[slot];
            if (at / P >= self.mapped[slot]) {
                var q: u32 = 0;
                while (!self.free(q)) q += 1;
                self.map(slot, q);
            }
            self.hist[slot][at] = x;
            self.pos[slot] += 1;
            self.content[self.table[slot][at / P]] = hashOf(self.hist[slot][0..self.pos[slot]]);
        }
    }
    /// The slot equals a cold run of its history: every page's content and the state.
    fn exact(self: *const Fake, slot: u32) bool {
        const n = self.pos[slot];
        for (0..(n + P - 1) / P) |k| {
            const end = @min(n, (k + 1) * P);
            if (self.content[self.table[slot][k]] != hashOf(self.hist[slot][0..end])) return false;
        }
        return true;
    }

    fn device(self: *Fake) kvcache.Device {
        return .{ .ctx = self, .vtable = &vt };
    }
    const vt: kvcache.Device.VTable = .{ .pageTokens = pageTokens, .save = save, .load = load, .pin = pin, .unpin = unpin, .attach = attach, .rebind = rebind, .demote = demote, .promote = promote, .reclaimable = reclaimable };
    fn reclaimable(ctx: *anyopaque, list: []const u32) u32 {
        const self = of(ctx);
        var n: u32 = 0;
        for (list) |q| {
            if (q & kvcache.Device.host_flag != 0) continue;
            n += @intFromBool(self.mask[q] == 0);
        }
        return n;
    }
    const HF = kvcache.Device.host_flag;
    fn demote(ctx: *anyopaque, list: []u32) anyerror!u32 {
        const self = of(ctx);
        var moved: u32 = 0;
        for (list) |*q| {
            if (q.* & HF != 0 or self.pins[q.*] != 1 or self.mask[q.*] != 0) continue;
            const h = for (0..self.host_n) |h| {
                if (!self.host_used[h]) break h;
            } else break;
            self.host_used[h] = true;
            self.host_content[h] = self.content[q.*];
            self.content[q.*] = 0xdead;
            self.pins[q.*] = 0;
            q.* = @as(u32, @intCast(h)) | HF;
            moved += 1;
        }
        return moved;
    }
    fn promote(ctx: *anyopaque, list: []u32) anyerror!bool {
        const self = of(ctx);
        var need: u32 = 0;
        for (list) |q| need += @intFromBool(q & HF != 0);
        if (self.freeCount() < need) return false;
        for (list, 0..) |*q, k| {
            if (q.* & HF == 0) continue;
            const h = q.* & ~HF;
            var f: u32 = 0;
            while (!self.free(f)) f += 1;
            self.pins[f] = 1;
            self.content[f] = self.host_content[h];
            self.logical[f] = @intCast(k);
            self.host_used[h] = false;
            q.* = f;
        }
        return true;
    }
    fn of(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }
    fn pageTokens(_: *anyopaque) u32 {
        return P;
    }
    fn save(ctx: *anyopaque, slot: u32, snapshot: u32) anyerror!void {
        const self = of(ctx);
        self.snaps[snapshot] = hashOf(self.hist[slot][0..self.pos[slot]]);
    }
    fn load(ctx: *anyopaque, slot: u32, snapshot: u32, position: u32) anyerror!void {
        const self = of(ctx);
        // The caller restores the history's tokens (the prompt prefix) itself; the state
        // must be the state after exactly them.
        if (self.snaps[snapshot] != hashOf(self.hist[slot][0..position])) self.violations += 1;
        self.pos[slot] = position;
    }
    fn pin(ctx: *anyopaque, slot: u32, tokens: u32, out: []u32) anyerror![]const u32 {
        const self = of(ctx);
        const n = (tokens + P - 1) / P;
        for (0..n) |k| {
            out[k] = self.table[slot][k];
            self.pins[out[k]] += 1;
        }
        return out[0..n];
    }
    fn unpin(ctx: *anyopaque, pages: []const u32) anyerror!void {
        const self = of(ctx);
        for (pages) |q| {
            if (q & kvcache.Device.host_flag != 0) {
                if (!self.host_used[q & ~kvcache.Device.host_flag]) return error.InvalidState;
                self.host_used[q & ~kvcache.Device.host_flag] = false;
                continue;
            }
            if (self.pins[q] == 0) return error.InvalidState;
            self.pins[q] -= 1;
        }
    }
    fn attach(ctx: *anyopaque, slot: u32, full: []const u32, partial: ?u32) anyerror!void {
        const self = of(ctx);
        if (self.mapped[slot] != 0) self.violations += 1;
        var fresh: ?u32 = null;
        if (partial) |_| {
            var q: u32 = 0;
            while (q < pool and !self.free(q)) q += 1;
            if (q == pool) return error.PoolExhausted;
            fresh = q;
        }
        for (full) |q| {
            if (q & kvcache.Device.host_flag != 0) self.violations += 1; // attached a host page
            self.map(slot, q);
        }
        if (fresh) |q| {
            self.content[q] = self.content[partial.?];
            self.map(slot, q);
        }
    }
    fn rebind(ctx: *anyopaque, slot: u32, pages: []const u32) anyerror!u32 {
        const self = of(ctx);
        var freed: u32 = 0;
        for (pages, 0..) |q, k| {
            const old = self.table[slot][k];
            if (q == old) continue;
            if (self.content[q] != self.content[old] or self.logical[q] != k) self.violations += 1;
            self.mask[old] &= ~(@as(u8, 1) << @intCast(slot));
            self.mask[q] |= @as(u8, 1) << @intCast(slot);
            self.table[slot][k] = q;
            freed += @intFromBool(self.free(old));
        }
        return freed;
    }
};

/// One request in `slot` through the cache: restore, prefill up to the checkpoint points
/// (taking them), then the rest; the slot must equal a cold run.
fn request(c: kvcache.Cache, fake: *Fake, slot: u32, prompt: []const u32) !u32 {
    fake.release(slot);
    // The history the slot will have (the fake's snapshot check reads it).
    @memcpy(fake.hist[slot][0..prompt.len], prompt);
    const start = try c.restore(fake.device(), slot, prompt);
    if (fake.pos[slot] != start) return error.WrongStart;
    var points: [session.checkpoint.max_points]u32 = undefined;
    var at = start;
    for (session.checkpoint.candidatePoints(prompt, start, B, &points)) |p| {
        try fake.run(slot, prompt[at..p]);
        try c.checkpoint(fake.device(), slot, prompt[0..p]);
        at = p;
    }
    try fake.run(slot, prompt[at..]);
    if (!fake.exact(slot)) return error.NotExact;
    return start;
}

/// Conversation `c` at turn `n`: [B sys…] then per turn [B user c k].
fn conversation(buf: []u32, sys_len: usize, c: u32, turns: u32) []const u32 {
    var n: usize = 0;
    buf[n] = B;
    n += 1;
    for (0..sys_len) |i| {
        buf[n] = @intCast(1000 + i);
        n += 1;
    }
    for (0..turns + 1) |k| {
        buf[n] = B;
        buf[n + 1] = 100 + c;
        buf[n + 2] = @intCast(200 + k);
        n += 3;
    }
    buf[n] = B;
    return buf[0 .. n + 1];
}

fn workload(kind: kvcache.Kind) !void {
    var fake: Fake = .{};
    const c = try kvcache.create(t.allocator, kind, 6, ctx_max, ctx_max / P + 1, B);
    defer c.deinit(t.allocator);
    var buf: [ctx_max]u32 = undefined;
    var restored: u32 = 0;
    // 10 conversations of 4 turns over 4 slots, interleaved.
    for (0..4) |turn| for (0..10) |conv| {
        const slot: u32 = @intCast(conv % slots);
        const start = try request(c, &fake, slot, conversation(&buf, 13, @intCast(conv), @intCast(turn)));
        restored += @intFromBool(start > 0);
    };
    try t.expectEqual(@as(u32, 0), fake.violations);
    const st = c.stats();
    try t.expect(restored >= 36); // all but the first request restore at least the system prompt
    try t.expectEqual(@as(u64, restored), st.restores);
    // Pressure evicts everything; then every page is free.
    for (0..slots) |s| fake.release(@intCast(s));
    while (c.evict(fake.device()) or c.evictHost(fake.device())) {}
    try t.expectEqual(@as(u32, pool), fake.freeCount());
}

test "kvcache flat: interleaved conversations restore exactly; every page comes back" {
    try workload(.flat);
}
test "kvcache radix: interleaved conversations restore exactly; every page comes back" {
    try workload(.radix);
}

test "kvcache radix: prefixes prefilled cold at the same time are deduplicated on insert" {
    var fake: Fake = .{};
    const c = try kvcache.create(t.allocator, .radix, 6, ctx_max, ctx_max / P + 1, B);
    defer c.deinit(t.allocator);
    var buf0: [ctx_max]u32 = undefined;
    var buf1: [ctx_max]u32 = undefined;
    const p0 = conversation(&buf0, 21, 0, 0);
    const p1 = conversation(&buf1, 21, 1, 0);
    // Both cold (nothing cached yet): prefill the system prompt in parallel.
    for ([_]u32{ 0, 1 }, [_][]const u32{ p0, p1 }) |slot, p| {
        fake.release(slot);
        @memcpy(fake.hist[slot][0..p.len], p);
        try t.expectEqual(@as(u32, 0), try c.restore(fake.device(), slot, p));
        try fake.run(slot, p[0..22]);
    }
    const before = fake.freeCount();
    try c.checkpoint(fake.device(), 0, p0[0..22]);
    try c.checkpoint(fake.device(), 1, p1[0..22]); // held already: only deduplicated
    try t.expect(c.stats().dedup_pages == 22 / P and c.stats().inserts == 1);
    // A longer prefix of slot 1 (its own turn) shares the system prompt's full pages.
    try fake.run(1, p1[22 .. p1.len - 1]);
    try c.checkpoint(fake.device(), 1, p1[0 .. p1.len - 1]);
    try t.expect(fake.exact(0) and fake.exact(1));
    try t.expectEqual(@as(u32, 0), fake.violations);
    try t.expectEqual(@as(u64, 22 / P), c.stats().dedup_pages);
    try t.expect(fake.freeCount() >= before); // slot 1's copies came back
    // Slot 1 now maps slot 0's system-prompt pages.
    for (0..22 / P) |k| try t.expectEqual(fake.table[0][k], fake.table[1][k]);
}

test "kvcache radix: tree invariants under random inserts, restores and evictions" {
    var fake: Fake = .{};
    const r = try kvcache.Radix.create(t.allocator, 5, ctx_max, ctx_max / P + 1, B);
    const c = r.cache();
    defer c.deinit(t.allocator);
    var prng = std.Random.DefaultPrng.init(18);
    const rnd = prng.random();
    var buf: [ctx_max]u32 = undefined;
    for (0..300) |_| {
        const slot = rnd.intRangeLessThan(u32, 0, slots);
        const p = conversation(&buf, rnd.intRangeLessThan(usize, 3, 9), rnd.intRangeLessThan(u32, 0, 4), rnd.intRangeLessThan(u32, 0, 4));
        _ = try request(c, &fake, slot, p);
        if (rnd.boolean() and rnd.boolean()) _ = c.evict(fake.device());
        try t.expect(r.checkTree());
    }
    try t.expectEqual(@as(u32, 0), fake.violations);
}

test "kvcache radix tiering: pressure demotes leaves to host; a later hit promotes them back, exact" {
    // A small pool: 3 conversations of ~40 tokens (10 pages each) exceed 24 free pages once
    // their checkpoints pin them; the host store takes the demoted pages.
    var fake: Fake = .{ .host_n = 64 };
    // Occupy most of the pool (other users): 256 - 24 pages held by slot 3.
    for (0..pool - 24) |q| fake.mask[q] = 1 << 3;
    const c = try kvcache.create(t.allocator, .radix, 6, ctx_max, ctx_max / P + 1, B);
    defer c.deinit(t.allocator);
    var buf: [ctx_max]u32 = undefined;
    for (0..3) |conv| {
        // Evict (demote) until the request's pages fit, as the backend's admission does.
        const pr = conversation(&buf, 30, @intCast(conv), 1);
        while (fake.freeCount() < (pr.len + P - 1) / P + 1) if (!c.evict(fake.device())) break;
        _ = try request(c, &fake, 0, pr);
        fake.release(0);
    }
    // Pressure: three evictions demote the three conversations' own (leaf) pages.
    for (0..3) |_| try t.expect(c.evict(fake.device()));
    const st = c.stats();
    try t.expect(st.demotions == 3 and st.demoted_pages > 0 and st.pressure_drops == 0);
    // Conversation 0 again: its checkpoint was demoted; restoring it promotes, exactly.
    const start = try request(c, &fake, 0, conversation(&buf, 30, 0, 1));
    try t.expect(start > 0);
    try t.expect(c.stats().promotions > 0);
    try t.expectEqual(@as(u32, 0), fake.violations);
    fake.release(0);
    while (c.evict(fake.device()) or c.evictHost(fake.device())) {}
    for (fake.host_used) |u| try t.expect(!u);
}

test "kvcache radix tiering: an internal node's segment (a long document) demotes; its child restores it" {
    var fake: Fake = .{ .host_n = 64 };
    const c = try kvcache.create(t.allocator, .radix, 6, ctx_max, ctx_max / P + 1, B);
    defer c.deinit(t.allocator);
    var buf: [ctx_max]u32 = undefined;
    // One conversation, two checkpoints: [B doc...] (the document, 8 pages) and the turn start.
    const pr = conversation(&buf, 30, 0, 1);
    _ = try request(c, &fake, 0, pr);
    fake.release(0);
    const free_before = fake.freeCount();
    // Pressure: the least recently used segments go to the host until nothing is left in
    // the pool; the document's 7 full pages move although its node has a child.
    while (fake.freeCount() < pool) if (!c.evict(fake.device())) break;
    const st = c.stats();
    try t.expect(st.demotions >= 2 and st.pressure_drops == 0);
    try t.expect(fake.freeCount() >= free_before + 7);
    // The next turn restores the child: the whole path is promoted, exactly.
    const start = try request(c, &fake, 0, conversation(&buf, 30, 0, 1));
    try t.expect(start > 30 and c.stats().promotions == 1);
    try t.expectEqual(@as(u32, 0), fake.violations);
    fake.release(0);
    while (c.evict(fake.device()) or c.evictHost(fake.device())) {}
    try t.expectEqual(@as(u32, pool), fake.freeCount());
    for (fake.host_used) |u| try t.expect(!u);
}
