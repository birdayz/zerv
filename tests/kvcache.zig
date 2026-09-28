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

const SourceRow = struct {
    op: enum { take, evict, acquire, release },
    tokens: []const u32 = &.{},
    index: u32 = 0,
    generation: u64 = 0,
    serial: u64 = 0,
    result: []const u8,
    state: []const struct { tokens: []const u32, generation: u64, serial: u64, active: bool, refs: u32 },
    candidate: ?u32,
};
fn sourceApply(c: kvcache.Cache, fake: *Fake, row: SourceRow) ![]const u8 {
    const h: kvcache.Handle = .{ .index = row.index, .generation = row.generation };
    switch (row.op) {
        .take => {
            fake.release(0);
            try fake.run(0, row.tokens);
            try c.checkpoint(fake.device(), 0, row.tokens);
            fake.release(0);
        },
        .evict => return if (c.evict(fake.device())) "yes" else "no",
        .acquire => {
            const source = try c.acquireSource(h);
            try t.expectEqual(h, source.lease.handle);
            try t.expectEqual(row.index, source.snapshot);
            try t.expectEqual(row.state[row.index].serial, source.lease.serial);
            try t.expectEqualSlices(u32, row.state[row.index].tokens, source.tokens);
        },
        .release => try c.releaseSource(.{ .handle = h, .serial = row.serial }),
    }
    return "ok";
}
fn releaseAllSources(r: *kvcache.Radix) void {
    for (r.sources, 0..) |s, i| if (s.active) {
        r.cache().releaseSource(.{ .handle = .{ .index = @intCast(i), .generation = s.generation }, .serial = s.serial }) catch @panic("source cleanup failed");
    };
}
test "cache sources: independent prefix-set ownership traces and generator provenance" {
    const Fixture = struct { generator_sha256: []const u8, cases: []const struct { capacity: u32, steps: []const SourceRow } };
    const parsed = try std.json.parseFromSlice(Fixture, t.allocator, @embedFile("fixtures/cache-sources.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("reference/generate_cache_source_fixture.py"), &digest, .{});
    try t.expectEqualStrings(parsed.value.generator_sha256, &std.fmt.bytesToHex(digest, .lower));
    var total: usize = 0;
    for (parsed.value.cases, 0..) |case, ci| {
        var fake: Fake = .{};
        const r = try kvcache.Radix.create(t.allocator, case.capacity, ctx_max, ctx_max / P + 1, B);
        const c = r.cache();
        r.tier = false;
        defer {
            releaseAllSources(r);
            c.deinit(t.allocator);
        }
        for (case.steps, 0..) |row, ri| {
            errdefer std.debug.print("source fixture case {d} row {d} {s}\n", .{ ci, ri, @tagName(row.op) });
            const got = sourceApply(c, &fake, row) catch |err| @errorName(err);
            try t.expectEqualStrings(row.result, got);
            for (row.state, 0..) |want, i| {
                try t.expectEqualSlices(u32, want.tokens, r.store.entryTokens(i));
                try t.expectEqualDeep(kvcache.SourceState{ .generation = want.generation, .serial = want.serial, .active = want.active, .refs = want.refs }, r.sources[i]);
            }
            const candidate = c.coldSource();
            try t.expectEqual(row.candidate, if (candidate) |h| h.index else null);
            if (candidate) |h| try t.expectEqual(r.sources[h.index].generation, h.generation);
            try t.expect(r.checkTree());
            try t.expectEqual(null, r.ownViolation());
            try t.expectEqual(@as(u32, 0), fake.violations);
            total += 1;
        }
        try t.expectEqual(@as(u32, pool), fake.freeCount());
    }
    try t.expectEqual(@as(usize, 2345), total);
}

test "cache sources: mixed ancestry survives eviction, blocked promotion, reparenting and unrelated progress" {
    var fake: Fake = .{ .host_n = 64 };
    const r = try kvcache.Radix.create(t.allocator, 4, ctx_max, ctx_max / P + 1, B);
    const c = r.cache();
    defer {
        releaseAllSources(r);
        c.deinit(t.allocator);
    }
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14 };
    try fake.run(0, tokens[0..8]);
    try c.checkpoint(fake.device(), 0, tokens[0..8]);
    try fake.run(0, tokens[8..12]);
    try c.checkpoint(fake.device(), 0, tokens[0..12]);
    fake.release(0);
    try t.expect(c.evict(fake.device())); // internal ancestor demotes first
    const source = try c.acquireSource(c.coldSource().?);
    try t.expect(source.pages[0] & Fake.HF != 0 and source.pages[2] & Fake.HF == 0);
    const pages = source.pages[0..3].*;
    const snapshot = fake.snaps[source.snapshot];
    try t.expect(!c.evict(fake.device()));
    try t.expect(!c.evictHost(fake.device()));
    @memcpy(fake.hist[1][0..tokens.len], &tokens);
    try t.expectEqual(@as(u32, 0), try c.restore(fake.device(), 1, &tokens));
    try t.expectEqual(@as(u64, 0), c.stats().promotions);
    // New ancestor would transfer leased segment ownership: it must not be inserted.
    try fake.run(0, tokens[0..4]);
    try c.checkpoint(fake.device(), 0, tokens[0..4]);
    fake.release(0);
    try t.expectEqual(@as(u32, 2), r.store.live());
    // Unrelated source still inserts, demotes and drops while the path is held.
    try fake.run(0, &.{ 21, 22, 23, 24 });
    try c.checkpoint(fake.device(), 0, &.{ 21, 22, 23, 24 });
    fake.release(0);
    try t.expect(c.evict(fake.device()));
    try t.expect(c.evictHost(fake.device()));
    try t.expectEqualSlices(u32, &pages, source.pages);
    try t.expectEqualSlices(u32, tokens[0..12], source.tokens);
    try t.expectEqual(snapshot, fake.snaps[source.snapshot]);
    try c.releaseSource(source.lease);
    try t.expectError(error.InvalidSource, c.releaseSource(source.lease));
    try t.expectEqual(@as(u32, 12), try c.restore(fake.device(), 1, &tokens));
    try t.expect(fake.exact(1));
    try t.expectEqual(@as(u32, 0), fake.violations);
}

fn sourceAllocation(a: std.mem.Allocator) !void {
    const r = try kvcache.Radix.create(a, 2, ctx_max, ctx_max / P + 1, B);
    const c = r.cache();
    defer c.deinit(a);
    var fake: Fake = .{};
    try fake.run(0, &.{ 1, 2, 3, 4 });
    try c.checkpoint(fake.device(), 0, &.{ 1, 2, 3, 4 });
    const source = try c.acquireSource(c.coldSource().?);
    try c.releaseSource(source.lease);
}
test "cache sources: allocation rollback, generation and serial exhaustion, unsupported flat" {
    try t.checkAllAllocationFailures(t.allocator, sourceAllocation, .{});
    var fa = std.testing.FailingAllocator.init(t.allocator, .{});
    const a = fa.allocator();
    const r = try kvcache.Radix.create(a, 1, ctx_max, ctx_max / P + 1, B);
    const c = r.cache();
    defer c.deinit(a);
    fa.fail_index = fa.alloc_index; // every operation below must be allocation-free
    fa.resize_fail_index = fa.resize_index;
    var fake: Fake = .{};
    try fake.run(0, &.{ 1, 2, 3, 4 });
    try c.checkpoint(fake.device(), 0, &.{ 1, 2, 3, 4 });
    fake.release(0);
    const h = c.coldSource().?;
    r.sources[0].serial = std.math.maxInt(u64);
    try t.expectError(error.SourceExhausted, c.acquireSource(h));
    try t.expectEqual(@as(u32, 0), r.sources[0].refs);
    r.sources[0].serial = 0;
    r.sources[0].refs = std.math.maxInt(u32);
    try t.expectError(error.SourceExhausted, c.acquireSource(h));
    try t.expectEqual(@as(u64, 0), r.sources[0].serial);
    try t.expect(!r.sources[0].active);
    r.sources[0].refs = 0;
    const source = try c.acquireSource(h);
    try t.expectError(error.Busy, c.acquireSource(h));
    try t.expect(!c.evict(fake.device()));
    try c.releaseSource(source.lease);
    r.sources[0].generation = std.math.maxInt(u64);
    try t.expect(c.evict(fake.device()));
    try fake.run(0, &.{ 5, 6, 7, 8 });
    try c.checkpoint(fake.device(), 0, &.{ 5, 6, 7, 8 });
    try t.expectEqual(@as(u32, 0), r.store.live());
    try t.expectError(error.InvalidSource, c.acquireSource(h));
    const flat = try kvcache.create(t.allocator, .flat, 1, ctx_max, ctx_max / P + 1, B);
    defer flat.deinit(t.allocator);
    try t.expectEqual(null, flat.coldSource());
    try t.expectError(error.UnsupportedSource, flat.acquireSource(h));
    try t.expectError(error.UnsupportedSource, flat.releaseSource(source.lease));
}

test "cache sources: GPU-only hits and new descendants retain overlapping leases" {
    var fake: Fake = .{};
    const r = try kvcache.Radix.create(t.allocator, 3, ctx_max, ctx_max / P + 1, B);
    const c = r.cache();
    defer {
        releaseAllSources(r);
        c.deinit(t.allocator);
    }
    const seq = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 10 };
    try fake.run(0, seq[0..4]);
    try c.checkpoint(fake.device(), 0, seq[0..4]);
    const parent = try c.acquireSource(c.coldSource().?);
    const page = parent.pages[0];
    try fake.run(0, seq[4..8]);
    try c.checkpoint(fake.device(), 0, seq[0..8]);
    const child = try c.acquireSource(c.coldSource().?);
    try t.expectEqual(@as(u32, 2), r.sources[parent.lease.handle.index].refs);
    @memcpy(fake.hist[1][0..seq.len], &seq);
    try t.expectEqual(@as(u32, 8), try c.restore(fake.device(), 1, &seq));
    try t.expect(fake.exact(1));
    try t.expectEqual(page, parent.pages[0]);
    try t.expectEqualSlices(u32, seq[0..4], parent.tokens);
    try c.releaseSource(parent.lease); // child must still protect the ancestor
    try t.expectEqual(@as(u32, 1), r.sources[parent.lease.handle.index].refs);
    try t.expectError(error.InvalidSource, c.releaseSource(parent.lease));
    try c.releaseSource(child.lease);
    fake.release(0);
    fake.release(1);
    while (c.evict(fake.device())) {}
    try t.expectEqual(@as(u32, pool), fake.freeCount());
    try t.expectEqual(@as(u32, 0), fake.violations);
}

const PressureRow = struct {
    op: enum { take, pressure, finish, discard },
    tokens: []const u32 = &.{},
    headroom: u32 = 0,
    max_tokens: u32 = 64,
    index: u32 = 0,
    generation: u64 = 0,
    success: bool = false,
    result: []const u8,
    decision: ?session.pressure.Decision,
    state: @FieldType(SourceRow, "state"),
    pending: ?u32,
    ready: []const []const u32,
};

test "pressure policy: independent capacity decisions and source/drop event traces" {
    const p = session.pressure;
    const Fixture = struct {
        generator_sha256: []const u8,
        source_oracle_sha256: []const u8,
        cases: []const struct { usage: p.Usage, candidates: []const p.Candidate, decision: ?p.Decision },
        traces: []const struct { capacity: u32, steps: []const PressureRow },
    };
    const parsed = try std.json.parseFromSlice(Fixture, t.allocator, @embedFile("fixtures/tiering-pressure.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    inline for (.{ .{ "reference/generate_tiering_pressure_fixture.py", "generator_sha256" }, .{ "reference/generate_cache_source_fixture.py", "source_oracle_sha256" } }) |entry| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(@embedFile(entry[0]), &digest, .{});
        try t.expectEqualStrings(@field(parsed.value, entry[1]), &std.fmt.bytesToHex(digest, .lower));
    }
    for (parsed.value.cases) |case| {
        try case.usage.validate();
        var select: p.Select = .{ .usage = case.usage };
        for (case.candidates) |candidate| select.consider(candidate);
        try t.expectEqualDeep(case.decision, select.decision());
    }
    var transitions: usize = 0;
    for (parsed.value.traces) |trace| {
        var fake: Fake = .{};
        const r = try kvcache.Radix.create(t.allocator, trace.capacity, ctx_max, ctx_max / P + 1, B);
        const c = r.cache();
        r.tier = false;
        defer {
            releaseAllSources(r);
            c.deinit(t.allocator);
        }
        var pending: ?kvcache.Source = null;
        var ready: [16][ctx_max]u32 = undefined;
        var lengths: [16]usize = @splat(0);
        for (trace.steps, 0..) |row, ri| {
            errdefer std.debug.print("pressure trace cap={d} row={d} op={s}\n", .{ trace.capacity, ri, @tagName(row.op) });
            var decision: ?p.Decision = null;
            var result: []const u8 = "ok";
            switch (row.op) {
                .take => {
                    fake.release(0);
                    try fake.run(0, row.tokens);
                    try c.checkpoint(fake.device(), 0, row.tokens);
                    fake.release(0);
                },
                .pressure => if (pending == null) {
                    const capacity = try c.sourceCapacity();
                    var select: p.Select = .{ .usage = .{ .slots = capacity.slots, .free_slots = capacity.free_slots, .slot_headroom = row.headroom, .host_pages = 0, .free_host_pages = 0, .host_headroom = 0 } };
                    try select.usage.validate();
                    for (0..capacity.slots) |i| if (c.sourceCandidate(@intCast(i))) |candidate| {
                        var backed = false;
                        for (&ready, lengths) |*tokens, n| if (std.mem.eql(u32, tokens[0..n], candidate.tokens)) {
                            backed = true;
                        };
                        select.consider(.{ .handle = candidate.handle, .used = candidate.used, .has_host = candidate.has_host, .backed = backed, .fits = candidate.tokens.len <= row.max_tokens });
                    };
                    decision = select.decision();
                    if (decision) |d| switch (d.action) {
                        .preserve => pending = try c.acquireSource(d.handle),
                        .discard => try c.discardSource(fake.device(), d.handle),
                    };
                },
                .finish => if (pending) |source| {
                    if (row.success) {
                        var exists = false;
                        for (&ready, lengths) |*tokens, n| if (std.mem.eql(u32, tokens[0..n], source.tokens)) {
                            exists = true;
                        };
                        if (!exists) {
                            const at = std.mem.indexOfScalar(usize, &lengths, 0) orelse return error.TooManyReady;
                            @memcpy(ready[at][0..source.tokens.len], source.tokens);
                            lengths[at] = source.tokens.len;
                        }
                    }
                    try c.releaseSource(source.lease);
                    pending = null;
                },
                .discard => c.discardSource(fake.device(), .{ .index = row.index, .generation = row.generation }) catch |err| {
                    result = @errorName(err);
                },
            }
            try t.expectEqualStrings(row.result, result);
            try t.expectEqualDeep(row.decision, decision);
            try t.expectEqual(row.pending, if (pending) |s| s.lease.handle.index else null);
            var live: u32 = 0;
            for (row.state, 0..) |want, i| {
                try t.expectEqualSlices(u32, want.tokens, r.store.entryTokens(i));
                try t.expectEqualDeep(kvcache.SourceState{ .generation = want.generation, .serial = want.serial, .active = want.active, .refs = want.refs }, r.sources[i]);
                live += @intFromBool(want.tokens.len != 0);
            }
            try t.expectEqualDeep(kvcache.Capacity{ .slots = trace.capacity, .free_slots = trace.capacity - live }, try c.sourceCapacity());
            var ready_count: usize = 0;
            for (&ready, lengths) |*tokens, n| if (n > 0) {
                ready_count += 1;
                var found = false;
                for (row.ready) |want| if (std.mem.eql(u32, want, tokens[0..n])) {
                    found = true;
                };
                try t.expect(found);
            };
            try t.expectEqual(row.ready.len, ready_count);
            try t.expect(r.checkTree());
            try t.expectEqual(null, r.ownViolation());
            try t.expectEqual(@as(u32, 0), fake.violations);
            transitions += 1;
        }
    }
    try t.expectEqual(@as(usize, 2048), parsed.value.cases.len);
    try t.expectEqual(@as(usize, 1560), transitions);
}

test "pressure metadata: host ancestry, leaf-only discard, generation and capacity validation" {
    var fake: Fake = .{ .host_n = 16 };
    const r = try kvcache.Radix.create(t.allocator, 4, ctx_max, ctx_max / P + 1, B);
    const c = r.cache();
    defer c.deinit(t.allocator);
    try fake.run(0, &.{ 1, 2, 3, 4 });
    try c.checkpoint(fake.device(), 0, &.{ 1, 2, 3, 4 });
    try fake.run(0, &.{ 5, 6, 7, 8 });
    try c.checkpoint(fake.device(), 0, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    fake.release(0);
    try t.expect(c.sourceCandidate(0) == null and c.sourceCandidate(4) == null);
    try t.expectError(error.Busy, c.discardSource(fake.device(), .{ .index = 0, .generation = 1 }));
    try t.expect(c.evict(fake.device())); // demote ancestor only
    const candidate = c.sourceCandidate(1).?;
    try t.expect(candidate.has_host);
    try t.expectEqual(@as(u32, 0), candidate.host_pages);
    const s = try c.acquireSource(candidate.handle);
    try t.expect(c.sourceCandidate(1) == null);
    try t.expectError(error.Busy, c.discardSource(fake.device(), candidate.handle));
    try c.releaseSource(s.lease);
    try c.discardSource(fake.device(), candidate.handle);
    try t.expectError(error.InvalidSource, c.discardSource(fake.device(), candidate.handle));
    const parent = c.sourceCandidate(0).?;
    try t.expectEqual(@as(u32, 1), parent.host_pages);
    try c.discardSource(fake.device(), parent.handle);
    r.sources[0].generation = std.math.maxInt(u64);
    try t.expectEqual(@as(u32, 3), (try c.sourceCapacity()).free_slots);
    const flat = try kvcache.create(t.allocator, .flat, 1, ctx_max, ctx_max / P + 1, B);
    defer flat.deinit(t.allocator);
    try t.expectError(error.UnsupportedSource, flat.sourceCapacity());
    try t.expectError(error.UnsupportedSource, flat.discardSource(fake.device(), candidate.handle));
    const p = session.pressure;
    const valid: p.Usage = .{ .slots = 2, .free_slots = 2, .slot_headroom = 0, .host_pages = 4, .free_host_pages = 4, .host_headroom = 1 };
    inline for (.{ "slots", "free_slots", "slot_headroom", "free_host_pages", "host_headroom" }) |field| {
        var bad = valid;
        @field(bad, field) = if (std.mem.eql(u8, field, "slots")) 0 else std.math.maxInt(u32);
        try t.expectError(error.InvalidCapacity, bad.validate());
    }
    var bad = valid;
    bad.host_pages = 0;
    bad.free_host_pages = 0;
    try t.expectError(error.InvalidCapacity, bad.validate());
}
