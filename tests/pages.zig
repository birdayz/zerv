//! KV page accounting (src/model/pages.zig) under random operation sequences, with the
//! device copies simulated on page contents: after every operation each sequence's table
//! must show exactly its own content, every page's holders must be consistent (`check`),
//! and freeing everything must free every page.
const std = @import("std");
const pages = @import("model").pages;
const t = std.testing;

const N = 48; // pool pages
const H = 64; // host pages
const S = 5; // slots
const L = 12; // pages per sequence at most

const Env = struct {
    pool: *pages.Pool,
    /// Simulated device: content id per pool page and host page.
    dev: [N]u64 = @splat(0),
    host: [H]u64 = @splat(0),
    /// Reference: each slot's expected content per logical page.
    want: [S][L]u64 = undefined,
    len: [S]u32 = @splat(0),
    swapped: [S]bool = @splat(false),
    /// Checkpoint entries: pinned pages (logical order) and their contents.
    entries: [8]struct { live: bool = false, n: u32 = 0, pages: [L]u32 = undefined, content: [L]u64 = undefined } = @splat(.{}),
    next_content: u64 = 1,
    ops: [11]u32 = @splat(0),

    fn tableOk(self: *const Env, slot: u32) bool {
        if (self.swapped[slot]) return true;
        if (self.pool.mapped[slot] != self.len[slot]) return false;
        const b = @as(u64, 1) << @intCast(slot);
        var seen: u32 = 0;
        for (0..N) |q| {
            if (self.pool.mask[q] & b == 0) continue;
            const l = self.pool.logical[q];
            if (l >= self.len[slot] or self.dev[q] != self.want[slot][l]) return false;
            seen += 1;
        }
        return seen == self.len[slot];
    }
    fn allOk(self: *const Env) bool {
        if (!self.pool.check()) return false;
        for (0..S) |s| if (!self.tableOk(@intCast(s))) return false;
        for (self.entries) |e| {
            if (!e.live) continue;
            for (e.pages[0..e.n], e.content[0..e.n]) |q, c| {
                if (q & pages.host_flag != 0) {
                    if (self.host[q & ~pages.host_flag] != c) return false;
                } else if (self.dev[q] != c or self.pool.pins[q] == 0) return false;
            }
        }
        return true;
    }

    /// Grow `slot` by `k` new pages (fresh content: the sequence wrote them).
    fn grow(self: *Env, slot: u32, k: u32) !void {
        var buf: [L]u32 = undefined;
        const free = self.pool.freeList(buf[0..k]);
        if (free.len < k) return;
        self.pool.checkMap(slot, free) catch return;
        for (free) |q| {
            self.dev[q] = self.next_content;
            self.want[slot][self.len[slot]] = self.next_content;
            self.next_content += 1;
            self.len[slot] += 1;
        }
        self.pool.commitMap(slot, free);
        self.ops[0] += 1;
    }
    fn release(self: *Env, slot: u32) !void {
        try self.pool.release(slot);
        self.len[slot] = 0;
        self.swapped[slot] = false;
        self.ops[1] += 1;
    }
    fn checkpoint(self: *Env, slot: u32, n: u32) !void {
        if (self.swapped[slot] or n == 0 or n > self.len[slot]) return;
        for (&self.entries) |*e| if (!e.live) {
            _ = try self.pool.pin(slot, n, &e.pages);
            e.n = n;
            @memcpy(e.content[0..n], self.want[slot][0..n]);
            e.live = true;
            self.ops[2] += 1;
            return;
        };
    }
    fn drop(self: *Env, i: usize) !void {
        const e = &self.entries[i];
        if (!e.live) return;
        try self.pool.releaseCheckpoint(e.pages[0..e.n]);
        e.live = false;
        self.ops[3] += 1;
    }
    /// Restore entry `i` into empty `slot`: the last page is treated as partial (copied).
    fn demote(self: *Env, i: usize, abort: bool) !void {
        const e = &self.entries[i];
        if (!e.live) return;
        var buf: [L]pages.Move = undefined;
        const moves = self.pool.demotePlan(e.pages[0..e.n], &buf);
        if (moves.len == 0) return;
        if (abort) return self.pool.demoteAbort(moves);
        for (moves) |m| {
            self.host[m.to] = self.dev[m.from];
            self.dev[m.from] = 0xdead;
        }
        self.pool.demoteCommit(e.pages[0..e.n], moves);
        self.ops[8] += 1;
    }
    fn promote(self: *Env, i: usize, abort: bool) !void {
        const e = &self.entries[i];
        if (!e.live) return;
        var buf: [L]pages.Move = undefined;
        const moves = self.pool.promotePlan(e.pages[0..e.n], &buf) orelse return;
        if (moves.len == 0) return;
        if (abort) return self.pool.promoteAbort(moves);
        for (moves) |m| self.dev[m.to] = self.host[m.from];
        self.pool.promoteCommit(e.pages[0..e.n], moves);
        self.ops[9] += 1;
    }
    fn attach(self: *Env, slot: u32, i: usize, partial: bool) !void {
        const e = &self.entries[i];
        if (!e.live or self.len[slot] != 0 or self.swapped[slot]) return;
        for (e.pages[0..e.n]) |q| if (q & pages.host_flag != 0) return; // promote first
        const full = e.pages[0 .. e.n - @intFromBool(partial)];
        const fresh = self.pool.attachPlan(slot, full, if (partial) e.pages[e.n - 1] else null) catch |err| switch (err) {
            error.PoolExhausted => return,
            else => return err,
        };
        var words: [L]u32 = undefined;
        @memcpy(words[0..full.len], full);
        if (fresh) |f| {
            self.dev[f] = self.dev[e.pages[e.n - 1]];
            words[full.len] = f;
        }
        self.pool.commitMap(slot, words[0..e.n]);
        @memcpy(self.want[slot][0..e.n], e.content[0..e.n]);
        self.len[slot] = e.n;
        self.ops[4] += 1;
    }
    /// Rebind `slot`'s first pages to entry `i`'s where the contents agree (dedup).
    fn rebind(self: *Env, slot: u32, i: usize) !void {
        const e = &self.entries[i];
        if (!e.live or self.swapped[slot]) return;
        var n: u32 = 0;
        while (n < e.n and n < self.len[slot] and e.pages[n] & pages.host_flag == 0 and self.want[slot][n] == e.content[n] and self.pool.logical[e.pages[n]] == n) n += 1;
        if (n == 0) return;
        var old: [L]u32 = undefined;
        try self.pool.rebindPlan(slot, 0, e.pages[0..n], &old);
        _ = self.pool.rebindCommit(slot, e.pages[0..n], old[0..n]);
        self.ops[5] += 1;
    }
    fn swapOut(self: *Env, slot: u32, abort: bool) !void {
        if (self.swapped[slot] or self.len[slot] == 0) return;
        _ = self.pool.swapOutPlan(slot) catch |err| switch (err) {
            error.SwapFull => return,
            else => return err,
        };
        if (abort) return self.pool.swapOutAbort(slot);
        for (0..N) |q| if (self.pool.exclusive(@intCast(q), slot)) {
            self.host[self.pool.link[q]] = self.dev[q];
            self.dev[q] = 0xdead; // freed pages may be overwritten
        };
        self.pool.swapOutCommit(slot);
        self.swapped[slot] = true;
        self.ops[6] += 1;
    }
    /// Deep swap-out of a swapped slot: its resident (shared, pinned) pages to host too.
    fn swapShared(self: *Env, slot: u32, abort: bool) !void {
        if (!self.swapped[slot]) return;
        var buf: [L]pages.Move = undefined;
        const moves = self.pool.swapSharedPlan(slot, &buf) catch |err| switch (err) {
            error.SwapFull => return,
            else => return err,
        };
        if (moves.len == 0) {
            try t.expect(!self.pool.holdsResident(slot));
            return;
        }
        if (abort) return self.pool.swapSharedAbort(moves);
        for (moves) |m| self.host[m.to] = self.dev[m.from];
        self.pool.swapSharedCommit(slot, moves);
        try t.expect(!self.pool.holdsResident(slot));
        // Pages no one else holds are free now: they may be overwritten.
        for (moves) |m| if (self.pool.isFree(m.from)) {
            self.dev[m.from] = 0xdead;
        };
        self.ops[10] += 1;
    }
    fn swapIn(self: *Env, slot: u32, abort: bool) !void {
        if (!self.swapped[slot]) return;
        var words: [L]u32 = undefined;
        if (!try self.pool.swapInPlan(slot, 0, &words)) return;
        if (abort) return self.pool.swapInAbort(slot);
        for (0..H) |h| if (self.pool.host_owner[h] == slot) {
            self.dev[self.pool.link[h]] = self.host[h];
        };
        const n = self.pool.swapInCommit(slot);
        self.pool.commitMap(slot, words[0..n]);
        self.swapped[slot] = false;
        self.ops[7] += 1;
    }
};

test "pages: random attach, rebind, pin, swap and release keep every sequence's content" {
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(N, S, L);
    try pool.setHost(H);
    var env: Env = .{ .pool = pool };
    var prng = std.Random.DefaultPrng.init(0x18d4);
    const r = prng.random();
    for (0..20000) |_| {
        const slot = r.intRangeLessThan(u32, 0, S);
        const i = r.intRangeLessThan(usize, 0, env.entries.len);
        switch (r.intRangeLessThan(u8, 0, 13)) {
            12 => try env.swapShared(slot, r.intRangeLessThan(u8, 0, 5) == 0),
            10 => try env.demote(i, r.intRangeLessThan(u8, 0, 5) == 0),
            11 => try env.promote(i, r.intRangeLessThan(u8, 0, 5) == 0),
            0, 1 => try env.grow(slot, r.intRangeAtMost(u32, 1, 3)),
            2 => try env.release(slot),
            3 => try env.checkpoint(slot, r.intRangeAtMost(u32, 1, L)),
            4 => try env.drop(i),
            5 => try env.attach(slot, i, r.boolean()),
            6 => try env.rebind(slot, i),
            7 => try env.swapOut(slot, r.intRangeLessThan(u8, 0, 5) == 0),
            8, 9 => try env.swapIn(slot, r.intRangeLessThan(u8, 0, 5) == 0),
            else => unreachable,
        }
        try t.expect(env.allOk());
    }
    for (env.ops) |n| try t.expect(n > 20); // every operation really ran
    for (0..S) |s| try env.release(@intCast(s));
    for (0..env.entries.len) |i| try env.drop(i);
    try t.expectEqual(@as(u32, N), pool.freeCount());
    try t.expectEqual(@as(u32, H), pool.hostFree());
}

test "pages: a swap moves only the slot's own pages; shared and pinned ones stay held" {
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(N, S, L);
    try pool.setHost(H);
    var env: Env = .{ .pool = pool };
    try env.grow(0, 4);
    try env.checkpoint(0, 3); // pages 0..2 pinned
    try env.attach(1, 0, true); // slot 1: 2 shared + 1 copied
    try env.grow(1, 2); // + 2 own
    const free_before = pool.freeCount();
    try env.swapOut(1, false);
    try t.expectEqual(@as(u32, 3), pool.hostOf(1)); // the copy and its 2 own pages
    try t.expectEqual(free_before + 3, pool.freeCount());
    try env.swapIn(1, false);
    try t.expect(env.allOk());
    try t.expectEqual(free_before, pool.freeCount());
    // Rebind errors: a page at another logical index, a free page.
    var old: [L]u32 = undefined;
    try t.expectError(error.PageInUse, pool.rebindPlan(1, 0, &.{env.entries[0].pages[1]}, &old)); // held already
    const other = for (0..N) |q| {
        if (pool.mask[q] == 1 and pool.logical[q] == 3) break @as(u32, @intCast(q)); // slot 0's 4th page
    } else unreachable;
    try t.expectError(error.InvalidState, pool.rebindPlan(1, 0, &.{other}, &old)); // wrong logical index
    const free_page = pool.freeList(old[0..1])[0];
    try t.expectError(error.InvalidState, pool.rebindPlan(1, 0, &.{free_page}, &old));
}

test "pages: a deep swap-out releases the shared and pinned pages a swapped slot kept" {
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(N, S, L);
    try pool.setHost(H);
    var env: Env = .{ .pool = pool };
    try env.grow(0, 4);
    try env.checkpoint(0, 3); // pages 0..2 pinned
    try env.attach(1, 0, true); // slot 1: 2 shared + 1 copied
    try env.grow(1, 2);
    try env.release(0); // the checkpoint alone pins pages 0..2 now, slot 1 maps two of them
    try env.swapOut(1, false);
    try t.expect(pool.holdsResident(1));
    try t.expectEqual(@as(u32, 0), pool.reclaimable(env.entries[0].pages[0..2])); // slot 1 maps them
    try env.swapShared(1, false);
    try t.expectEqual(@as(u32, 5), pool.hostOf(1));
    try t.expectEqual(@as(u32, 2), pool.reclaimable(env.entries[0].pages[0..2]));
    try env.drop(0);
    try t.expectEqual(@as(u32, N), pool.freeCount()); // nothing on the pool is held
    try env.swapIn(1, false); // all five come back from host
    try t.expect(env.allOk());
    try env.release(1);
    try t.expectEqual(@as(u32, N), pool.freeCount());
    try t.expectEqual(@as(u32, H), pool.hostFree());
}
