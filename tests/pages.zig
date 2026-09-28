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

test "checkpoint source maps validate mixed ancestry and preserve logical indices on suffix demotion" {
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(N, S, L);
    try pool.setHost(H);
    var env: Env = .{ .pool = pool };
    try env.grow(0, 4);
    try env.checkpoint(0, 4);
    try env.release(0);
    const map = env.entries[0].pages[0..4];
    try pool.checkCheckpoint(map);
    var moves: [L]pages.Move = undefined;
    const plan = pool.demotePlan(map[2..], &moves);
    try t.expectEqual(@as(usize, 2), plan.len);
    pool.demoteCommit(map[2..], plan);
    try pool.checkCheckpoint(map);
    try t.expectEqual(@as(u16, 2), pool.host_logical[map[2] & ~pages.host_flag]);
    try t.expectError(error.PagesMissing, pool.checkCheckpoint(&.{ map[0], map[1], map[3], map[2] }));
    try t.expectError(error.PagesMissing, pool.checkCheckpoint(&.{map[2]}));
    try t.expectError(error.PagesMissing, pool.checkCheckpoint(&.{pages.host_flag | H}));
    try t.expectError(error.PagesMissing, pool.checkCheckpoint(&.{N}));
    try t.expectError(error.InvalidToken, pool.checkCheckpoint(&.{}));
    try pool.releaseCheckpoint(map);
    try t.expectError(error.PagesMissing, pool.checkCheckpoint(map));
    try env.grow(1, 1);
    try env.swapOut(1, false);
    const host = for (0..H) |i| {
        if (pool.host_owner[i] == 1) break @as(u32, @intCast(i));
    } else return error.MissingSwap;
    try t.expectError(error.PagesMissing, pool.checkCheckpoint(&.{host | pages.host_flag}));
}

const PreparationCase = struct {
    input: struct { first: u32, window: u32, hit: u32, alias: i32, cancel: bool, busy: u32, reserve: u32, shared: u32, shift: u32 },
    moves: []struct { index: u32, from_page: u32, to: u32 },
    status: []const u8,
    copied: u32,
    freed: u32,
    pages: []u32,
    pins: []u8,
    masks: []u64,
    host_used: []bool,
    host_logical: []u16,
};

test "preparation: independent named-owner oracle, hit after plan and late alias atomicity" {
    const Oracle = struct { generator_sha256: []const u8, cases: []PreparationCase };
    const oracle = try std.json.parseFromSlice(Oracle, t.allocator, @embedFile("fixtures/preparation.json"), .{ .ignore_unknown_fields = true });
    defer oracle.deinit();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("reference/generate_preparation_fixture.py"), &digest, .{});
    try t.expectEqualStrings(oracle.value.generator_sha256, &std.fmt.bytesToHex(digest, .lower));
    try t.expectEqual(@as(usize, 1362), oracle.value.cases.len);
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    for (oracle.value.cases) |c| {
        try pool.init(8, 2, 4);
        try pool.setHost(8);
        var list: [4]u32 = undefined;
        for (&list, 0..) |*q, i| {
            q.* = (@as(u32, @intCast(i)) * 3 + c.input.shift) % 8;
            pool.logical[q.*] = @intCast(i);
            pool.pins[q.*] = 1 + @as(u8, @intFromBool(c.input.shared & (@as(u32, 1) << @intCast(i)) != 0));
        }
        for (0..c.input.busy) |h| {
            pool.host_owner[h] = 0xfe; // unrelated committed checkpoint
            pool.host_logical[h] = @intCast(h % 4);
        }
        const generation = try pool.prepare(&list, c.input.first, c.input.window, c.input.reserve);
        try t.expectEqual(c.moves.len != 0, generation != null);
        if (generation) |g| {
            const moves = try pool.preparedMoves(g);
            try t.expectEqual(c.moves.len, moves.len);
            for (c.moves, moves) |want, got| {
                try t.expectEqual(want.index, got.index);
                try t.expectEqual(want.from_page, got.from);
                try t.expectEqual(want.to, got.to);
                try t.expectError(error.InvalidState, pool.releaseCheckpoint(&.{got.to | pages.host_flag}));
                try t.expectError(error.PagesMissing, pool.checkCheckpoint(&.{got.to | pages.host_flag}));
            }
            try t.expectEqual(8 - c.input.busy - @as(u32, @intCast(moves.len)), pool.hostFree());
            try t.expectError(error.PendingPreparation, pool.commitPreparation(g, &list));
            try t.expectError(error.PendingPreparation, pool.abortPreparation(g));
            try t.expectError(error.InvalidState, pool.setHost(8));
            try t.expectError(error.InvalidState, pool.prepare(&list, 0, 1, 0));
        }
        if (c.input.hit != 0) {
            const full = if (c.input.hit == 2) @as(usize, 3) else 4;
            const fresh = try pool.attachPlan(0, list[0..full], if (full == 3) list[3] else null);
            pool.commitMap(0, list[0..full]);
            if (fresh) |q| pool.commitMap(0, &.{q});
        }
        if (c.input.alias >= 0) pool.pins[list[@intCast(c.input.alias)]] += 1;
        if (generation) |g| {
            try pool.ackPreparation(g);
            if (std.mem.eql(u8, c.status, "commit")) {
                const result = try pool.commitPreparation(g, &list);
                try t.expectEqual(c.copied, result.copied);
                try t.expectEqual(c.freed, result.freed);
            } else {
                if (std.mem.eql(u8, c.status, "conflict")) {
                    const before_list = list;
                    const before_pins = pool.pins[0..8].*;
                    const before_masks = pool.mask[0..8].*;
                    const before_host = pool.host_owner[0..8].*;
                    const before_logical = pool.host_logical[0..8].*;
                    try t.expectError(error.InvalidState, pool.commitPreparation(g, &list));
                    try t.expectEqualSlices(u32, &before_list, &list);
                    try t.expectEqualSlices(u8, &before_pins, pool.pins[0..8]);
                    try t.expectEqualSlices(u64, &before_masks, pool.mask[0..8]);
                    try t.expectEqualSlices(u8, &before_host, pool.host_owner[0..8]);
                    try t.expectEqualSlices(u16, &before_logical, pool.host_logical[0..8]);
                }
                try pool.abortPreparation(g);
            }
            try t.expectError(error.InvalidState, pool.ackPreparation(g));
            try t.expectError(error.InvalidState, pool.abortPreparation(g));
        }
        try t.expectEqualSlices(u32, c.pages, &list);
        try t.expectEqualSlices(u8, c.pins, pool.pins[0..8]);
        try t.expectEqualSlices(u64, c.masks, pool.mask[0..8]);
        try t.expectEqualSlices(u16, c.host_logical, pool.host_logical[0..8]);
        for (c.host_used, 0..) |used, h| try t.expectEqual(@as(u8, if (used) 0xfe else 0xff), pool.host_owner[h]);
        try pool.checkCheckpoint(&list);
        try t.expect(pool.check());
    }
}

test "preparation: stale incarnation, bounded options, source validation and destination ownership" {
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(8, 2, 4);
    try pool.setHost(8);
    var list = [_]u32{ 0, 1, 2, 3 };
    for (list, 0..) |q, i| {
        pool.pins[q] = 1;
        pool.logical[q] = @intCast(i);
    }
    for ([_]u32{ 0, 3, 5, std.math.maxInt(u32) }) |window| try t.expectError(error.InvalidToken, pool.prepare(&list, 0, window, 0));
    try t.expectError(error.InvalidToken, pool.prepare(&list, 5, 1, 0));
    try t.expectError(error.InvalidToken, pool.prepare(&list, 0, 1, 9));
    try t.expectError(error.InvalidToken, pool.prepare(&.{}, 0, 1, 0));
    try t.expectError(error.PagesMissing, pool.prepare(&.{8}, 0, 1, 0));
    try t.expectEqual(@as(?u64, null), try pool.prepare(&list, 0, 4, 8));
    try t.expectEqual(@as(u32, 8), pool.hostFree());
    const old = (try pool.prepare(&list, 0, 4, 0)).?;
    try pool.ackPreparation(old);
    try t.expectError(error.InvalidState, pool.ackPreparation(old));
    try pool.abortPreparation(old);
    const current = (try pool.prepare(&list, 0, 4, 0)).?;
    try t.expect(current > old);
    try t.expectError(error.InvalidState, pool.preparedMoves(old));
    try t.expectError(error.InvalidState, pool.ackPreparation(old));
    try t.expectError(error.InvalidState, pool.abortPreparation(old));
    try t.expectError(error.InvalidState, pool.commitPreparation(old, &list));
    try t.expectEqual(@as(u32, 4), pool.hostFree());
    try pool.ackPreparation(current);
    try t.expectError(error.InvalidState, pool.commitPreparation(current, list[0..3]));
    list[3] = 4;
    try t.expectError(error.InvalidState, pool.commitPreparation(current, &list));
    list[3] = 3;
    pool.logical[3] = 2;
    try t.expectError(error.InvalidState, pool.commitPreparation(current, &list));
    pool.logical[3] = 3;
    const moves = try pool.preparedMoves(current);
    const last_host = moves[moves.len - 1].to;
    const owner = pool.host_owner[last_host];
    pool.host_owner[last_host] = 0xfe;
    try t.expectError(error.InvalidState, pool.commitPreparation(current, &list));
    try t.expectError(error.InvalidState, pool.abortPreparation(current));
    try t.expectEqual(@as(u32, 4), pool.hostFree());
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, &list);
    try t.expectEqualSlices(u8, &.{ 1, 1, 1, 1 }, pool.pins[0..4]);
    pool.host_owner[last_host] = owner;
    try pool.abortPreparation(current);
    const half = (try pool.prepare(&list, 0, 2, 0)).?;
    try pool.ackPreparation(half);
    try t.expectEqual(pages.PreparedResult{ .copied = 2, .freed = 2 }, try pool.commitPreparation(half, &list));
    const rest = (try pool.prepare(&list, 0, 4, 0)).?;
    try t.expectEqual(@as(u32, 2), (try pool.preparedMoves(rest))[0].index);
    try pool.ackPreparation(rest);
    try pool.abortPreparation(rest);
    pool.preparation.generation = std.math.maxInt(u64);
    try t.expectError(error.PreparationExhausted, pool.prepare(&list, 0, 4, 0));
    try t.expectEqual(@as(u32, 6), pool.hostFree());
    try pool.checkCheckpoint(&list);
}

test "preparation: skip live pages, hit then leave, and disjoint demotion reservations" {
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(8, 2, 4);
    try pool.setHost(8);
    var list = [_]u32{ 0, 1, 2, 3 };
    for (list, 0..) |q, i| {
        pool.pins[q] = 1;
        pool.logical[q] = @intCast(i);
    }
    pool.commitMap(0, list[0..2]);
    const g = (try pool.prepare(&list, 0, 4, 1)).?;
    const plan = try pool.preparedMoves(g);
    try t.expectEqual(@as(usize, 2), plan.len);
    try t.expectEqual(@as(u32, 2), plan[0].index);
    try t.expectEqual(@as(u32, 3), plan[1].index);
    // Another cache's ordinary demotion must not reuse the reserved host pages.
    var other = [_]u32{ 4, 5 };
    for (other, 0..) |q, i| {
        pool.pins[q] = 1;
        pool.logical[q] = @intCast(i);
    }
    var buf: [2]pages.Move = undefined;
    const disjoint = pool.demotePlan(&other, &buf);
    try t.expectEqual(@as(usize, 2), disjoint.len);
    for (disjoint) |m| for (plan) |p| try t.expect(m.to != p.to);
    pool.demoteCommit(&other, disjoint);
    try t.expectEqual(@as(u32, 4), pool.hostFree());
    try t.expectEqual(@as(?u32, null), try pool.attachPlan(1, &list, null));
    pool.commitMap(1, &list);
    try pool.release(1); // leaving before completion makes these pages reclaimable again
    try pool.ackPreparation(g);
    try t.expectEqual(pages.PreparedResult{ .copied = 2, .freed = 2 }, try pool.commitPreparation(g, &list));
    try t.expectEqual(@as(u64, 1), pool.mask[0]);
    try t.expectEqual(@as(u64, 1), pool.mask[1]);
    try t.expectEqual(@as(u32, 2), pool.mapped[0]);
    try pool.checkCheckpoint(&list);
    try pool.checkCheckpoint(&other);
    try t.expect(pool.check());
    try pool.release(0);
    try pool.releaseCheckpoint(&list);
    try pool.releaseCheckpoint(&other);
    try t.expectEqual(@as(u32, 8), pool.freeCount());
    try t.expectEqual(@as(u32, 8), pool.hostFree());
}
