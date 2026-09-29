//! The shared KV pool end to end without a GPU (docs/specs/concurrent.md, "18d.2"–"18d.4"):
//! the real batcher drives a backend made of the real page accounting (`model.pages.Pool`)
//! and the real prefix-cache policies (`session.kvcache`), with simulated page contents.
//! Every logits row is computed by reading all earlier positions through the slot's page
//! table, so a missing, wrong, stale or shared-and-overwritten page changes the output. Each
//! generation checks every row against its solo value; at the end every page is free.
const std = @import("std");
const batcher = @import("serve").batcher;
const session = @import("session");
const pages = @import("model").pages;
const kvcache = session.kvcache;
const t = std.testing;

const V = 8; // logits per row
const P = 4; // tokens per page
const S = 4; // slots
const L = 24; // pages per sequence
const B = 9; // message boundary token

fn mixh(h: u64, x: u64) u64 {
    return (h ^ x) *% 0x100000001b3 +% 0x9e3779b97f4a7c15;
}
/// Value stored at a position: a function of the history up to and including it.
fn posValue(hist: []const u32) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (hist) |x| h = mixh(h, x);
    return h;
}
/// The logits after `hist`: a function of every position's stored value (what a correct
/// attention read would see).
fn soloLogits(hist: []const u32, out: []f32) void {
    var h: u64 = 7;
    for (0..hist.len) |k| h = mixh(h, posValue(hist[0 .. k + 1]));
    for (out, 0..) |*x, i| x.* = @floatFromInt((mixh(h, i) >> 40) % 1000);
}
fn pick(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits, 0..) |x, i| if (x > logits[best]) {
        best = i;
    };
    return @intCast(best + 20);
}

const Sys = struct {
    io: std.Io,
    pool: *pages.Pool,
    cache: ?kvcache.Cache,
    dev: []u64, // pool page q, offset o: dev[q * P + o]
    host: []u64,
    hist: [S][L * P]u32 = undefined,
    pos: [S]u32 = @splat(0),
    snaps: [64]struct { hash: u64 = 0, len: u32 = 0 } = @splat(.{}),
    rows_out: [S * V]f32 = undefined,
    prefill_out: [S * V]f32 = undefined,
    flight: ?struct { n: usize, slots: [S]u32, count: [S]usize, total: [S]usize, tokens: [S][3]u32 } = null,
    violations: u32 = 0,
    /// Per decode batch (so the time slice fires).
    delay_ns: u64 = 0,
    swaps: u32 = 0,
    restores: u32 = 0,
    demoted: u32 = 0,
    promoted: u32 = 0,
    deep_swaps: u32 = 0,
    radix: bool = false,

    fn table(self: *const Sys, slot: u32, k: u32) ?u32 {
        const b = @as(u64, 1) << @intCast(slot);
        for (0..self.pool.pages) |q| {
            if (self.pool.mask[q] & b != 0 and self.pool.logical[q] == k) return @intCast(q);
        }
        return null;
    }
    /// Write the value of `slot`'s position `at` (its page must be mapped).
    fn write(self: *Sys, slot: u32, at: u32) void {
        const q = self.table(slot, at / P) orelse {
            self.violations += 1;
            return;
        };
        self.dev[q * P + at % P] = posValue(self.hist[slot][0 .. at + 1]);
    }
    /// Logits read through the table (what the device computes).
    fn readLogits(self: *Sys, slot: u32, out: []f32) void {
        var h: u64 = 7;
        for (0..self.pos[slot]) |k| {
            const q = self.table(slot, @intCast(k / P)) orelse {
                self.violations += 1;
                return;
            };
            h = mixh(h, self.dev[q * P + k % P]);
        }
        for (out, 0..) |*x, i| x.* = @floatFromInt((mixh(h, i) >> 40) % 1000);
    }
    /// All or nothing, like `Model.ensurePages` (a failed admission holds no new page).
    /// After every cache call: radix segment ownership holds (each page a node does not own
    /// is an ancestor's at the same index) and every pool page's pins are exactly the live
    /// segments naming it (no orphaned or missing pin).
    fn checkCache(self: *Sys) void {
        const c = self.cache orelse return;
        if (!self.radix) return;
        const r: *kvcache.Radix = @ptrCast(@alignCast(c.ctx));
        if (r.ownViolation() != null) self.violations += 1;
        for (0..self.pool.pages) |q| {
            if (r.segmentPins(@intCast(q)) != self.pool.pins[q]) self.violations += 1;
        }
    }
    fn mapFree(self: *Sys, slot: u32, want: u32) bool {
        if (self.pool.mapped[slot] >= want) return true;
        while (self.pool.freeCount() < want - self.pool.mapped[slot]) {
            if (!self.makeRoom(null)) return false;
        }
        while (self.pool.mapped[slot] < want) {
            var buf: [1]u32 = undefined;
            const list = self.pool.freeList(&buf);
            if (list.len == 0) {
                if (!self.makeRoom(null)) return false;
                continue;
            }
            self.pool.checkMap(slot, list) catch {
                self.violations += 1;
                return false;
            };
            self.pool.commitMap(slot, list);
        }
        return true;
    }

    /// Like `ModelBackend.makeRoom`: a checkpoint goes first; then a swapped sequence other
    /// than `except` gives up its resident pages (a deep swap-out).
    fn makeRoom(self: *Sys, except: ?u32) bool {
        if (self.cache) |c| if (c.evict(self.device())) {
            self.checkCache();
            return true;
        };
        for (0..S) |i| {
            const slot: u32 = @intCast(i);
            if (except != null and except.? == slot or !self.pool.holdsResident(slot)) continue;
            var buf: [L]pages.Move = undefined;
            const moves: []const pages.Move = while (true) {
                break self.pool.swapSharedPlan(slot, &buf) catch |e| {
                    if (e == error.SwapFull) if (self.cache) |c| if (c.evictHost(self.device())) {
                        self.checkCache();
                        continue;
                    };
                    if (e != error.SwapFull) self.violations += 1;
                    break &.{};
                };
            };
            if (moves.len == 0) continue;
            for (moves) |m| @memcpy(self.host[m.to * P ..][0..P], self.dev[m.from * P ..][0..P]);
            self.pool.swapSharedCommit(slot, moves);
            self.deep_swaps += 1;
            return true;
        }
        return false;
    }

    // ---- batcher backend
    pub fn reset(self: *Sys, slot: u32) !void {
        try self.pool.release(slot);
        self.pos[slot] = 0;
    }
    pub fn begin(self: *Sys, slot: u32, prompt: []const u32) !u32 {
        try self.reset(slot);
        @memcpy(self.hist[slot][0..prompt.len], prompt);
        const c = self.cache orelse return 0;
        const start = try c.restore(self.device(), slot, prompt);
        self.checkCache();
        self.restores += @intFromBool(start > 0);
        return start;
    }
    pub fn checkpoint(self: *Sys, slot: u32, prefix: []const u32) !void {
        const c = self.cache orelse return;
        if (self.pos[slot] != prefix.len) self.violations += 1;
        try c.checkpoint(self.device(), slot, prefix);
        self.checkCache();
    }
    pub fn packFits(_: *Sys, remaining: []const usize) bool {
        return remaining.len <= 2;
    }
    pub fn prefillUnit(self: *Sys, items: []const batcher.Item) !batcher.Unit {
        if (self.flight == null) {
            var f: @TypeOf(self.flight.?) = .{ .n = items.len, .slots = undefined, .count = undefined, .total = undefined, .tokens = undefined };
            for (items, 0..) |item, i| {
                const n = @min(3, item.tokens.len);
                f.slots[i] = item.slot;
                f.count[i] = n;
                f.total[i] = item.tokens.len;
                @memcpy(f.tokens[i][0..n], item.tokens[0..n]);
            }
            self.flight = f;
        }
        const f = self.flight.?;
        self.flight = null;
        var unit: batcher.Unit = .{ .done = true, .logits = self.prefill_out[0 .. f.n * V] };
        for (0..f.n) |i| {
            const slot = f.slots[i];
            for (f.tokens[i][0..f.count[i]]) |x| {
                self.hist[slot][self.pos[slot]] = x;
                self.write(slot, self.pos[slot]);
                self.pos[slot] += 1;
            }
            unit.consumed[i] = f.count[i];
            self.readLogits(slot, self.prefill_out[i * V ..][0..V]);
        }
        return unit;
    }
    pub fn abortChunk(self: *Sys) void {
        self.flight = null;
    }
    pub fn admit(self: *Sys, slot: u32, tokens: usize) bool {
        return self.mapFree(slot, @intCast((tokens + P - 1) / P));
    }
    pub fn release(self: *Sys, slot: u32) void {
        self.pool.release(slot) catch {
            self.violations += 1;
        };
    }
    pub fn grow(self: *Sys, slot: u32) !bool {
        if (self.pool.swapped[slot] != 0) return error.SlotSwapped;
        return self.mapFree(slot, self.pos[slot] / P + 1);
    }
    pub fn swapOut(self: *Sys, slot: u32) !bool {
        while (true) {
            _ = self.pool.swapOutPlan(slot) catch |e| switch (e) {
                error.SwapFull => {
                    if (self.cache) |c| if (c.evictHost(self.device())) {
                        self.checkCache();
                        continue;
                    };
                    return false;
                },
                else => return e,
            };
            break;
        }
        for (0..self.pool.pages) |q| if (self.pool.exclusive(@intCast(q), slot)) {
            const h = self.pool.link[q];
            @memcpy(self.host[h * P ..][0..P], self.dev[q * P ..][0..P]);
            @memset(self.dev[q * P ..][0..P], 0xdead); // freed: may be overwritten
        };
        self.pool.swapOutCommit(slot);
        self.swaps += 1;
        return true;
    }
    pub fn swapIn(self: *Sys, slot: u32, spare: u32) !bool {
        var words: [L]u32 = undefined;
        while (!try self.pool.swapInPlan(slot, spare, &words)) {
            if (!self.makeRoom(slot)) return false;
        }
        for (0..self.pool.host_pages) |h| if (self.pool.host_owner[h] == slot) {
            @memcpy(self.dev[self.pool.link[h] * P ..][0..P], self.host[h * P ..][0..P]);
        };
        const n = self.pool.swapInCommit(slot);
        self.pool.commitMap(slot, words[0..n]);
        return true;
    }
    pub fn checkRow(_: *Sys, _: batcher.Row) !void {}
    pub fn decodeBatch(self: *Sys, rows: []const batcher.Row) ![]const f32 {
        if (self.delay_ns > 0) try self.io.sleep(.fromNanoseconds(self.delay_ns), .awake);
        for (rows, 0..) |row, r| {
            self.hist[row.slot][self.pos[row.slot]] = row.token;
            self.write(row.slot, self.pos[row.slot]);
            self.pos[row.slot] += 1;
            self.readLogits(row.slot, self.rows_out[r * V ..][0..V]);
        }
        return self.rows_out[0 .. rows.len * V];
    }

    // ---- kvcache device
    fn device(self: *Sys) kvcache.Device {
        return .{ .ctx = self, .vtable = &dev_vt };
    }
    const dev_vt: kvcache.Device.VTable = .{ .pageTokens = dPage, .save = dSave, .load = dLoad, .pin = dPin, .unpin = dUnpin, .attach = dAttach, .rebind = dRebind, .demote = dDemote, .promote = dPromote, .reclaimable = dReclaimable };
    fn dReclaimable(ctx: *anyopaque, list: []const u32) u32 {
        return of(ctx).pool.reclaimable(list);
    }
    fn dDemote(ctx: *anyopaque, list: []u32) anyerror!u32 {
        const self = of(ctx);
        var buf: [L]pages.Move = undefined;
        const moves = self.pool.demotePlan(list, &buf);
        for (moves) |m| {
            @memcpy(self.host[m.to * P ..][0..P], self.dev[m.from * P ..][0..P]);
            @memset(self.dev[m.from * P ..][0..P], 0xdead); // freed: may be overwritten
        }
        self.pool.demoteCommit(list, moves);
        self.demoted += @intCast(moves.len);
        return @intCast(moves.len);
    }
    fn dPromote(ctx: *anyopaque, list: []u32) anyerror!bool {
        const self = of(ctx);
        var buf: [L]pages.Move = undefined;
        const moves = self.pool.promotePlan(list, &buf) orelse return false;
        for (moves) |m| @memcpy(self.dev[m.to * P ..][0..P], self.host[m.from * P ..][0..P]);
        self.pool.promoteCommit(list, moves);
        self.promoted += @intCast(moves.len);
        return true;
    }
    fn of(ctx: *anyopaque) *Sys {
        return @ptrCast(@alignCast(ctx));
    }
    fn dPage(_: *anyopaque) u32 {
        return P;
    }
    fn dSave(ctx: *anyopaque, slot: u32, snap: u32) anyerror!void {
        const self = of(ctx);
        self.snaps[snap] = .{ .hash = posValue(self.hist[slot][0..self.pos[slot]]), .len = self.pos[slot] };
    }
    fn dLoad(ctx: *anyopaque, slot: u32, snap: u32, position: u32) anyerror!void {
        const self = of(ctx);
        if (self.snaps[snap].len != position or self.snaps[snap].hash != posValue(self.hist[slot][0..position])) self.violations += 1;
        self.pos[slot] = position;
    }
    fn dPin(ctx: *anyopaque, slot: u32, tokens: u32, out: []u32) anyerror![]const u32 {
        return of(ctx).pool.pin(slot, (tokens + P - 1) / P, out);
    }
    fn dUnpin(ctx: *anyopaque, list: []const u32) anyerror!void {
        return of(ctx).pool.releaseCheckpoint(list);
    }
    fn dAttach(ctx: *anyopaque, slot: u32, full: []const u32, partial: ?u32) anyerror!void {
        const self = of(ctx);
        const fresh = try self.pool.attachPlan(slot, full, partial);
        var words: [L]u32 = undefined;
        @memcpy(words[0..full.len], full);
        if (fresh) |f| {
            @memcpy(self.dev[f * P ..][0..P], self.dev[partial.? * P ..][0..P]);
            words[full.len] = f;
        }
        self.pool.commitMap(slot, words[0 .. full.len + @intFromBool(fresh != null)]);
    }
    fn dRebind(ctx: *anyopaque, slot: u32, list: []const u32) anyerror!u32 {
        const self = of(ctx);
        var old: [L]u32 = undefined;
        try self.pool.rebindPlan(slot, 0, list, &old);
        for (list, old[0..list.len]) |q, o| {
            if (!std.mem.eql(u64, self.dev[q * P ..][0..P], self.dev[o * P ..][0..P])) self.violations += 1; // not byte-identical
        }
        return self.pool.rebindCommit(slot, list, old[0..list.len]);
    }
};

const Bt = batcher.Batcher(*Sys);

/// One conversation turn: [B sys...] [B user c k] [B], then `steps` generated tokens.
fn gen(b: *Bt, prompt: []const u32, steps: u32, out: *anyerror!void) void {
    out.* = genInner(b, prompt, steps);
}
fn genInner(b: *Bt, prompt: []const u32, steps: u32) !void {
    const slot = while (true) break b.join() catch |e| switch (e) {
        error.NoSlot => {
            try b.io.sleep(.fromMicroseconds(200), .awake);
            continue;
        },
        else => return e,
    };
    defer b.leave(slot);
    var hist: [L * P]u32 = undefined;
    @memcpy(hist[0..prompt.len], prompt);
    const start = try b.begin(slot, prompt);
    var points: [session.checkpoint.max_points]u32 = undefined;
    var at = start;
    for (session.checkpoint.candidatePoints(prompt, start, B, &points)) |p| {
        _ = try b.prefill(slot, prompt[at..p]);
        b.sampled(slot);
        try b.checkpoint(slot, prompt[0..p]);
        at = p;
    }
    var logits = try b.prefill(slot, prompt[at..]);
    var n: usize = prompt.len;
    var want: [V]f32 = undefined;
    for (0..steps) |_| {
        soloLogits(hist[0..n], &want);
        if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
        const token = pick(logits);
        b.sampled(slot);
        hist[n] = token;
        n += 1;
        logits = try b.step(slot, token);
    }
    soloLogits(hist[0..n], &want);
    if (!std.mem.eql(f32, &want, logits)) return error.WrongLogits;
    b.sampled(slot);
}

fn makePrompt(buf: []u32, sys: usize, conv: u32, turn: u32) []const u32 {
    var n: usize = 0;
    buf[n] = B;
    n += 1;
    for (0..sys) |i| {
        buf[n] = @intCast(1000 + i);
        n += 1;
    }
    for (0..turn + 1) |k| {
        buf[n] = B;
        buf[n + 1] = 100 + conv;
        buf[n + 2] = @intCast(200 + k);
        n += 3;
    }
    buf[n] = B;
    return buf[0 .. n + 1];
}

fn watchStall(b: *Bt, kind: ?kvcache.Kind, tier: bool, seed: u64) void {
    b.io.sleep(.fromSeconds(10), .awake) catch return;
    b.mutex.lockUncancelable(b.io);
    defer b.mutex.unlock(b.io);
    std.debug.print("STALL kind={any} tier={} seed={d} epoch={d} blocked={any} swapped={d} pack={any}\n", .{ kind, tier, seed, b.admit_epoch, b.blocked, b.swapped_slots, b.pack });
    for (b.slot[0..S], 0..) |s, i| std.debug.print("slot {d}: used={} closing={} op={any} running={} decoding={} admitted={} base={d} done={d} tokens={d} failed={d} swapped={} swap_epoch={d} order={d} wait={d}\n", .{ i, s.used, s.closing, s.op, s.running, s.decoding, s.admitted, s.base, s.done, s.tokens.len, s.failed_epoch, s.swapped, s.swap_epoch, s.order, s.wait_ns });
    @panic("KV system progress deadline exceeded");
}

const Outcome = struct { restores: u32, swaps: u32, demoted: u32 = 0, promoted: u32 = 0, deep_swaps: u32 = 0 };
fn run(kind: ?kvcache.Kind, pool_pages: u32, slice_ms: ?i64, seed: u64) !Outcome {
    return runDelay(kind, pool_pages, slice_ms, seed, 0);
}
fn runDelay(kind: ?kvcache.Kind, pool_pages: u32, slice_ms: ?i64, seed: u64, delay_ns: u64) !Outcome {
    return runShape(kind, pool_pages, slice_ms, seed, delay_ns, .{});
}
/// Workload shape: `distinct` conversations with `sys`-token system prompts and up to
/// `turns` turns; `tier`: radix demotes to host (false: drops).
const Shape = struct { distinct: u32 = 5, turns: u32 = 3, sys: usize = 9, tier: bool = true };
fn runShape(kind: ?kvcache.Kind, pool_pages: u32, slice_ms: ?i64, seed: u64, delay_ns: u64, shape: Shape) !Outcome {
    const io = t.io;
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    try pool.init(pool_pages, S, L);
    try pool.setHost(400);
    const dev = try t.allocator.alloc(u64, pool_pages * P);
    defer t.allocator.free(dev);
    const host = try t.allocator.alloc(u64, 400 * P);
    defer t.allocator.free(host);
    var sys: Sys = .{ .io = io, .pool = pool, .cache = if (kind) |k| try kvcache.createTiered(t.allocator, k, 6, L * P, L, B, shape.tier) else null, .dev = dev, .host = host, .delay_ns = delay_ns, .radix = kind != null and kind.? == .radix };
    defer if (sys.cache) |c| c.deinit(t.allocator);
    var b = try Bt.init(io, &sys, .{ .slots = S, .vocab = V, .stall = .{ .ns = 0 }, .order = .shortest, .pack = 2, .swap_slice = if (slice_ms) |ms| .fromMilliseconds(ms) else null });
    var task = try io.concurrent(Bt.run, .{&b});
    var watchdog = try io.concurrent(watchStall, .{ &b, kind, shape.tier, seed });
    defer watchdog.cancel(io);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var bufs: [16][L * P]u32 = undefined;
    var results: [16]anyerror!void = undefined;
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (0..16) |i| {
        const conv = r.intRangeLessThan(u32, 0, shape.distinct);
        const p = makePrompt(&bufs[i], shape.sys + conv % 7, conv, r.intRangeLessThan(u32, 0, shape.turns));
        try group.concurrent(io, gen, .{ &b, p, r.intRangeAtMost(u32, 8, 40), &results[i] });
        if (r.boolean()) try io.sleep(.fromMicroseconds(r.intRangeLessThan(u32, 0, 400)), .awake);
    }
    try group.await(io);
    b.stop();
    task.await(io);
    for (results) |res| try res;
    try t.expectEqual(@as(u32, 0), sys.violations);
    try t.expect(pool.check());
    if (sys.cache) |c| while (c.evict(sys.device()) or c.evictHost(sys.device())) {};
    try t.expectEqual(pool_pages, pool.freeCount());
    try t.expectEqual(@as(u32, 400), pool.hostFree());
    return .{ .restores = sys.restores, .swaps = sys.swaps, .demoted = sys.demoted, .promoted = sys.promoted, .deep_swaps = sys.deep_swaps };
}

test "kv system: no cache, roomy pool" {
    _ = try run(null, 120, null, 1);
}
test "kv system: flat and radix, roomy pool: restores, exact" {
    for ([_]kvcache.Kind{ .flat, .radix }) |k| {
        const r = try run(k, 120, null, 2);
        try t.expect(r.restores > 0);
    }
}
test "kv system: tight pool, swaps and time slice, every policy, several seeds" {
    for ([_]?kvcache.Kind{ null, .flat, .radix }) |k| {
        var swaps: u32 = 0;
        var restores: u32 = 0;
        var demoted: u32 = 0;
        var promoted: u32 = 0;
        for (0..4) |seed| {
            const r = try runDelay(k, 30, 1, 100 + seed, 100_000);
            swaps += r.swaps;
            restores += r.restores;
            demoted += r.demoted;
            promoted += r.promoted;
        }
        std.debug.print("kv system {?}: swaps {d} restores {d} demoted {d} promoted {d}\n", .{ k, swaps, restores, demoted, promoted });
        // Tiering ran (whether a demoted conversation returns depends on timing; promotion
        // is tested deterministically in tests/kvcache.zig).
        if (k == .radix) try t.expect(demoted > 0);
        // Every policy really ran under pressure: swaps, and restores with a cache.
        try t.expect(swaps > 0);
        if (k != null) try t.expect(restores > 0);
    }
}
test "kv system: radix without the host tier, tight pool" {
    for (0..4) |seed| _ = try runShape(.radix, 30, 1, 200 + seed, 100_000, .{ .tier = false });
}
test "kv system: distinct long conversations fill the pool with checkpoints, every policy and tier" {
    // Like bench/workloads/multiturn-distinct-v1.json: few checkpoints fit, all distinct;
    // swapped sequences' shared pages and prompts between segments fill the pool (the
    // 2026-09-28 tier-off hang: every sequence waited, nothing ran).
    var deep: u32 = 0;
    var swaps: u32 = 0;
    for ([_]kvcache.Kind{ .flat, .radix }) |k| for ([_]bool{ true, false }) |tier| for (0..3) |seed| {
        const r = try runShape(k, 40, 1, 300 + seed, 50_000, .{ .distinct = 16, .turns = 2, .sys = 40, .tier = tier });
        deep += r.deep_swaps;
        swaps += r.swaps;
    };
    std.debug.print("kv system distinct: swaps {d} deep swap-outs {d}\n", .{ swaps, deep });
    try t.expect(swaps > 0);
    try t.expect(deep > 0);
}

const PreparedCommit = struct {
    sys: *Sys,
    generation: u64,
    first: u32,
    calls: u32 = 0,
    fail: bool = false,
    fn callback(self: *@This()) kvcache.PreparationCommit {
        return .{ .ctx = self, .apply = apply };
    }
    fn apply(ctx: *anyopaque, list: []u32, first: u32) !kvcache.PreparedResult {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        self.calls += 1;
        try t.expectEqual(self.first, first);
        if (self.fail) return error.CommitFault;
        const result = try self.sys.pool.commitPreparation(self.generation, list);
        return .{ .copied = result.copied, .freed = result.freed };
    }
};

fn insertPreparationFixture(sys: *Sys, tokens: []const u32) !void {
    try sys.pool.release(0);
    sys.pos[0] = 0;
    try t.expect(sys.mapFree(0, @intCast((tokens.len + P - 1) / P)));
    for (tokens, 0..) |x, i| {
        sys.hist[0][i] = x;
        sys.write(0, @intCast(i));
    }
    sys.pos[0] = @intCast(tokens.len);
    try sys.cache.?.checkpoint(sys.device(), 0, tokens);
    try sys.pool.release(0);
    sys.checkCache();
}

test "preparation: independent joint prefix/object oracle through real cache and pool" {
    const Case = struct {
        input: struct { nodes: [][]const u32, selected: u32, late: []const u32, protect: i32, window: u32, hit: bool, mode: []const u8, reserve: u32 },
        first: u32,
        planned: u32,
        result: []const u8,
        callbacks: u32,
        copied: u32,
        freed: u32,
        nodes: []struct { tokens: []const u32, first: u32, host: []bool, pins: []u32, refs: u32, active: bool },
    };
    const Oracle = struct { generator_sha256: []const u8, cases: []Case };
    const oracle = try std.json.parseFromSlice(Oracle, t.allocator, @embedFile("fixtures/preparation-cache.json"), .{ .ignore_unknown_fields = true });
    defer oracle.deinit();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("reference/generate_preparation_cache_fixture.py"), &digest, .{});
    try t.expectEqualStrings(oracle.value.generator_sha256, &std.fmt.bytesToHex(digest, .lower));
    try t.expectEqual(@as(usize, 3872), oracle.value.cases.len);
    const pool = try t.allocator.create(pages.Pool);
    defer t.allocator.destroy(pool);
    for (oracle.value.cases) |case| {
        try pool.init(64, S, L);
        try pool.setHost(32);
        var dev: [64 * P]u64 = @splat(0);
        var host: [32 * P]u64 = @splat(0);
        const r = try kvcache.Radix.create(t.allocator, 16, L * P, L, B);
        const cache = r.cache();
        defer cache.deinit(t.allocator);
        var sys: Sys = .{ .io = t.io, .pool = pool, .cache = cache, .dev = &dev, .host = &host, .radix = true };
        for (case.input.nodes) |tokens| try insertPreparationFixture(&sys, tokens);
        const candidate = cache.preparationCandidate(case.input.selected) orelse return error.MissingCandidate;
        try t.expectEqual(case.first, candidate.first);
        if (r.children[case.input.selected] != 0) try t.expectEqual(@as(?kvcache.Candidate, null), cache.sourceCandidate(case.input.selected));
        const source = try cache.acquireSource(candidate.handle);
        const g = try pool.prepare(source.pages, candidate.first, case.input.window, case.input.reserve);
        try t.expectEqual(case.planned != 0, g != null);
        var commit: PreparedCommit = .{ .sys = &sys, .generation = g orelse 0, .first = candidate.first };
        if (g) |generation| {
            const moves = try pool.preparedMoves(generation);
            try t.expectEqual(case.planned, moves.len);
            try t.expectError(error.PendingPreparation, cache.finishPreparation(source.lease, commit.callback()));
            try t.expect(r.sources[case.input.selected].active);
            try t.expectEqual(@as(u32, 1), commit.calls);
            commit.calls = 0;
            const count = if (std.mem.eql(u8, case.input.mode, "cancel")) @min(1, moves.len) else moves.len;
            for (moves[0..count]) |m| @memcpy(host[m.to * P ..][0..P], dev[m.from * P ..][0..P]);
        } else try cache.releaseSource(source.lease);
        if (case.input.late.len != 0) try insertPreparationFixture(&sys, case.input.late);
        var protected: ?kvcache.Lease = null;
        if (case.input.protect >= 0) {
            const i: u32 = @intCast(case.input.protect);
            if (r.store.entries[i].len > 0) protected = (try cache.acquireSource(.{ .index = i, .generation = r.sources[i].generation })).lease;
        }
        if (case.input.hit) {
            const tokens = case.input.nodes[case.input.selected];
            @memcpy(sys.hist[1][0..tokens.len], tokens);
            sys.hist[1][tokens.len] = 9999;
            try t.expectEqual(tokens.len, try cache.restore(sys.device(), 1, sys.hist[1][0 .. tokens.len + 1]));
        }
        if (g) |generation| {
            try pool.ackPreparation(generation);
            commit.fail = std.mem.eql(u8, case.input.mode, "fault");
            if (std.mem.eql(u8, case.result, "commit")) {
                const result = try cache.finishPreparation(source.lease, commit.callback());
                try t.expectEqual(case.copied, result.copied);
                try t.expectEqual(case.freed, result.freed);
                try t.expectError(error.InvalidSource, cache.releaseSource(source.lease));
            } else if (!std.mem.eql(u8, case.result, "cancel")) {
                const err: anyerror = if (std.mem.eql(u8, case.result, "Busy")) error.Busy else if (std.mem.eql(u8, case.result, "CommitFault")) error.CommitFault else error.InvalidState;
                try t.expectError(err, cache.finishPreparation(source.lease, commit.callback()));
            }
        }
        try t.expectEqual(case.callbacks, commit.calls);
        for (case.nodes, 0..) |node, i| {
            try t.expectEqualSlices(u32, node.tokens, r.store.entryTokens(i));
            try t.expectEqual(node.first, r.own[i]);
            try t.expectEqual(node.refs, r.sources[i].refs);
            try t.expectEqual(node.active, r.sources[i].active);
            const list = r.store.entryPages(i);
            try t.expectEqual(node.host.len, list.len);
            for (list, node.host, node.pins, 0..) |q, is_host, pins, k| {
                try t.expectEqual(is_host, q & pages.host_flag != 0);
                if (!is_host) try t.expectEqual(pins, pool.pins[q]);
                const bytes = if (is_host) host[(q & ~pages.host_flag) * P ..][0..P] else dev[q * P ..][0..P];
                for (0..@min(P, node.tokens.len - k * P)) |o| try t.expectEqual(posValue(node.tokens[0 .. k * P + o + 1]), bytes[o]);
            }
        }
        try t.expectEqual(@as(u32, 0), r.store.entries[case.nodes.len].len);
        if (case.input.hit) {
            var actual: [V]f32 = undefined;
            var want: [V]f32 = undefined;
            sys.readLogits(1, &actual);
            soloLogits(case.input.nodes[case.input.selected], &want);
            try t.expectEqualSlices(f32, &want, &actual);
            try pool.release(1);
        }
        sys.checkCache();
        try t.expectEqual(@as(u32, 0), sys.violations);
        if (g) |generation| if (pool.preparation.count != 0) {
            try pool.abortPreparation(generation);
            try cache.releaseSource(source.lease);
        };
        if (protected) |lease| try cache.releaseSource(lease);
        while (cache.evict(sys.device()) or cache.evictHost(sys.device())) {}
        try t.expectEqual(@as(u32, 64), pool.freeCount());
        try t.expectEqual(@as(u32, 32), pool.hostFree());
    }
}
