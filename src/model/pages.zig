//! KV page accounting of the shared pool (docs/specs/concurrent.md, "18d.2"–"18d.4"): who
//! holds every pool page (slots, as a bitmask, and checkpoint pins), each held page's
//! logical index in its holders' tables, the host swap store's pages, and the steps of the
//! operations that copy (attach, rebind, swap out, swap in) as plan / commit / abort, so the
//! caller does the device copies in between. Pure bookkeeping: host tests drive it without
//! a device (tests/pages.zig). Page ids are u32, slots < 64.
//!
//! Invariants (`check`): a page is free iff no slot holds it and no pin; a slot's held pages
//! have distinct logical indices 0..mapped (a swapped slot: its resident pages plus its host
//! pages cover 0..swapped); a host page is owned by a swapped slot.
const std = @import("std");

pub const max_pages = 1 << 16;
pub const max_slots = 64;
const no_owner: u8 = 0xff;

pub const Error = error{ InvalidSlot, SlotSwapped, InvalidToken, PageInUse, InvalidState, PagesMissing, PoolExhausted, SwapFull, NoSwap };

pub const Pool = struct {
    pages: u32,
    slots: u32,
    /// Logical pages one sequence may hold at most.
    seq_pages: u32,
    host_pages: u32 = 0,
    mask: [max_pages]u64 = @splat(0),
    pins: [max_pages]u8 = @splat(0),
    logical: [max_pages]u16 = @splat(0),
    host_owner: [max_pages]u8 = @splat(no_owner),
    host_logical: [max_pages]u16 = @splat(0),
    /// The partner of a page during a copy plan: pool page -> host page (swap out), host page
    /// -> pool page (swap in), pool page -> old page (rebind).
    link: [max_pages]u32 = undefined,
    mapped: [max_slots]u32 = @splat(0),
    /// Logical pages of a swapped-out slot (0: not swapped).
    swapped: [max_slots]u32 = @splat(0),

    /// In place (the struct is large).
    pub fn init(self: *Pool, pages: u32, slots: u32, seq_pages: u32) Error!void {
        if (pages == 0 or pages > max_pages or slots == 0 or slots > max_slots or seq_pages == 0 or seq_pages > std.math.maxInt(u16)) return error.InvalidToken;
        self.* = .{ .pages = pages, .slots = slots, .seq_pages = seq_pages };
    }
    pub fn setHost(self: *Pool, host_pages: u32) Error!void {
        if (host_pages > max_pages) return error.InvalidToken;
        self.host_pages = host_pages;
    }

    fn bit(slot: u32) u64 {
        return @as(u64, 1) << @intCast(slot);
    }
    pub fn isFree(self: *const Pool, q: u32) bool {
        return self.mask[q] == 0 and self.pins[q] == 0;
    }
    /// Held by `slot` alone and pinned by nothing.
    pub fn exclusive(self: *const Pool, q: u32, slot: u32) bool {
        return self.mask[q] == bit(slot) and self.pins[q] == 0;
    }
    pub fn freeCount(self: *const Pool) u32 {
        var n: u32 = 0;
        for (0..self.pages) |q| n += @intFromBool(self.isFree(@intCast(q)));
        return n;
    }
    pub fn hostFree(self: *const Pool) u32 {
        var n: u32 = 0;
        for (self.host_owner[0..self.host_pages]) |o| n += @intFromBool(o == no_owner);
        return n;
    }
    pub fn hostOf(self: *const Pool, slot: u32) u32 {
        var n: u32 = 0;
        for (self.host_owner[0..self.host_pages]) |o| n += @intFromBool(o == slot);
        return n;
    }
    fn slotOk(self: *const Pool, slot: u32) Error!void {
        if (slot >= self.slots) return error.InvalidSlot;
    }

    /// Up to `out.len` lowest free pages (not taken).
    pub fn freeList(self: *const Pool, out: []u32) []const u32 {
        var n: usize = 0;
        var q: u32 = 0;
        while (n < out.len and q < self.pages) : (q += 1) {
            if (!self.isFree(q)) continue;
            out[n] = q;
            n += 1;
        }
        return out[0..n];
    }

    /// Can `pages` (free, distinct) be appended to `slot`'s table?
    pub fn checkMap(self: *const Pool, slot: u32, pages: []const u32) Error!void {
        try self.slotOk(slot);
        if (self.swapped[slot] != 0) return error.SlotSwapped;
        if (pages.len > self.seq_pages - self.mapped[slot]) return error.InvalidToken;
        for (pages, 0..) |q, i| {
            if (q >= self.pages or !self.isFree(q)) return error.PageInUse;
            for (pages[0..i]) |o| if (o == q) return error.PageInUse;
        }
    }
    /// After the table write: `slot` holds `pages` (free or shared) at its next logical
    /// indices.
    pub fn commitMap(self: *Pool, slot: u32, pages: []const u32) void {
        for (pages, 0..) |q, k| {
            self.mask[q] |= bit(slot);
            self.logical[q] = @intCast(self.mapped[slot] + k);
        }
        self.mapped[slot] += @intCast(pages.len);
    }

    /// Drop every hold of `slot`, host pages included.
    pub fn release(self: *Pool, slot: u32) Error!void {
        try self.slotOk(slot);
        const b = bit(slot);
        for (self.mask[0..self.pages]) |*m| m.* &= ~b;
        self.mapped[slot] = 0;
        if (self.swapped[slot] != 0) {
            for (self.host_owner[0..self.host_pages]) |*o| if (o.* == slot) {
                o.* = no_owner;
            };
            self.swapped[slot] = 0;
        }
    }

    /// Pin `slot`'s logical pages 0..n into `out` (logical order).
    pub fn pin(self: *Pool, slot: u32, n: u32, out: []u32) Error![]const u32 {
        try self.slotOk(slot);
        if (self.swapped[slot] != 0) return error.SlotSwapped;
        if (n == 0) return error.InvalidToken;
        if (n > self.mapped[slot] or n > out.len) return error.PagesMissing;
        var found: u32 = 0;
        for (self.mask[0..self.pages], 0..) |m, q| {
            if (m & bit(slot) == 0 or self.logical[q] >= n) continue;
            out[self.logical[q]] = @intCast(q);
            found += 1;
        }
        if (found != n) return error.InvalidState;
        for (out[0..n]) |q| if (self.pins[q] == std.math.maxInt(u8)) return error.PageInUse;
        for (out[0..n]) |q| self.pins[q] += 1;
        return out[0..n];
    }
    pub fn unpin(self: *Pool, pages: []const u32) Error!void {
        for (pages) |q| if (q >= self.pages or self.pins[q] == 0) return error.InvalidState;
        for (pages) |q| self.pins[q] -= 1;
    }

    /// Attach plan: `full` held pages shared, `partial` copied into a free page (returned;
    /// not taken until `commitMap`). Nothing changes.
    pub fn attachPlan(self: *const Pool, slot: u32, full: []const u32, partial: ?u32) Error!?u32 {
        try self.slotOk(slot);
        if (self.mapped[slot] != 0 or self.swapped[slot] != 0) return error.InvalidState;
        if (full.len + @intFromBool(partial != null) > self.seq_pages) return error.InvalidToken;
        for (full) |q| if (q >= self.pages or self.isFree(q)) return error.InvalidState;
        if (partial) |q| {
            if (q >= self.pages or self.isFree(q)) return error.InvalidState;
            var f: u32 = 0;
            while (f < self.pages and !self.isFree(f)) f += 1;
            if (f == self.pages) return error.PoolExhausted;
            return f;
        }
        return null;
    }

    /// Rebind plan for `slot`'s logical pages first..first + pages.len: each new page must be
    /// held (not free), at the same logical index, not yet held by the slot unless it is the
    /// same page. Sets `link[new] = old` (read with `rebindOld`). Nothing changes.
    pub fn rebindPlan(self: *Pool, slot: u32, first: u32, pages: []const u32, old: []u32) Error!void {
        try self.slotOk(slot);
        if (self.swapped[slot] != 0) return error.SlotSwapped;
        if (first + pages.len > self.mapped[slot] or old.len < pages.len) return error.PagesMissing;
        var found: usize = 0;
        for (self.mask[0..self.pages], 0..) |m, q| {
            const l = self.logical[q];
            if (m & bit(slot) == 0 or l < first or l >= first + pages.len) continue;
            old[l - first] = @intCast(q);
            found += 1;
        }
        if (found != pages.len) return error.InvalidState;
        for (pages, 0..) |q, i| {
            if (q >= self.pages or self.isFree(q)) return error.InvalidState;
            if (q == old[i]) continue;
            if (self.mask[q] & bit(slot) != 0) return error.PageInUse;
            if (self.logical[q] != first + i) return error.InvalidState;
        }
    }
    /// After the table write: move `slot`'s holds from `old` to `pages`; old pages freed.
    pub fn rebindCommit(self: *Pool, slot: u32, pages: []const u32, old: []const u32) u32 {
        var freed: u32 = 0;
        for (pages, old) |q, o| {
            if (q == o) continue;
            self.mask[o] &= ~bit(slot);
            self.mask[q] |= bit(slot);
            freed += @intFromBool(self.isFree(o));
        }
        return freed;
    }

    /// Swap-out plan: reserve a host page for every page `slot` holds alone (`link[q]` = its
    /// host page; iterate with `swapMoves`). Shared or pinned pages stay resident, held.
    /// Returns the pages to move; `swapOutCommit` or `swapOutAbort` must follow.
    pub fn swapOutPlan(self: *Pool, slot: u32) Error!u32 {
        try self.slotOk(slot);
        if (self.host_pages == 0) return error.NoSwap;
        if (self.swapped[slot] != 0) return error.SlotSwapped;
        if (self.mapped[slot] == 0) return error.PagesMissing;
        var own: u32 = 0;
        for (0..self.pages) |q| own += @intFromBool(self.exclusive(@intCast(q), slot));
        if (self.hostFree() < own) return error.SwapFull;
        var h: u32 = 0;
        for (0..self.pages) |q| {
            if (!self.exclusive(@intCast(q), slot)) continue;
            while (self.host_owner[h] != no_owner) h += 1;
            self.host_owner[h] = @intCast(slot);
            self.host_logical[h] = self.logical[q];
            self.link[q] = h;
            h += 1;
        }
        return own;
    }
    pub fn swapOutAbort(self: *Pool, slot: u32) void {
        for (self.host_owner[0..self.host_pages]) |*o| if (o.* == slot) {
            o.* = no_owner;
        };
    }
    /// After the copies: the moved pages are free, the slot is swapped.
    pub fn swapOutCommit(self: *Pool, slot: u32) void {
        const n = self.mapped[slot];
        for (0..self.pages) |q| {
            if (self.exclusive(@intCast(q), slot)) self.mask[q] = 0;
        }
        self.mapped[slot] = 0;
        self.swapped[slot] = n;
    }

    /// Swap-in plan: false (nothing changes) unless the pool has free pages for the host
    /// pages plus `spare`. Otherwise every host page h of `slot` gets a free pool page
    /// (`link[h]`, taken for the slot), and `words` receives the slot's full table (resident
    /// pages at their logical index). `swapInCommit` or `swapInAbort` must follow.
    pub fn swapInPlan(self: *Pool, slot: u32, spare: u32, words: []u32) Error!bool {
        try self.slotOk(slot);
        const n = self.swapped[slot];
        if (n == 0) return error.InvalidState;
        if (words.len < n) return error.InvalidToken;
        const moved = self.hostOf(slot);
        if (self.freeCount() < @as(u64, moved) + spare) return false;
        var taken: u32 = 0;
        for (self.mask[0..self.pages], 0..) |m, q| {
            if (m & bit(slot) == 0) continue;
            const i = self.logical[q];
            if (i >= n) return error.InvalidState;
            words[i] = @intCast(q);
            taken += 1;
        }
        var p: u32 = 0;
        for (self.host_owner[0..self.host_pages], 0..) |o, h| {
            if (o != slot) continue;
            while (!self.isFree(p)) p += 1;
            const i = self.host_logical[h];
            if (i >= n) return error.InvalidState;
            words[i] = p;
            self.mask[p] |= bit(slot);
            self.link[h] = p;
            taken += 1;
            p += 1;
        }
        if (taken != n) return error.InvalidState;
        return true;
    }
    pub fn swapInAbort(self: *Pool, slot: u32) void {
        for (self.host_owner[0..self.host_pages], 0..) |o, h| {
            if (o == slot) self.mask[self.link[h]] &= ~bit(slot);
        }
    }
    /// After the copies (and before `commitMap(slot, words[0..n])`): the host pages are free,
    /// the slot's holds are dropped for `commitMap` to set them again with the table.
    pub fn swapInCommit(self: *Pool, slot: u32) u32 {
        const n = self.swapped[slot];
        for (self.host_owner[0..self.host_pages]) |*o| if (o.* == slot) {
            o.* = no_owner;
        };
        for (self.mask[0..self.pages]) |*m| m.* &= ~bit(slot);
        self.swapped[slot] = 0;
        return n;
    }

    /// The invariants (tests; O(pages × slots)).
    pub fn check(self: *const Pool) bool {
        for (0..self.slots) |s| {
            const b = bit(@intCast(s));
            const n = if (self.swapped[s] != 0) self.swapped[s] else self.mapped[s];
            if (self.swapped[s] != 0 and self.mapped[s] != 0) return false;
            var seen: [std.math.maxInt(u16) + 1]bool = undefined;
            @memset(seen[0..n], false);
            var count: u32 = 0;
            for (self.mask[0..self.pages], 0..) |m, q| {
                if (m & b == 0) continue;
                const l = self.logical[q];
                if (l >= n or seen[l]) return false;
                seen[l] = true;
                count += 1;
            }
            for (self.host_owner[0..self.host_pages], 0..) |o, h| {
                if (o != s) continue;
                if (self.swapped[s] == 0) return false;
                const l = self.host_logical[h];
                if (l >= n or seen[l]) return false;
                seen[l] = true;
                count += 1;
            }
            if (count != n) return false;
        }
        for (self.mask[0..self.pages]) |m| if (m >> @intCast(self.slots) != 0) return false;
        return true;
    }
};
