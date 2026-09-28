//! Immutable snapshot residency, separate from checkpoint identity and physical I/O.
//! One serialized caller. See docs/specs/snapshot-residency.md for completion ownership.
const std = @import("std");

pub const Handle = struct { index: u32, generation: u64 };
pub const Phase = enum { free, saving, resident, writing, disk, reading };
pub const Direction = enum { read, write };
pub const Entry = struct {
    generation: u64 = 0,
    serial: u64 = 0,
    phase: Phase = .free,
    hot: ?u32 = null,
    disk: ?u32 = null,
    leases: u32 = 0,
};
pub const Transfer = struct {
    handle: Handle,
    serial: u64,
    hot: u32,
    disk: u32,
    direction: Direction,
};
pub const Error = error{ InvalidOptions, InvalidHandle, InvalidState, InvalidTransfer, Busy, NoEntry, NoHotSlot, NoDiskSlot, SerialExhausted, LeaseOverflow };

pub const Table = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,
    hot_owners: []?u32,
    disk_owners: []?u32,

    pub fn init(a: std.mem.Allocator, count: u32, hot: u32, disk: u32) (Error || std.mem.Allocator.Error)!Table {
        if (count == 0 or count > std.math.maxInt(u16) or hot == 0 or hot > count or disk > count) return error.InvalidOptions;
        const entries = try a.alloc(Entry, count);
        errdefer a.free(entries);
        const hot_owners = try a.alloc(?u32, hot);
        errdefer a.free(hot_owners);
        const disk_owners = try a.alloc(?u32, disk);
        @memset(entries, .{});
        @memset(hot_owners, null);
        @memset(disk_owners, null);
        return .{ .allocator = a, .entries = entries, .hot_owners = hot_owners, .disk_owners = disk_owners };
    }

    /// No bytes are owned here. Caller must drain I/O/GPU work and resident leases first.
    pub fn deinit(self: *Table) Error!void {
        for (self.entries) |e| if (pending(e.phase) or e.leases != 0) return error.Busy;
        self.allocator.free(self.entries);
        self.allocator.free(self.hot_owners);
        self.allocator.free(self.disk_owners);
        self.* = undefined;
    }

    fn pending(phase: Phase) bool {
        return phase == .saving or phase == .writing or phase == .reading;
    }
    fn firstFree(owners: []const ?u32) ?u32 {
        for (owners, 0..) |owner, i| if (owner == null) return @intCast(i);
        return null;
    }
    fn entry(self: *const Table, h: Handle) Error!*Entry {
        if (h.index >= self.entries.len) return error.InvalidHandle;
        const e = &self.entries[h.index];
        if (e.phase == .free or e.generation != h.generation) return error.InvalidHandle;
        return e;
    }
    pub fn inspect(self: *const Table, h: Handle) Error!Entry {
        return (try self.entry(h)).*;
    }

    pub fn reserve(self: *Table) Error!Handle {
        for (self.entries, 0..) |*e, i| {
            if (e.phase != .free or e.generation == std.math.maxInt(u64)) continue;
            const hot = firstFree(self.hot_owners) orelse return error.NoHotSlot;
            const h: Handle = .{ .index = @intCast(i), .generation = e.generation + 1 };
            e.* = .{ .generation = h.generation, .phase = .saving, .hot = hot };
            self.hot_owners[hot] = h.index;
            return h;
        }
        return error.NoEntry;
    }

    pub fn saved(self: *Table, h: Handle, success: bool) Error!void {
        const e = try self.entry(h);
        if (e.phase != .saving) return error.InvalidState;
        if (success) {
            e.phase = .resident;
        } else {
            self.hot_owners[e.hot.?] = null;
            e.* = .{ .generation = e.generation };
        }
    }

    /// Read-only borrow of immutable bytes. Release only after the consumer's GPU access.
    pub fn acquireResident(self: *Table, h: Handle) Error!u32 {
        const e = try self.entry(h);
        if (e.phase != .resident) return error.Busy;
        if (e.leases == std.math.maxInt(u32)) return error.LeaseOverflow;
        e.leases += 1;
        return e.hot.?;
    }
    pub fn releaseResident(self: *Table, h: Handle) Error!void {
        const e = try self.entry(h);
        if (e.phase != .resident or e.leases == 0) return error.InvalidState;
        e.leases -= 1;
    }

    /// Null: already backed, demotion completed without I/O. Otherwise both slots stay held.
    pub fn spill(self: *Table, h: Handle) Error!?Transfer {
        const e = try self.entry(h);
        if (e.phase != .resident or e.leases != 0) return error.Busy;
        if (e.disk != null) {
            self.hot_owners[e.hot.?] = null;
            e.hot = null;
            e.phase = .disk;
            return null;
        }
        const disk = firstFree(self.disk_owners) orelse return error.NoDiskSlot;
        if (e.serial == std.math.maxInt(u64)) return error.SerialExhausted;
        self.disk_owners[disk] = h.index;
        e.disk = disk;
        e.serial += 1;
        e.phase = .writing;
        return .{ .handle = h, .serial = e.serial, .hot = e.hot.?, .disk = disk, .direction = .write };
    }

    /// Null: already resident. A pending read is not resident/usable until complete(true).
    pub fn restore(self: *Table, h: Handle) Error!?Transfer {
        const e = try self.entry(h);
        if (e.phase == .resident) return null;
        if (e.phase != .disk) return error.Busy;
        const hot = firstFree(self.hot_owners) orelse return error.NoHotSlot;
        if (e.serial == std.math.maxInt(u64)) return error.SerialExhausted;
        self.hot_owners[hot] = h.index;
        e.hot = hot;
        e.serial += 1;
        e.phase = .reading;
        return .{ .handle = h, .serial = e.serial, .hot = hot, .disk = e.disk.?, .direction = .read };
    }

    /// ALL transfer chunks/GPU accesses must have ended, even for cancellation/failure.
    /// Success requires exact byte counts and the adapter's integrity checks.
    pub fn complete(self: *Table, transfer: Transfer, success: bool) Error!void {
        const e = try self.entry(transfer.handle);
        const write = transfer.direction == .write;
        if (e.phase != (if (write) Phase.writing else Phase.reading) or
            e.serial != transfer.serial or e.hot != transfer.hot or e.disk != transfer.disk) return error.InvalidTransfer;
        if (write) {
            if (success) {
                self.hot_owners[e.hot.?] = null;
                e.hot = null;
                e.phase = .disk;
            } else {
                self.disk_owners[e.disk.?] = null;
                e.disk = null;
                e.phase = .resident;
            }
        } else if (success) {
            e.phase = .resident;
        } else {
            self.hot_owners[e.hot.?] = null;
            e.hot = null;
            e.phase = .disk;
        }
    }

    pub fn discardBacking(self: *Table, h: Handle) Error!void {
        const e = try self.entry(h);
        if (e.phase != .resident) return error.Busy;
        if (e.disk) |d| self.disk_owners[d] = null;
        e.disk = null;
    }
    pub fn drop(self: *Table, h: Handle) Error!void {
        const e = try self.entry(h);
        if (pending(e.phase) or e.leases != 0) return error.Busy;
        if (e.hot) |hot| self.hot_owners[hot] = null;
        if (e.disk) |disk| self.disk_owners[disk] = null;
        e.* = .{ .generation = e.generation };
    }
};
