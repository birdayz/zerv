//! Bounded disk-only read-ahead. No request tokens, page maps or GPU ownership.
//! The active archive job pins its record until exact handoff or disk drain.
const archive = @import("archive.zig");
pub const Key = struct {
    slot: u32,
    order: u64,
    pub fn eql(a: Key, b: Key) bool {
        return a.slot == b.slot and a.order == b.order;
    }
};
pub const Owner = struct {
    held: ?struct { key: Key, record: u32, window: u32, canceled: bool = false, done_mask: u64 = 0, completed: u64 = 0 } = null,
    starts: u64 = 0,
    handoffs: u64 = 0,
    cancellations: u64 = 0,
    submitted_bytes: u64 = 0,
    completed_bytes: u64 = 0,
    discarded_bytes: u64 = 0,

    pub fn start(self: *Owner, a: *archive.Archive, key: Key, record: u32, window: u32) !void {
        if (self.held != null) return error.Busy;
        if (window != 1 and window != 2) return error.InvalidOptions;
        try a.startRead(key.slot, record);
        self.held = .{ .key = key, .record = record, .window = window };
        self.starts += 1;
    }
    pub fn cancel(self: *Owner) void {
        if (self.held) |*h| h.canceled = true;
    }
    pub fn matches(self: *const Owner, key: Key) bool {
        const h = self.held orelse return false;
        return h.key.eql(key) and !h.canceled;
    }
    pub fn occupancy(self: *const Owner, a: *const archive.Archive) u32 {
        const h = self.held orelse return 0;
        return a.jobs[h.key.slot].pending;
    }
    /// Never invokes device callbacks. Cancellation always drains, even with no issue credit.
    pub fn poll(self: *Owner, a: *archive.Archive, key: ?Key, allow_issue: bool) !archive.Progress {
        const h = if (self.held) |*h| h else return .{ .done = true };
        if (key == null or !h.key.eql(key.?)) h.canceled = true;
        for (a.pending) |pending| if (pending) |ticket| {
            if (ticket.slot != h.key.slot) continue;
            const bit = @as(u64, 1) << @intCast(ticket.ticket.slot);
            if (h.done_mask & bit != 0) continue;
            if (a.store.poll(ticket.ticket) catch @panic("read-ahead lost its staging ticket")) |completion| {
                h.done_mask |= bit;
                if (completion.exact()) {
                    h.completed += a.store.slot_bytes;
                    self.completed_bytes += a.store.slot_bytes;
                }
            }
        };
        const before_next = a.jobs[h.key.slot].next;
        const slot = h.key.slot;
        const p = a.advanceWith(.{ .ctx = self, .start = noStart, .poll = noPoll }, h.key.slot, h.canceled, .{
            .allow_start = allow_issue and a.store.freeSlots() > 2,
            .max_pending = h.window,
            .allow_upload = false,
        }) catch |e| {
            // Archive errors are terminal only after the exact job's owners drain.
            if (a.active(h.key.slot)) @panic("read-ahead returned with unresolved ownership");
            self.discarded_bytes += h.completed;
            self.held = null;
            if (e == error.Canceled) {
                self.cancellations += 1;
                return .{ .done = true, .progressed = true };
            }
            return e;
        };
        self.submitted_bytes += @as(u64, a.jobs[slot].next - before_next) * a.store.slot_bytes;
        if (p.done) @panic("staging-only read completed without handoff");
        return p;
    }
    /// Caller validates/populates private destination maps before this non-yielding handoff.
    pub fn take(self: *Owner, a: *const archive.Archive, key: Key) !u32 {
        const h = self.held orelse return error.InvalidReadAhead;
        if (!self.matches(key) or !a.active(key.slot) or a.jobs[key.slot].record != h.record or a.jobs[key.slot].writing) return error.InvalidReadAhead;
        self.held = null;
        self.handoffs += 1;
        return h.record;
    }
    fn noStart(_: *anyopaque, _: u32, _: u64, _: []u8, _: bool) !void {
        @panic("read-ahead attempted GPU upload before handoff");
    }
    fn noPoll(_: *anyopaque) !bool {
        @panic("read-ahead attempted GPU acknowledgment before handoff");
    }
};
