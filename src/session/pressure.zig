//! Pure optional-cache preservation decisions. No device, storage or borrowed tokens.
//! Contract: docs/specs/tiering-pressure.md. Call only with eligible unleased leaves.
const cache = @import("kvcache.zig");
pub const Usage = struct {
    slots: u32,
    free_slots: u32,
    slot_headroom: u32,
    host_pages: u32,
    free_host_pages: u32,
    host_headroom: u32,

    pub fn validate(self: Usage) error{InvalidCapacity}!void {
        if (self.slots == 0 or self.free_slots > self.slots or self.slot_headroom >= self.slots or
            self.free_host_pages > self.host_pages or
            (if (self.host_pages == 0) self.host_headroom != 0 else self.host_headroom >= self.host_pages))
            return error.InvalidCapacity;
    }
    pub fn pressured(self: Usage) bool {
        return self.free_slots <= self.slot_headroom or (self.host_pages != 0 and self.free_host_pages <= self.host_headroom);
    }
};
pub const Candidate = struct { handle: cache.Handle, used: u64, has_host: bool, backed: bool, fits: bool };
pub const Decision = struct { handle: cache.Handle, action: enum { preserve, discard } };

/// Streaming selection avoids an allocation or a fixed-size array in the caller.
/// Usage must be validated before starting the scan; candidates come from one cache epoch.
pub const Select = struct {
    usage: Usage,
    best: ?Candidate = null,

    pub fn consider(self: *Select, c: Candidate) void {
        if (!self.usage.pressured() or (!c.backed and !c.fits)) return;
        if (self.usage.free_slots > self.usage.slot_headroom and !c.has_host) return;
        if (self.best) |b| {
            if (c.used > b.used or (c.used == b.used and c.handle.index >= b.handle.index)) return;
        }
        self.best = c;
    }
    pub fn decision(self: Select) ?Decision {
        const c = self.best orelse return null;
        return .{ .handle = c.handle, .action = if (c.backed) .discard else .preserve };
    }
};
