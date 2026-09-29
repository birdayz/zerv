//! Bounded asynchronous scratch-file I/O through borrowed RAM staging slots.
//! Linux only; no GPU or cache-policy dependency. See docs/specs/nvme-store.md.
const std = @import("std");
const linux = std.os.linux;

pub const max_depth = 32;
pub const Direction = enum { read, write };
pub const Ticket = struct { slot: u32, generation: u64 };
pub const Alignment = struct {
    memory: u32,
    offset: u32,

    fn valid(self: Alignment) bool {
        return self.memory != 0 and self.offset != 0 and std.math.isPowerOfTwo(self.memory) and std.math.isPowerOfTwo(self.offset);
    }
};
pub const Options = struct {
    dir_fd: i32 = linux.AT.FDCWD,
    /// New basename in dir_fd; exclusively created then unlinked, never reopened.
    name: [:0]const u8,
    file_bytes: u64,
    slot_bytes: usize,
    /// Filesystem contract supplied by deployment when STATX_DIOALIGN is unavailable.
    /// May strengthen, but never weaken, requirements reported by the filesystem.
    alignment: ?Alignment = null,
};

pub const Completion = struct {
    expected: u32,
    /// Kernel CQE result: bytes transferred or negative errno. Shorts are failures.
    result: i32,
    pub fn exact(self: Completion) bool {
        return self.result >= 0 and @as(u32, @intCast(self.result)) == self.expected;
    }
};

/// Resolve the deployment contract without opening a file. Reported unsupported I/O
/// cannot be overridden, and configuration may only strengthen known requirements.
pub fn resolveAlignment(reported: ?Alignment, configured: ?Alignment) !Alignment {
    if (reported) |r| if (!r.valid()) return error.DirectIoUnsupported;
    if (configured) |c| {
        if (!c.valid()) return error.InvalidAlignment;
        if (reported) |r| if (c.memory % r.memory != 0 or c.offset % r.offset != 0) return error.InvalidAlignment;
        return c;
    }
    return reported orelse error.DirectIoUnsupported;
}

const State = enum(u8) { free, held, queued, active, done };
const Slot = struct {
    state: std.atomic.Value(State) = .init(.free),
    generation: u64 = 0,
    direction: Direction = .read,
    offset: u64 = 0,
    length: u32 = 0,
    result: i32 = 0,
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    /// Owned scratch descriptor. Do not close, truncate or perform concurrent I/O on it.
    fd: i32,
    memory: []u8,
    slot_bytes: usize,
    file_bytes: u64,
    offset_alignment: u32,
    slots: []Slot,
    ring: linux.IoUring,
    thread: std.Thread = undefined,
    epoch: std.atomic.Value(u32) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),

    /// The memory and returned owner have stable addresses until destroy. Caller owns
    /// memory, imports it into Vulkan separately, and serializes all public API calls.
    pub fn create(a: std.mem.Allocator, options: Options, memory: []u8) !*Store {
        const chunk = options.slot_bytes;
        if (options.name.len == 0 or std.mem.eql(u8, options.name, ".") or std.mem.eql(u8, options.name, "..") or
            std.mem.indexOfScalar(u8, options.name, '/') != null or std.mem.indexOfScalar(u8, options.name, 0) != null or
            chunk < 4096 or chunk > 64 << 20 or chunk % 4096 != 0 or memory.len == 0 or memory.len % chunk != 0 or
            memory.len / chunk > max_depth or @intFromPtr(memory.ptr) % 4096 != 0 or
            options.file_bytes == 0 or options.file_bytes > 1 << 40 or options.file_bytes % 4096 != 0) return error.InvalidOptions;
        if (options.alignment) |alignment| if (!alignment.valid()) return error.InvalidAlignment;
        const rc = linux.openat(options.dir_fd, options.name, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true, .DIRECT = true }, 0o600);
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .EXIST => return error.FileExists,
            else => return error.OpenFailed,
        }
        const fd: i32 = @intCast(rc);
        errdefer _ = linux.close(fd);
        if (linux.errno(linux.unlinkat(options.dir_fd, options.name, 0)) != .SUCCESS) return error.UnlinkFailed;
        var stat: linux.Statx = undefined;
        if (linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .DIOALIGN = true }, &stat)) != .SUCCESS) return error.DirectIoUnsupported;
        const reported: ?Alignment = if (stat.mask.DIOALIGN) .{ .memory = stat.dio_mem_align, .offset = stat.dio_offset_align } else null;
        const alignment = try resolveAlignment(reported, options.alignment);
        if (@intFromPtr(memory.ptr) % alignment.memory != 0 or chunk % alignment.memory != 0 or
            chunk % alignment.offset != 0 or options.file_bytes % alignment.offset != 0) return error.InvalidAlignment;
        if (linux.errno(linux.fallocate(fd, 0, 0, @intCast(options.file_bytes))) != .SUCCESS) return error.PreallocateFailed;
        const self = try a.create(Store);
        errdefer a.destroy(self);
        const slots = try a.alloc(Slot, memory.len / chunk);
        errdefer a.free(slots);
        for (slots) |*s| s.* = .{};
        var ring = try linux.IoUring.init(64, 0);
        errdefer ring.deinit();
        self.* = .{ .allocator = a, .fd = fd, .memory = memory, .slot_bytes = chunk, .file_bytes = options.file_bytes, .offset_alignment = alignment.offset, .slots = slots, .ring = ring };
        self.thread = try std.Thread.spawn(.{}, worker, .{self});
        return self;
    }

    /// Reusable tickets only. Used for the combined optional-I/O allowance.
    pub fn freeSlots(self: *const Store) u32 {
        var count: u32 = 0;
        for (self.slots) |*s| count += @intFromBool(s.state.load(.acquire) == .free and s.generation != std.math.maxInt(u64));
        return count;
    }

    pub fn acquire(self: *Store) !Ticket {
        for (self.slots, 0..) |*s, i| {
            if (s.state.load(.acquire) != .free) continue;
            if (s.generation == std.math.maxInt(u64)) continue; // never let an old ticket alias
            s.generation += 1;
            s.state.store(.held, .release);
            return .{ .slot = @intCast(i), .generation = s.generation };
        }
        return error.QueueFull;
    }

    fn slot(self: *Store, ticket: Ticket) !*Slot {
        if (ticket.slot >= self.slots.len) return error.InvalidTicket;
        const s = &self.slots[ticket.slot];
        if (s.generation != ticket.generation or s.state.load(.acquire) == .free) return error.InvalidTicket;
        return s;
    }

    fn span(self: *Store, index: usize) []u8 {
        return self.memory[index * self.slot_bytes ..][0..self.slot_bytes];
    }

    /// Borrow only while held or done; do not retain CPU/GPU access across submit.
    pub fn buffer(self: *Store, ticket: Ticket) ![]u8 {
        const s = try self.slot(ticket);
        return switch (s.state.load(.acquire)) {
            .held, .done => self.span(ticket.slot),
            else => error.ResourceInUse,
        };
    }

    pub fn submit(self: *Store, ticket: Ticket, direction: Direction, offset: u64, length: usize) !void {
        const s = try self.slot(ticket);
        if (s.state.load(.acquire) != .held) return error.ResourceInUse;
        if (length == 0 or length > self.slot_bytes or offset > self.file_bytes or length > self.file_bytes - offset or
            offset % self.offset_alignment != 0 or length % self.offset_alignment != 0) return error.InvalidRange;
        for (self.slots) |*other| {
            if (other == s) continue;
            switch (other.state.load(.acquire)) {
                .queued, .active, .done => if ((direction == .write or other.direction == .write) and
                    offset < other.offset + other.length and other.offset < offset + length) return error.RangeInUse,
                else => {},
            }
        }
        s.direction = direction;
        s.offset = offset;
        s.length = @intCast(length);
        s.state.store(.queued, .release);
        self.wake();
    }

    pub fn poll(self: *Store, ticket: Ticket) !?Completion {
        const s = try self.slot(ticket);
        return switch (s.state.load(.acquire)) {
            .queued, .active => null,
            .done => .{ .expected = s.length, .result = s.result },
            else => error.InvalidState,
        };
    }

    /// Cancellation of submitted work is drain-then-discard, never early buffer reuse.
    pub fn release(self: *Store, ticket: Ticket) !void {
        const s = try self.slot(ticket);
        switch (s.state.load(.acquire)) {
            .held, .done => s.state.store(.free, .release),
            else => return error.ResourceInUse,
        }
    }

    pub fn destroy(self: *Store) !void {
        for (self.slots) |*s| if (s.state.load(.acquire) != .free) return error.ResourceInUse;
        self.stopping.store(true, .release);
        self.wake();
        self.thread.join();
        self.ring.deinit();
        _ = linux.close(self.fd);
        const a = self.allocator;
        a.free(self.slots);
        a.destroy(self);
    }

    fn wake(self: *Store) void {
        _ = self.epoch.fetchAdd(1, .release);
        _ = linux.futex_3arg(&self.epoch.raw, .{ .cmd = .WAKE, .private = true }, 1);
    }

    fn fatal(e: anyerror) noreturn {
        // Unknown kernel ownership: never return/unwind to code that could free staging.
        std.debug.panic("scratch io_uring lost ownership: {}", .{e});
    }

    fn worker(self: *Store) void {
        while (true) {
            const epoch = self.epoch.load(.acquire);
            var count: u32 = 0;
            var outstanding: u32 = 0;
            for (self.slots, 0..) |*s, i| {
                if (s.state.load(.acquire) != .queued) continue;
                s.state.store(.active, .release);
                const bytes = self.span(i)[0..s.length];
                switch (s.direction) {
                    .read => _ = self.ring.read(i, self.fd, .{ .buffer = bytes }, s.offset) catch |e| fatal(e),
                    .write => _ = self.ring.write(i, self.fd, bytes, s.offset) catch |e| fatal(e),
                }
                count += 1;
                outstanding |= @as(u32, 1) << @intCast(i);
            }
            if (count == 0) {
                if (self.stopping.load(.acquire)) return;
                switch (linux.errno(linux.futex_4arg(&self.epoch.raw, .{ .cmd = .WAIT, .private = true }, epoch, null))) {
                    .SUCCESS, .AGAIN, .INTR => {},
                    else => fatal(error.WorkerWaitFailed),
                }
                continue;
            }
            var submitted: u32 = 0;
            while (submitted < count) {
                const n = self.ring.submit() catch |e| switch (e) {
                    error.SignalInterrupt => continue,
                    else => fatal(e),
                };
                if (n == 0 or n > count - submitted) fatal(error.SubmissionProgress);
                submitted += n;
            }
            while (outstanding != 0) {
                const cqe = self.ring.copy_cqe() catch |e| switch (e) {
                    error.SignalInterrupt => continue,
                    else => fatal(e),
                };
                if (cqe.user_data >= self.slots.len) fatal(error.InvalidCompletion);
                const bit = @as(u32, 1) << @intCast(cqe.user_data);
                if (outstanding & bit == 0) fatal(error.DuplicateCompletion);
                outstanding &= ~bit;
                const s = &self.slots[@intCast(cqe.user_data)];
                s.result = cqe.res;
                s.state.store(.done, .release);
            }
        }
    }
};
