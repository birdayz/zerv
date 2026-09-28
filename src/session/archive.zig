//! Immutable write-through prefix archive. No model math/GPU ownership.
//! Spec: docs/specs/disk-prefix-cache.md. One scheduler-side caller.
const std = @import("std");
const storage = @import("storage");
const Sha256 = std.crypto.hash.sha2.Sha256;
const none = std.math.maxInt(u32);

pub const Device = struct {
    ctx: *anyopaque,
    /// Submit precisely bytes.len logical bytes; borrow the span until poll acknowledges.
    /// One operation at a time. Errors must leave no device access; otherwise fail-stop.
    start: *const fn (*anyopaque, slot: u32, offset: u64, bytes: []u8, importing: bool) anyerror!void,
    /// Nonblocking: false retains ownership; true or a terminal error releases it.
    poll: *const fn (*anyopaque) anyerror!bool,
};
pub const Options = struct { records: u32, context: u32, slots: u32, max_bytes: u64 };
pub const Stats = struct {
    writes: u64 = 0,
    reads: u64 = 0,
    write_bytes: u64 = 0,
    read_bytes: u64 = 0,
    evictions: u64 = 0,
    skips: u64 = 0,
    failures: u64 = 0,
    cancellations: u64 = 0,
    device_starts: u64 = 0,
    device_polls: u64 = 0,
    device_pending_polls: u64 = 0,
};
pub const Entry = struct {
    phase: enum { free, writing, ready, bad } = .free,
    len: u32 = 0,
    bytes: u64 = 0,
    chunks: u32 = 0,
    readers: u32 = 0,
    used: u64 = 0,
};
const Job = struct {
    active: bool = false,
    record: u32 = 0,
    writing: bool = false,
    next: u32 = 0,
    pending: u32 = 0,
    failure: ?anyerror = null,
};
const Pending = struct { ticket: storage.Ticket, slot: u32, chunk: u32 };
pub const Progress = struct { done: bool = false, progressed: bool = false, position: u32 = 0 };

pub const Archive = struct {
    allocator: std.mem.Allocator,
    store: *storage.Store,
    options: Options,
    max_chunks: usize,
    entries: []Entry,
    tokens: []u32,
    blocks: []u32,
    digests: [][32]u8,
    owners: []u32,
    jobs: []Job,
    pending: []?Pending,
    /// A held (capture) or disk-done (upload) ticket still borrowed by the device.
    device_pending: ?Pending = null,
    free_chunks: u32,
    tick: u64 = 0,
    stats: Stats = .{},

    pub fn init(a: std.mem.Allocator, store: *storage.Store, o: Options) !Archive {
        if (o.records == 0 or o.records > 4096 or o.context == 0 or o.slots == 0 or o.slots > 64 or o.max_bytes == 0 or
            store.file_bytes % store.slot_bytes != 0) return error.InvalidOptions;
        const max_chunks = std.math.divCeil(u64, o.max_bytes, store.slot_bytes) catch return error.InvalidOptions;
        const token_count = std.math.mul(usize, o.records, o.context) catch return error.InvalidOptions;
        const chunk_count = std.math.mul(usize, o.records, std.math.cast(usize, max_chunks) orelse return error.InvalidOptions) catch return error.InvalidOptions;
        const block_count = store.file_bytes / store.slot_bytes;
        if (block_count > std.math.maxInt(u32) or max_chunks > std.math.maxInt(u32)) return error.InvalidOptions;
        var metadata = std.math.mul(u64, token_count, 4) catch return error.InvalidOptions;
        metadata = std.math.add(u64, metadata, std.math.mul(u64, chunk_count, 36) catch return error.InvalidOptions) catch return error.InvalidOptions;
        metadata = std.math.add(u64, metadata, block_count * 4 + @as(u64, o.records) * @sizeOf(Entry) + @as(u64, o.slots) * @sizeOf(Job) + storage.max_depth * @sizeOf(?Pending)) catch return error.InvalidOptions;
        if (metadata > 256 << 20) return error.MetadataBudgetExceeded;
        const entries = try a.alloc(Entry, o.records);
        errdefer a.free(entries);
        const tokens = try a.alloc(u32, token_count);
        errdefer a.free(tokens);
        const blocks = try a.alloc(u32, chunk_count);
        errdefer a.free(blocks);
        const digests = try a.alloc([32]u8, chunk_count);
        errdefer a.free(digests);
        const owners = try a.alloc(u32, @intCast(block_count));
        errdefer a.free(owners);
        const jobs = try a.alloc(Job, o.slots);
        errdefer a.free(jobs);
        const pending = try a.alloc(?Pending, store.memory.len / store.slot_bytes);
        @memset(entries, .{});
        @memset(owners, none);
        @memset(jobs, .{});
        @memset(pending, null);
        return .{ .allocator = a, .store = store, .options = o, .max_chunks = @intCast(max_chunks), .entries = entries, .tokens = tokens, .blocks = blocks, .digests = digests, .owners = owners, .jobs = jobs, .pending = pending, .free_chunks = @intCast(block_count) };
    }
    pub fn deinit(self: *Archive) !void {
        for (self.jobs) |j| if (j.active) return error.Busy;
        if (self.device_pending != null) return error.Busy;
        for (self.pending) |p| if (p != null) return error.Busy;
        const a = self.allocator;
        a.free(self.entries);
        a.free(self.tokens);
        a.free(self.blocks);
        a.free(self.digests);
        a.free(self.owners);
        a.free(self.jobs);
        a.free(self.pending);
        self.* = undefined;
    }
    pub fn prefix(self: *const Archive, i: u32) []const u32 {
        return self.tokens[@as(usize, i) * self.options.context ..][0..self.entries[i].len];
    }
    fn index(self: *const Archive, i: u32, chunk: u32) usize {
        return @as(usize, i) * self.max_chunks + chunk;
    }
    fn touch(self: *Archive, i: u32) void {
        self.tick +|= 1;
        self.entries[i].used = self.tick;
    }
    /// Pure lookup: longest ready prefix leaving at least one token for logits.
    pub fn lookup(self: *const Archive, prompt: []const u32) ?u32 {
        var best: ?u32 = null;
        for (self.entries, 0..) |e, i| {
            if (e.phase != .ready or e.len >= prompt.len or (best != null and e.len <= self.entries[best.?].len)) continue;
            if (std.mem.eql(u32, self.prefix(@intCast(i)), prompt[0..e.len])) best = @intCast(i);
        }
        return best;
    }
    fn freeRecord(self: *Archive, i: u32) void {
        const e = &self.entries[i];
        for (0..e.chunks) |k| self.owners[self.blocks[self.index(i, @intCast(k))]] = none;
        self.free_chunks += e.chunks;
        e.* = .{};
    }
    fn evict(self: *Archive) bool {
        var victim: ?u32 = null;
        for (self.entries, 0..) |e, i| {
            if (e.phase != .ready or e.readers != 0) continue;
            if (victim == null or e.used < self.entries[victim.?].used) victim = @intCast(i);
        }
        const v = victim orelse return false;
        self.freeRecord(v);
        self.stats.evictions += 1;
        return true;
    }
    fn checkSlot(self: *Archive, slot: u32) !void {
        if (slot >= self.jobs.len) return error.InvalidSlot;
        if (self.jobs[slot].active) return error.Busy;
    }
    /// False: duplicate or budget/leased-capacity miss. Caller continues inference.
    pub fn startWrite(self: *Archive, slot: u32, tokens: []const u32, bytes: u64) !bool {
        try self.checkSlot(slot);
        if (tokens.len == 0 or tokens.len > self.options.context or bytes == 0 or bytes > self.options.max_bytes) return error.InvalidOptions;
        for (self.entries, 0..) |e, i| {
            if ((e.phase == .ready or e.phase == .writing) and e.len == tokens.len and std.mem.eql(u32, self.prefix(@intCast(i)), tokens)) return false;
        }
        const chunks: u32 = @intCast(try std.math.divCeil(u64, bytes, self.store.slot_bytes));
        var possible: u64 = self.free_chunks;
        var available_entry = false;
        for (self.entries) |e| {
            if (e.phase == .free) available_entry = true;
            if (e.phase == .ready and e.readers == 0) {
                possible += e.chunks;
                available_entry = true;
            }
        }
        if (possible < chunks or !available_entry) {
            self.stats.skips += 1;
            return false;
        }
        while (self.free_chunks < chunks) if (!self.evict()) return error.InvalidState;
        var chosen: ?u32 = null;
        while (chosen == null) {
            for (self.entries, 0..) |e, i| if (e.phase == .free) {
                chosen = @intCast(i);
                break;
            };
            if (chosen == null and !self.evict()) return error.InvalidState;
        }
        const i = chosen.?;
        var n: u32 = 0;
        for (self.owners, 0..) |*owner, b| {
            if (owner.* != none) continue;
            owner.* = i;
            self.blocks[self.index(i, n)] = @intCast(b);
            n += 1;
            if (n == chunks) break;
        }
        self.free_chunks -= chunks;
        @memcpy(self.tokens[@as(usize, i) * self.options.context ..][0..tokens.len], tokens);
        self.entries[i] = .{ .phase = .writing, .len = @intCast(tokens.len), .bytes = bytes, .chunks = chunks };
        self.touch(i);
        self.jobs[slot] = .{ .active = true, .record = i, .writing = true };
        return true;
    }
    pub fn startRead(self: *Archive, slot: u32, i: u32) !void {
        try self.checkSlot(slot);
        if (i >= self.entries.len or self.entries[i].phase != .ready) return error.InvalidRecord;
        self.entries[i].readers += 1; // bounded by slots <=64
        self.touch(i);
        self.jobs[slot] = .{ .active = true, .record = i };
    }
    pub fn writing(self: *const Archive, slot: u32) bool {
        return self.jobs[slot].writing;
    }
    pub fn active(self: *const Archive, slot: u32) bool {
        return self.jobs[slot].active;
    }
    fn fail(j: *Job, err: anyerror) void {
        if (j.failure == null) j.failure = err;
    }

    fn release(self: *Archive, held: Pending) !void {
        try self.store.release(held.ticket);
        self.jobs[held.slot].pending -= 1;
    }
    fn submitDisk(self: *Archive, held: Pending) !void {
        const j = &self.jobs[held.slot];
        try self.store.submit(held.ticket, if (j.writing) .write else .read, @as(u64, self.blocks[self.index(j.record, held.chunk)]) * self.store.slot_bytes, self.store.slot_bytes);
        self.pending[held.ticket.slot] = held;
    }
    fn startDevice(self: *Archive, dev: Device, held: Pending) !void {
        const j = &self.jobs[held.slot];
        const e = &self.entries[j.record];
        const bytes = try self.store.buffer(held.ticket);
        const offset = @as(u64, held.chunk) * self.store.slot_bytes;
        try dev.start(dev.ctx, held.slot, offset, bytes[0..@intCast(@min(bytes.len, e.bytes - offset))], !j.writing);
        self.device_pending = held;
        self.stats.device_starts += 1;
    }

    /// At most one device poll/start, one disk completion and one acquisition per call.
    /// Never waits. An error is returned only after this job's device AND disk leases drain.
    pub fn advance(self: *Archive, dev: Device, slot: u32, cancel: bool) !Progress {
        if (slot >= self.jobs.len or !self.jobs[slot].active) return error.InvalidSlot;
        const j = &self.jobs[slot];
        const e = &self.entries[j.record];
        if (cancel) fail(j, error.Canceled);
        if (e.phase == .bad) fail(j, error.CorruptRecord);
        var progress = false;
        if (self.device_pending) |held| if (held.slot == slot) {
            self.stats.device_polls += 1;
            const done = dev.poll(dev.ctx) catch |err| done: {
                fail(j, err); // callback contract: terminal, no remaining access
                break :done true;
            };
            if (done) {
                self.device_pending = null;
                if (j.writing and j.failure == null) {
                    const bytes = try self.store.buffer(held.ticket);
                    Sha256.hash(bytes, &self.digests[self.index(j.record, held.chunk)], .{});
                    self.submitDisk(held) catch |err| fail(j, err);
                }
                if (!j.writing or j.failure != null) try self.release(held);
                progress = true;
            } else self.stats.device_pending_polls += 1;
        };
        for (self.pending) |*p| {
            const held = p.* orelse continue;
            if (held.slot != slot) continue;
            // Leave completed reads on disk-owned tickets until an upload can start.
            // Canceled jobs can drain without waiting for another job's device quantum.
            if (!j.writing and j.failure == null and self.device_pending != null) continue;
            const c = (try self.store.poll(held.ticket)) orelse continue;
            if (!c.exact()) fail(j, error.DiskIoFailed) else {
                if (j.writing) self.stats.write_bytes += self.store.slot_bytes else self.stats.read_bytes += self.store.slot_bytes;
                if (!j.writing and j.failure == null) {
                    const bytes = try self.store.buffer(held.ticket);
                    var digest: [32]u8 = undefined;
                    Sha256.hash(bytes, &digest, .{});
                    if (!std.mem.eql(u8, &digest, &self.digests[self.index(j.record, held.chunk)])) {
                        fail(j, error.CorruptRecord);
                    } else self.startDevice(dev, held) catch |err| fail(j, err);
                }
            }
            p.* = null;
            if (j.writing or j.failure != null) try self.release(held);
            progress = true;
            break;
        }
        if (!j.writing and j.failure != null and j.failure.? != error.Canceled) e.phase = .bad;
        if (j.failure == null and j.next < e.chunks and (!j.writing or self.device_pending == null)) {
            const ticket: ?storage.Ticket = self.store.acquire() catch |err| if (err == error.QueueFull) null else return err;
            if (ticket) |t| {
                const held: Pending = .{ .ticket = t, .slot = slot, .chunk = j.next };
                if (j.writing) {
                    @memset(try self.store.buffer(t), 0);
                    self.startDevice(dev, held) catch |err| fail(j, err);
                } else self.submitDisk(held) catch |err| fail(j, err);
                if (j.failure != null) try self.store.release(t) else {
                    j.pending += 1;
                    j.next += 1;
                }
                progress = true;
            }
        }
        if (j.pending != 0 or (j.failure == null and j.next != e.chunks)) return .{ .progressed = progress };
        const failure = j.failure;
        const position = if (j.writing) 0 else e.len;
        if (j.writing) {
            if (failure == null) {
                e.phase = .ready;
                self.stats.writes += 1;
            } else self.freeRecord(j.record);
        } else {
            e.readers -= 1;
            if (failure == null) self.stats.reads += 1;
            if (e.phase == .bad and e.readers == 0) self.freeRecord(j.record);
        }
        j.* = .{};
        if (failure) |err| {
            if (err == error.Canceled) self.stats.cancellations += 1 else self.stats.failures += 1;
            return err;
        }
        return .{ .done = true, .progressed = true, .position = position };
    }
};
