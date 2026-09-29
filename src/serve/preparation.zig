//! One leased GPU→host preparation. Scheduler-thread owned; no per-job allocation.
const std = @import("std");
const gpu = @import("gpu");
const model = @import("model");
const cache = @import("session").kvcache;

pub const Options = struct { target: u32, window: u32 = 1, host_headroom: u32 = 0 };
pub const Progress = struct { pending: bool = false, progressed: bool = false, reclaimed: bool = false };
pub const Counters = struct { submitted: u64 = 0, copied: u64 = 0, committed: u64 = 0, freed: u64 = 0, aborted: u64 = 0, hold_ns: u64 = 0 };
pub const Preparation = struct {
    m: *model.Model,
    c: cache.Cache,
    io: std.Io,
    commands: gpu.Commands,
    options: Options,
    held: ?struct { source: cache.Source, generation: u64, first: u32, pages: u32 } = null,
    ready: bool = false,
    canceled: bool = false,
    submitted_ns: i96 = 0,
    failed_generation: [model.max_snapshots]u64 = @splat(0),
    counters: Counters = .{},

    pub fn init(m: *model.Model, c: cache.Cache, io: std.Io, options: Options) !Preparation {
        const capacity = try c.sourceCapacity();
        if (capacity.slots > model.max_snapshots or capacity.slots != m.snapshot_slots or options.target == 0 or options.target > m.pool.pages or m.swap_pages == 0 or options.host_headroom > m.swap_pages or (options.window != 1 and options.window != 2 and options.window != 4)) return error.InvalidOptions;
        return .{ .m = m, .c = c, .io = io, .commands = try gpu.Commands.init(m.device), .options = options };
    }
    pub fn deinit(self: *Preparation) !void {
        if (self.held != null) return error.ResourceInUse;
        try self.commands.deinit();
    }
    fn release(self: *Preparation) void {
        self.counters.hold_ns += @intCast(@max(0, std.Io.Clock.awake.now(self.io).nanoseconds - self.submitted_ns));
        self.held = null;
        self.ready = false;
        self.canceled = false;
    }
    fn abort(self: *Preparation) !void {
        const held = self.held orelse return error.InvalidState;
        try self.m.pool.abortPreparation(held.generation);
        try self.c.releaseSource(held.source.lease);
        self.failed_generation[held.source.lease.handle.index] = held.source.lease.handle.generation;
        self.counters.aborted += 1;
        self.release();
    }
    fn commit(ctx: *anyopaque, pages: []u32, first: u32) !cache.PreparedResult {
        const self: *Preparation = @ptrCast(@alignCast(ctx));
        const held = self.held orelse return error.InvalidState;
        if (!self.ready or first != held.first) return error.InvalidState;
        const result = try self.m.pool.commitPreparation(held.generation, pages);
        return .{ .copied = result.copied, .freed = result.freed };
    }
    /// In a packed chunk, acknowledge only. The held source stays protected until publication.
    pub fn poll(self: *Preparation, cancel: bool, publish: bool) !Progress {
        const held = self.held orelse return .{};
        self.canceled = self.canceled or cancel;
        var progressed = false;
        if (!self.ready) {
            var done = self.commands.poll() catch |e| {
                if (self.commands.state == .pending or self.m.device.lost) @panic("preparation GPU DMA ownership unresolved");
                try self.m.pool.ackPreparation(held.generation);
                self.ready = true;
                self.canceled = true;
                if (publish) try self.abort();
                return e;
            };
            if (!done and std.Io.Clock.awake.now(self.io).nanoseconds - self.submitted_ns >= self.m.options.timeout_ns) {
                done = self.commands.poll() catch @panic("preparation GPU DMA deadline unresolved");
                if (!done) @panic("preparation GPU DMA deadline exceeded; ownership unresolved");
            }
            if (!done) return .{ .pending = true };
            try self.m.pool.ackPreparation(held.generation);
            self.ready = true;
            self.counters.copied += held.pages;
            progressed = true;
        }
        if (!publish) return .{ .pending = true, .progressed = progressed };
        if (self.canceled) {
            try self.abort();
        } else {
            const result = self.c.finishPreparation(held.source.lease, .{ .ctx = self, .apply = commit }) catch {
                try self.abort();
                return .{ .progressed = true, .reclaimed = true };
            };
            self.counters.committed += result.copied;
            self.counters.freed += result.freed;
            self.release();
        }
        return .{ .progressed = true, .reclaimed = true };
    }
    /// Cold unleased segments, bounded by snapshot capacity; pool eligibility is authoritative.
    pub fn start(self: *Preparation) !bool {
        if (self.held != null) return error.InvalidState;
        if (self.m.freePages() >= self.options.target or self.m.hostFreePages() <= self.options.host_headroom) return false;
        const capacity = try self.c.sourceCapacity();
        var tried: [model.max_snapshots]bool = @splat(false);
        for (0..capacity.slots) |_| {
            var best: ?cache.PreparationCandidate = null;
            for (0..capacity.slots) |i| {
                if (tried[i]) continue;
                const candidate = self.c.preparationCandidate(@intCast(i)) orelse continue;
                if (self.failed_generation[i] == candidate.handle.generation) continue;
                if (best == null or candidate.used < best.?.used) best = candidate;
            }
            const candidate = best orelse return false;
            tried[candidate.handle.index] = true;
            const source = try self.c.acquireSource(candidate.handle);
            const generation = self.m.pool.prepare(source.pages, candidate.first, self.options.window, self.options.host_headroom) catch |e| {
                try self.c.releaseSource(source.lease);
                return e;
            };
            if (generation == null) {
                try self.c.releaseSource(source.lease);
                continue;
            }
            const pages: u32 = @intCast((try self.m.pool.preparedMoves(generation.?)).len);
            self.held = .{ .source = source, .generation = generation.?, .first = candidate.first, .pages = pages };
            self.submitted_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
            self.m.prepareSubmit(&self.commands, generation.?) catch |e| {
                if (self.commands.state == .pending or self.m.device.lost) @panic("preparation submission ownership unresolved");
                try self.m.pool.ackPreparation(generation.?);
                try self.abort();
                return e;
            };
            self.counters.submitted += pages;
            return true;
        }
        return false;
    }
};
