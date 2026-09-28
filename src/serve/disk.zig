//! Stable owner of a RAM-staged immutable prefix archive. Scheduler thread only.
//! Lifetime/error contract: docs/specs/disk-prefix-cache.md.
const std = @import("std");
const gpu = @import("gpu");
const model = @import("model");
const storage = @import("storage");
const archive = @import("session").archive;
const kvcache = @import("session").kvcache;
const linux = std.os.linux;
pub const staging_bytes = 8 << 20;
pub const Options = struct { directory: []const u8, bytes: u64, records: u32 = 64, alignment: ?storage.Alignment = null, headroom_slots: ?u32 = null, headroom_mib: ?u64 = null };
pub const Disk = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    submitted_ns: i96 = 0,
    m: *model.Model,
    raw: []u8,
    memory: []u8,
    imported: gpu.Buffer,
    commands: gpu.Commands,
    store: *storage.Store,
    catalog: archive.Archive,
    maps: []u32,
    positions: [65]u32 = @splat(0),
    source: ?struct { cache: kvcache.Cache, view: kvcache.Source } = null,
    source_span: ?struct { offset: u64, bytes: []u8 } = null,
    cpu_copy: bool = false,
    source_cpu_quanta: u64 = 0,
    source_gpu_quanta: u64 = 0,
    headroom_slots: u32,
    headroom_pages: u32,
    failed_generation: [model.max_snapshots]u64 = @splat(0),
    source_cancel_requested: bool = false,
    source_started_ns: i96 = 0,
    source_hold_ns: i96 = 0,
    source_max_hold_ns: i96 = 0,

    pub fn create(a: std.mem.Allocator, io: std.Io, m: *model.Model, o: Options) !*Disk {
        if (m.options.mtp or !m.options.kv_share or m.state_layout.slots < 2 or o.bytes == 0 or o.bytes % (1 << 20) != 0) return error.InvalidOptions;
        if (m.snapshot_slots == 0) return error.InvalidOptions;
        const headroom_slots = o.headroom_slots orelse @as(u32, if (m.snapshot_slots >= 3) 1 else 0);
        const S = m.state_layout;
        const page_bytes = @as(u64, S.caches) * S.piece() * S.kv.bytes();
        const headroom_pages = if (o.headroom_mib) |mib|
            std.math.divCeil(u64, std.math.mul(u64, mib, 1 << 20) catch return error.InvalidOptions, page_bytes) catch return error.InvalidOptions
        else
            @min(std.math.divCeil(u64, 256 << 20, page_bytes) catch unreachable, m.swap_pages / 4);
        if (headroom_slots >= m.snapshot_slots or (if (m.swap_pages == 0) headroom_pages != 0 else headroom_pages >= m.swap_pages)) return error.InvalidOptions;
        const max_bytes = try m.archiveBytes(m.state_layout.context);
        const alignment: usize = @intCast(@max(4096, m.device.host_import_alignment, if (o.alignment) |al| al.memory else 1));
        if (alignment > 1 << 20 or !std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;
        const self = try a.create(Disk);
        errdefer a.destroy(self);
        const rc = linux.mmap(null, staging_bytes + alignment, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        if (linux.errno(rc) != .SUCCESS) return error.MmapFailed;
        const raw = @as([*]u8, @ptrFromInt(rc))[0 .. staging_bytes + alignment];
        errdefer _ = linux.munmap(raw.ptr, raw.len);
        const memory = @as([*]u8, @ptrFromInt(std.mem.alignForward(usize, rc, alignment)))[0..staging_bytes];
        const path = try a.dupeZ(u8, o.directory);
        defer a.free(path);
        const dir_rc = linux.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
        if (linux.errno(dir_rc) != .SUCCESS) return error.OpenDirectoryFailed;
        const fd: i32 = @intCast(dir_rc);
        defer _ = linux.close(fd);
        const store = try storage.Store.create(a, .{ .dir_fd = fd, .name = "zerv-prefix-scratch", .file_bytes = o.bytes, .slot_bytes = 1 << 20, .alignment = o.alignment }, memory);
        errdefer store.destroy() catch @panic("new store busy");
        var catalog = try archive.Archive.init(a, store, .{ .records = o.records, .context = m.state_layout.context, .slots = m.state_layout.slots + 1, .max_bytes = max_bytes });
        errdefer catalog.deinit() catch @panic("new archive busy");
        const maps = try a.alloc(u32, (@as(usize, m.state_layout.slots) + 1) * m.state_layout.seq_pages);
        errdefer a.free(maps);
        self.* = .{ .allocator = a, .io = io, .m = m, .raw = raw, .memory = memory, .store = store, .catalog = catalog, .maps = maps, .imported = undefined, .commands = undefined, .headroom_slots = headroom_slots, .headroom_pages = @intCast(headroom_pages) };
        self.imported = try gpu.Buffer.initImported(m.device, memory);
        errdefer self.imported.deinit() catch @panic("new import busy");
        self.commands = try gpu.Commands.init(m.device);
        return self;
    }
    pub fn destroy(self: *Disk) void {
        if (self.source != null) @panic("cache source destroyed before drain");
        self.catalog.deinit() catch @panic("archive destroyed before drain");
        self.commands.deinit() catch @panic("archive GPU copy pending");
        self.imported.deinit() catch @panic("archive import retained");
        self.store.destroy() catch @panic("archive disk I/O pending");
        _ = linux.munmap(self.raw.ptr, self.raw.len);
        const a = self.allocator;
        a.free(self.maps);
        a.destroy(self);
    }
    fn map(self: *Disk, slot: u32, tokens: u32) []u32 {
        const S = self.m.state_layout;
        return self.maps[@as(usize, slot) * S.seq_pages ..][0 .. std.math.divCeil(u32, tokens, S.page) catch unreachable];
    }
    pub fn startWrite(self: *Disk, slot: u32, tokens: []const u32) !bool {
        const n: u32 = @intCast(tokens.len);
        try self.m.archiveMap(slot, n, self.map(slot, n), false);
        if (!try self.catalog.startWrite(slot, tokens, try self.m.archiveBytes(n))) return false;
        self.positions[slot] = n;
        return true;
    }
    /// Caller already reserved/reset private target pages; publication is in poll.
    pub fn startRead(self: *Disk, slot: u32, record: u32) !void {
        const n = self.catalog.entries[record].len;
        try self.m.archiveMap(slot, n, self.map(slot, n), true);
        try self.catalog.startRead(slot, record);
        self.positions[slot] = n;
    }
    /// One optional write job independent of all request slots. No request-state pause.
    pub fn startSource(self: *Disk, cache: kvcache.Cache) !bool {
        return self.startSourceHandle(cache, cache.coldSource() orelse return false);
    }
    pub fn startSourceHandle(self: *Disk, cache: kvcache.Cache, handle: kvcache.Handle) !bool {
        if (self.source != null) return false;
        const view = try cache.acquireSource(handle);
        errdefer cache.releaseSource(view.lease) catch @panic("invalid cache source lease");
        const n: u32 = @intCast(view.tokens.len);
        if (!try self.catalog.startWrite(self.m.state_layout.slots, view.tokens, try self.m.archiveBytes(n))) {
            try cache.releaseSource(view.lease);
            return false;
        }
        self.source = .{ .cache = cache, .view = view };
        self.source_cancel_requested = false;
        self.source_started_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
        self.positions[self.m.state_layout.slots] = n;
        return true;
    }
    fn releaseSource(self: *Disk) void {
        const held = self.source.?;
        held.cache.releaseSource(held.view.lease) catch @panic("invalid cache source lease");
        self.source = null;
        const held_ns = std.Io.Clock.awake.now(self.io).nanoseconds - self.source_started_ns;
        self.source_hold_ns += held_ns;
        self.source_max_hold_ns = @max(self.source_max_hold_ns, held_ns);
    }
    pub fn pollSource(self: *Disk, cancel: bool) !archive.Progress {
        return self.pollSourceWith(cancel, true);
    }
    pub fn pollSourceWith(self: *Disk, cancel: bool, allow_start: bool) !archive.Progress {
        if (self.source == null) return .{ .done = true, .progressed = false };
        const p = self.pollWith(self.m.state_layout.slots, cancel, .{ .allow_start = allow_start, .max_pending = staging_bytes / (1 << 20) - 2 }) catch |err| {
            self.releaseSource(); // archive errors are returned only after both owners drain
            return err;
        };
        if (p.done) self.releaseSource();
        return p;
    }
    fn startCopy(ctx: *anyopaque, slot: u32, offset: u64, bytes: []u8, importing: bool) !void {
        const self: *Disk = @ptrCast(@alignCast(ctx));
        const host_offset = @intFromPtr(bytes.ptr) - @intFromPtr(self.memory.ptr);
        self.source_span = null;
        self.cpu_copy = false;
        if (slot == self.m.state_layout.slots) {
            if (importing) return error.InvalidState;
            const view = (self.source orelse return error.InvalidState).view;
            self.cpu_copy = !(self.m.archiveSourceSubmit(&self.commands, &self.imported, host_offset, view.snapshot, @intCast(view.tokens.len), view.pages, offset, bytes.len) catch |e| {
                if (self.commands.state == .pending) @panic("source GPU DMA ownership unresolved");
                return e;
            });
            self.source_span = .{ .offset = offset, .bytes = bytes };
            if (self.cpu_copy) self.source_cpu_quanta += 1 else self.source_gpu_quanta += 1;
            self.submitted_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
            return;
        }
        self.m.archiveSubmit(&self.commands, &self.imported, host_offset, slot, self.positions[slot], self.map(slot, self.positions[slot]), offset, bytes.len, importing) catch |e| {
            // Returning this span to the disk worker after an uncertain GPU fence would
            // allow concurrent DMA into freed/reused memory. There is no safe fallback.
            if (self.commands.state == .pending) @panic("archive GPU DMA ownership unresolved");
            return e;
        };
        self.submitted_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
    }
    fn pollCopy(ctx: *anyopaque) !bool {
        const self: *Disk = @ptrCast(@alignCast(ctx));
        const done = self.cpu_copy or (self.commands.poll() catch |e| {
            if (self.commands.state == .pending) @panic("archive GPU DMA ownership unresolved");
            return e;
        });
        if (done) if (self.source_span) |span| {
            try self.m.state_layout.clearArchiveTail(self.m.snapshotBytes(), @intCast(self.source.?.view.tokens.len), span.offset, span.bytes);
            self.source_span = null;
        };
        if (!done and std.Io.Clock.awake.now(self.io).nanoseconds - self.submitted_ns >= self.m.options.timeout_ns)
            @panic("archive GPU DMA deadline exceeded; ownership unresolved");
        return done;
    }
    pub fn poll(self: *Disk, slot: u32, cancel: bool) !archive.Progress {
        return self.pollWith(slot, cancel, .{});
    }
    fn pollWith(self: *Disk, slot: u32, cancel: bool, options: archive.Advance) !archive.Progress {
        const writing = self.catalog.writing(slot);
        const p = self.catalog.advanceWith(.{ .ctx = self, .start = startCopy, .poll = pollCopy }, slot, cancel, options) catch |e| {
            if (!writing) {
                try self.m.select(slot);
                try self.m.reset();
                try self.m.releasePages(slot);
            }
            if (e == error.DiskIoFailed or e == error.CorruptRecord) return .{ .done = true, .progressed = true };
            return e;
        };
        if (p.done and !writing) try self.m.archiveRestored(slot, p.position);
        return p;
    }
};
