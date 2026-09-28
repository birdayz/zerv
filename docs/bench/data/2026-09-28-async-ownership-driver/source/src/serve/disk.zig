//! Stable owner of a RAM-staged immutable prefix archive. Scheduler thread only.
//! Lifetime/error contract: docs/specs/disk-prefix-cache.md.
const std = @import("std");
const gpu = @import("gpu");
const model = @import("model");
const storage = @import("storage");
const archive = @import("session").archive;
const linux = std.os.linux;
pub const staging_bytes = 8 << 20;
pub const Options = struct { directory: []const u8, bytes: u64, records: u32 = 64, alignment: ?storage.Alignment = null };
pub const Disk = struct {
    allocator: std.mem.Allocator,
    m: *model.Model,
    raw: []u8,
    memory: []u8,
    imported: gpu.Buffer,
    commands: gpu.Commands,
    store: *storage.Store,
    catalog: archive.Archive,
    maps: []u32,
    positions: [64]u32 = @splat(0),

    pub fn create(a: std.mem.Allocator, m: *model.Model, o: Options) !*Disk {
        if (m.options.mtp or !m.options.kv_share or m.state_layout.slots < 2 or o.bytes == 0 or o.bytes % (1 << 20) != 0) return error.InvalidOptions;
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
        var catalog = try archive.Archive.init(a, store, .{ .records = o.records, .context = m.state_layout.context, .slots = m.state_layout.slots, .max_bytes = max_bytes });
        errdefer catalog.deinit() catch @panic("new archive busy");
        const maps = try a.alloc(u32, @as(usize, m.state_layout.slots) * m.state_layout.seq_pages);
        errdefer a.free(maps);
        self.* = .{ .allocator = a, .m = m, .raw = raw, .memory = memory, .store = store, .catalog = catalog, .maps = maps, .imported = undefined, .commands = undefined };
        self.imported = try gpu.Buffer.initImported(m.device, memory);
        errdefer self.imported.deinit() catch @panic("new import busy");
        self.commands = try gpu.Commands.init(m.device);
        return self;
    }
    pub fn destroy(self: *Disk) void {
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
    fn copy(ctx: *anyopaque, slot: u32, offset: u64, bytes: []u8, importing: bool) !void {
        const self: *Disk = @ptrCast(@alignCast(ctx));
        const host_offset = @intFromPtr(bytes.ptr) - @intFromPtr(self.memory.ptr);
        self.m.archiveCopy(&self.commands, &self.imported, host_offset, slot, self.positions[slot], self.map(slot, self.positions[slot]), offset, bytes.len, importing) catch |e| {
            // Returning this span to the disk worker after an uncertain GPU fence would
            // allow concurrent DMA into freed/reused memory. There is no safe fallback.
            if (self.m.device.pending != 0) @panic("archive GPU DMA ownership unresolved");
            return e;
        };
    }
    pub fn poll(self: *Disk, slot: u32, cancel: bool) !archive.Progress {
        const writing = self.catalog.writing(slot);
        const p = self.catalog.advance(.{ .ctx = self, .copy = copy }, slot, cancel) catch |e| {
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
