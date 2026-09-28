const std = @import("std");
const archive = @import("session").archive;
const storage = @import("storage");
const linux = std.os.linux;
const t = std.testing;
const fixture = @embedFile("fixtures/storage/positional.bin");
const Sha256 = @import("fast_sha256.zig").Sha256;
fn hex(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
const Fake = struct {
    output: [3][fixture.len]u8 = undefined,
    fail_import: bool = false,
    fail_start: bool = false,
    fail_poll: bool = false,
    ready: bool = true,
    pending: ?struct { slot: u32, offset: u64, bytes: []u8, importing: bool } = null,
    fn start(ctx: *anyopaque, slot: u32, offset: u64, bytes: []u8, importing: bool) !void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        if (self.pending != null) return error.DeviceBusy;
        if (slot >= 3 or offset + bytes.len > fixture.len) return error.BadCopy;
        if (self.fail_start) return error.InjectedStartError;
        self.pending = .{ .slot = slot, .offset = offset, .bytes = bytes, .importing = importing };
    }
    fn poll(ctx: *anyopaque) !bool {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        const p = self.pending orelse return error.NoCopy;
        if (!self.ready) return false;
        self.pending = null;
        if (self.fail_poll or (p.importing and self.fail_import)) return error.InjectedDeviceError;
        if (p.importing) {
            @memcpy(self.output[p.slot][@intCast(p.offset)..][0..p.bytes.len], p.bytes);
        } else @memcpy(p.bytes, fixture[@intCast(p.offset)..][0..p.bytes.len]);
        return true;
    }
    fn device(self: *Fake) archive.Device {
        return .{ .ctx = self, .start = start, .poll = poll };
    }
};
fn finish(a: *archive.Archive, fake: *Fake, slot: u32, cancel: bool) !u32 {
    for (0..10000) |_| {
        const p = try a.advance(fake.device(), slot, cancel);
        if (p.done) return p.position;
        if (!p.progressed) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    @panic("archive did not drain: never free DMA-owned staging");
}
fn open(dir: std.Io.Dir, mem: []u8) !*storage.Store {
    return storage.Store.create(t.allocator, .{ .dir_fd = dir.handle, .name = "archive", .file_bytes = 65536, .slot_bytes = 4096, .alignment = .{ .memory = 4096, .offset = 4096 } }, mem);
}
const opts: archive.Options = .{ .records = 4, .context = 32, .slots = 3, .max_bytes = fixture.len };
fn initFailure(a: std.mem.Allocator, store: *storage.Store) !void {
    var x = try archive.Archive.init(a, store, opts);
    try x.deinit();
}

test "archive independent POSIX SHA256 goldens, eviction, exact partial tails and no steady allocation" {
    const Case = struct { length: u32, chunk: u32, hashes: [][]const u8 };
    const Oracle = struct { source_sha256: []const u8, generator_sha256: []const u8, cases: []Case };
    const oracle = try std.json.parseFromSlice(Oracle, t.allocator, @embedFile("fixtures/archive/oracle.json"), .{ .ignore_unknown_fields = true });
    defer oracle.deinit();
    try t.expectEqualStrings(oracle.value.source_sha256, &hex(fixture));
    try t.expectEqualStrings(oracle.value.generator_sha256, &hex(@embedFile("reference/generate_archive_fixture.py")));
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 8192);
    defer t.allocator.free(mem);
    const independent = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 4096);
    defer t.allocator.free(independent);
    const store = try open(tmp.dir, mem);
    defer store.destroy() catch @panic("pending disk");
    try t.checkAllAllocationFailures(t.allocator, initFailure, .{store});
    var failing = t.FailingAllocator.init(t.allocator, .{});
    var a = try archive.Archive.init(failing.allocator(), store, opts);
    defer a.deinit() catch @panic("pending archive");
    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    var fake: Fake = .{};
    for (oracle.value.cases, 0..) |case, ci| {
        const tokens = [_]u32{ 9, @intCast(ci + 1), 9 };
        try t.expect(try a.startWrite(0, tokens[0..2], case.length));
        try t.expect(a.lookup(&tokens) == null);
        try t.expectError(error.Busy, a.deinit());
        try t.expectEqual(@as(u32, 0), try finish(&a, &fake, 0, false));
        const i = a.lookup(&tokens) orelse return error.MissingRecord;
        for (case.hashes, 0..) |digest, k| {
            const idx = i * a.max_chunks + k;
            try t.expectEqualStrings(digest, &std.fmt.bytesToHex(a.digests[idx], .lower));
            try t.expectEqual(@as(usize, 4096), linux.pread(store.fd, independent.ptr, 4096, @as(i64, a.blocks[idx]) * 4096));
            try t.expectEqualStrings(digest, &hex(independent));
        }
        try t.expect(!try a.startWrite(1, tokens[0..2], case.length));
        // Simultaneous immutable readers; corruption/eviction cannot alias their blocks.
        try a.startRead(0, i);
        try a.startRead(1, i);
        @memset(&fake.output[0], 0xcc);
        @memset(&fake.output[1], 0xdd);
        try t.expectEqual(@as(u32, 2), try finish(&a, &fake, 1, false));
        try t.expectEqual(@as(u32, 2), try finish(&a, &fake, 0, false));
        for (0..2) |slot| {
            try t.expectEqualSlices(u8, fixture[0..case.length], fake.output[slot][0..case.length]);
            for (fake.output[slot][case.length..]) |b| try t.expectEqual(@as(u8, if (slot == 0) 0xcc else 0xdd), b);
        }
    }
    try t.expect(a.stats.evictions > 0);
    try t.expect(!failing.has_induced_failure);
    try t.expectEqual(@as(u64, 4), a.stats.writes);
    try t.expectEqual(@as(u64, 8), a.stats.reads);
}

test "archive cancellation drains, leases block eviction, corruption and I/O failures never publish" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 8192);
    defer t.allocator.free(mem);
    const poison = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 4096);
    defer t.allocator.free(poison);
    const store = try open(tmp.dir, mem);
    defer store.destroy() catch @panic("pending disk");
    var a = try archive.Archive.init(t.allocator, store, opts);
    defer a.deinit() catch @panic("pending archive");
    var fake: Fake = .{};
    try t.expect(try a.startWrite(0, &.{1}, 65000));
    _ = try a.advance(fake.device(), 0, false); // at least one actual submitted write
    try t.expectError(error.Canceled, finish(&a, &fake, 0, true));
    try t.expect(a.lookup(&.{ 1, 2 }) == null);
    try t.expectEqual(@as(u32, 16), a.free_chunks);
    try t.expect(try a.startWrite(0, &.{1}, 65000));
    _ = try finish(&a, &fake, 0, false);
    const i = a.lookup(&.{ 1, 2 }).?;
    try a.startRead(1, i);
    try t.expect(!try a.startWrite(0, &.{2}, 8193)); // all extents leased
    try t.expectError(error.Canceled, finish(&a, &fake, 1, true));
    try t.expect(a.lookup(&.{ 1, 2 }) != null); // cancel doesn't corrupt the immutable record
    // Independent disk corruption while idle. Hash validation must fail before import.
    @memset(poison, 0x55);
    try t.expectEqual(@as(usize, 4096), linux.pwrite(store.fd, poison.ptr, 4096, @as(i64, a.blocks[i * a.max_chunks]) * 4096));
    try a.startRead(0, i);
    try a.startRead(1, i);
    @memset(&fake.output[0], 0xaa);
    try t.expectError(error.CorruptRecord, finish(&a, &fake, 0, false));
    try t.expectError(error.CorruptRecord, finish(&a, &fake, 1, false));
    try t.expect(a.lookup(&.{ 1, 2 }) == null);
    for (fake.output[0][0..4096]) |b| try t.expectEqual(@as(u8, 0xaa), b);
    try t.expectEqual(@as(u32, 16), a.free_chunks);
    try t.expect(try a.startWrite(0, &.{3}, 8193));
    _ = try finish(&a, &fake, 0, false);
    const j = a.lookup(&.{ 3, 4 }).?;
    try t.expectEqual(linux.E.SUCCESS, linux.errno(linux.ftruncate(store.fd, 0)));
    try a.startRead(0, j);
    try t.expectError(error.DiskIoFailed, finish(&a, &fake, 0, false));
    try t.expect(a.lookup(&.{ 3, 4 }) == null);
    _ = linux.close(store.fd);
    store.fd = -1;
    try t.expect(try a.startWrite(0, &.{5}, 8193));
    try t.expectError(error.DiskIoFailed, finish(&a, &fake, 0, false));
    try t.expectEqual(@as(u32, 16), a.free_chunks);
}

test "archive options, longest prefix and device-error drain" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 8192);
    defer t.allocator.free(mem);
    const store = try open(tmp.dir, mem);
    defer store.destroy() catch @panic("pending disk");
    var bad = opts;
    bad.records = 0;
    try t.expectError(error.InvalidOptions, archive.Archive.init(t.allocator, store, bad));
    bad = opts;
    bad.context = std.math.maxInt(u32);
    try t.expectError(error.MetadataBudgetExceeded, archive.Archive.init(t.allocator, store, bad));
    var a = try archive.Archive.init(t.allocator, store, opts);
    defer a.deinit() catch @panic("pending archive");
    var fake: Fake = .{};
    try t.expectError(error.InvalidSlot, a.startWrite(3, &.{1}, 4));
    try t.expectError(error.InvalidOptions, a.startWrite(0, &.{}, 4));
    try t.expectError(error.InvalidOptions, a.startWrite(0, &.{1}, 65537));
    try t.expect(try a.startWrite(0, &.{1}, 8193));
    try t.expect(try a.startWrite(1, &.{ 1, 2 }, 12288));
    _ = try finish(&a, &fake, 1, false);
    _ = try finish(&a, &fake, 0, false);
    const best = a.lookup(&.{ 1, 2, 3 }).?;
    try t.expectEqual(@as(u32, 2), a.entries[best].len);
    try t.expectEqual(@as(u32, 1), a.entries[a.lookup(&.{ 1, 2 }).?].len);
    try t.expect(a.lookup(&.{1}) == null);
    fake.fail_import = true;
    try a.startRead(2, best);
    try t.expectError(error.InjectedDeviceError, finish(&a, &fake, 2, false));
    try t.expectEqual(@as(u32, 1), a.entries[a.lookup(&.{ 1, 2, 3 }).?].len);
}

test "async capture keeps staging leased, submits no early disk write, cancellation waits for device" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 8192);
    defer t.allocator.free(mem);
    const store = try open(tmp.dir, mem);
    defer store.destroy() catch @panic("pending disk");
    var a = try archive.Archive.init(t.allocator, store, opts);
    defer a.deinit() catch @panic("pending archive");
    var fake: Fake = .{ .ready = false };
    try t.expect(try a.startWrite(0, &.{1}, 8193));
    _ = try a.advance(fake.device(), 0, false);
    const ticket = a.device_pending.?.ticket;
    try t.expectError(error.InvalidState, store.poll(ticket)); // held, never submitted
    const other = try store.acquire();
    try t.expectError(error.QueueFull, store.acquire());
    for (0..4) |_| {
        const p = try a.advance(fake.device(), 0, true);
        try t.expect(!p.done and !p.progressed);
        try t.expect(a.active(0));
        try t.expectError(error.Busy, a.startWrite(0, &.{2}, 4));
        try t.expectError(error.Busy, a.deinit());
        try t.expectError(error.QueueFull, store.acquire());
        try t.expectError(error.InvalidState, store.poll(ticket));
        try t.expectEqual(@as(u64, 0), a.stats.write_bytes);
        try t.expectEqual(@as(u64, 0), a.stats.writes);
    }
    fake.ready = true;
    try t.expectError(error.Canceled, a.advance(fake.device(), 0, true));
    try t.expect(fake.pending == null and a.device_pending == null);
    try t.expect(!a.active(0));
    try t.expectEqual(@as(u32, 16), a.free_chunks);
    const reused = try store.acquire();
    try t.expectEqual(ticket.slot, reused.slot);
    try t.expect(reused.generation > ticket.generation);
    try t.expectError(error.InvalidTicket, store.buffer(ticket));
    try store.release(reused);
    try store.release(other);
    // Same request slot can now retry; all bytes still match the independent payload.
    try t.expect(try a.startWrite(0, &.{1}, 8193));
    _ = try finish(&a, &fake, 0, false);
    try a.startRead(0, a.lookup(&.{ 1, 2 }).?);
    _ = try finish(&a, &fake, 0, false);
    try t.expectEqualSlices(u8, fixture[0..8193], fake.output[0][0..8193]);
}

test "async upload keeps disk-done ticket until acknowledgment, other slot disk progress and cancel" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 8192);
    defer t.allocator.free(mem);
    const store = try open(tmp.dir, mem);
    defer store.destroy() catch @panic("pending disk");
    var a = try archive.Archive.init(t.allocator, store, opts);
    defer a.deinit() catch @panic("pending archive");
    var fake: Fake = .{};
    try t.expect(try a.startWrite(0, &.{1}, 8193));
    _ = try finish(&a, &fake, 0, false);
    const record = a.lookup(&.{ 1, 2 }).?;
    fake.ready = false;
    @memset(&fake.output[0], 0xcc);
    try a.startRead(0, record);
    for (0..10000) |_| {
        _ = try a.advance(fake.device(), 0, false);
        if (fake.pending != null) break;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    try t.expect(fake.pending != null);
    const held = a.device_pending.?.ticket;
    try t.expect((try store.poll(held)).?.exact()); // disk is done, GPU is not
    for (0..4) |_| {
        const p = try a.advance(fake.device(), 0, true);
        try t.expect(!p.done);
        try t.expectEqual(held, a.device_pending.?.ticket);
        try t.expectError(error.Busy, a.deinit());
        for (fake.output[0][0..4096]) |b| try t.expectEqual(@as(u8, 0xcc), b);
    }
    // Wait for canceled DISK requests, not for the deliberately blocked device.
    // Their real worker completion is independent of the number/speed of our polls.
    for (0..10000) |_| {
        if (a.jobs[0].pending == 1) break;
        _ = try a.advance(fake.device(), 0, true);
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    try t.expectEqual(@as(u32, 1), a.jobs[0].pending);
    // Another reader can submit/drain disk work while slot 0 owns the device.
    try a.startRead(1, record);
    _ = try a.advance(fake.device(), 1, false);
    try t.expect(a.jobs[1].pending != 0);
    try t.expectError(error.Canceled, finish(&a, &fake, 1, true));
    try t.expect(a.active(0));
    fake.ready = true;
    try t.expectError(error.Canceled, finish(&a, &fake, 0, true));
    try t.expect(fake.pending == null and a.device_pending == null);
    try t.expectEqual(record, a.lookup(&.{ 1, 2 }).?);
    try a.startRead(0, record);
    _ = try finish(&a, &fake, 0, false);
    try t.expectEqualSlices(u8, fixture[0..8193], fake.output[0][0..8193]);
}

test "async device start and completion faults drain before slot reuse" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 8192);
    defer t.allocator.free(mem);
    const store = try open(tmp.dir, mem);
    defer store.destroy() catch @panic("pending disk");
    var a = try archive.Archive.init(t.allocator, store, opts);
    defer a.deinit() catch @panic("pending archive");
    for ([_]bool{ false, true }) |importing| {
        for ([_]bool{ false, true }) |at_start| {
            var fake: Fake = .{};
            try t.expect(try a.startWrite(0, &.{1}, 8193));
            if (importing) {
                _ = try finish(&a, &fake, 0, false);
                try a.startRead(0, a.lookup(&.{ 1, 2 }).?);
            }
            fake.fail_start = at_start;
            fake.fail_poll = !at_start;
            try t.expectError(if (at_start) error.InjectedStartError else error.InjectedDeviceError, finish(&a, &fake, 0, false));
            try t.expect(fake.pending == null and a.device_pending == null);
            try t.expect(!a.active(0));
            try t.expectEqual(@as(u32, 16), a.free_chunks);
            for (a.pending) |p| try t.expect(p == null);
            try t.expect(a.lookup(&.{ 1, 2 }) == null);
        }
    }
}
