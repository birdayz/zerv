const std = @import("std");
const storage = @import("storage");
const linux = std.os.linux;
const t = std.testing;
const fixture = @embedFile("fixtures/storage/positional.bin");
const fast_sha = @import("fast_sha256.zig").Sha256;

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    fast_sha.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn wait(s: *storage.Store, ticket: storage.Ticket) !storage.Completion {
    // A timeout never cancels or frees a DMA-owned buffer; fail-stop the test process.
    for (0..10000) |_| {
        if (try s.poll(ticket)) |c| return c;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    @panic("scratch I/O did not complete in 10 seconds; no early staging free");
}

fn memory(size: usize) ![]align(4096) u8 {
    return t.allocator.alignedAlloc(u8, .fromByteUnits(4096), size);
}

fn open(dir: std.Io.Dir, bytes: []u8, chunk: usize) !*storage.Store {
    return storage.Store.create(t.allocator, .{ .dir_fd = dir.handle, .name = "scratch", .file_bytes = fixture.len, .slot_bytes = chunk, .alignment = .{ .memory = 4096, .offset = 4096 } }, bytes);
}

test "storage independent positional-I/O fixture provenance" {
    const Manifest = struct { sha256: []const u8, generator_sha256: []const u8, bytes: usize };
    const m = try std.json.parseFromSlice(Manifest, t.allocator, @embedFile("fixtures/storage/manifest.json"), .{ .ignore_unknown_fields = true });
    defer m.deinit();
    try t.expectEqual(fixture.len, m.value.bytes);
    try t.expectEqualStrings(m.value.sha256, &hash(fixture));
    try t.expectEqualStrings(m.value.generator_sha256, &hash(@embedFile("reference/generate_disk_fixture.py")));
}

test "storage worker reads synchronous writes and synchronous reads verify worker writes" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try memory(4 * 8192);
    defer t.allocator.free(mem);
    const oracle = try memory(fixture.len);
    defer t.allocator.free(oracle);
    const s = try open(tmp.dir, mem, 8192);
    defer s.destroy() catch @panic("outstanding storage ticket");
    // The path was unlinked immediately, not held for pathname-based cleanup at exit.
    const absent = linux.openat(tmp.dir.handle, "scratch", .{ .ACCMODE = .RDONLY }, 0);
    try t.expectEqual(linux.E.NOENT, linux.errno(absent));
    @memcpy(oracle, fixture);
    try t.expectEqual(fixture.len, linux.pwrite(s.fd, oracle.ptr, oracle.len, 0));
    var tickets: [4]storage.Ticket = undefined;
    // More rounds than ring/slot capacity, mixed 4K/8K lengths and nonsequential offsets.
    for (0..128) |round| {
        for (&tickets, 0..) |*ticket, j| {
            ticket.* = try s.acquire();
            const buf = try s.buffer(ticket.*);
            @memset(buf, 0xa5);
            const offset = ((round + j * 2) % 8) * 8192;
            const len: usize = if (j % 2 == 0) 4096 else 8192;
            try s.submit(ticket.*, .read, offset, len);
        }
        try t.expectError(error.QueueFull, s.acquire());
        try t.expectError(error.ResourceInUse, s.destroy());
        for (tickets, 0..) |ticket, j| {
            try t.expect((try wait(s, ticket)).exact());
            const offset = ((round + j * 2) % 8) * 8192;
            const len: usize = if (j % 2 == 0) 4096 else 8192;
            const buf = try s.buffer(ticket);
            try t.expectEqualSlices(u8, fixture[offset..][0..len], buf[0..len]);
            if (len < buf.len) for (buf[len..]) |b| try t.expectEqual(@as(u8, 0xa5), b);
            try s.release(ticket);
            try t.expectError(error.InvalidTicket, s.poll(ticket));
        }
    }
    // Poison the file through the independent syscall, then write it through io_uring.
    @memset(oracle, 0);
    try t.expectEqual(fixture.len, linux.pwrite(s.fd, oracle.ptr, oracle.len, 0));
    for (0..2) |batch| {
        for (&tickets, 0..) |*ticket, j| {
            ticket.* = try s.acquire();
            const offset = (batch * 4 + (3 - j)) * 8192;
            @memcpy(try s.buffer(ticket.*), fixture[offset..][0..8192]);
            try s.submit(ticket.*, .write, offset, 8192);
        }
        for (tickets) |ticket| {
            try t.expect((try wait(s, ticket)).exact());
            try s.release(ticket);
        }
    }
    try t.expectEqual(fixture.len, linux.pread(s.fd, oracle.ptr, oracle.len, 0));
    try t.expectEqualSlices(u8, fixture, oracle);
}

test "storage bounds, generations, range ownership and held cancellation" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try memory(2 * 4096);
    defer t.allocator.free(mem);
    const s = try open(tmp.dir, mem, 4096);
    defer s.destroy() catch @panic("outstanding storage ticket");
    const a = try s.acquire();
    const b = try s.acquire();
    try t.expectError(error.InvalidState, s.poll(a));
    try t.expectError(error.ResourceInUse, s.destroy());
    try t.expectError(error.InvalidTicket, s.buffer(.{ .slot = 32, .generation = 0 }));
    try t.expectError(error.InvalidRange, s.submit(a, .read, std.math.maxInt(u64), 4096));
    try t.expectError(error.InvalidRange, s.submit(a, .read, fixture.len, 4096));
    try t.expectError(error.InvalidRange, s.submit(a, .read, 0, 0));
    try t.expectError(error.InvalidRange, s.submit(a, .read, 0, 8192));
    try t.expectError(error.InvalidRange, s.submit(a, .read, 1, 4096));
    try t.expectError(error.InvalidRange, s.submit(a, .read, 0, 1));
    @memcpy(try s.buffer(a), fixture[0..4096]);
    try s.submit(a, .write, 0, 4096);
    try t.expectError(error.ResourceInUse, s.submit(a, .write, 8192, 4096));
    try t.expectError(error.RangeInUse, s.submit(b, .read, 0, 4096));
    try t.expectError(error.RangeInUse, s.submit(b, .write, 0, 4096));
    try t.expect((try wait(s, a)).exact());
    // A completed but unacknowledged write still owns its range.
    try t.expectError(error.RangeInUse, s.submit(b, .read, 0, 4096));
    try s.release(a);
    const c = try s.acquire();
    try t.expectEqual(a.slot, c.slot);
    try t.expect(a.generation != c.generation);
    try t.expectError(error.InvalidTicket, s.release(a));
    try s.submit(b, .read, 0, 4096);
    try s.submit(c, .read, 0, 4096); // overlapping reads are permitted
    for ([_]storage.Ticket{ b, c }) |ticket| {
        try t.expect((try wait(s, ticket)).exact());
        try t.expectEqualSlices(u8, fixture[0..4096], try s.buffer(ticket));
        try s.release(ticket);
        try t.expectError(error.InvalidTicket, s.release(ticket));
    }
    const canceled = try s.acquire();
    try s.release(canceled); // a held slot has never passed ownership to the worker
}

test "storage existing files and symlinks are never overwritten; init rejects bad budgets" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try memory(4096 * 33);
    defer t.allocator.free(mem);
    const base: storage.Options = .{ .dir_fd = tmp.dir.handle, .name = "scratch", .file_bytes = fixture.len, .slot_bytes = 4096 };
    var bad = base;
    bad.file_bytes = 0;
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, bad, mem[0..4096]));
    bad.file_bytes = std.math.maxInt(u64);
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, bad, mem[0..4096]));
    bad = base;
    bad.slot_bytes = 0;
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, bad, mem[0..4096]));
    bad = base;
    bad.name = "../scratch";
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, bad, mem[0..4096]));
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, base, mem));
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, base, mem[1..4097]));
    try t.expectError(error.InvalidOptions, storage.Store.create(t.allocator, base, mem[0..0]));
    const rc = linux.openat(tmp.dir.handle, "scratch", .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true }, 0o600);
    try t.expectEqual(linux.E.SUCCESS, linux.errno(rc));
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    try t.expectEqual(@as(usize, 4), linux.write(fd, "keep", 4));
    try t.expectError(error.FileExists, storage.Store.create(t.allocator, base, mem[0..4096]));
    var check: [4]u8 = undefined;
    try t.expectEqual(@as(usize, 4), linux.pread(fd, &check, 4, 0));
    try t.expectEqualStrings("keep", &check);
    try t.expectEqual(linux.E.SUCCESS, linux.errno(linux.symlinkat("scratch", tmp.dir.handle, "link")));
    bad = base;
    bad.name = "link";
    try t.expectError(error.FileExists, storage.Store.create(t.allocator, bad, mem[0..4096]));
}

test "storage short and negative completions are not successful and remain releasable" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try memory(4096);
    defer t.allocator.free(mem);
    const s = try open(tmp.dir, mem, 4096);
    defer s.destroy() catch @panic("outstanding storage ticket");
    // Deliberate external fault injection, only while idle. Not a production operation.
    try t.expectEqual(linux.E.SUCCESS, linux.errno(linux.ftruncate(s.fd, 0)));
    const a = try s.acquire();
    try s.submit(a, .read, 0, 4096);
    const short = try wait(s, a);
    try t.expect(!short.exact());
    try t.expectEqual(@as(i32, 0), short.result);
    try s.release(a);
    _ = linux.close(s.fd);
    s.fd = -1;
    const b = try s.acquire();
    try s.submit(b, .read, 0, 4096);
    const failed = try wait(s, b);
    try t.expect(!failed.exact());
    try t.expectEqual(-@as(i32, @intFromEnum(linux.E.BADF)), failed.result);
    try s.release(b);
}

fn initWithAllocator(a: std.mem.Allocator, dir: std.Io.Dir, mem: []u8) !void {
    const s = try storage.Store.create(a, .{ .dir_fd = dir.handle, .name = "oom", .file_bytes = fixture.len, .slot_bytes = 4096, .alignment = .{ .memory = 4096, .offset = 4096 } }, mem);
    try s.destroy();
}

test "storage initialization allocation failures clean up and steady state allocates nothing" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const mem = try memory(storage.max_depth * 4096);
    defer t.allocator.free(mem);
    try t.checkAllAllocationFailures(t.allocator, initWithAllocator, .{ tmp.dir, mem });
    var fail = t.FailingAllocator.init(t.allocator, .{});
    const s = try storage.Store.create(fail.allocator(), .{ .dir_fd = tmp.dir.handle, .name = "steady", .file_bytes = storage.max_depth * 4096, .slot_bytes = 4096, .alignment = .{ .memory = 4096, .offset = 4096 } }, mem);
    defer s.destroy() catch @panic("outstanding storage ticket");
    fail.fail_index = fail.alloc_index;
    fail.resize_fail_index = fail.resize_index;
    var tickets: [storage.max_depth]storage.Ticket = undefined;
    for (&tickets, 0..) |*ticket, i| {
        ticket.* = try s.acquire();
        @memcpy(try s.buffer(ticket.*), fixture[(i % 16) * 4096 ..][0..4096]);
        try s.submit(ticket.*, .write, i * 4096, 4096);
    }
    try t.expectError(error.QueueFull, s.acquire());
    for (tickets) |ticket| {
        try t.expect((try wait(s, ticket)).exact());
        try s.release(ticket);
    }
    try t.expect(!fail.has_induced_failure);
}

test "direct I/O alignment is reported or configured, never guessed or weakened" {
    const base: storage.Alignment = .{ .memory = 4096, .offset = 512 };
    try t.expectEqual(base, try storage.resolveAlignment(base, null));
    try t.expectEqual(base, try storage.resolveAlignment(null, base));
    try t.expectError(error.DirectIoUnsupported, storage.resolveAlignment(null, null));
    try t.expectError(error.DirectIoUnsupported, storage.resolveAlignment(.{ .memory = 0, .offset = 0 }, base));
    try t.expectError(error.InvalidAlignment, storage.resolveAlignment(null, .{ .memory = 3, .offset = 512 }));
    try t.expectError(error.InvalidAlignment, storage.resolveAlignment(base, .{ .memory = 512, .offset = 512 }));
    try t.expectError(error.InvalidAlignment, storage.resolveAlignment(base, .{ .memory = 4096, .offset = 256 }));
    const stronger: storage.Alignment = .{ .memory = 8192, .offset = 4096 };
    try t.expectEqual(stronger, try storage.resolveAlignment(base, stronger));
}
