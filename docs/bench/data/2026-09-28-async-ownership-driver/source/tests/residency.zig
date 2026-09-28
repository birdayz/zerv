const std = @import("std");
const r = @import("session").residency;
const storage = @import("storage");
const linux = std.os.linux;
const t = std.testing;
const Sha256 = @import("fast_sha256.zig").Sha256;

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
const Row = struct {
    op: enum { reserve, saved, acquire, release, spill, restore, complete, discard, drop },
    handle: ?r.Handle,
    transfer: ?r.Transfer,
    success: bool,
    @"error": ?[]const u8,
    returned_handle: ?r.Handle,
    returned_transfer: ?r.Transfer,
    slot: ?u32,
    entries: []r.Entry,
    hot_owners: []?u32,
    disk_owners: []?u32,
};
const Result = struct { handle: ?r.Handle = null, transfer: ?r.Transfer = null, slot: ?u32 = null };
fn apply(table: *r.Table, row: Row) !Result {
    switch (row.op) {
        .reserve => return .{ .handle = try table.reserve() },
        .saved => try table.saved(row.handle.?, row.success),
        .acquire => return .{ .slot = try table.acquireResident(row.handle.?) },
        .release => try table.releaseResident(row.handle.?),
        .spill => return .{ .transfer = try table.spill(row.handle.?) },
        .restore => return .{ .transfer = try table.restore(row.handle.?) },
        .complete => try table.complete(row.transfer.?, row.success),
        .discard => try table.discardBacking(row.handle.?),
        .drop => try table.drop(row.handle.?),
    }
    return .{};
}
fn invariants(table: *const r.Table) !void {
    for (table.entries, 0..) |e, i| {
        if (e.hot) |hot| try t.expectEqual(@as(?u32, @intCast(i)), table.hot_owners[hot]);
        if (e.disk) |disk| try t.expectEqual(@as(?u32, @intCast(i)), table.disk_owners[disk]);
        switch (e.phase) {
            .free => try t.expect(e.hot == null and e.disk == null and e.leases == 0),
            .saving => try t.expect(e.hot != null and e.disk == null and e.leases == 0),
            .resident => try t.expect(e.hot != null),
            .disk => try t.expect(e.hot == null and e.disk != null and e.leases == 0),
            .writing, .reading => try t.expect(e.hot != null and e.disk != null and e.leases == 0),
        }
    }
    for (table.hot_owners, 0..) |owner, hot| if (owner) |i| try t.expectEqual(@as(?u32, @intCast(hot)), table.entries[i].hot);
    for (table.disk_owners, 0..) |owner, disk| if (owner) |i| try t.expectEqual(@as(?u32, @intCast(disk)), table.entries[i].disk);
}
fn drainModel(table: *r.Table) !void {
    // Metadata-only test cleanup: no real I/O was submitted by the trace player.
    for (table.entries, 0..) |e, i| {
        const h: r.Handle = .{ .index = @intCast(i), .generation = e.generation };
        switch (e.phase) {
            .saving => try table.saved(h, false),
            .reading, .writing => try table.complete(.{ .handle = h, .serial = e.serial, .hot = e.hot.?, .disk = e.disk.?, .direction = if (e.phase == .writing) .write else .read }, false),
            else => {},
        }
        for (0..e.leases) |_| try table.releaseResident(h);
    }
}

test "residency exact independent ownership traces and provenance" {
    const bytes = @embedFile("fixtures/residency/traces.json");
    const Manifest = struct { sha256: []const u8, generator_sha256: []const u8, rows: usize };
    const m = try std.json.parseFromSlice(Manifest, t.allocator, @embedFile("fixtures/residency/manifest.json"), .{ .ignore_unknown_fields = true });
    defer m.deinit();
    try t.expectEqualStrings(m.value.sha256, &hash(bytes));
    try t.expectEqualStrings(m.value.generator_sha256, &hash(@embedFile("reference/generate_residency_fixture.py")));
    const Case = struct { count: u32, hot: u32, disk: u32, rows: []Row };
    const cases = try std.json.parseFromSlice([]Case, t.allocator, bytes, .{ .ignore_unknown_fields = true });
    defer cases.deinit();
    var n: usize = 0;
    for (cases.value, 0..) |case, ci| {
        var table = try r.Table.init(t.allocator, case.count, case.hot, case.disk);
        defer {
            drainModel(&table) catch @panic("trace cleanup failed");
            table.deinit() catch @panic("trace still pending");
        }
        for (case.rows, 0..) |row, ri| {
            errdefer std.debug.print("residency fixture case {d}, row {d}, op {s}\n", .{ ci, ri, @tagName(row.op) });
            if (apply(&table, row)) |got| {
                try t.expect(row.@"error" == null);
                try t.expectEqualDeep(row.returned_handle, got.handle);
                try t.expectEqualDeep(row.returned_transfer, got.transfer);
                try t.expectEqual(row.slot, got.slot);
            } else |err| {
                try t.expectEqualStrings(row.@"error" orelse "unexpected native error", @errorName(err));
            }
            try t.expectEqualDeep(row.entries, table.entries);
            try t.expectEqualDeep(row.hot_owners, table.hot_owners);
            try t.expectEqualDeep(row.disk_owners, table.disk_owners);
            try invariants(&table);
            n += 1;
        }
    }
    try t.expectEqual(m.value.rows, n);
    std.debug.print("residency: {d} independent full-state transitions exact\n", .{n});
}

fn allocated(a: std.mem.Allocator) !void {
    var table = try r.Table.init(a, 8, 2, 8);
    try table.deinit();
}
test "residency bounds, no allocation after init, nonwrapping identities and leases" {
    try t.expectError(error.InvalidOptions, r.Table.init(t.allocator, 0, 1, 0));
    try t.expectError(error.InvalidOptions, r.Table.init(t.allocator, 65536, 1, 1));
    try t.expectError(error.InvalidOptions, r.Table.init(t.allocator, 2, 0, 1));
    try t.expectError(error.InvalidOptions, r.Table.init(t.allocator, 2, 3, 1));
    try t.expectError(error.InvalidOptions, r.Table.init(t.allocator, 2, 1, 3));
    try t.checkAllAllocationFailures(t.allocator, allocated, .{});
    var fail = t.FailingAllocator.init(t.allocator, .{});
    var table = try r.Table.init(fail.allocator(), 3, 1, 3);
    defer table.deinit() catch @panic("pending residency");
    fail.fail_index = fail.alloc_index;
    fail.resize_fail_index = fail.resize_index;
    // Deliberately inject near-limit metadata, not billions of operations.
    table.entries[0].generation = std.math.maxInt(u64);
    const h = try table.reserve();
    try t.expectEqual(@as(u32, 1), h.index);
    try t.expectError(error.Busy, table.deinit());
    try table.saved(h, true);
    table.entries[h.index].serial = std.math.maxInt(u64);
    try t.expectError(error.SerialExhausted, table.spill(h));
    try invariants(&table);
    table.entries[h.index].serial = 0;
    table.entries[h.index].leases = std.math.maxInt(u32);
    try t.expectError(error.LeaseOverflow, table.acquireResident(h));
    table.entries[h.index].leases = 0;
    _ = try table.acquireResident(h);
    _ = try table.acquireResident(h);
    try t.expectError(error.Busy, table.deinit());
    try table.releaseResident(h);
    try t.expectError(error.Busy, table.drop(h));
    try table.releaseResident(h);
    const write = (try table.spill(h)).?;
    try t.expectError(error.Busy, table.deinit());
    try table.complete(write, true);
    table.entries[h.index].serial = std.math.maxInt(u64);
    try t.expectError(error.SerialExhausted, table.restore(h));
    try invariants(&table);
    table.entries[h.index].serial = write.serial;
    const read = (try table.restore(h)).?;
    try table.complete(read, true);
    try t.expectEqual(@as(?r.Transfer, null), try table.spill(h));
    try table.drop(h);
    try t.expectError(error.InvalidHandle, table.complete(read, true));
    const replacement = try table.reserve();
    try t.expect(replacement.generation != h.generation);
    try t.expectError(error.InvalidHandle, table.complete(read, false));
    try table.saved(replacement, false);
    try t.expectError(error.InvalidHandle, table.inspect(.{ .index = 3, .generation = 1 }));
    try t.expect(!fail.has_induced_failure);
}

fn wait(s: *storage.Store, ticket: storage.Ticket) !storage.Completion {
    for (0..10000) |_| {
        if (try s.poll(ticket)) |c| return c;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    @panic("pending disk I/O: do not free DMA-owned memory");
}

test "twelve snapshots survive two resident slots via worker and independent POSIX bytes" {
    const fixture = @embedFile("fixtures/storage/positional.bin");
    const block = 4096;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const staging = try t.allocator.alignedAlloc(u8, .fromByteUnits(block), 2 * block);
    defer t.allocator.free(staging);
    const hot = try t.allocator.alloc(u8, 2 * block);
    defer t.allocator.free(hot);
    const oracle = try t.allocator.alignedAlloc(u8, .fromByteUnits(block), block);
    defer t.allocator.free(oracle);
    const s = try storage.Store.create(t.allocator, .{ .dir_fd = tmp.dir.handle, .name = "residency", .file_bytes = 12 * block, .slot_bytes = block, .alignment = .{ .memory = block, .offset = block } }, staging);
    defer s.destroy() catch @panic("outstanding DMA");
    var table = try r.Table.init(t.allocator, 12, 2, 12);
    defer table.deinit() catch @panic("pending snapshots");
    var handles: [12]r.Handle = undefined;
    var moves: [2]r.Transfer = undefined;
    var tickets: [2]storage.Ticket = undefined;
    // Two writes pending together, acknowledged in reverse order. More records than RAM.
    for (0..6) |wave| {
        for (0..2) |j| {
            const i = wave * 2 + j;
            handles[i] = try table.reserve();
            const slot = (try table.inspect(handles[i])).hot.?;
            @memcpy(hot[slot * block ..][0..block], fixture[i * block ..][0..block]);
            try table.saved(handles[i], true);
            moves[j] = (try table.spill(handles[i])).?;
            tickets[j] = try s.acquire();
            @memcpy(try s.buffer(tickets[j]), hot[slot * block ..][0..block]);
            try s.submit(tickets[j], .write, moves[j].disk * block, block);
            try t.expectError(error.Busy, table.drop(handles[i]));
        }
        for (0..2) |k| {
            const j = 1 - k;
            const i = wave * 2 + j;
            try t.expect((try wait(s, tickets[j])).exact());
            try s.release(tickets[j]);
            // Independent positional read before publication. No transport self-oracle.
            try t.expectEqual(@as(usize, block), linux.pread(s.fd, oracle.ptr, block, moves[j].disk * block));
            try t.expectEqualSlices(u8, fixture[i * block ..][0..block], oracle);
            if (i == 5) {
                // Cancellation only after the actual write drained: keep the RAM source.
                try table.complete(moves[j], false);
                try t.expectEqual(r.Phase.resident, (try table.inspect(handles[i])).phase);
                try t.expectEqualSlices(u8, fixture[i * block ..][0..block], hot[moves[j].hot * block ..][0..block]);
                const stale = moves[j];
                moves[j] = (try table.spill(handles[i])).?;
                try t.expectError(error.InvalidTransfer, table.complete(stale, true));
                const retry = try s.acquire();
                @memcpy(try s.buffer(retry), hot[moves[j].hot * block ..][0..block]);
                try s.submit(retry, .write, moves[j].disk * block, block);
                try t.expect((try wait(s, retry)).exact());
                try s.release(retry);
            }
            try table.complete(moves[j], true);
            @memset(hot[moves[j].hot * block ..][0..block], 0xcc);
            try invariants(&table);
        }
    }
    try t.expectError(error.NoEntry, table.reserve());
    for (0..6) |wave| {
        for (0..2) |j| {
            const i = 11 - (wave * 2 + j);
            moves[j] = (try table.restore(handles[i])).?;
            tickets[j] = try s.acquire();
            try s.submit(tickets[j], .read, moves[j].disk * block, block);
            try t.expectError(error.Busy, table.acquireResident(handles[i]));
        }
        for (0..2) |k| {
            const j = 1 - k;
            const i = 11 - (wave * 2 + j);
            try t.expect((try wait(s, tickets[j])).exact());
            const bytes = try s.buffer(tickets[j]);
            try t.expectEqualSlices(u8, fixture[i * block ..][0..block], bytes);
            @memcpy(hot[moves[j].hot * block ..][0..block], bytes);
            try s.release(tickets[j]);
            try table.complete(moves[j], true);
            const slot = try table.acquireResident(handles[i]);
            try t.expectEqualSlices(u8, fixture[i * block ..][0..block], hot[slot * block ..][0..block]);
            try t.expectError(error.Busy, table.spill(handles[i]));
            try table.releaseResident(handles[i]);
            try t.expectEqual(@as(?r.Transfer, null), try table.spill(handles[i]));
            @memset(hot[slot * block ..][0..block], 0xdd);
            try invariants(&table);
        }
    }
    // Actual short completion: drain before rejecting destination, retain the disk record.
    try t.expectEqual(linux.E.SUCCESS, linux.errno(linux.ftruncate(s.fd, 0)));
    const read = (try table.restore(handles[0])).?;
    const ticket = try s.acquire();
    try s.submit(ticket, .read, read.disk * block, block);
    const completion = try wait(s, ticket);
    try t.expect(!completion.exact());
    try s.release(ticket);
    try table.complete(read, false);
    try t.expectEqual(r.Phase.disk, (try table.inspect(handles[0])).phase);
    try t.expectError(error.Busy, table.acquireResident(handles[0]));
    for (handles) |h| try table.drop(h);
    // Real negative write CQE: source must remain resident and byte-exact after drain.
    const h = try table.reserve();
    const hot_slot = (try table.inspect(h)).hot.?;
    @memcpy(hot[hot_slot * block ..][0..block], fixture[0..block]);
    try table.saved(h, true);
    const write = (try table.spill(h)).?;
    _ = linux.close(s.fd); // deliberate idle-fd fault injection, never production behavior
    s.fd = -1;
    const bad = try s.acquire();
    @memcpy(try s.buffer(bad), hot[hot_slot * block ..][0..block]);
    try s.submit(bad, .write, write.disk * block, block);
    const failed = try wait(s, bad);
    try t.expectEqual(-@as(i32, @intFromEnum(linux.E.BADF)), failed.result);
    try s.release(bad);
    try table.complete(write, false);
    const kept = try table.acquireResident(h);
    try t.expectEqualSlices(u8, fixture[0..block], hot[kept * block ..][0..block]);
    try table.releaseResident(h);
    try table.drop(h);
    try invariants(&table);
}
