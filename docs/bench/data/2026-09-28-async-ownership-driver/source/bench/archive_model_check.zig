//! Exact model archive adapter gate, optionally through the production disk owner.
//! Usage: MODEL PREFIX_TOKENS [SCRATCH_DIR ALIGNMENT]; all bytes and four vocabulary rows exact.
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const model = zerv.model;
const quantum = 1 << 20;
pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 5) return error.Usage;
    const disk_mode = args.len == 5;
    const n = try std.fmt.parseInt(u32, args[2], 10);
    if (n == 0 or n > 80000) return error.Usage;
    var file = try zerv.artifact.MappedFile.open(init.io, args[1], 64 << 30);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 32 << 30, .cooperative_matrix = true, .subgroup_size_control = true, .storage16 = true, .pipeline_binaries = true, .host_import = true });
    defer device.deinit() catch @panic("live device");
    const m = try a.create(model.Model);
    defer a.destroy(m);
    try m.init(&device, &container, .{ .context = std.mem.alignForward(u32, n + 128, 128), .prefill_rows = 512, .prefill_precision = .f16, .embedding_memory = .host, .snapshots = if (disk_mode) 2 else 0, .snapshot_memory = .host, .kv_type = .f16, .slots = 2, .batch_rows = 2, .kv_share = true, .kv_pages = (n + 255) / 128 + 1 });
    defer m.deinit();
    var backend: zerv.serve.ModelBackend = .{ .m = m };
    defer backend.deinitDisk();
    defer backend.deinitCache(a);
    if (disk_mode) {
        try backend.initCache(a, .radix, 128, false);
        const alignment = try std.fmt.parseInt(u32, args[4], 10);
        try backend.initDisk(a, .{ .directory = args[3], .bytes = std.mem.alignForward(u64, try m.archiveBytes(n), quantum) * 2, .records = 4, .alignment = if (alignment == 0) null else .{ .memory = alignment, .offset = alignment } });
    }
    const mem = try a.alignedAlloc(u8, .fromByteUnits(65536), quantum);
    defer a.free(mem);
    if (device.host_import_alignment > 65536) return error.UnsupportedAlignment;
    var imported = try gpu.Buffer.initImported(&device, mem);
    defer imported.deinit() catch @panic("live import");
    var c = try gpu.Commands.init(&device);
    defer c.deinit() catch @panic("live copies");
    const tokens = try a.alloc(u32, n);
    defer a.free(tokens);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 150000);
    try m.ensurePages(0, n + 4);
    const start = std.Io.Clock.awake.now(init.io);
    _ = try m.prefill(tokens);
    const bytes = try m.archiveBytes(n);
    const gold = try a.alloc(u8, @intCast(bytes));
    defer a.free(gold);
    const map = try a.alloc(u32, (n + 127) / 128);
    defer a.free(map);
    try m.archiveMap(0, n, map, false);
    const capture_start = std.Io.Clock.awake.now(init.io);
    var at: usize = 0;
    while (at < gold.len) {
        const count = @min(quantum, gold.len - at);
        try m.archiveCopy(&c, &imported, 0, 0, n, map, at, count, false);
        @memcpy(gold[at..][0..count], mem[0..count]);
        at += count;
    }
    const capture_ns = capture_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
    var disk_write_ns: i96 = 0;
    if (disk_mode) {
        const write_start = std.Io.Clock.awake.now(init.io);
        try std.testing.expectError(error.PendingIo, backend.checkpoint(0, tokens));
        _ = try drain(&backend, init.io, 0, false);
        disk_write_ns = write_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
    }
    const continuation = [_]u32{ 11, 220, 42, 1000 };
    const logits = try a.alloc(f32, continuation.len * model.config.vocab);
    defer a.free(logits);
    for (continuation, 0..) |t, i| {
        try std.testing.expect(try backend.grow(0));
        @memcpy(logits[i * model.config.vocab ..][0..model.config.vocab], try m.step(t));
    }
    try m.releasePages(0);
    // Eviction (not metadata destruction) releases hot pins; the archive must survive.
    if (backend.cache) |cache| while (cache.evict(backend.device())) {};
    backend.deinitCache(a);
    // Reuse and poison every physical byte (including former source) before restoring.
    try c.reset();
    try c.begin();
    @memset(mem, 0xde);
    try c.barrier(.host, .transfer);
    try c.barrier(.compute, .transfer);
    var buffers: [33]*gpu.Buffer = undefined;
    buffers[0] = &m.state;
    for (m.kv[0..m.state_layout.kv_buffers], buffers[1..][0..m.state_layout.kv_buffers]) |*b, *p| p.* = b;
    for (buffers[0 .. m.state_layout.kv_buffers + 1]) |b| {
        var offset: u64 = 0;
        while (offset < b.size) : (offset += @min(quantum, b.size - offset)) try c.copy(&imported, 0, b, offset, @min(quantum, b.size - offset));
    }
    try c.barrier(.transfer, .compute);
    try c.end();
    try c.run(m.options.timeout_ns);
    const physical = try a.alloc(u32, map.len + 1);
    defer a.free(physical);
    for (physical, 0..) |*q, i| q.* = @intCast(physical.len - i - 1);
    try m.mapPages(1, physical);
    try m.archiveMap(1, n, map, true);
    try std.testing.expectError(error.InvalidRange, m.archiveCopy(&c, &imported, 0, 1, n, map, 1, 4, true));
    try std.testing.expectError(error.InvalidRange, m.archiveCopy(&c, &imported, 0, 1, n, map, bytes, 4, true));
    const restore_start = std.Io.Clock.awake.now(init.io);
    if (backend.disk_archive) |d| {
        const prompt = try a.alloc(u32, n + 1);
        defer a.free(prompt);
        @memcpy(prompt[0..n], tokens);
        prompt[n] = 11;
        const record = d.catalog.lookup(prompt) orelse return error.MissingRecord;
        try d.startRead(1, record);
        _ = try drain(&backend, init.io, 1, false);
    } else {
        at = 0;
        while (at < gold.len) {
            const count = @min(quantum, gold.len - at);
            @memcpy(mem[0..count], gold[at..][0..count]);
            try m.archiveCopy(&c, &imported, 0, 1, n, map, at, count, true);
            at += count;
        }
        try m.archiveRestored(1, n);
    }
    const restore_ns = restore_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
    at = 0;
    while (at < gold.len) {
        const count = @min(quantum, gold.len - at);
        @memset(mem, 0xa5);
        try m.archiveCopy(&c, &imported, 0, 1, n, map, at, count, false);
        if (!std.mem.eql(u8, gold[at..][0..count], mem[0..count])) return error.WrongState;
        at += count;
    }
    for (continuation, 0..) |t, i| {
        const got = try m.step(t);
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongLogits;
    }
    if (backend.disk_archive) |d| {
        const prompt = try a.alloc(u32, n + 1);
        defer a.free(prompt);
        @memcpy(prompt[0..n], tokens);
        prompt[n] = 11;
        // Exercise production begin, cancellation drain, successful retry, and cold
        // fallback after physical corruption, not only the owner copy callback.
        try std.testing.expectError(error.PendingIo, backend.begin(1, prompt));
        _ = try backend.pollCache(1, false); // one submitted read
        try std.testing.expectError(error.Canceled, drain(&backend, init.io, 1, true));
        try std.testing.expectEqual(@as(u32, 0), m.slotPosition(1));
        try std.testing.expectError(error.PendingIo, backend.begin(1, prompt));
        const restored = try drain(&backend, init.io, 1, false);
        try std.testing.expectEqual(n, restored);
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(1));
            const got = try m.step(t);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongRetryLogits;
        }
        const record = d.catalog.lookup(prompt) orelse return error.MissingRecord;
        // Corrupt the LAST chunk: preceding chunks can already have uploaded state.
        const block = d.catalog.blocks[@as(usize, record) * d.catalog.max_chunks + d.catalog.entries[record].chunks - 1];
        const disk_offset = @as(i64, block) * quantum;
        if (std.os.linux.pread(d.store.fd, d.memory.ptr, quantum, disk_offset) != quantum) return error.CorruptReadFailed;
        d.memory[0] ^= 0x80;
        const written = std.os.linux.pwrite(d.store.fd, d.memory.ptr, quantum, disk_offset);
        if (written != quantum) return error.CorruptWriteFailed;
        try std.testing.expectError(error.PendingIo, backend.begin(1, prompt));
        try std.testing.expectEqual(@as(u32, 0), try drain(&backend, init.io, 1, false));
        try std.testing.expectEqual(@as(u32, 0), m.mappedPages(1));
        try std.testing.expectEqual(null, d.catalog.lookup(prompt));
    }
    std.debug.print("{{\"prefix\":{d},\"state_bytes\":{d},\"exact_state\":true,\"exact_vocab_rows\":4,\"elapsed_ns\":{d},\"capture_ns\":{d},\"restore_ns\":{d},\"disk_write_ns\":{d},\"disk\":{}}}\n", .{ n, bytes, start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds, capture_ns, restore_ns, disk_write_ns, disk_mode });
}

fn drain(b: *zerv.serve.ModelBackend, io: std.Io, slot: u32, cancel: bool) !u32 {
    while (true) {
        const p = try b.pollCache(slot, cancel);
        if (p.done) return p.position;
        if (!p.progressed) try std.Io.sleep(io, .fromMicroseconds(100), .awake);
    }
}
