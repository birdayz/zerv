//! Exact model archive adapter gate, optionally through the production disk owner.
//! Usage: MODEL PREFIX_TOKENS [SCRATCH_DIR ALIGNMENT]; all bytes and four vocabulary rows exact.
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const model = zerv.model;
const quantum = 1 << 20;
var max_poll_ns: i96 = 0;
pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 5 and args.len != 6 and args.len != 7 and args.len != 8) return error.Usage;
    const chunk_mib = if (args.len >= 7) try std.fmt.parseInt(u32, args[6], 10) else 1;
    const prepare_window = if (args.len == 8) try std.fmt.parseInt(u32, args[7], 10) else 1;
    const window = try zerv.serve.disk.Window.init(chunk_mib, 8 << 20);
    const demand_mode = args.len >= 6 and std.mem.eql(u8, args[5], "demand");
    const prepare_mode = args.len >= 6 and std.mem.eql(u8, args[5], "prepare");
    const disk_mode = args.len >= 5 and !(prepare_mode and std.mem.eql(u8, args[3], "-"));
    const cache_mode = disk_mode or prepare_mode;
    const prefill_deferred = args.len >= 6 and std.mem.eql(u8, args[5], "prefill-chunk");
    const prefill_mode = prefill_deferred or (args.len >= 6 and std.mem.eql(u8, args[5], "prefill"));
    const pressure_mode = prefill_mode or (args.len >= 6 and std.mem.eql(u8, args[5], "pressure"));
    const source_mode = demand_mode or prepare_mode or pressure_mode or (args.len >= 6 and std.mem.eql(u8, args[5], "source"));
    if (args.len >= 6 and !source_mode and !std.mem.eql(u8, args[5], "disk")) return error.Usage;
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
    try m.init(&device, &container, .{ .context = std.mem.alignForward(u32, n + 128, 128), .prefill_rows = 512, .prefill_precision = .f16, .embedding_memory = .host, .snapshots = if (cache_mode) 2 else 0, .snapshot_memory = .host, .kv_type = .f16, .slots = 2, .batch_rows = 2, .swap_bytes = if (source_mode) 2 << 30 else 0, .kv_share = true, .kv_pages = (n + 255) / 128 + 1 });
    defer m.deinit();
    var backend: zerv.serve.ModelBackend = .{ .m = m };
    defer backend.deinitDisk();
    defer backend.deinitCache(a);
    defer backend.deinitPreparation() catch @panic("preparation still owned");
    if (cache_mode) try backend.initCache(a, .radix, 128, source_mode);
    if (disk_mode) {
        const alignment = try std.fmt.parseInt(u32, args[4], 10);
        try backend.initDisk(a, init.io, .{ .directory = args[3], .bytes = std.mem.alignForward(u64, try m.archiveBytes(n), window.chunk_bytes) * 2, .records = 4, .chunk_mib = chunk_mib, .alignment = if (alignment == 0) null else .{ .memory = alignment, .offset = alignment } });
    }
    if (backend.disk_archive) |d| {
        try std.testing.expectEqual(window.chunk_bytes, d.store.slot_bytes);
        try std.testing.expectEqual(window.staging_bytes, d.memory.len);
        try std.testing.expectEqual(@as(usize, 8), d.store.slots.len);
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
    // Independent slot's solo logits, later replayed before acknowledging archive DMA.
    const other_tokens = [_]u32{ 17, 29 };
    const other_logits = try a.alloc(f32, other_tokens.len * model.config.vocab);
    defer a.free(other_logits);
    if (disk_mode) {
        try backend.reset(1);
        try m.ensurePages(1, other_tokens.len);
        for (other_tokens, 0..) |token, i| @memcpy(other_logits[i * model.config.vocab ..][0..model.config.vocab], try m.step(token));
        try backend.reset(1);
        try m.select(0);
    }
    var pack_tokens: [2][128]u32 = undefined;
    const pack_gold = try a.alloc(f32, 2 * model.config.vocab);
    defer a.free(pack_gold);
    var prefill_source_quanta: u64 = 0;
    if (prefill_mode or prepare_mode or demand_mode) {
        for (&pack_tokens, 0..) |*prompt, slot| {
            for (prompt, 0..) |*token, i| token.* = @intCast(31 + slot * 1000 + i * 7);
            try backend.reset(@intCast(slot));
            try m.ensurePages(@intCast(slot), prompt.len);
            @memcpy(pack_gold[slot * model.config.vocab ..][0..model.config.vocab], try m.prefill(prompt));
        }
        try backend.reset(1);
        try backend.reset(0);
    }
    try m.ensurePages(0, n + 4);
    const start = std.Io.Clock.awake.now(init.io);
    if (source_mode) {
        if (n <= 128) return error.Usage;
        _ = try m.prefill(tokens[0..128]);
        try backend.cache.?.checkpoint(backend.device(), 0, tokens[0..128]);
        if (pressure_mode) {
            const idle = try backend.pollMaintenance(false, false);
            try std.testing.expect(!idle.pending and !idle.progressed);
            try std.testing.expectEqual(@as(u64, 0), backend.disk_archive.?.catalog.stats.writes);
        }
        if (prepare_mode and disk_mode) {
            try std.testing.expect(try backend.disk_archive.?.startWrite(0, tokens[0..128]));
            _ = try drain(&backend, init.io, 0, false);
            @memcpy(other_logits[0..model.config.vocab], try m.step(tokens[128]));
            if (n > 129) _ = try m.prefill(tokens[129..]);
        } else _ = try m.prefill(tokens[128..]);
    } else _ = try m.prefill(tokens);
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
        if (source_mode) try m.state_layout.clearArchiveTail(m.snapshotBytes(), n, at, mem[0..count]);
        @memcpy(gold[at..][0..count], mem[0..count]);
        at += count;
    }
    const capture_ns = capture_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
    var disk_write_ns: i96 = 0;
    const continuation = [_]u32{ 11, 220, 42, 1000 };
    const logits = try a.alloc(f32, continuation.len * model.config.vocab);
    defer a.free(logits);
    if (demand_mode) {
        const d = backend.disk_archive orelse return error.Usage;
        try std.testing.expect(try d.startWrite(0, tokens));
        _ = try drain(&backend, init.io, 0, false);
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(0));
            @memcpy(logits[i * model.config.vocab ..][0..model.config.vocab], try m.step(t));
        }
        try backend.reset(0);
        while (backend.cache.?.evict(backend.device()) or backend.cache.?.evictHost(backend.device())) {}
        backend.demand_mode = .prefetch;
        backend.prefetch_chunks = prepare_window;
        const prompt = try a.alloc(u32, n + 1);
        defer a.free(prompt);
        @memcpy(prompt[0..n], tokens);
        prompt[n] = continuation[0];
        var queued: zerv.serve.batcher.Demand = .{ .slot = 1, .order = 7, .tokens = prompt };
        const starts = d.catalog.stats.device_starts;
        for (0..prepare_window + 1) |_| _ = try backend.pollDemand(queued, false, false);
        try std.testing.expectEqual(starts, d.catalog.stats.device_starts);
        try std.testing.expectEqual(prepare_window, d.read_ahead.occupancy(&d.catalog));
        try std.testing.expectEqual(@as(u32, 0), m.mappedPages(1));
        // Independent packed computation proceeds while this request has staging only.
        try m.ensurePages(0, pack_tokens[0].len);
        var unit = try m.prefillPackedSegment(&.{.{ .slot = 0, .tokens = &pack_tokens[0] }});
        while (!unit.done) {
            _ = try backend.pollDemand(queued, false, false);
            try std.testing.expectEqual(starts, d.catalog.stats.device_starts);
            try std.testing.expectEqual(@as(u32, 0), m.mappedPages(1));
            unit = try m.prefillPackedSegment(&.{});
        }
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(pack_gold[0..model.config.vocab]), std.mem.sliceAsBytes(unit.logits.?))) return error.WrongReadAheadPackedLogits;
        try backend.reset(0);
        // Generation reuse cancels the old disk pin; it cannot hand old tickets to a new begin.
        queued.order = 8;
        while (d.read_ahead.held != null) _ = try backend.pollDemand(queued, false, false);
        try std.testing.expectEqual(@as(u64, 1), d.read_ahead.cancellations);
        _ = try backend.pollDemand(queued, false, false);
        try std.testing.expectError(error.CacheReclaimPending, backend.beginWithDemand(1, 7, prompt));
        while (d.read_ahead.held != null) _ = try backend.pollDemand(null, true, false);
        try std.testing.expectEqual(@as(u64, 2), d.read_ahead.cancellations);
        queued.order = 9;
        for (0..prepare_window + 1) |_| _ = try backend.pollDemand(queued, false, false);
        try std.testing.expectError(error.PendingIo, backend.beginWithDemand(1, queued.order, prompt));
        try std.testing.expect(d.read_ahead.held == null);
        try std.testing.expectEqual(@as(u64, 1), d.read_ahead.handoffs);
        try std.testing.expectEqual(n, try drain(&backend, init.io, 1, false));
        try m.archiveMap(1, n, map, false);
        at = 0;
        while (at < gold.len) {
            const count = @min(quantum, gold.len - at);
            try m.archiveCopy(&c, &imported, 0, 1, n, map, at, count, false);
            try m.state_layout.clearArchiveTail(m.snapshotBytes(), n, at, mem[0..count]);
            if (!std.mem.eql(u8, gold[at..][0..count], mem[0..count])) return error.WrongReadAheadState;
            at += count;
        }
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(1));
            const got = try m.step(t);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongReadAheadLogits;
        }
        try backend.reset(1);
        // An unrelated live slot exhausts private target admission. Drain the old
        // reader before fallback, and suppress speculative restart for this identity.
        queued.order = 10;
        _ = try backend.pollDemand(queued, false, false);
        try m.ensurePages(0, m.state_layout.context);
        try std.testing.expectError(error.CacheReclaimPending, backend.beginWithDemand(1, queued.order, prompt));
        while (d.read_ahead.held != null) _ = try backend.pollDemand(queued, false, false);
        const attempts = d.read_ahead.starts;
        for (0..4) |_| _ = try backend.pollDemand(queued, false, false);
        try std.testing.expectEqual(attempts, d.read_ahead.starts);
        try std.testing.expectEqual(@as(u32, 8), d.store.freeSlots());
        try backend.reset(0);
        try std.testing.expectError(error.PendingIo, backend.beginWithDemand(1, queued.order, prompt));
        try std.testing.expectEqual(n, try drain(&backend, init.io, 1, false));
        try backend.reset(1);
        // Foreground reads preempt optional issue and drain the speculative record pin.
        queued.order = 11;
        _ = try backend.pollDemand(queued, false, false);
        const issued = d.read_ahead.submitted_bytes;
        while (d.read_ahead.held != null) _ = try backend.pollDemand(queued, false, true);
        _ = try backend.pollDemand(queued, false, true);
        try std.testing.expectEqual(issued, d.read_ahead.submitted_bytes);
        try std.testing.expectEqual(@as(u64, 4), d.read_ahead.cancellations);
        // Corrupt every digest so any disk completion order must fail before upload.
        // Exercise the production miss/reset path after a speculative handoff.
        queued.order = 12;
        _ = try backend.pollDemand(queued, false, false);
        const record = d.read_ahead.held.?.record;
        for (0..d.catalog.entries[record].chunks) |chunk| d.catalog.digests[d.catalog.max_chunks * record + chunk][0] ^= 1;
        const uploads = d.catalog.stats.device_starts;
        try std.testing.expectError(error.PendingIo, backend.beginWithDemand(1, queued.order, prompt));
        try std.testing.expectEqual(@as(u32, 0), try drain(&backend, init.io, 1, false));
        try std.testing.expectEqual(uploads, d.catalog.stats.device_starts);
        try std.testing.expectEqual(@as(u32, 0), m.slotPosition(1));
        try std.testing.expectEqual(@as(u32, 0), m.mappedPages(1));
        try std.testing.expectEqual(@as(u32, 0), d.catalog.entries[record].readers);
        try std.testing.expectEqual(@as(u64, 2), d.read_ahead.handoffs);
        _ = try backend.pollDemand(null, true, false);
        try std.testing.expectEqual(@as(u32, 8), d.store.freeSlots());
        // A source-held host prefix is a deferred hit, never silent cold recompute.
        backend.demand_mode = .protect;
        try backend.reset(0);
        try m.ensurePages(0, 129);
        _ = try m.prefill(tokens[0..128]);
        const cache = backend.cache.?;
        try cache.checkpoint(backend.device(), 0, tokens[0..128]);
        @memcpy(logits[0..model.config.vocab], try m.step(tokens[128]));
        try backend.reset(0);
        while (cache.evict(backend.device())) {}
        try std.testing.expect(try d.startSource(cache));
        _ = try d.pollSource(false);
        const short: zerv.serve.batcher.Demand = .{ .slot = 1, .order = 13, .tokens = tokens[0..129] };
        _ = try backend.pollDemand(short, false, false);
        try std.testing.expect(d.source_cancel_requested);
        try std.testing.expectError(error.CacheReclaimPending, backend.beginWithDemand(1, short.order, short.tokens));
        while (d.source != null) _ = try backend.pollMaintenance(false, false);
        try std.testing.expectEqual(@as(u32, 128), try backend.beginWithDemand(1, short.order, short.tokens));
        try std.testing.expect(try backend.grow(1));
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits[0..model.config.vocab]), std.mem.sliceAsBytes(try m.step(tokens[128])))) return error.WrongDemandHostHit;
        try backend.reset(1);
        _ = try backend.pollDemand(null, true, false);
        try std.testing.expectEqual(@as(u32, 8), d.store.freeSlots());
        try std.testing.expect(!backend.fatal.load(.acquire));
        std.debug.print("{{\"source_conflict\":true,\"admission_drain\":true,\"demand\":true,\"tokens\":{d},\"disk\":true,\"exact_state\":true,\"exact_vocab_rows\":4,\"exact_packed_rows\":1,\"window\":{d},\"handoffs\":2,\"cancellations\":4,\"corrupt_miss\":true,\"foreground_preemption\":true,\"submitted_bytes\":{d}}}\n", .{ n, prepare_window, d.read_ahead.submitted_bytes });
        return;
    }
    if (prepare_mode) {
        const cache = backend.cache.?;
        try cache.checkpoint(backend.device(), 0, tokens);
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(0));
            @memcpy(logits[i * model.config.vocab ..][0..model.config.vocab], try m.step(t));
        }
        try backend.reset(0);
        try std.testing.expectError(error.InvalidOptions, backend.initPreparation(init.io, .{ .target = m.pool.pages + 1 }));
        try std.testing.expectError(error.InvalidOptions, backend.initPreparation(init.io, .{ .target = 1, .host_headroom = m.swap_pages + 1 }));
        try std.testing.expect(backend.preparation_owner == null);
        try backend.initPreparation(init.io, .{ .target = m.pool.pages, .window = prepare_window, .host_headroom = m.swap_pages });
        const owner = &backend.preparation_owner.?;
        try std.testing.expect(!try owner.start()); // reserve is never evicted for preparation
        owner.options.host_headroom = 0;
        try std.testing.expect(!(try backend.pollMaintenance(false, true)).pending);
        try std.testing.expect(owner.held == null); // foreground read priority
        // Suppress the one-page ancestor to exercise multi-page suffix windows.
        const ancestor = cache.preparationCandidate(0) orelse return error.MissingCandidate;
        owner.failed_generation[ancestor.handle.index] = ancestor.handle.generation;
        try std.testing.expect(try owner.start());
        // Stop drains, never publishes, and retains the incarnation suppression.
        while (owner.held != null) {
            _ = try owner.poll(true, true);
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        try std.testing.expectEqual(@as(u64, 1), owner.counters.aborted);
        owner.failed_generation = @splat(0); // explicit diagnostic retry, not production policy
        owner.failed_generation[ancestor.handle.index] = ancestor.handle.generation;
        try std.testing.expect(try owner.start());
        const first_pages = owner.held.?.pages;
        try std.testing.expectEqual(@min(prepare_window, map.len - 1), first_pages);
        const first_moves = try m.pool.preparedMoves(owner.held.?.generation);
        const tail_freed: u64 = @intFromBool(n % 128 != 0 and first_moves[first_moves.len - 1].index == map.len - 1);
        try std.testing.expectError(error.InvalidSlot, backend.reset(m.state_layout.slots));
        try std.testing.expect(!backend.fatal.load(.acquire));
        if (backend.disk_archive) |d| {
            // A real foreground upload uses private target pages while preparation owns
            // a disjoint source/reservation. Neither optional issue nor publication occurs.
            try backend.reset(1);
            try m.ensurePages(1, 129);
            const record = d.catalog.lookup(tokens) orelse return error.MissingRecord;
            try d.startRead(1, record);
            while (true) {
                const read = try backend.pollCache(1, false);
                _ = try backend.pollMaintenanceInChunk(false, true);
                try std.testing.expect(owner.held != null);
                try std.testing.expectEqual(@as(u64, 0), owner.counters.committed);
                if (read.done) break;
                try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
            }
            try std.testing.expectEqual(@as(u32, 128), m.slotPosition(1));
            try m.select(1);
            const got = try m.step(tokens[128]);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(other_logits[0..model.config.vocab]))) return error.WrongConcurrentReadLogits;
            try backend.reset(1);
        }
        const prompt = try a.alloc(u32, n + 1);
        defer a.free(prompt);
        @memcpy(prompt[0..n], tokens);
        prompt[n] = continuation[0];
        try backend.reset(0);
        try std.testing.expectEqual(n, try cache.restore(backend.device(), 0, prompt));
        while (!owner.ready) {
            _ = try backend.pollMaintenanceInChunk(false, false);
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        try std.testing.expect(owner.held != null);
        try std.testing.expectEqual(@as(u64, 0), owner.counters.committed);
        _ = try backend.pollMaintenance(false, false);
        try std.testing.expectEqual(@as(u64, first_pages), owner.counters.committed);
        try std.testing.expectEqual(tail_freed, owner.counters.freed); // full GPU pages retain masks; partial hit copied privately
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(0));
            const got = try m.step(t);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongLivePreparationLogits;
        }
        try backend.reset(0);
        try std.testing.expectEqual(n, try cache.restore(backend.device(), 0, prompt));
        try m.archiveMap(0, n, map, false);
        at = 0;
        while (at < gold.len) {
            const count = @min(quantum, gold.len - at);
            try m.archiveCopy(&c, &imported, 0, 0, n, map, at, count, false);
            try m.state_layout.clearArchiveTail(m.snapshotBytes(), n, at, mem[0..count]);
            if (!std.mem.eql(u8, gold[at..][0..count], mem[0..count])) return error.WrongPreparationState;
            at += count;
        }
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(0));
            const got = try m.step(t);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongRestoredPreparationLogits;
        }
        try backend.reset(0);
        try std.testing.expect(try owner.start());
        const packed_pages = owner.held.?.pages;
        for (0..2) |slot| {
            try backend.reset(@intCast(slot));
            try m.ensurePages(@intCast(slot), pack_tokens[slot].len);
        }
        const items = [_]model.PackItem{ .{ .slot = 0, .tokens = &pack_tokens[0] }, .{ .slot = 1, .tokens = &pack_tokens[1] } };
        var pack_result = try m.prefillPackedSegment(&items);
        while (!pack_result.done) {
            _ = try backend.pollMaintenanceInChunk(false, false);
            try std.testing.expect(owner.held != null);
            try std.testing.expectEqual(@as(u64, first_pages), owner.counters.committed);
            pack_result = try m.prefillPackedSegment(&.{});
        }
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(pack_gold), std.mem.sliceAsBytes(pack_result.logits.?))) return error.WrongPreparationPackedLogits;
        while (owner.held != null) {
            _ = try owner.poll(false, true);
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        try std.testing.expectEqual(@as(u64, first_pages) + packed_pages, owner.counters.committed);
        try std.testing.expectEqual(tail_freed + packed_pages, owner.counters.freed);
        try backend.reset(0);
        try backend.reset(1);
        // The ancestor is still GPU-resident. A late descendant lease must make its
        // completed transaction abort atomically, without renaming or losing pages.
        owner.failed_generation = @splat(0);
        const suffix = cache.preparationCandidate(1);
        if (suffix) |candidate| owner.failed_generation[candidate.handle.index] = candidate.handle.generation;
        const before_host = m.hostFreePages();
        const before_gpu = m.freePages();
        try std.testing.expect(try owner.start());
        try std.testing.expectEqual(ancestor.handle.index, owner.held.?.source.lease.handle.index);
        const descendant = cache.sourceCandidate(1) orelse return error.MissingCandidate;
        const protection = try cache.acquireSource(descendant.handle);
        const stale_lease = owner.held.?.source.lease;
        const committed_before = owner.counters.committed;
        while (owner.held != null) {
            _ = try owner.poll(false, true);
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        try std.testing.expectEqual(committed_before, owner.counters.committed);
        try std.testing.expectEqual(@as(u64, 2), owner.counters.aborted);
        try std.testing.expectEqual(before_host, m.hostFreePages());
        try std.testing.expectEqual(before_gpu, m.freePages());
        try std.testing.expectError(error.InvalidSource, cache.releaseSource(stale_lease));
        try cache.releaseSource(protection.lease);
        std.debug.print("{{\"preparation\":true,\"tokens\":{d},\"disk\":{},\"exact_state\":true,\"exact_vocab_rows\":8,\"concurrent_read_rows\":{d},\"aborts\":2,\"committed\":{d},\"freed\":{d},\"window\":{d},\"first_pages\":{d},\"packed_pages\":{d},\"exact_packed_rows\":2}}\n", .{ n, disk_mode, @as(u32, @intFromBool(disk_mode)), owner.counters.committed, owner.counters.freed, prepare_window, first_pages, packed_pages });
        return;
    }
    if (source_mode) {
        const d = backend.disk_archive.?;
        const cache = backend.cache.?;
        try cache.checkpoint(backend.device(), 0, tokens);
        try std.testing.expect(try d.startSource(cache));
        // Submit the last KV quantum but don't acknowledge it. Advance the source
        // through its partial page while the fence is still owned, then cancel.
        while (true) {
            _ = try d.pollSource(false);
            if (d.catalog.device_pending) |held| {
                if (@as(u64, held.chunk) * d.store.slot_bytes + d.store.slot_bytes >= bytes) break;
            }
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        try std.testing.expectEqual(gpu.Commands.State.pending, d.commands.state);
        for (continuation, 0..) |t, i| {
            try std.testing.expect(try backend.grow(0));
            @memcpy(logits[i * model.config.vocab ..][0..model.config.vocab], try m.step(t));
        }
        try std.testing.expectError(error.Canceled, drainSource(d, init.io, true));
        try std.testing.expect(d.source == null);
        try m.releasePages(0);
        try backend.reset(0);
        try std.testing.expect(cache.evict(backend.device())); // host ancestor, GPU suffix
        const write_start = std.Io.Clock.awake.now(init.io);
        if (pressure_mode) {
            try std.testing.expect((try backend.pollMaintenance(false, false)).pending);
        } else try std.testing.expect(try d.startSource(cache));
        const view = d.source.?.view;
        try std.testing.expect(view.pages[0] & zerv.session.kvcache.Device.host_flag != 0);
        try std.testing.expect(view.pages[view.pages.len - 1] & zerv.session.kvcache.Device.host_flag == 0);
        while (d.commands.state != .pending) {
            if (pressure_mode) _ = try backend.pollMaintenance(false, false) else _ = try d.pollSource(false);
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        // Reuse the originating request slot while capture retains only cache ownership.
        try backend.reset(0);
        try m.ensurePages(0, other_tokens.len);
        for (other_tokens, 0..) |token, i| {
            const got = try m.step(token);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(other_logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongIndependentLogits;
        }
        if (prefill_mode) {
            for (0..2) |slot| {
                try backend.reset(@intCast(slot));
                try m.ensurePages(@intCast(slot), pack_tokens[slot].len);
            }
            const items = [_]model.PackItem{ .{ .slot = 0, .tokens = &pack_tokens[0] }, .{ .slot = 1, .tokens = &pack_tokens[1] } };
            var unit = try m.prefillPackedSegment(&items);
            var units: u32 = 1;
            const before_gpu = d.source_gpu_quanta;
            while (!unit.done) {
                // Mutable capture AND restore must still be rejected, even when source
                // capture is allowed. Rejection must happen before touching the command.
                try std.testing.expectError(error.InvalidState, m.archiveSubmit(&c, &imported, 0, 0, n, view.pages, 0, 4, false));
                try std.testing.expectError(error.InvalidState, m.archiveSubmit(&c, &imported, 0, 0, n, view.pages, 0, 4, true));
                const before = d.source_gpu_quanta + d.source_cpu_quanta;
                const read_pending = units % 3 == 0;
                if (!prefill_deferred) _ = try backend.pollMaintenanceInChunk(false, read_pending);
                if (read_pending) try std.testing.expectEqual(before, d.source_gpu_quanta + d.source_cpu_quanta);
                unit = try m.prefillPackedSegment(&.{});
                units += 1;
            }
            try std.testing.expectEqual(@as(u32, model.prefill_segments), units);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(pack_gold), std.mem.sliceAsBytes(unit.logits.?))) return error.WrongPackedLogits;
            prefill_source_quanta = d.source_gpu_quanta - before_gpu;
            try std.testing.expect(if (prefill_deferred) prefill_source_quanta == 0 else prefill_source_quanta > 0);
            try backend.reset(1);
            try backend.reset(0);
        }
        if (pressure_mode) {
            while ((try backend.pollMaintenance(false, false)).pending) try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        } else _ = try drainSource(d, init.io, false);
        disk_write_ns = write_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
        try std.testing.expect(d.source == null and d.source_cpu_quanta > 0 and d.source_gpu_quanta > 0);
        if (pressure_mode) {
            try std.testing.expect(d.catalog.containsReady(tokens));
            try std.testing.expectEqual(@as(u64, 1), d.catalog.stats.writes);
            try std.testing.expectEqual(@as(u32, 1), (try cache.sourceCapacity()).free_slots);
        } else try std.testing.expect(!try d.startSource(cache)); // clean backing: no rewrite
        try std.testing.expect(d.source == null);
    } else if (disk_mode) {
        const write_start = std.Io.Clock.awake.now(init.io);
        try backend.checkpoint(0, tokens);
        // Retain the old paused-write diagnostic independently of production admission.
        try std.testing.expect(try backend.disk_archive.?.startWrite(0, tokens));
        _ = try backend.pollCache(0, false); // submit capture, deliberately do not acknowledge
        try std.testing.expectEqual(gpu.Commands.State.pending, backend.disk_archive.?.commands.state);
        // An unrelated rejected operation must not misclassify that owned transfer as fatal.
        try std.testing.expectError(error.InvalidSlot, backend.reset(m.state_layout.slots));
        try std.testing.expect(!backend.fatal.load(.acquire));
        try backend.reset(1);
        try m.ensurePages(1, other_tokens.len);
        for (other_tokens, 0..) |token, i| {
            const got = try m.step(token);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(other_logits[i * model.config.vocab ..][0..model.config.vocab]))) return error.WrongIndependentLogits;
        }
        try backend.reset(1);
        try m.select(0);
        _ = try drain(&backend, init.io, 0, false);
        disk_write_ns = write_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
    }
    if (!source_mode) for (continuation, 0..) |t, i| {
        try std.testing.expect(try backend.grow(0));
        @memcpy(logits[i * model.config.vocab ..][0..model.config.vocab], try m.step(t));
    };
    try m.releasePages(0);
    // Eviction (not metadata destruction) releases hot pins; the archive must survive.
    if (backend.cache) |cache| while (cache.evict(backend.device()) or cache.evictHost(backend.device())) {};
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
        // Cancel after an actual GPU upload is submitted, not only a disk read.
        while (d.catalog.device_pending == null) {
            _ = try backend.pollCache(1, false);
            try std.Io.sleep(init.io, .fromMicroseconds(100), .awake);
        }
        try std.testing.expectEqual(gpu.Commands.State.pending, d.commands.state);
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
        const chunk = d.store.slot_bytes;
        const disk_offset = @as(i64, block) * @as(i64, @intCast(chunk));
        if (std.os.linux.pread(d.store.fd, d.memory.ptr, chunk, disk_offset) != chunk) return error.CorruptReadFailed;
        d.memory[0] ^= 0x80;
        const written = std.os.linux.pwrite(d.store.fd, d.memory.ptr, chunk, disk_offset);
        if (written != chunk) return error.CorruptWriteFailed;
        try std.testing.expectError(error.PendingIo, backend.begin(1, prompt));
        try std.testing.expectEqual(@as(u32, 0), try drain(&backend, init.io, 1, false));
        try std.testing.expectEqual(@as(u32, 0), m.mappedPages(1));
        try std.testing.expectEqual(null, d.catalog.lookup(prompt));
    }
    const stats = if (backend.disk_archive) |d| d.catalog.stats else zerv.session.archive.Stats{};
    std.debug.print("{{\"prefix\":{d},\"state_bytes\":{d},\"exact_state\":true,\"exact_vocab_rows\":4,\"elapsed_ns\":{d},\"capture_ns\":{d},\"restore_ns\":{d},\"disk_write_ns\":{d},\"disk\":{},\"independent_rows\":{d},\"device_starts\":{d},\"device_pending_polls\":{d},\"max_poll_ns\":{d},\"source\":{},\"source_cpu_quanta\":{d},\"source_gpu_quanta\":{d},\"pressure\":{},\"prefill_source_quanta\":{d},\"exact_packed_rows\":{d}}}\n", .{ n, bytes, start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds, capture_ns, restore_ns, disk_write_ns, disk_mode, if (disk_mode) @as(u32, 2) else 0, stats.device_starts, stats.device_pending_polls, max_poll_ns, source_mode, if (backend.disk_archive) |d| d.source_cpu_quanta else 0, if (backend.disk_archive) |d| d.source_gpu_quanta else 0, pressure_mode, prefill_source_quanta, if (prefill_mode) @as(u32, 2) else 0 });
}

fn drain(b: *zerv.serve.ModelBackend, io: std.Io, slot: u32, cancel: bool) !u32 {
    while (true) {
        const start = std.Io.Clock.awake.now(io);
        const p = try b.pollCache(slot, cancel);
        max_poll_ns = @max(max_poll_ns, start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
        if (p.done) return p.position;
        if (!p.progressed) try std.Io.sleep(io, .fromMicroseconds(100), .awake);
    }
}

fn drainSource(d: *zerv.serve.disk.Disk, io: std.Io, cancel: bool) !void {
    while (true) {
        const start = std.Io.Clock.awake.now(io);
        const p = try d.pollSource(cancel);
        max_poll_ns = @max(max_poll_ns, start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
        if (p.done) return;
        if (!p.progressed) try std.Io.sleep(io, .fromMicroseconds(100), .awake);
    }
}
