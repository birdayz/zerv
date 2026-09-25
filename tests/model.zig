const std = @import("std");
const zerv = @import("zerv");
const config = zerv.model.config;
const layout = zerv.model.layout;
const t = std.testing;

fn artifactBytes(s: *const config.Spec, format: zerv.matvec.Format) u64 {
    const shape: zerv.matvec.Shape = .{ .format = format, .columns = @intCast(s.k), .rows = @intCast(s.rows) };
    return shape.weightBytes() catch unreachable;
}

fn actualFormat(s: *const config.Spec) zerv.matvec.Format {
    const name = config.specName(s);
    if (s.role != .matrix) return if (s.role == .embedding) .q4_0 else .f32;
    if (std.mem.eql(u8, name, "output.weight")) return .q6_k;
    if (std.mem.endsWith(u8, name, "ssm_out.weight")) return .q5_k;
    if (std.mem.endsWith(u8, name, "ssm_alpha.weight") or std.mem.endsWith(u8, name, "ssm_beta.weight")) return .f32;
    if (std.mem.endsWith(u8, name, "ffn_down.weight")) {
        const il = std.fmt.parseInt(u32, name[4..std.mem.indexOfScalarPos(u8, name, 4, '.').?], 10) catch unreachable;
        if (il < 8) return .q4_1;
    }
    return .q4_0;
}

test "qwen35 inventory is complete, unique and matches the verified artifact totals" {
    const specs = config.tensors();
    try t.expectEqual(@as(usize, 3 + 16 * 11 + 48 * 14), specs.len);
    var total: u64 = 0;
    var names: std.StringHashMapUnmanaged(void) = .empty;
    defer names.deinit(t.allocator);
    for (&specs) |*s| {
        try t.expect(!names.contains(config.specName(s)));
        try names.put(t.allocator, config.specName(s), {});
        total += artifactBytes(s, actualFormat(s));
    }
    // Observed GGUF: all tensors before blk.64 occupy exactly these payload bytes.
    try t.expectEqual(@as(u64, 15780284416), total);
    try t.expect(config.isAttention(3) and config.isAttention(63) and !config.isAttention(0) and !config.isAttention(62));
    try t.expectEqual(@as(u32, 15), config.attentionIndex(63));
    try t.expectEqual(@as(u32, 47), config.linearIndex(62));
    try t.expectEqual(@as(u32, 0), config.linearIndex(0));
}

test "weight banks: params first in bank 0, aligned, bounded and non-overlapping" {
    const specs = config.tensors();
    var items: [config.tensor_count]layout.Item = undefined;
    for (&specs, &items) |*s, *item| item.* = .{ .role = s.role, .bytes = artifactBytes(s, actualFormat(s)) };
    var placements: [config.tensor_count]layout.Placement = undefined;
    const capacity: u64 = 0xf000_0000;
    const banks = try layout.place(&items, capacity, &placements);
    try t.expectEqual(@as(u8, 4), banks.count);
    var params_end: u64 = 0;
    var first_matrix: u64 = std.math.maxInt(u64);
    for (items, placements) |item, p| {
        try t.expect(p.offset % 32 == 0 and p.offset + item.bytes <= banks.bytes[p.bank] and banks.bytes[p.bank] <= capacity);
        if (item.role != .matrix) {
            try t.expectEqual(@as(u8, 0), p.bank);
            params_end = @max(params_end, p.offset + item.bytes);
        } else if (p.bank == 0) first_matrix = @min(first_matrix, p.offset);
    }
    try t.expect(params_end <= first_matrix);
    // No overlaps within a bank.
    for (items, placements, 0..) |a, pa, i| for (items[i + 1 ..], placements[i + 1 ..]) |b, pb| {
        if (pa.bank == pb.bank) try t.expect(pa.offset + a.bytes <= pb.offset or pb.offset + b.bytes <= pa.offset);
    };
    try t.expectError(error.BankOverflow, layout.place(&items, 64 * 1024 * 1024, &placements));
    const tiny = [_]layout.Item{ .{ .role = .matrix, .bytes = 100 }, .{ .role = .matrix, .bytes = 100 } };
    var tiny_out: [2]layout.Placement = undefined;
    try t.expectError(error.BankOverflow, layout.place(&tiny, 96, &tiny_out));
    const b = try layout.place(&tiny, 128, &tiny_out);
    try t.expectEqual(@as(u8, 2), b.count);
    try t.expectEqual(@as(u8, 1), tiny_out[1].bank);
}

test "arena layouts: context bounds and disjoint regions" {
    const capacity: u64 = 0xf000_0000;
    // One cache (a layer's K and V, 8 KiB per token) must fit one KV buffer.
    const max = layout.maxContext(capacity, .f32, 256);
    try t.expectEqual(@as(u32, 491520), max);
    try t.expectError(error.ContextTooLarge, layout.state(max + 2, capacity, false, .f32, 256));
    try t.expectError(error.InvalidContext, layout.state(0, capacity, false, .f32, 256));
    // Capacity is counted in whole pages; a partial last page is allocated in full.
    try t.expectEqual(@as(u32, 256), layout.maxContext(5 * 512 * 1024, .f32, 256));
    try t.expectEqual(@as(u32, 2), (try layout.state(258, capacity, false, .f32, 256)).pages);
    // Page 0: one page of the context, the layout before paging (per-token capacity).
    try t.expectEqual(@as(u32, 320), layout.maxContext(5 * 512 * 1024, .f32, 0));
    try t.expectEqual(max, layout.maxContext(capacity, .f32, 0));
    {
        const one = try layout.state(8194, capacity, false, .f32, 0);
        try t.expectEqual(@as(u32, 1), one.pages);
        try t.expectEqual(@as(u32, 8194), one.page);
        // Cache i at i x cache elements; V after K: the pre-paging offsets.
        try t.expectEqual(one.kcache(1), one.vcache(0) + 4 * 256 * 8194);
        try t.expectEqual(@as(u64, one.pstride(0)), one.kvElements(0));
        try t.expectEqual(@as(u64, 16) * 2 * 1024 * 8194 * 4, one.kvBytes());
    }
    // Page sizes: multiples of 128 up to max_kv_page; a page may exceed the context.
    const p128 = try layout.state(8192, capacity, false, .f32, 128);
    try t.expectEqual(@as(u32, 64), p128.pages);
    try t.expectEqual(@as(u32, 1), (try layout.state(8192, capacity, false, .f32, 16384)).pages);
    try t.expectError(error.InvalidKvPage, layout.state(8192, capacity, false, .f32, 192));
    try t.expectError(error.InvalidKvPage, layout.state(8192, capacity, false, .f32, 100));
    try t.expectError(error.InvalidKvPage, layout.state(8192, capacity, false, .f32, layout.max_kv_page + 128));
    try t.expectEqual(@as(u32, 64), layout.ptabWords(8192));
    try t.expectEqual(@as(u32, 65), layout.ptabWords(8193));
    {
        // 8k tokens: one KV buffer holds all 16 caches.
        const s = try layout.state(8192, capacity, false, .f32, 256);
        try t.expectEqual(@as(u32, 1), s.kv_buffers);
        try t.expectEqual(@as(u32, 16), s.per_buffer);
        try t.expectEqual(s.conv, layout.State.ssm_words);
        try t.expectEqual(@as(u64, layout.State.ssm_words + layout.State.conv_words), s.words);
        try t.expect(s.hp == null);
        // Pages of 256 tokens (docs/specs/concurrent.md, "Addressing"): in every page,
        // cache i's K at i x page_piece, its V after the K.
        try t.expectEqual(@as(u32, 32), s.pages);
        try t.expectEqual(@as(u32, 0), s.kcache(0));
        try t.expectEqual(@as(u32, 256), s.page);
        try t.expectEqual(s.kcache(1), s.vcache(0) + 4 * 256 * 256);
        try t.expectEqual(s.vcache(15) + 4 * 256 * 256, s.pstride(15));
        try t.expectEqual(s.pstride(0), s.pstride(15));
        try t.expectEqual(@as(u64, s.pstride(0)) * s.pages, s.kvElements(0));
        try t.expectEqual(@as(u64, 16) * 2 * 1024 * 8192 * 4, s.kvBytes());
    }
    {
        // 64k tokens: a cache is 512 MiB, 7 per 3.75 GiB buffer; 16 caches in 3 buffers.
        const s = try layout.state(65536, capacity, false, .f32, 256);
        try t.expectEqual(@as(u32, 7), s.per_buffer);
        try t.expectEqual(@as(u32, 3), s.kv_buffers);
        try t.expectEqual(@as(u32, 2), s.kvBuffer(15));
        try t.expectEqual(s.kcache(14), s.kcache(7));
        try t.expectEqual(@as(u32, 0), s.kcache(7));
        try t.expectEqual(2 * s.cacheElements(), s.kvElements(2));
        try t.expectEqual(7 * s.piece(), s.pstride(6));
        try t.expectEqual(2 * s.piece(), s.pstride(14));
        try t.expect(s.kvBufferBytes(0) <= capacity);
    }
    {
        // MTP: a 17th cache and the pending h after conv. 32k tokens (the old cap was
        // 27,786): caches of 256 MiB, 15 per buffer, 2 buffers.
        const m = try layout.state(32768, capacity, true, .f32, 256);
        try t.expectEqual(@as(u32, 17), m.caches);
        try t.expectEqual(@as(u32, layout.State.ssm_words + layout.State.conv_words), m.hp.?);
        try t.expectEqual(@as(u64, m.hp.? + 5120), m.words);
        try t.expectEqual(@as(u32, 15), m.per_buffer);
        try t.expectEqual(@as(u32, 2), m.kv_buffers);
        try t.expectEqual(m.kvBuffer(layout.mtp_attention), m.kv_buffers - 1);
        // A forced small buffer: 3 caches each, 6 buffers, the last holding 2.
        const f = try layout.state(1024, 3 * 2 * 1024 * 1024 * 4, true, .f32, 256);
        try t.expectEqual(@as(u32, 6), f.kv_buffers);
        try t.expectEqual(2 * f.cacheElements(), f.kvElements(5));
        try t.expectEqual(f.vcache(2) + 4 * 256 * 256, f.pstride(0));
        try t.expectEqual(@as(u64, f.pstride(0)) * f.pages, f.kvElements(0));
        try t.expectError(error.ContextTooLarge, layout.state(1026, 2 * 1024 * 1024 * 4, false, .f32, 256));
    }
    {
        // f16 KV (docs/specs/model.md, "KV precision"): half the bytes, offsets in halves,
        // twice the tokens per buffer; an even context.
        try t.expectEqual(@as(u32, 2 * 491520), layout.maxContext(capacity, .f16, 256));
        const h = try layout.state(65536, capacity, false, .f16, 256);
        const w = try layout.state(65536, capacity, false, .f32, 256);
        try t.expectEqual(w.cacheElements(), h.cacheElements());
        try t.expectEqual(w.kvBytes() / 2, h.kvBytes());
        // 256 MiB per cache: 15 per 3.75 GiB buffer (f32: 7 at this context).
        try t.expectEqual(@as(u32, 15), h.per_buffer);
        try t.expectEqual(@as(u32, 2), h.kv_buffers);
        try t.expectEqual(h.vcache(0) + 4 * 256 * 256, h.kcache(1));
        try t.expectEqual(h.kvElements(0) * 2, h.kvBufferBytes(0));
        try t.expect(h.kvBufferBytes(0) <= capacity);
        try t.expectError(error.InvalidContext, layout.state(8191, capacity, false, .f16, 256));
        try t.expectError(error.InvalidContext, layout.state(8191, capacity, false, .f32, 256));
        try t.expectError(error.ContextTooLarge, layout.state(2 * 491520 + 2, capacity, false, .f16, 256));
    }
    for ([_]u32{ 1, 13, 512 }) |rows| {
        const a = try layout.act(8192, rows, 1000, false, 1, false);
        // Regions are ordered, 64-word aligned and each holds `rows` rows of its width.
        var previous: u64 = 0;
        inline for (layout.widths, 0..) |entry, i| {
            const start: u64 = @field(a, entry[0]);
            try t.expect(start % 64 == 0);
            if (i > 0) try t.expect(start >= previous);
            previous = start + @as(u64, entry[1]) * rows;
        }
        try t.expect(a.scores >= previous);
        // Scores: decode attention's rows only (prefill attention is fused).
        try t.expectEqual(@as(u64, a.part), a.scores + 24 * 8192);
        // Decode attention scratch: 128 chunks of 64 keys per head.
        try t.expectEqual(@as(u32, 128), a.chunks);
        try t.expectEqual(@as(u64, a.amax), a.part + std.mem.alignForward(u64, 1000, 64));
        try t.expectEqual(@as(u64, a.apart), a.amax + 24 * 128);
        try t.expectEqual(@as(u64, a.asum), a.apart + 24 * 128 * 256);
        // The per-(row, head) global maxima (block 17c) follow.
        try t.expectEqual(@as(u64, a.gmax), a.asum + 24 * 128);
        // 8-chunk block partials (16 blocks) follow.
        try t.expectEqual(@as(u32, 16), a.blocks);
        try t.expectEqual(@as(u64, a.bpart), a.gmax + 64);
        try t.expectEqual(@as(u64, a.bsum), a.bpart + 24 * 16 * 256);
        // The page table (32 pages, one 64-word block) ends the arena.
        try t.expectEqual(@as(u64, a.ptab), @as(u64, a.bsum) + 24 * 16);
        try t.expectEqual(a.words, @as(u64, a.ptab) + 64);
        try t.expectEqual(rows, a.rows);
        try t.expect(a.x16 == null);
        // f16 mode: the f16 input copy (rows x ffn halves) is last, 16-byte aligned; the
        // other regions do not move.
        const h = try layout.act(8192, rows, 1000, true, 1, false);
        try t.expectEqual(a.asum, h.asum);
        try t.expectEqual(@as(u64, h.x16.?), std.mem.alignForward(u64, a.words, 64));
        try t.expectEqual(h.words, @as(u64, h.x16.?) + std.mem.alignForward(u64, 8704 * @as(u64, rows), 64));
        try t.expect(a.spec == null and a.decode_rows == 1);
        if (rows >= 5) {
            // Speculative verification: 5 rows of decode attention scratch, then the verify
            // slots (48 linear layers x 5 rows x 26720 words); earlier regions do not move.
            const v = try layout.act(8192, rows, 1000, false, 5, false);
            try t.expectEqual(@as(u64, v.part), v.scores + 24 * 5 * 8192);
            try t.expectEqual(@as(u64, v.amax), v.part + std.mem.alignForward(u64, 1000, 64));
            try t.expectEqual(@as(u64, v.apart), v.amax + 24 * 128 * 5);
            try t.expectEqual(@as(u64, v.asum), v.apart + 24 * 128 * 256 * 5);
            try t.expectEqual(@as(u64, v.gmax), v.asum + 24 * 128 * 5);
            try t.expectEqual(@as(u64, v.bpart), v.gmax + std.mem.alignForward(u64, 24 * 5, 64));
            try t.expectEqual(@as(u64, v.spec.?), v.bsum + 24 * 16 * 5);
            try t.expectEqual(@as(u64, v.ptab), @as(u64, v.spec.?) + std.mem.alignForward(u64, 48 * 5 * 26720, 64));
            try t.expectEqual(v.words, @as(u64, v.ptab) + 64);
            const slot = v.specSlot(47);
            try t.expectEqual(slot.mixed, v.spec.? + 47 * 5 * 26720);
            try t.expectEqual(slot.beta + 5 * 48, v.spec.? + 48 * 5 * 26720);
            try t.expectError(error.InvalidContext, layout.act(8192, rows, 1000, false, rows + 1, false));
            // MTP regions after the verify slots; the earlier regions do not move.
            const m = try layout.act(8192, rows, 1000, false, 5, true);
            try t.expectEqual(v.spec, m.spec);
            const mr = m.mtp.?;
            try t.expectEqual(mr.cat, v.ptab);
            try t.expectEqual(@as(u64, mr.mh), mr.cat + std.mem.alignForward(u64, 10240 * @as(u64, rows), 64));
            try t.expectEqual(mr.mo, mr.mh + 5 * 5120);
            try t.expectEqual(mr.hrows, mr.mo + 5 * 5120);
            try t.expectEqual(@as(u64, mr.logits), mr.hrows + std.mem.alignForward(u64, 5120 * (@as(u64, rows) + 1), 64));
            try t.expectEqual(mr.part, mr.logits + 248320);
            try t.expectEqual(m.ptab, mr.part + 768);
            try t.expectEqual(m.words, @as(u64, m.ptab) + 64);
            try t.expect(v.mtp == null);
        }
    }
    try t.expectError(error.InvalidContext, layout.act(0, 1, 0, false, 1, false));
    try t.expectError(error.InvalidContext, layout.act(8192, 8, 0, false, 1, true));
    try t.expectError(error.InvalidContext, layout.act(8192, 0, 0, false, 1, false));
    try t.expectEqual(@as(u32, 2), (try layout.act(65, 1, 0, false, 1, false)).chunks);
    try t.expectEqual(@as(u32, 1), (try layout.act(64, 1, 0, false, 1, false)).chunks);
    // Without a prefill score matrix, context x chunk no longer bounds the arena (262144
    // x 512 fits); only decode's 24 x decode rows x context does.
    _ = try layout.act(262144, 512, 0, false, 5, false);
    try t.expectError(error.ContextTooLarge, layout.act(50_000_000, 1, 0, false, 1, false));
    // io block: prefill token ids and RoPE rows follow the logits.
    try t.expectEqual(@as(u32, layout.io.logits + 248320), layout.io.tokens);
    try t.expectEqual(layout.io.words(512), @as(u64, layout.io.tokens) + 512 + 512 * 64 + layout.io.spec_words + layout.io.batch_max * layout.io.slot_entry);
    // MTP words after the prefill RoPE rows: positions, tokens, drafts, sources, RoPE rows.
    const sp = layout.io.spec(512);
    try t.expectEqual(layout.io.rope(512) + 512 * 64, sp.pos);
    try t.expectEqual(sp.pos + 9, sp.tok);
    try t.expectEqual(sp.tok + 5, sp.draft);
    try t.expectEqual(sp.draft + 4, sp.prob);
    try t.expectEqual(sp.prob + 4, sp.src);
    try t.expect(sp.src + 3 <= sp.rope);
    try t.expectEqual(layout.io.batch(512), sp.rope + 9 * 64);
    // Slot entries (docs/specs/concurrent.md, "18b.2 design"): the single-slot entry in the
    // control words after sin, before the logits; the batch table last.
    try t.expectEqual(@as(u32, layout.io.sin + 32), layout.io.slot);
    try t.expect(layout.io.slot + layout.io.slot_entry <= layout.io.logits);
    try t.expectEqual(layout.io.words(512), @as(u64, layout.io.batch(512)) + 32 * 4);
}

test "slots: state arena slot strides, KV pool pages and per-slot page tables" {
    const capacity: u64 = 0xf000_0000;
    const one = try layout.state(8192, capacity, false, .f32, 128);
    // One slot: exactly the single-sequence layout.
    const s1 = try layout.stateWith(8192, capacity, false, .f32, 128, 1, 0);
    try t.expectEqual(one.words, s1.words);
    try t.expectEqual(one.pages, s1.pages);
    try t.expectEqual(@as(u32, 64), s1.seq_pages);
    // Four slots: slot s at s x slot_words (slot 0's words rounded up to 64); the pool is
    // 4 x the pages of one context by default, or as given.
    const s4 = try layout.stateWith(8192, capacity, false, .f32, 128, 4, 0);
    try t.expectEqual(std.mem.alignForward(u64, layout.State.ssm_words + layout.State.conv_words, 64), s4.slot_words);
    try t.expectEqual(3 * @as(u64, s4.slot_words) + one.words, s4.words);
    try t.expectEqual(@as(u32, 256), s4.pages);
    try t.expectEqual(@as(u32, 64), s4.seq_pages);
    try t.expectEqual(4 * one.kvBytes(), s4.kvBytes());
    const pool = try layout.stateWith(8192, capacity, true, .f16, 128, 8, 100);
    try t.expectEqual(@as(u32, 100), pool.pages);
    try t.expectEqual(pool.slot_words, std.mem.alignForward(u32, layout.State.ssm_words + layout.State.conv_words + 5120, 64));
    // The pool must hold one full sequence; slots are 1..max_slots.
    try t.expectError(error.InvalidSlots, layout.stateWith(8192, capacity, false, .f32, 128, 2, 63));
    try t.expectError(error.InvalidSlots, layout.stateWith(8192, capacity, false, .f32, 128, 0, 0));
    try t.expectError(error.InvalidSlots, layout.stateWith(8192, capacity, false, .f32, 128, layout.max_slots + 1, 0));
    // A pool too large for one KV buffer (one sequence's cache is 64 MiB, the pool's 128).
    _ = try layout.stateWith(8192, 100 << 20, false, .f32, 128, 1, 0);
    try t.expectError(error.ContextTooLarge, layout.stateWith(8192, 100 << 20, false, .f32, 128, 2, 0));
    // Page tables: ptabWords(context) per slot; the regions before them do not move.
    const a1 = try layout.act(8192, 16, 1000, false, 5, false);
    const a4 = try layout.actWith(8192, 16, 1000, false, 5, false, true, 4);
    try t.expectEqual(a1.ptab, a4.ptab);
    try t.expectEqual(@as(u32, 64), a4.ptab_words);
    // Then the batch's own r/f rows (2 x decode rows x hidden), only with several slots.
    try t.expectEqual(a4.brf.?, a4.ptab + 256);
    try t.expectEqual(a4.words, @as(u64, a4.brf.?) + 2 * 5 * 5120);
    try t.expect(a1.brf == null);
    const ab = a4.batch();
    try t.expectEqual(a4.brf.?, ab.r);
    try t.expectEqual(a4.brf.? + 5 * 5120, ab.f);
    try t.expectEqual(a4.h, ab.h);
    try t.expectEqual(a1.r, a1.batch().r);
    // Batched decode without verify: the attention scratch rows, no verify slots.
    const b = try layout.actWith(8192, 16, 1000, false, 8, false, false, 4);
    try t.expect(b.spec == null);
    try t.expectEqual(@as(u32, 8), b.decode_rows);
    try t.expectError(error.InvalidContext, layout.actWith(8192, 16, 1000, false, 1, false, true, 1));
    try t.expectError(error.InvalidSlots, layout.actWith(8192, 16, 1000, false, 1, false, false, 0));
}

test "fitContext is the largest fitting multiple of 32" {
    const shape: zerv.model.ContextShape = .{ .kv_capacity = 0xf000_0000, .rows = 512, .part_words = 1 << 20, .x16 = false, .decode_rows = 4, .mtp = true };
    const room: u64 = 6 * 1024 * 1024 * 1024;
    const got = zerv.model.fitContext(room, 262144, shape);
    try t.expect(got > 0 and got % 32 == 0);
    try t.expect(try zerv.model.contextBytes(got, shape) <= room);
    try t.expect(try zerv.model.contextBytes(got + 32, shape) > room);
    // The limit binds; no room gives 0.
    try t.expectEqual(@as(u32, 4096), zerv.model.fitContext(room, 4096, shape));
    try t.expectEqual(@as(u32, 0), zerv.model.fitContext(1024, 262144, shape));
}

test "qwen35 metadata gate rejects other artifacts" {
    var file = try zerv.artifact.MappedFile.open(t.io, "tests/fixtures/gguf/default.gguf", 4096);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(t.allocator, file.bytes, .{});
    defer container.deinit();
    try t.expectError(error.UnsupportedModel, config.hyper(&container));
}

test "tensor checks reject wrong shapes, types and embedding formats" {
    const specs = config.tensors();
    const bytes: [4]u8 = @splat(0);
    for (&specs) |*s| {
        var tensor: zerv.artifact.gguf.Tensor = .{ .name = config.specName(s), .kind = .f32, .rank = if (s.rows == 1) 1 else 2, .dims = .{ s.k, s.rows, 1, 1 }, .offset = 0, .size = 4, .data = &bytes };
        tensor.kind = switch (s.role) {
            .param => .f32,
            .embedding => .q4_0,
            .matrix => .q4_0,
        };
        try config.check(s, &tensor);
        var wrong = tensor;
        wrong.dims[0] += 32;
        try t.expectError(error.WrongTensorShape, config.check(s, &wrong));
        wrong = tensor;
        wrong.dims[2] = 2;
        try t.expectError(error.WrongTensorShape, config.check(s, &wrong));
        wrong = tensor;
        wrong.kind = if (s.role == .param) .q4_0 else if (s.role == .embedding) .q4_1 else .q8_0;
        try t.expectError(error.WrongTensorType, config.check(s, &wrong));
        wrong.kind = .f16;
        try t.expectError(error.WrongTensorType, config.check(s, &wrong));
    }
    try t.expectError(error.WrongTensorType, config.matrixFormat(.bf16));
    try t.expectEqual(zerv.matvec.Format.q5_k, try config.matrixFormat(.q5_k));
    // The MTP inventory: 15 blk.64 tensors; only eh_proj may be Q8_0 (multi-row only).
    const mtp = config.mtpTensors();
    var rows_only: u32 = 0;
    for (&mtp) |*s| {
        try t.expect(std.mem.startsWith(u8, config.specName(s), "blk.64."));
        if (s.rows_only) {
            rows_only += 1;
            try t.expectEqualStrings("blk.64.nextn.eh_proj.weight", config.specName(s));
            try t.expectEqual(@as(u64, 10240), s.k);
            var q8: zerv.artifact.gguf.Tensor = undefined;
            q8.kind = .q8_0;
            q8.rank = 2;
            q8.dims = .{ 10240, 5120, 1, 1 };
            try config.check(s, &q8);
        }
    }
    try t.expectEqual(@as(u32, 1), rows_only);
}

test "prefill GEMM geometry: split-K selection, validation and extents" {
    const gemm = zerv.model.gemm;
    // Tile 256 x 32. Large outputs need no split; small M or few row tiles split K.
    try t.expectEqual(@as(u32, 256), gemm.tile_m);
    try t.expectEqual(@as(u32, 32), gemm.Tile.narrow.rows());
    try t.expectEqual(@as(u32, 64), gemm.Tile.wide.rows());
    // Tile rule: wide only for >= 512-row plans of projections with M >= 4096.
    try t.expectEqual(gemm.Tile.wide, gemm.tileFor(5120, 512));
    try t.expectEqual(gemm.Tile.wide, gemm.tileFor(17408, 1024));
    try t.expectEqual(gemm.Tile.narrow, gemm.tileFor(1024, 512));
    try t.expectEqual(gemm.Tile.narrow, gemm.tileFor(17408, 256));
    try t.expectError(error.InvalidShape, gemm.module(.f32_k, .wide));
    _ = try gemm.module(.q4_0, .wide);
    // Wide tiles: half the row tiles, so a split can start earlier (68 x 8 = 544 >= 384).
    try t.expectEqual(@as(u32, 0), gemm.splitChunk(17408, 512, 5120, 384, .wide));
    try t.expectEqual(@as(u32, 3), gemm.splitCount(17408, gemm.splitChunk(5120, 512, 17408, 384, .wide))); // 20 x 8 = 160
    try t.expectEqual(@as(u32, 0), gemm.splitChunk(17408, 512, 5120, 384, .narrow)); // 68 x 16 tiles
    const chunk = gemm.splitChunk(5120, 512, 17408, 384, .narrow); // 20 x 16 = 320 tiles
    try t.expect(chunk > 0 and chunk % 256 == 0);
    try t.expectEqual(@as(u32, 2), gemm.splitCount(17408, chunk));
    try t.expectEqual(@as(u32, 20), gemm.splitCount(5120, gemm.splitChunk(48, 512, 5120, 384, .narrow)));
    try t.expectEqual(@as(u32, 0), gemm.splitChunk(64, 64, 256, 384, .narrow)); // K < 512
    for ([_]u32{ 48, 1024, 5120, 6144, 10240, 12288, 17408 }) |M| for ([_]u32{ 5120, 6144, 17408 }) |K| for ([_]u32{ 1, 13, 256, 512 }) |rows| {
        const c = gemm.splitChunk(M, rows, K, 384, .narrow);
        if (c == 0) continue;
        try t.expect(c % 256 == 0 and gemm.splitCount(K, c) >= 2 and gemm.splitCount(K, c) <= K / 256);
    };
    // 17408 x 16 rows is 68 tiles, split to reach the target.
    try t.expectEqual(@as(u32, 1024), gemm.splitChunk(17408, 16, 5120, 384, .narrow));
    try t.expectEqualSlices(u32, &.{ 68, 1, 5 }, &(try gemm.validate(.q4_0, .narrow, .{ .a_base = 0, .a_rs = 2880, .x_base = 0, .x_rs = 5120, .y_base = 5120 * 16, .y_rs = 17408, .y_bs = 16 * 17408, .m = 17408, .k = 5120, .k_chunk = 1024 }, .{ .rows = 16, .batches = 5 }, 17408 * 2880, (5120 * 16 + 5 * 16 * 17408) * 4)));
    const push: gemm.Push = .{ .a_base = 2, .a_rs = 2880, .x_base = 0, .x_rs = 5120, .y_base = 5120 * 16, .y_rs = 1024, .m = 1024, .k = 5120 };
    const act_bytes: u64 = (5120 * 16 + 1024 * 16) * 4;
    const groups = try gemm.validate(.q4_0, .narrow, push, .{ .rows = 16 }, 2 + 1024 * 2880 + 2, act_bytes);
    try t.expectEqualSlices(u32, &.{ 4, 1, 1 }, &groups);
    try t.expectError(error.InvalidRange, gemm.validate(.q4_0, .narrow, push, .{ .rows = 16 }, 1024 * 2880, act_bytes));
    try t.expectError(error.InvalidRange, gemm.validate(.q4_0, .narrow, push, .{ .rows = 17 }, 2 + 1024 * 2880 + 2, act_bytes));
    var split = push;
    split.k_chunk = 1280;
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, split, .{ .rows = 16, .batches = 3 }, 1 << 30, 1 << 30));
    _ = try gemm.validate(.q4_0, .narrow, split, .{ .rows = 16, .batches = 4 }, 1 << 30, 1 << 30);
    split.k_chunk = 1000; // not a multiple of 256
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, split, .{ .rows = 16, .batches = 6 }, 1 << 30, 1 << 30));
    try t.expectError(error.InvalidShape, gemm.validate(.q4_0, .narrow, push, .{ .rows = 0 }, 1 << 30, 1 << 30));
}

test "split-K decode attention geometry and push layouts" {
    const att = zerv.model.attention;
    try t.expectEqual(@as(u32, 6), att.heads_per_kv);
    try t.expectEqual(@as(u32, 64), att.chunk);
    // Scores: one workgroup per chunk pair (block 17c).
    try t.expectEqualSlices(u32, &.{ 1, 64, 4 }, &att.groups(.scores, 128, 1));
    try t.expectEqualSlices(u32, &.{ 1, 65, 4 }, &att.groups(.scores, 129, 1));
    try t.expectEqualSlices(u32, &.{ 1, 128, 4 }, &att.groups(.pv, 128, 1));
    try t.expectEqualSlices(u32, &.{ 24, 1, 1 }, &att.groups(.combine, 128, 1));
    // Speculative verify: query rows in grid z.
    try t.expectEqualSlices(u32, &.{ 5, 64, 4 }, &att.groups(.scores, 128, 5)); // rows fastest (L2 reuse)
    try t.expectEqualSlices(u32, &.{ 24, 1, 3 }, &att.groups(.combine, 128, 3));
    // The global-max pass (block 17c): one workgroup per (row, KV head).
    try t.expectEqualSlices(u32, &.{ 3, 1, 4 }, &att.groups(.gmax, 128, 3));
    try t.expectEqualSlices(u32, &.{ 16, 24, 3 }, &att.groups(.block, 128, 3));
    try t.expectEqualSlices(u32, &.{ 17, 24, 1 }, &att.groups(.block, 129, 1));
    // Push blocks mirror the GLSL declarations (std430 scalars).
    try t.expectEqual(@as(u32, 48), att.pushBytes(.scores));
    try t.expectEqual(@as(u32, 52), att.pushBytes(.pv));
    try t.expectEqual(@as(u32, 40), att.pushBytes(.combine));
    try t.expectEqual(@as(u32, 24), att.pushBytes(.gmax));
    try t.expectEqual(@as(u32, 36), att.pushBytes(.block));
    for ([_]att.Pass{ .scores, .gmax, .pv, .block, .combine }) |pass| {
        for ([_]zerv.model.KvType{ .f32, .f16 }) |kv| {
            const code = att.module(pass, kv);
            try t.expect(code.len > 20 and std.mem.readInt(u32, code[0..4], .little) == 0x07230203);
        }
    }
}

test "prefill plans and chunk policy" {
    const model = zerv.model;
    var plans: [model.max_plans]model.Plan = undefined;
    // Default capacity: plans of 32/64/128/256/512 rows (split-K sized per plan).
    const n = model.makePlans(512, &plans);
    try t.expectEqual(@as(u8, 5), n);
    for (plans[0..n], [_]u32{ 32, 64, 128, 256, 512 }) |plan, rows| try t.expectEqual(rows, plan.rows);
    const P = plans[0..n];
    const Case = struct { remaining: u32, rows: u32, plan: usize };
    for ([_]Case{
        .{ .remaining = 1, .rows = 1, .plan = 0 },      .{ .remaining = 32, .rows = 32, .plan = 0 },
        .{ .remaining = 33, .rows = 33, .plan = 1 },    .{ .remaining = 128, .rows = 128, .plan = 2 },
        .{ .remaining = 151, .rows = 151, .plan = 3 },  .{ .remaining = 324, .rows = 324, .plan = 4 },
        .{ .remaining = 3223, .rows = 512, .plan = 4 }, .{ .remaining = 512, .rows = 512, .plan = 4 },
    }) |c| {
        const got = model.chunkFor(P, 512, c.remaining);
        try t.expectEqual(c.rows, got.rows);
        try t.expectEqual(c.plan, got.plan);
    }
    for (1..2000) |len| {
        var left: u32 = @intCast(len);
        while (left > 0) {
            const c = model.chunkFor(P, 512, left);
            try t.expect(c.rows >= 1 and c.rows <= left and c.rows <= P[c.plan].rows);
            left -= c.rows;
        }
    }
    // Small capacities (verification modes) clip the table.
    try t.expectEqual(@as(u8, 1), model.makePlans(13, &plans));
    try t.expectEqual(@as(u32, 13), plans[0].rows);
    try t.expectEqual(@as(u8, 2), model.makePlans(60, &plans));
    try t.expectEqual(@as(u32, 60), plans[1].rows);
    try t.expectEqual(@as(u8, 4), model.makePlans(129, &plans));
    try t.expectEqual(@as(u32, 129), plans[3].rows);
}
