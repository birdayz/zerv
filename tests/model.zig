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
    const max = layout.maxContext(capacity);
    try t.expect(max >= 16384 and max < 32768);
    const s = try layout.state(max, capacity);
    try t.expect(s.words * 4 <= capacity);
    try t.expectError(error.ContextTooLarge, layout.state(max + 1, capacity));
    try t.expectError(error.InvalidContext, layout.state(0, capacity));
    try t.expectEqual(s.conv, layout.State.ssm_words);
    try t.expectEqual(s.kv, layout.State.ssm_words + layout.State.conv_words);
    try t.expectEqual(s.vcache(15) + 4 * 256 * max, @as(u32, @intCast(s.words)));
    try t.expectEqual(s.kcache(1), s.vcache(0) + 4 * 256 * max);
    for ([_]u32{ 1, 13, 512 }) |rows| {
        const a = try layout.act(8192, rows, 1000, false);
        // Regions are ordered, 64-word aligned and each holds `rows` rows of its width.
        var previous: u64 = 0;
        inline for (layout.widths, 0..) |entry, i| {
            const start: u64 = @field(a, entry[0]);
            try t.expect(start % 64 == 0);
            if (i > 0) try t.expect(start >= previous);
            previous = start + @as(u64, entry[1]) * rows;
        }
        try t.expect(a.scores >= previous);
        try t.expectEqual(@as(u64, a.part), a.scores + std.mem.alignForward(u64, 24 * 8192 * @as(u64, rows), 64));
        // Decode attention scratch: 128 chunks of 64 keys per head.
        try t.expectEqual(@as(u32, 128), a.chunks);
        try t.expectEqual(@as(u64, a.amax), a.part + std.mem.alignForward(u64, 1000, 64));
        try t.expectEqual(@as(u64, a.apart), a.amax + 24 * 128);
        try t.expectEqual(@as(u64, a.asum), a.apart + 24 * 128 * 256);
        try t.expectEqual(a.words, @as(u64, a.asum) + 24 * 128);
        try t.expectEqual(rows, a.rows);
        try t.expect(a.x16 == null);
        // f16 mode: the f16 input copy (rows x ffn halves) is last, 16-byte aligned; the
        // other regions do not move.
        const h = try layout.act(8192, rows, 1000, true);
        try t.expectEqual(a.asum, h.asum);
        try t.expectEqual(@as(u64, h.x16.?), std.mem.alignForward(u64, a.words, 64));
        try t.expectEqual(h.words, @as(u64, h.x16.?) + std.mem.alignForward(u64, 8704 * @as(u64, rows), 64));
    }
    try t.expectError(error.InvalidContext, layout.act(0, 1, 0, false));
    try t.expectError(error.InvalidContext, layout.act(8192, 0, 0, false));
    try t.expectEqual(@as(u32, 2), (try layout.act(65, 1, 0, false)).chunks);
    try t.expectEqual(@as(u32, 1), (try layout.act(64, 1, 0, false)).chunks);
    try t.expectError(error.ContextTooLarge, layout.act(262144, 512, 0, false));
    // io block: prefill token ids and RoPE rows follow the logits.
    try t.expectEqual(@as(u32, layout.io.logits + 248320), layout.io.tokens);
    try t.expectEqual(layout.io.words(512), @as(u64, layout.io.tokens) + 512 + 512 * 64);
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
    try t.expectEqualSlices(u32, &.{ 128, 4, 1 }, &att.groups(.scores, 128));
    try t.expectEqualSlices(u32, &.{ 128, 4, 1 }, &att.groups(.pv, 128));
    try t.expectEqualSlices(u32, &.{ 24, 1, 1 }, &att.groups(.combine, 128));
    // Push blocks mirror the GLSL declarations (7 words each, std430 scalars).
    try t.expectEqual(@as(u32, 28), att.pushBytes(.scores));
    try t.expectEqual(@as(u32, 28), att.pushBytes(.pv));
    try t.expectEqual(@as(u32, 28), att.pushBytes(.combine));
    for ([_]att.Pass{ .scores, .pv, .combine }) |pass| {
        const code = att.module(pass);
        try t.expect(code.len > 20 and std.mem.readInt(u32, code[0..4], .little) == 0x07230203);
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
