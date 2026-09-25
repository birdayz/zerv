//! Real-model gate for batched decode (docs/specs/concurrent.md, "18b.2 design"): is every
//! row of a batched decode bit for bit the logits of the same sequence decoded alone?
//!
//! Phase A (a one-slot model, the production single-sequence configuration): for each of
//! K sequences (prompt P_i of p_i tokens, then T teacher-forced tokens X_i, fixed pseudo-
//! random: the check is about arithmetic), prefill and T steps; the logits are the reference.
//! Phase B (a K-slot model with batched decode commands for 1..K rows):
//!   1. single-slot operations in slots 0, 3 and K-1 (select, reset, prefill, step) equal
//!      the reference;
//!   2. a schedule where sequence i joins at global step 2i (prefill into slot sigma(i) while
//!      the others are mid-decode) and leaves after T steps; each global step decodes the
//!      active sequences in one batch (1..K rows, over 5 rows in projection groups) in a
//!      fresh random row order; every row equals the reference;
//!   3. the same schedule after every slot's pages are released and remapped to a random
//!      permutation of a larger pool (non-identity, interleaved page tables);
//!   4. error cases (duplicate slot, page in use, unmapped slot);
//!   5. timings: decodeBatch per row count at a common position, and step().
//! Usage: zerv-batch-check MODEL [f32|f16 [PAGE_TOKENS [CONTEXT [f32|f16]]]] > report.jsonl
//! (KV type, page tokens, context, decode precision; exit status 1 on any mismatch). With
//! decode precision f16 the reference is the same mode on a one-slot model, each sequence
//! decoded alone through one-row batches (batch invariance within the mode).
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const model = zerv.model;

const vocab = model.config.vocab;
const K = 8;
const T = 20;
// Prompt lengths across the 64-key chunk and 128-token page edges, and longer.
const prompt_lens = [K]u32{ 40, 61, 127, 128, 200, 255, 511, 700 };
const max_prompt = 700;

fn equalBits(a: []const f32, b: []const f32) bool {
    for (a, b) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return false;
    return true;
}

fn firstDiff(a: []const f32, b: []const f32) usize {
    for (a, b, 0..) |x, y, i| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return i;
    return a.len;
}

const Stats = struct { compared: u64 = 0, failures: u64 = 0 };

fn check(out: *std.Io.Writer, stats: *Stats, what: []const u8, seq: usize, t: i64, got: []const f32, want: []const f32) !void {
    stats.compared += 1;
    if (equalBits(got, want)) return;
    stats.failures += 1;
    const i = firstDiff(got, want);
    try out.print("{{\"mismatch\":\"{s}\",\"seq\":{d},\"t\":{d},\"index\":{d},\"got\":{e},\"want\":{e}}}\n", .{ what, seq, t, i, got[i], want[i] });
}

fn median(ns: []u64) f64 {
    std.mem.sort(u64, ns, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(ns[ns.len / 2])) / 1e6;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or args.len > 6) return error.Usage;
    const kv_type: model.KvType = if (args.len >= 3) std.meta.stringToEnum(model.KvType, args[2]) orelse return error.Usage else .f32;
    const page: u32 = if (args.len >= 4) (if (std.mem.eql(u8, args[3], "context")) 0 else try std.fmt.parseInt(u32, args[3], 10)) else model.layout.default_kv_page;
    const context: u32 = if (args.len >= 5) try std.fmt.parseInt(u32, args[4], 10) else 2048;
    const precision: model.DecodePrecision = if (args.len >= 6) std.meta.stringToEnum(model.DecodePrecision, args[5]) orelse return error.Usage else .f32;
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buffer);
    const out = &stdout.interface;

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .storage16 = kv_type == .f16, .cooperative_matrix = precision == .f16 });
    defer device.deinit() catch @panic("live device resources");

    var prng = std.Random.DefaultPrng.init(0x18b2);
    const random = prng.random();
    var prompts: [K][max_prompt]u32 = undefined;
    var xs: [K][T]u32 = undefined;
    for (&prompts) |*p| for (p) |*token| {
        token.* = random.intRangeLessThan(u32, 0, 150000);
    };
    for (&xs) |*x| for (x) |*token| {
        token.* = random.intRangeLessThan(u32, 0, 150000);
    };

    // Phase A: references from the one-slot model.
    const ref_prefill = try a.alloc(f32, K * vocab);
    const ref = try a.alloc(f32, K * T * vocab);
    var step_ns: [T]u64 = undefined;
    {
        var m: model.Model = undefined;
        try m.init(&device, &container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .kv_page_tokens = page, .batch_rows = if (precision == .f16) 1 else 0, .decode_precision = precision });
        defer m.deinit();
        for (0..K) |i| {
            try m.reset();
            @memcpy(ref_prefill[i * vocab ..][0..vocab], try m.prefill(prompts[i][0..prompt_lens[i]]));
            for (0..T) |t| {
                const t0 = std.Io.Clock.awake.now(io).nanoseconds;
                const logits = if (precision == .f16) try m.decodeBatch(&.{.{ .slot = 0, .token = xs[i][t] }}) else try m.step(xs[i][t]);
                step_ns[t] = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);
                @memcpy(ref[(i * T + t) * vocab ..][0..vocab], logits);
            }
        }
        try out.print("{{\"phase\":\"reference\",\"slots\":1,\"kv\":\"{s}\",\"page\":{d},\"context\":{d},\"decode_precision\":\"{s}\",\"step_ms_median\":{d:.3}}}\n", .{ @tagName(kv_type), m.state_layout.page, context, @tagName(precision), median(&step_ns) });
        try out.flush();
    }

    // Phase B: K slots, batched decode of up to K rows; a pool with spare pages for the
    // permuted tables (the default tables use the first K x seq_pages).
    var m: model.Model = undefined;
    const seq_pages = model.layout.stateWith(context, 0xf000_0000, false, kv_type, page, 1, 0) catch unreachable;
    const pool = K * seq_pages.seq_pages + 5;
    try m.init(&device, &container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .kv_page_tokens = page, .slots = K, .batch_rows = K, .kv_pages = pool, .decode_precision = precision });
    defer m.deinit();
    try out.print("{{\"phase\":\"batched\",\"slots\":{d},\"pool_pages\":{d},\"seq_pages\":{d},\"page\":{d}}}\n", .{ K, m.state_layout.pages, m.state_layout.seq_pages, m.state_layout.page });
    var stats: Stats = .{};

    // 1. Single-slot operations in other slots.
    for ([_]u32{ 0, 3, K - 1 }) |slot| {
        const i: usize = 5;
        try m.select(slot);
        try m.reset();
        try check(out, &stats, "slot-prefill", i, -1, try m.prefill(prompts[i][0..prompt_lens[i]]), ref_prefill[i * vocab ..][0..vocab]);
        for (0..T) |t| {
            const logits = if (precision == .f16) try m.decodeBatch(&.{.{ .slot = slot, .token = xs[i][t] }}) else try m.step(xs[i][t]);
            try check(out, &stats, "slot-step", i, @intCast(t), logits, ref[(i * T + t) * vocab ..][0..vocab]);
        }
    }
    try out.print("{{\"part\":\"single-slot\",\"compared\":{d},\"failures\":{d}}}\n", .{ stats.compared, stats.failures });
    try out.flush();

    // 2 and 3. The join/leave schedule, on the default tables, then on permuted ones.
    var sizes_seen: u32 = 0;
    for (0..2) |round| {
        if (round == 1) {
            for (0..K) |s| try m.releasePages(@intCast(s));
            var perm: [model.max_pool_pages]u32 = undefined;
            const all = perm[0..m.state_layout.pages];
            for (all, 0..) |*p, i| p.* = @intCast(i);
            random.shuffle(u32, all);
            const sp = m.state_layout.seq_pages;
            for (0..K) |s| try m.mapPages(@intCast(s), all[s * sp ..][0..sp]);
        }
        var sigma: [K]u32 = undefined;
        for (&sigma, 0..) |*s, i| s.* = @intCast(i);
        random.shuffle(u32, &sigma);
        var done: [K]u32 = @splat(0);
        const before = stats;
        var g: u32 = 0;
        while (g < 2 * (K - 1) + T) : (g += 1) {
            // Joins: prefill into the sequence's slot while others are mid-decode.
            for (0..K) |i| if (g == 2 * i) {
                try m.select(sigma[i]);
                try m.reset();
                try check(out, &stats, "join-prefill", i, -1, try m.prefill(prompts[i][0..prompt_lens[i]]), ref_prefill[i * vocab ..][0..vocab]);
            };
            var rows: [K]model.BatchRow = undefined;
            var seqs: [K]usize = undefined;
            var n: usize = 0;
            for (0..K) |i| if (g >= 2 * i and done[i] < T) {
                seqs[n] = i;
                n += 1;
            };
            if (n == 0) continue;
            random.shuffle(usize, seqs[0..n]);
            for (seqs[0..n], rows[0..n]) |i, *row| row.* = .{ .slot = sigma[i], .token = xs[i][done[i]] };
            const logits = try m.decodeBatch(rows[0..n]);
            sizes_seen |= @as(u32, 1) << @intCast(n - 1);
            for (seqs[0..n], 0..) |i, r| {
                try check(out, &stats, if (round == 0) "batch" else "batch-permuted-pages", i, done[i], logits[r * vocab ..][0..vocab], ref[(i * T + done[i]) * vocab ..][0..vocab]);
                done[i] += 1;
            }
        }
        try out.print("{{\"part\":\"{s}\",\"compared\":{d},\"failures\":{d}}}\n", .{ if (round == 0) "schedule" else "schedule-permuted-pages", stats.compared - before.compared, stats.failures - before.failures });
        try out.flush();
    }

    // 4. Error cases.
    var errors_ok = true;
    if (m.decodeBatch(&.{ .{ .slot = 1, .token = 1 }, .{ .slot = 1, .token = 2 } })) |_| {
        errors_ok = false;
    } else |e| errors_ok = errors_ok and e == error.InvalidBatch;
    const owned = blk: {
        for (0..m.state_layout.pages) |p| if (m.page_owner[p] == 2) break :blk @as(u32, @intCast(p));
        unreachable;
    };
    try m.releasePages(6);
    if (m.mapPages(6, &.{owned})) |_| {
        errors_ok = false;
    } else |e| errors_ok = errors_ok and e == error.PageInUse;
    if (m.decodeBatch(&.{.{ .slot = 6, .token = 1 }})) |_| {
        errors_ok = false;
    } else |e| errors_ok = errors_ok and e == error.PagesMissing;
    try out.print("{{\"part\":\"errors\",\"ok\":{}}}\n", .{errors_ok});

    // 5. Timings: every slot at a common position, decodeBatch per row count.
    {
        var free: [model.max_pool_pages]u32 = undefined;
        var nfree: usize = 0;
        for (0..m.state_layout.pages) |p| if (m.page_owner[p] == 0xff) {
            free[nfree] = @intCast(p);
            nfree += 1;
        };
        const need = m.state_layout.seq_pages - m.mappedPages(6);
        try m.mapPages(6, free[0..need]);
    }
    const timing_prompt = @min(max_prompt, context - 64);
    for (0..K) |s| {
        try m.select(@intCast(s));
        try m.reset();
        _ = try m.prefill(prompts[s % K][0..timing_prompt]);
    }
    var b: u32 = 1;
    while (b <= K) : (b += 1) {
        var ns: [5]u64 = undefined;
        for (&ns) |*slot_ns| {
            var rows: [K]model.BatchRow = undefined;
            for (rows[0..b], 0..) |*row, r| row.* = .{ .slot = @intCast(r), .token = 1000 + @as(u32, @intCast(r)) };
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            _ = try m.decodeBatch(rows[0..b]);
            slot_ns.* = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);
        }
        const ms = median(&ns);
        try out.print("{{\"timing\":\"decodeBatch\",\"rows\":{d},\"position\":{d},\"ms_median\":{d:.3},\"tok_s\":{d:.1}}}\n", .{ b, m.slotPosition(0), ms, @as(f64, @floatFromInt(b)) * 1000 / ms });
        try out.flush();
    }

    const passed = stats.failures == 0 and errors_ok and sizes_seen == (1 << K) - 1;
    try out.print("{{\"summary\":true,\"compared\":{d},\"failures\":{d},\"batch_sizes_seen\":{d},\"errors_ok\":{},\"passed\":{}}}\n", .{ stats.compared, stats.failures, @popCount(sizes_seen), errors_ok, passed });
    try out.flush();
    if (!passed) std.process.exit(1);
}
