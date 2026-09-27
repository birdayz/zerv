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
//!      Joins run segment by segment (`prefillSegment`) with decode batches of the running
//!      sequences between the segments (docs/specs/concurrent.md, "18c.2 design");
//!   3. the same schedule after every slot's pages are released and remapped to a random
//!      permutation of a larger pool (non-identity, interleaved page tables);
//!   4. error cases (duplicate slot, page in use, unmapped slot);
//!   5. timings: decodeBatch per row count at a common position, and step().
//! Usage: zerv-batch-check MODEL [f32|f16 [PAGE_TOKENS [CONTEXT [f32|f16[@v1|@v2][@all] [pack]]]]] > report.jsonl
//! `shared`: the shared-KV-pool gate (docs/specs/concurrent.md, "18d.2 design"): a pool
//! for about 3 of the 8 sequences, pages mapped on demand and recycled as sequences leave;
//! every logits row bitwise equal to the sequence run alone.
//! `swap`: the swap gate (docs/specs/concurrent.md, "18d.3 design"): prompt-only admission
//! into a pool too small for every sequence's growth; a decode step that finds no free page
//! swaps the youngest running sequence out to the host store, and swapped sequences come
//! back (on fresh pages) before new ones join; every logits row bitwise equal to the
//! sequence run alone.
//! `pack`: the packed-prefill gate instead (docs/specs/concurrent.md, "18d.1 design"; f16
//! prefill mode): every sequence prefilled alone is the reference; then the sequences join
//! in packed chunks of 2..8 (re-packed chunk by chunk, several chunk grids) with decode
//! batches of the joined ones between segments; every logits row must be bitwise equal.
//! (KV type, page tokens, context, decode precision; exit status 1 on any mismatch). With
//! decode precision f16 the reference is the same mode on a one-slot model, each sequence
//! decoded alone through one-row batches (batch invariance within the mode).
const std = @import("std");
const w = @import("matvec_workload");
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
    if (args.len < 2 or args.len > 7) return error.Usage;
    const pack_mode = args.len == 7 and std.mem.eql(u8, args[6], "pack");
    const shared_mode = args.len == 7 and std.mem.eql(u8, args[6], "shared");
    const swap_mode = args.len == 7 and std.mem.eql(u8, args[6], "swap");
    if (args.len == 7 and !pack_mode and !shared_mode and !swap_mode) return error.Usage;
    const kv_type: model.KvType = if (args.len >= 3) std.meta.stringToEnum(model.KvType, args[2]) orelse return error.Usage else .f32;
    const page: u32 = if (args.len >= 4) (if (std.mem.eql(u8, args[3], "context")) 0 else try std.fmt.parseInt(u32, args[3], 10)) else model.layout.default_kv_page;
    const context: u32 = if (args.len >= 5) try std.fmt.parseInt(u32, args[4], 10) else 2048;
    // DECODE_PRECISION[@v1|@v2]: `Options.decode_f16_kernel` for f16.
    var precision: model.DecodePrecision = .f32;
    var dkernel: model.DecodeF16Kernel = .v2;
    var dformats: model.DecodeF16Formats = .q4_0;
    if (args.len >= 6) {
        var pp = std.mem.splitScalar(u8, args[5], '@');
        precision = std.meta.stringToEnum(model.DecodePrecision, pp.first()) orelse return error.Usage;
        while (pp.next()) |k| {
            if (std.meta.stringToEnum(model.DecodeF16Kernel, k)) |v| {
                dkernel = v;
            } else if (std.mem.eql(u8, k, "all")) {
                dformats = .all;
            } else return error.Usage;
        }
    }
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buffer);
    const out = &stdout.interface;

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .storage16 = kv_type == .f16, .cooperative_matrix = precision == .f16 or pack_mode, .subgroup_size_control = precision == .f16 or pack_mode });
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

    if (swap_mode) return swapCheck(a, &device, &container, out, kv_type, page, context, precision, dkernel, dformats, &prompts, &xs);
    if (shared_mode) return sharedCheck(a, &device, &container, out, kv_type, page, context, precision, dkernel, dformats, &prompts, &xs);
    if (pack_mode) return packCheck(a, io, &device, &container, out, kv_type, page, context, precision, dkernel, dformats, &prompts, &xs);

    // Phase A: references from the one-slot model.
    const ref_prefill = try a.alloc(f32, K * vocab);
    const ref = try a.alloc(f32, K * T * vocab);
    var step_ns: [T]u64 = undefined;
    {
        var m: model.Model = undefined;
        try m.init(&device, &container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .kv_page_tokens = page, .batch_rows = if (precision != .f32) 1 else 0, .decode_precision = precision, .decode_f16_kernel = dkernel, .decode_f16_formats = dformats });
        defer m.deinit();
        for (0..K) |i| {
            try m.reset();
            @memcpy(ref_prefill[i * vocab ..][0..vocab], try m.prefill(prompts[i][0..prompt_lens[i]]));
            for (0..T) |t| {
                const t0 = std.Io.Clock.awake.now(io).nanoseconds;
                const logits = if (precision != .f32) try m.decodeBatch(&.{.{ .slot = 0, .token = xs[i][t] }}) else try m.step(xs[i][t]);
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
    try m.init(&device, &container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .kv_page_tokens = page, .slots = K, .batch_rows = K, .kv_pages = pool, .decode_precision = precision, .decode_f16_kernel = dkernel, .decode_f16_formats = dformats });
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
            const logits = if (precision != .f32) try m.decodeBatch(&.{.{ .slot = slot, .token = xs[i][t] }}) else try m.step(xs[i][t]);
            try check(out, &stats, "slot-step", i, @intCast(t), logits, ref[(i * T + t) * vocab ..][0..vocab]);
        }
    }
    try out.print("{{\"part\":\"single-slot\",\"compared\":{d},\"failures\":{d}}}\n", .{ stats.compared, stats.failures });
    try out.flush();

    // 2 and 3. The join/leave schedule, on the default tables, then on permuted ones.
    var sizes_seen: u32 = 0;
    var interleaved: u32 = 0;
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
            // Joins: prefill into the sequence's slot while others are mid-decode, segment by
            // segment. Round 0 runs the segments back to back (the main schedule then sees
            // every batch size); round 1 runs a decode batch of the running sequences between
            // every two segments, as the server schedules them.
            for (0..K) |i| if (g == 2 * i) {
                try m.select(sigma[i]);
                try m.reset();
                var rem: []const u32 = prompts[i][0..prompt_lens[i]];
                while (true) {
                    const seg = try m.prefillSegment(rem);
                    if (seg.consumed > 0) {
                        rem = rem[seg.consumed..];
                        if (rem.len == 0) {
                            try check(out, &stats, "join-prefill-segments", i, -1, seg.logits.?, ref_prefill[i * vocab ..][0..vocab]);
                            break;
                        }
                    }
                    if (round == 0) continue;
                    var mrows: [K]model.BatchRow = undefined;
                    var mseqs: [K]usize = undefined;
                    var mn: usize = 0;
                    for (0..i) |j| if (done[j] < T) {
                        mseqs[mn] = j;
                        mn += 1;
                    };
                    if (mn == 0) continue;
                    random.shuffle(usize, mseqs[0..mn]);
                    for (mseqs[0..mn], mrows[0..mn]) |j, *row| row.* = .{ .slot = sigma[j], .token = xs[j][done[j]] };
                    const ml = try m.decodeBatch(mrows[0..mn]);
                    interleaved += 1;
                    for (mseqs[0..mn], 0..) |j, r| {
                        try check(out, &stats, "batch-between-segments", j, done[j], ml[r * vocab ..][0..vocab], ref[(j * T + done[j]) * vocab ..][0..vocab]);
                        done[j] += 1;
                    }
                }
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
    // A chunk in flight: its slot is not decodable, no other slot can be selected; an
    // aborted chunk leaves its slot usable only after a reset.
    {
        try m.select(5);
        try m.reset();
        const first = try m.prefillSegment(prompts[7][0..prompt_lens[7]]);
        errors_ok = errors_ok and first.consumed == 0;
        if (m.select(4)) |_| {
            errors_ok = false;
        } else |e| errors_ok = errors_ok and e == error.ChunkInFlight;
        if (m.decodeBatch(&.{.{ .slot = 5, .token = 1 }})) |_| {
            errors_ok = false;
        } else |e| errors_ok = errors_ok and e == error.ChunkInFlight;
        m.abortChunk();
        if (m.prefillSegment(prompts[7][0..8])) |_| {
            errors_ok = false;
        } else |e| errors_ok = errors_ok and e == error.SlotNeedsReset;
        try m.reset();
        try check(out, &stats, "prefill-after-abort", 5, -1, try m.prefill(prompts[5][0..prompt_lens[5]]), ref_prefill[5 * vocab ..][0..vocab]);
    }
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
    // Per-phase GPU time of one batched step (instrumented commands), 1 and K rows.
    for ([_]u32{ 1, K }) |pb| {
        var rec: Recorder = .{ .marks = try w.Marks.init(&device, max_marks) };
        defer rec.marks.deinit();
        var cmd = try gpu.Commands.init(&device);
        defer cmd.deinit() catch @panic("profile commands");
        try cmd.begin();
        try rec.marks.begin(&cmd);
        try m.recordBatchProfile(&cmd, pb, .{ .context = &rec, .mark = Recorder.mark });
        try cmd.end();
        var totals: [phase_count]f64 = @splat(0);
        var total: f64 = 0;
        for (0..3) |_| {
            var rows: [K]model.BatchRow = undefined;
            for (rows[0..pb], 0..) |*row, r| row.* = .{ .slot = @intCast(r), .token = 1000 + @as(u32, @intCast(r)) };
            _ = try m.decodeBatchProfiled(rows[0..pb], &cmd);
            total = try rec.totals(&totals); // the last run's
        }
        try out.print("{{\"profile\":\"decodeBatch\",\"rows\":{d},\"gpu_ms\":{d:.3},\"phases_ms\":{{", .{ pb, total / 1e6 });
        var first = true;
        inline for (@typeInfo(model.Phase).@"enum".fields, 0..) |f, i| if (totals[i] > 0) {
            try out.print("{s}\"{s}\":{d:.3}", .{ if (first) "" else ",", f.name, totals[i] / 1e6 });
            first = false;
        };
        try out.writeAll("}}\n");
        try out.flush();
    }

    const passed = stats.failures == 0 and errors_ok and sizes_seen == (1 << K) - 1 and interleaved > 0;
    // Hash of the reference logits (phase A): equal across decode kernels of one arithmetic.
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(std.mem.sliceAsBytes(ref));
    try out.print("{{\"reference_logits_hash\":\"{x}\"}}\n", .{hasher.final()});
    try out.print("{{\"summary\":true,\"compared\":{d},\"failures\":{d},\"batch_sizes_seen\":{d},\"batches_between_segments\":{d},\"errors_ok\":{},\"passed\":{}}}\n", .{ stats.compared, stats.failures, @popCount(sizes_seen), interleaved, errors_ok, passed });
    try out.flush();
    if (!passed) std.process.exit(1);
}

const phase_count = @typeInfo(model.Phase).@"enum".fields.len;
const max_marks = 1024;
/// GPU timestamps between the recorded phases (as zerv-model-profile).
const Recorder = struct {
    marks: w.Marks,
    phases: [max_marks]model.Phase = undefined,

    fn mark(context: *anyopaque, commands: *gpu.Commands, phase: model.Phase, layer: i32) error{ProbeFailed}!void {
        _ = layer;
        const self: *Recorder = @ptrCast(@alignCast(context));
        const index = self.marks.mark(commands) catch return error.ProbeFailed;
        self.phases[index] = phase;
    }
    fn totals(self: *Recorder, out: *[phase_count]f64) !f64 {
        var gaps: [max_marks]f64 = undefined;
        try self.marks.read(gaps[0 .. self.marks.count - 1]);
        out.* = @splat(0);
        var sum: f64 = 0;
        for (gaps[0 .. self.marks.count - 1], 1..) |gap, i| {
            out[@intFromEnum(self.phases[i])] += gap;
            sum += gap;
        }
        return sum;
    }
};

/// The packed-prefill gate (`pack` mode; docs/specs/concurrent.md, "18d.1 design").
fn packCheck(a: std.mem.Allocator, io: std.Io, device: *gpu.Device, container: *zerv.artifact.gguf.Container, out: *std.Io.Writer, kv_type: model.KvType, page: u32, context: u32, precision: model.DecodePrecision, dkernel: model.DecodeF16Kernel, dformats: model.DecodeF16Formats, prompts: *const [K][max_prompt]u32, xs: *const [K][T]u32) !void {
    _ = io;
    var m: model.Model = undefined;
    try m.init(device, container, .{ .context = context, .prefill_rows = 512, .prefill_precision = .f16, .kv_type = kv_type, .kv_page_tokens = page, .slots = K, .batch_rows = K, .decode_precision = precision, .decode_f16_kernel = dkernel, .decode_f16_formats = dformats });
    defer m.deinit();
    if (!m.packable()) return error.NotPackable;
    // References: each sequence alone (one-sequence chunks), then T single-row steps.
    const ref_prefill = try a.alloc(f32, K * vocab);
    const ref = try a.alloc(f32, K * T * vocab);
    for (0..K) |i| {
        try m.select(@intCast(i));
        try m.reset();
        @memcpy(ref_prefill[i * vocab ..][0..vocab], try m.prefill(prompts[i][0..prompt_lens[i]]));
        for (0..T) |t| @memcpy(ref[(i * T + t) * vocab ..][0..vocab], try m.decodeBatch(&.{.{ .slot = @intCast(i), .token = xs[i][t] }}));
    }
    var stats: Stats = .{};
    var packs_seen: u32 = 0; // bit n: a chunk of n sequences ran
    var interleaved: u64 = 0;
    // Three rounds with different join orders and pack caps.
    const orders = [3][K]u32{ .{ 0, 1, 2, 3, 4, 5, 6, 7 }, .{ 7, 3, 5, 1, 6, 0, 4, 2 }, .{ 2, 4, 6, 0, 1, 3, 5, 7 } };
    const caps = [3]u32{ 8, 3, 2 };
    for (orders, caps) |order, cap| {
        for (0..K) |i| {
            try m.select(@intCast(i));
            try m.reset();
        }
        var consumed: [K]u32 = @splat(0);
        var steps: [K]u32 = @splat(0);
        var joined: [K]bool = @splat(false);
        var done_count: u32 = 0;
        while (done_count < K) {
            // Next packed chunk: pending sequences in join order while they fit.
            var items: [8]model.PackItem = undefined;
            var ids: [8]u32 = undefined;
            var n: u32 = 0;
            var remaining: [8]u32 = undefined;
            for (order) |i| {
                if (joined[i] or n == cap) continue;
                remaining[n] = prompt_lens[i] - consumed[i];
                if (m.packSpan(remaining[0 .. n + 1]) > 512) continue;
                items[n] = .{ .slot = i, .tokens = prompts[i][consumed[i]..prompt_lens[i]] };
                ids[n] = i;
                n += 1;
            }
            if (n > 0) {
                packs_seen |= @as(u32, 1) << @intCast(n);
                while (true) {
                    const seg = try m.prefillPackedSegment(items[0..n]);
                    if (seg.done) {
                        for (ids[0..n], 0..) |i, s| {
                            consumed[i] += seg.consumed[s];
                            if (consumed[i] == prompt_lens[i]) {
                                try check(out, &stats, "packed-prefill", i, -1, seg.logits.?[s * vocab ..][0..vocab], ref_prefill[i * vocab ..][0..vocab]);
                                joined[i] = true;
                            }
                        }
                        break;
                    }
                    // A decode batch of the joined, unfinished sequences between segments.
                    var rows: [K]model.BatchRow = undefined;
                    var who: [K]u32 = undefined;
                    var b: usize = 0;
                    for (0..K) |i| if (joined[i] and steps[i] < T) {
                        rows[b] = .{ .slot = @intCast(i), .token = xs[i][steps[i]] };
                        who[b] = @intCast(i);
                        b += 1;
                    };
                    if (b == 0) continue;
                    interleaved += 1;
                    const logits = try m.decodeBatch(rows[0..b]);
                    for (who[0..b], 0..) |i, r| {
                        try check(out, &stats, "decode-between-packed", i, steps[i], logits[r * vocab ..][0..vocab], ref[(i * T + steps[i]) * vocab ..][0..vocab]);
                        steps[i] += 1;
                        if (steps[i] == T) done_count += 1;
                    }
                }
                continue;
            }
            // Everyone joined: finish the decode steps.
            var rows: [K]model.BatchRow = undefined;
            var who: [K]u32 = undefined;
            var b: usize = 0;
            for (0..K) |i| if (steps[i] < T) {
                rows[b] = .{ .slot = @intCast(i), .token = xs[i][steps[i]] };
                who[b] = @intCast(i);
                b += 1;
            };
            const logits = try m.decodeBatch(rows[0..b]);
            for (who[0..b], 0..) |i, r| {
                try check(out, &stats, "decode-after-packed", i, steps[i], logits[r * vocab ..][0..vocab], ref[(i * T + steps[i]) * vocab ..][0..vocab]);
                steps[i] += 1;
                if (steps[i] == T) done_count += 1;
            }
        }
        try out.flush();
    }
    // Error cases: a duplicate slot, more sequences than the pack holds.
    var errors_ok = true;
    for (0..K) |i| {
        try m.select(@intCast(i));
        try m.reset();
    }
    if (m.prefillPackedSegment(&.{ .{ .slot = 1, .tokens = prompts[1][0..10] }, .{ .slot = 1, .tokens = prompts[2][0..10] } })) |_| {
        errors_ok = false;
    } else |e| errors_ok = errors_ok and e == error.InvalidBatch;
    const passed = stats.failures == 0 and errors_ok and packs_seen & 0b1_1111_1100 != 0 and interleaved > 0;
    try out.print("{{\"summary\":true,\"mode\":\"pack\",\"compared\":{d},\"failures\":{d},\"pack_sizes_seen\":\"{b}\",\"batches_between_segments\":{d},\"errors_ok\":{},\"passed\":{}}}\n", .{ stats.compared, stats.failures, packs_seen, interleaved, errors_ok, passed });
    try out.flush();
    if (!passed) std.process.exit(1);
}

/// The shared-KV-pool gate (`shared` mode; docs/specs/concurrent.md, "18d.2 design").
fn sharedCheck(a: std.mem.Allocator, device: *gpu.Device, container: *zerv.artifact.gguf.Container, out: *std.Io.Writer, kv_type: model.KvType, page: u32, context: u32, precision: model.DecodePrecision, dkernel: model.DecodeF16Kernel, dformats: model.DecodeF16Formats, prompts: *const [K][max_prompt]u32, xs: *const [K][T]u32) !void {
    const pt = if (page == 0) context else page;
    // Room for about three of the longest sequences at once: joins must wait for leaves.
    const pool: u32 = 3 * ((max_prompt + T + pt - 1) / pt);
    var m: model.Model = undefined;
    try m.init(device, container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .kv_page_tokens = page, .slots = K, .batch_rows = K, .kv_pages = pool, .kv_share = true, .decode_precision = precision, .decode_f16_kernel = dkernel, .decode_f16_formats = dformats });
    defer m.deinit();
    if (m.freePages() != pool) return error.PoolNotEmpty;
    // References: each sequence alone in slot 0 (its pages mapped, then released).
    const ref_prefill = try a.alloc(f32, K * vocab);
    const ref = try a.alloc(f32, K * T * vocab);
    for (0..K) |i| {
        try m.select(0);
        try m.reset();
        try m.releasePages(0);
        try m.ensurePages(0, prompt_lens[i] + T);
        @memcpy(ref_prefill[i * vocab ..][0..vocab], try m.prefill(prompts[i][0..prompt_lens[i]]));
        for (0..T) |t| @memcpy(ref[(i * T + t) * vocab ..][0..vocab], try m.decodeBatch(&.{.{ .slot = 0, .token = xs[i][t] }}));
    }
    try m.releasePages(0);
    var stats: Stats = .{};
    var waits: u64 = 0;
    var max_live: u32 = 0;
    // Sequences join in this order into rotating slots; a join waits (decode steps of the
    // running ones) until the pool has its pages; a sequence leaves after T steps.
    const order = [K]u32{ 6, 0, 7, 2, 5, 1, 4, 3 };
    var slot_of: [K]?u32 = @splat(null);
    var steps: [K]u32 = @splat(0);
    var slot_busy: [K]bool = @splat(false);
    var next_join: usize = 0;
    var done: u32 = 0;
    var next_slot: u32 = 0;
    while (done < K) {
        if (next_join < K) {
            const i = order[next_join];
            const need = prompt_lens[i] + T;
            const need_pages = (need + pt - 1) / pt;
            if (m.freePages() >= need_pages) {
                var slot = next_slot;
                while (slot_busy[slot]) slot = (slot + 1) % K;
                next_slot = (slot + 3) % K;
                slot_busy[slot] = true;
                slot_of[i] = slot;
                try m.select(slot);
                try m.reset();
                try m.ensurePages(slot, need);
                try check(out, &stats, "shared-prefill", i, -1, try m.prefill(prompts[i][0..prompt_lens[i]]), ref_prefill[i * vocab ..][0..vocab]);
                next_join += 1;
                continue;
            }
            waits += 1;
        }
        // One batch of every live sequence.
        var rows: [K]model.BatchRow = undefined;
        var who: [K]u32 = undefined;
        var b: usize = 0;
        for (0..K) |i| if (slot_of[i] != null and steps[i] < T) {
            rows[b] = .{ .slot = slot_of[i].?, .token = xs[i][steps[i]] };
            who[b] = @intCast(i);
            b += 1;
        };
        if (b == 0) return error.Deadlock;
        max_live = @max(max_live, @as(u32, @intCast(b)));
        const logits = try m.decodeBatch(rows[0..b]);
        for (who[0..b], 0..) |i, r| {
            try check(out, &stats, "shared-decode", i, steps[i], logits[r * vocab ..][0..vocab], ref[(i * T + steps[i]) * vocab ..][0..vocab]);
            steps[i] += 1;
            if (steps[i] == T) {
                // Leave: its pages go back to the pool (reused by the next joins).
                try m.releasePages(slot_of[i].?);
                slot_busy[slot_of[i].?] = false;
                slot_of[i] = null;
                done += 1;
            }
        }
    }
    // Error cases: a request beyond the pool, and beyond the context.
    var errors_ok = true;
    if (m.ensurePages(1, context)) |_| {
        errors_ok = context <= pool * pt;
        try m.releasePages(1);
    } else |e| errors_ok = e == error.PoolExhausted;
    if (m.ensurePages(1, context + 1)) |_| errors_ok = false else |e| errors_ok = errors_ok and e == error.ContextFull;
    const all_free = m.freePages() == pool;
    const passed = stats.failures == 0 and errors_ok and all_free and waits > 0 and max_live >= 2;
    try out.print("{{\"summary\":true,\"mode\":\"shared\",\"pool_pages\":{d},\"compared\":{d},\"failures\":{d},\"join_waits\":{d},\"max_live\":{d},\"errors_ok\":{},\"all_pages_returned\":{},\"passed\":{}}}\n", .{ pool, stats.compared, stats.failures, waits, max_live, errors_ok, all_free, passed });
    try out.flush();
    if (!passed) std.process.exit(1);
}

/// The swap gate (`swap` mode; docs/specs/concurrent.md, "18d.3 design").
fn swapCheck(a: std.mem.Allocator, device: *gpu.Device, container: *zerv.artifact.gguf.Container, out: *std.Io.Writer, kv_type: model.KvType, page: u32, context: u32, precision: model.DecodePrecision, dkernel: model.DecodeF16Kernel, dformats: model.DecodeF16Formats, prompts: *const [K][max_prompt]u32, xs: *const [K][T]u32) !void {
    const pt = if (page == 0) context else page;
    // The pool holds the prompts of about 3 sequences plus growth only for some.
    const pool: u32 = 3 * ((max_prompt + T + pt - 1) / pt);
    var m: model.Model = undefined;
    try m.init(device, container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .kv_page_tokens = page, .slots = K, .batch_rows = K, .kv_pages = pool, .kv_share = true, .swap_bytes = 2 << 30, .decode_precision = precision, .decode_f16_kernel = dkernel, .decode_f16_formats = dformats });
    defer m.deinit();
    const host_pages = m.hostFreePages();
    if (m.freePages() != pool or host_pages < pool) return error.PoolNotEmpty;
    const ref_prefill = try a.alloc(f32, K * vocab);
    const ref = try a.alloc(f32, K * T * vocab);
    for (0..K) |i| {
        try m.select(0);
        try m.reset();
        try m.releasePages(0);
        try m.ensurePages(0, prompt_lens[i] + T);
        @memcpy(ref_prefill[i * vocab ..][0..vocab], try m.prefill(prompts[i][0..prompt_lens[i]]));
        for (0..T) |t| @memcpy(ref[(i * T + t) * vocab ..][0..vocab], try m.decodeBatch(&.{.{ .slot = 0, .token = xs[i][t] }}));
    }
    try m.releasePages(0);
    var stats: Stats = .{};
    var swap_outs: u32 = 0;
    var swap_ins: u32 = 0;
    var self_swaps: u32 = 0;
    var forced: u32 = 0;
    var batches: u32 = 0;
    var max_live: u32 = 0;
    // Join order = age (index into `order`); slots rotate as in the shared gate.
    const order = [K]u32{ 6, 0, 7, 2, 5, 1, 4, 3 };
    var age: [K]u32 = undefined;
    for (order, 0..) |i, k| age[i] = @intCast(k);
    var slot_of: [K]?u32 = @splat(null);
    var swapped: [K]bool = @splat(false);
    var steps: [K]u32 = @splat(0);
    var slot_busy: [K]bool = @splat(false);
    var next_join: usize = 0;
    var done: u32 = 0;
    var next_slot: u32 = 0;
    while (done < K) {
        var running: u32 = 0;
        var oldest_swapped: ?u32 = null;
        for (0..K) |i| if (slot_of[i] != null and steps[i] < T) {
            if (swapped[i]) {
                if (oldest_swapped == null or age[i] < age[oldest_swapped.?]) oldest_swapped = @intCast(i);
            } else running += 1;
        };
        // Swapped sequences come back first (with a spare page per running one).
        if (oldest_swapped) |i| {
            if (try m.swapIn(slot_of[i].?, running)) {
                swapped[i] = false;
                swap_ins += 1;
                continue;
            }
        } else if (next_join < K) {
            const i = order[next_join];
            if (m.freePages() >= (prompt_lens[i] + pt - 1) / pt) {
                var slot = next_slot;
                while (slot_busy[slot]) slot = (slot + 1) % K;
                next_slot = (slot + 3) % K;
                slot_busy[slot] = true;
                slot_of[i] = slot;
                try m.select(slot);
                try m.reset();
                try m.ensurePages(slot, prompt_lens[i]); // the prompt only
                try check(out, &stats, "swap-prefill", i, -1, try m.prefill(prompts[i][0..prompt_lens[i]]), ref_prefill[i * vocab ..][0..vocab]);
                next_join += 1;
                continue;
            }
        }
        // Growth, oldest first: a missing page swaps out the youngest running sequence.
        var by_age: [K]u32 = undefined;
        var n_by: usize = 0;
        for (order) |i| if (slot_of[i] != null and steps[i] < T and !swapped[i]) {
            by_age[n_by] = i;
            n_by += 1;
        };
        for (by_age[0..n_by]) |i| {
            if (swapped[i]) continue;
            while (true) {
                m.ensurePages(slot_of[i].?, prompt_lens[i] + steps[i] + 1) catch |e| {
                    if (e != error.PoolExhausted) return e;
                    var victim: ?u32 = null;
                    for (by_age[0..n_by]) |j| if (!swapped[j]) {
                        victim = j;
                    };
                    const v = victim.?;
                    try m.swapOut(slot_of[v].?);
                    swapped[v] = true;
                    swap_outs += 1;
                    if (v == i) {
                        self_swaps += 1;
                        break;
                    }
                    continue;
                };
                break;
            }
        }
        // Forced swaps besides the pressure ones: every third batch the youngest running
        // sequence goes out (and comes back on fresh pages), so many swaps are compared.
        batches += 1;
        if (batches % 3 == 0) {
            var live: u32 = 0;
            var youngest: ?u32 = null;
            for (by_age[0..n_by]) |i| if (!swapped[i]) {
                live += 1;
                youngest = i;
            };
            if (live >= 2) {
                try m.swapOut(slot_of[youngest.?].?);
                swapped[youngest.?] = true;
                swap_outs += 1;
                forced += 1;
            }
        }
        var rows: [K]model.BatchRow = undefined;
        var who: [K]u32 = undefined;
        var b: usize = 0;
        for (0..K) |i| if (slot_of[i] != null and steps[i] < T and !swapped[i]) {
            rows[b] = .{ .slot = slot_of[i].?, .token = xs[i][steps[i]] };
            who[b] = @intCast(i);
            b += 1;
        };
        if (b == 0) return error.Deadlock;
        max_live = @max(max_live, @as(u32, @intCast(b)));
        const logits = try m.decodeBatch(rows[0..b]);
        for (who[0..b], 0..) |i, r| {
            try check(out, &stats, "swap-decode", i, steps[i], logits[r * vocab ..][0..vocab], ref[(i * T + steps[i]) * vocab ..][0..vocab]);
            steps[i] += 1;
            if (steps[i] == T) {
                try m.releasePages(slot_of[i].?);
                slot_busy[slot_of[i].?] = false;
                slot_of[i] = null;
                done += 1;
            }
        }
    }
    // Error cases: swapping an empty slot, swapping twice, dropping a swapped slot.
    var errors_ok = true;
    if (m.swapOut(1)) |_| errors_ok = false else |e| errors_ok = e == error.PagesMissing;
    try m.select(1);
    try m.reset();
    try m.ensurePages(1, 100);
    try m.swapOut(1);
    if (m.swapOut(1)) |_| errors_ok = false else |e| errors_ok = errors_ok and e == error.SlotSwapped;
    if (m.ensurePages(1, 200)) |_| errors_ok = false else |e| errors_ok = errors_ok and e == error.SlotSwapped;
    try m.releasePages(1);
    const all_free = m.freePages() == pool and m.hostFreePages() == host_pages;
    const passed = stats.failures == 0 and errors_ok and all_free and swap_outs - forced > 0 and forced >= 10 and swap_ins == swap_outs and max_live >= 2;
    try out.print("{{\"summary\":true,\"mode\":\"swap\",\"pool_pages\":{d},\"host_pages\":{d},\"compared\":{d},\"failures\":{d},\"swap_outs\":{d},\"swap_ins\":{d},\"self_swaps\":{d},\"forced\":{d},\"max_live\":{d},\"errors_ok\":{},\"all_pages_returned\":{},\"passed\":{}}}\n", .{ pool, host_pages, stats.compared, stats.failures, swap_outs, swap_ins, self_swaps, forced, max_live, errors_ok, all_free, passed });
    try out.flush();
    if (!passed) std.process.exit(1);
}
