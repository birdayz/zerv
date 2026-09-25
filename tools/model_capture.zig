//! Verification tool: run a teacher-forced token sequence through the native model and
//! write named intermediates + logits in the libllama capture format. Also checks that
//! capture and plain steps produce bit-identical logits and that reset reproduces them.
//! Usage: zerv-model-capture MODEL TOKENS.json NAMES.txt OUT_DIR CONTEXT[:KV_MIB][@OPTION...] [CHUNK CAPTURE_MIB [PRECISION[@native|@spirv]]]
//! KV_MIB caps each KV buffer (forces the KV caches into several buffers; model.md).
//! OPTION: a KV cache type, f32 (default) or f16 (model.md, "KV precision"), or a matvec
//! accumulation, fma (default) or separate (matvec-push.md), or page=N|page=context, the KV
//! page tokens (concurrent.md, "Addressing"; default 128).
//! CHUNK > 0 selects batched prefill (teacher-forced in chunks of CHUNK tokens).
//! CHUNK:SPLIT additionally ends a chunk at token SPLIT, as a prefix-cache restore at
//! SPLIT does (docs/specs/prefix-cache.md); the plain replays are split there as well.
//! PRECISION: fp32 (default) or f16 (explicit f16 prefill projections, block 14).
const std = @import("std");
const zerv = @import("zerv");
const model = zerv.model;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 6 and args.len != 8 and args.len != 9) return error.Usage;
    const io = init.io;
    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var tokens_file = try zerv.artifact.MappedFile.open(io, args[2], 16 * 1024 * 1024);
    defer tokens_file.deinit();
    const Tokens = struct { n_prompt: u32, n_vocab: u32, tokens: []const u32 };
    const tokens = try std.json.parseFromSliceLeaky(Tokens, a, tokens_file.bytes, .{});
    var names_file = try zerv.artifact.MappedFile.open(io, args[3], 1024 * 1024);
    defer names_file.deinit();
    var names: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, names_file.bytes, "\n\r ");
    while (it.next()) |name| try names.append(a, name);
    var parts = std.mem.splitScalar(u8, args[5], '@');
    const ctx_arg = parts.first();
    var kv_type: model.KvType = .f32;
    var accumulation: zerv.matvec.Accumulation = .fma;
    var kv_page: u32 = model.layout.default_kv_page;
    while (parts.next()) |option| {
        if (std.mem.startsWith(u8, option, "page=")) kv_page = try parsePage(option["page=".len..]) //
        else if (std.meta.stringToEnum(model.KvType, option)) |kv| kv_type = kv //
        else if (std.meta.stringToEnum(zerv.matvec.Accumulation, option)) |acc| accumulation = acc //
        else return error.Usage;
    }
    const ctx_colon = std.mem.indexOfScalar(u8, ctx_arg, ':');
    const context = try std.fmt.parseInt(u32, ctx_arg[0 .. ctx_colon orelse ctx_arg.len], 10);
    const kv_mib: u64 = if (ctx_colon) |c| try std.fmt.parseInt(u64, ctx_arg[c + 1 ..], 10) else 0;
    const chunk_arg = if (args.len >= 8) args[6] else "0";
    const colon = std.mem.indexOfScalar(u8, chunk_arg, ':');
    const chunk: u32 = try std.fmt.parseInt(u32, chunk_arg[0 .. colon orelse chunk_arg.len], 10);
    const split: u32 = if (colon) |c| try std.fmt.parseInt(u32, chunk_arg[c + 1 ..], 10) else 0;
    if (split > 0 and (chunk == 0 or split >= tokens.tokens.len)) return error.Usage;
    const capture_mib: u64 = if (args.len >= 8) try std.fmt.parseInt(u64, args[7], 10) else 512;
    // PRECISION[@native|@spirv]: the gemm_f16x machine code (`Model.Options.gemm_code`).
    var precision: model.gemm.Precision = .fp32;
    var gemm_code: model.gemm.Code = .spirv;
    if (args.len == 9) {
        var pp = std.mem.splitScalar(u8, args[8], '@');
        precision = std.meta.stringToEnum(model.gemm.Precision, pp.first()) orelse return error.Usage;
        while (pp.next()) |option| gemm_code = std.meta.stringToEnum(model.gemm.Code, option) orelse return error.Usage;
    }

    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .cooperative_matrix = model.gemm.deviceNeeds(precision).cooperative_matrix, .subgroup_size_control = model.gemm.deviceNeeds(precision).subgroup_size_control, .storage16 = kv_type == .f16, .pipeline_binaries = model.gemm.needsPipelineBinaries(precision, gemm_code) });
    defer device.deinit() catch @panic("live device resources");
    const load_start = std.Io.Clock.awake.now(io);
    var m: model.Model = undefined;
    try m.init(&device, &container, .{ .context = context, .prefill_rows = chunk, .prefill_precision = precision, .kv_capacity = kv_mib * 1024 * 1024, .kv_type = kv_type, .kv_page_tokens = kv_page, .matvec_accumulation = accumulation, .gemm_code = gemm_code });
    defer m.deinit();
    std.debug.print("gemm_f16x: {s}\n", .{m.gemmCodeStatus()});
    const load_ns = load_start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
    std.debug.print("loaded banks={d} kv_buffers={d} kv_pages={d}x{d} pipelines={d} kernels={d} load_ms={d}\n", .{ m.bank_count, m.state_layout.kv_buffers, m.state_layout.pages, m.state_layout.page, m.pipeline_count, device.kernels, @divTrunc(load_ns, std.time.ns_per_ms) });

    var capture_buffer = try zerv.gpu.Buffer.init(&device, capture_mib * 1024 * 1024, .host);
    defer capture_buffer.deinit() catch @panic("capture buffer in use");
    const entries = try a.alloc(model.Capture.Entry, 8192);
    var capture: model.Capture = .{ .buffer = &capture_buffer, .filter = names.items, .entries = entries };
    var capture_commands = try zerv.gpu.Commands.init(&device);
    defer capture_commands.deinit() catch @panic("capture command pending");
    try capture_commands.begin();
    if (chunk == 0) try m.record(&capture_commands, .{ .capture = &capture });
    try capture_commands.end();
    // Prefill: one capture command per recorded plan (each plan has its own row count, so
    // its own entry table); all write into the same capture buffer, one run at a time.
    var plan_commands: [model.max_plans]zerv.gpu.Commands = undefined;
    var plan_captures: [model.max_plans]model.Capture = undefined;
    var live_plans: usize = 0;
    defer for (plan_commands[0..live_plans]) |*c| c.deinit() catch @panic("plan capture command pending");
    if (chunk > 0) for (0..m.plan_count) |p| {
        plan_captures[p] = .{ .buffer = &capture_buffer, .filter = names.items, .entries = try a.alloc(model.Capture.Entry, 8192) };
        plan_commands[p] = try zerv.gpu.Commands.init(&device);
        live_plans += 1;
        try plan_commands[p].begin();
        try m.recordPrefill(&plan_commands[p], p, .{ .capture = &plan_captures[p] });
        try plan_commands[p].end();
    };

    const dir = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer dir.close(io);
    const blob = try dir.createFile(io, "tensors.bin", .{ .exclusive = true });
    defer blob.close(io);
    var blob_buffer: [1 << 16]u8 = undefined;
    var blob_writer: std.Io.File.Writer = .init(blob, io, &blob_buffer);
    const index = try dir.createFile(io, "index.jsonl", .{ .exclusive = true });
    defer index.close(io);
    var index_buffer: [1 << 16]u8 = undefined;
    var index_writer: std.Io.File.Writer = .init(index, io, &index_buffer);
    const logits_file = try dir.createFile(io, "logits.bin", .{ .exclusive = true });
    defer logits_file.close(io);
    var logits_buffer: [1 << 16]u8 = undefined;
    var logits_writer: std.Io.File.Writer = .init(logits_file, io, &logits_buffer);

    const n = tokens.tokens.len;
    if (chunk > 0) return prefillMode(io, a, &m, plan_captures[0..live_plans], plan_commands[0..live_plans], &capture_buffer, dir, tokens.tokens, tokens.n_prompt, chunk, split, context, load_ns, &blob_writer.interface, &index_writer.interface, &logits_writer.interface);
    const reference = try a.alloc(f32, n * model.config.vocab);
    var offset: u64 = 0;
    for (tokens.tokens, 0..) |token, t| {
        const logits = try m.run(&capture_commands, token);
        @memcpy(reference[t * model.config.vocab ..][0..model.config.vocab], logits);
        try logits_writer.interface.writeAll(std.mem.sliceAsBytes(logits));
        const mapped = try capture_buffer.mapped();
        for (capture.entries[0..capture.count]) |*e| {
            const bytes = mapped[e.offset_words * 4 ..][0 .. @as(usize, e.words) * 4];
            for (std.mem.bytesAsSlice(f32, bytes)) |v| if (!std.math.isFinite(v)) return error.NonFiniteCapture;
            try blob_writer.interface.writeAll(bytes);
            var name_buf: [64]u8 = undefined;
            const name = if (e.layer < 0) model.Capture.entryName(e) else try std.fmt.bufPrint(&name_buf, "{s}-{d}", .{ model.Capture.entryName(e), e.layer });
            try index_writer.interface.print("{{\"token\":{d},\"name\":\"{s}\",\"op\":\"zerv\",\"ne\":[{d},1,1,1],\"offset\":{d},\"count\":{d}}}\n", .{ t, name, e.words, offset, e.words });
            offset += @as(u64, e.words) * 4;
        }
    }
    try blob_writer.interface.flush();
    try index_writer.interface.flush();
    try logits_writer.interface.flush();

    // Gate 4: plain steps equal capture steps; reset reproduces the sequence.
    var mismatches: [2]usize = .{ 0, 0 };
    var step_ns: [2]std.ArrayList(i96) = .{ .empty, .empty };
    for (0..2) |pass| {
        try m.reset();
        for (tokens.tokens, 0..) |token, t| {
            const start = std.Io.Clock.awake.now(io);
            const logits = try m.step(token);
            try step_ns[pass].append(a, start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
            const expected = reference[t * model.config.vocab ..][0..model.config.vocab];
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(expected))) mismatches[pass] += 1;
        }
    }
    // Bounds: invalid token and a full context are rejected without GPU work.
    if (m.step(model.config.vocab)) |_| return error.ExpectedInvalidToken else |e| if (e != error.InvalidToken) return e;
    const before_fill = m.position;
    while (m.position < context) _ = try m.step(0);
    if (m.step(0)) |_| return error.ExpectedContextFull else |e| if (e != error.ContextFull) return e;
    const filled = m.position - before_fill;
    try m.reset();
    const after_reset = try m.step(tokens.tokens[0]);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(after_reset), std.mem.sliceAsBytes(reference[0..model.config.vocab]))) return error.ResetMismatch;
    const summary = try dir.createFile(io, "native-summary.json", .{ .exclusive = true });
    defer summary.close(io);
    var summary_buffer: [4096]u8 = undefined;
    var summary_writer: std.Io.File.Writer = .init(summary, io, &summary_buffer);
    try std.json.Stringify.value(.{
        .tokens = n,
        .context = context,
        .banks = m.bank_count,
        .pipelines = m.pipeline_count,
        .load_ns = load_ns,
        .capture_entries_per_token = capture.count,
        .plain_vs_capture_logit_mismatches = mismatches[0],
        .reset_replay_logit_mismatches = mismatches[1],
        .plain_step_ns = step_ns[0].items,
        .replay_step_ns = step_ns[1].items,
        .bounds_checked = .{ .invalid_token = true, .context_full_after_steps = filled, .reset_after_full_matches = true },
    }, .{}, &summary_writer.interface);
    try summary_writer.interface.flush();
    std.debug.print("tokens={d} captures/token={d} mismatches plain={d} replay={d}\n", .{ n, capture.count, mismatches[0], mismatches[1] });
    if (mismatches[0] != 0 or mismatches[1] != 0) return error.NondeterministicLogits;
}

/// Prefill `tokens` with a chunk boundary at `split` (0 = none), as a restore at `split` does.
fn splitPrefill(m: *model.Model, tokens: []const u32, split: u32) ![]const f32 {
    if (split > 0 and split < tokens.len) _ = try m.prefill(tokens[0..split]);
    return m.prefill(tokens[if (split < tokens.len) split else 0..]);
}

fn prefillMode(io: std.Io, a: std.mem.Allocator, m: *model.Model, captures: []model.Capture, commands: []zerv.gpu.Commands, capture_buffer: *zerv.gpu.Buffer, dir: std.Io.Dir, tokens: []const u32, n_prompt: u32, chunk: u32, split: u32, context: u32, load_ns: i96, blob: *std.Io.Writer, index: *std.Io.Writer, logits_out: *std.Io.Writer) !void {
    const V = model.config.vocab;
    const nan_row = try a.alloc(f32, V);
    @memset(nan_row, std.math.nan(f32));
    var logit_positions: std.ArrayList(usize) = .empty;
    const final_logits = try a.alloc(f32, V);
    var offset: u64 = 0;
    var start: usize = 0;
    var plans_used: [model.max_plans]u32 = @splat(0);
    while (start < tokens.len) {
        const until: usize = if (start < split) split else tokens.len;
        const next = m.nextChunk(@intCast(until - start));
        const count = next.rows;
        const plan = next.plan;
        plans_used[plan] += 1;
        const capture = &captures[plan];
        const logits = try m.runChunk(&commands[plan], plan, tokens[start..][0..count]);
        @memcpy(final_logits, logits);
        const mapped = try capture_buffer.mapped();
        const words = std.mem.bytesAsSlice(f32, mapped);
        for (0..count) |r| {
            for (capture.entries[0..capture.count]) |*e| {
                if (e.last_only and r != count - 1) continue;
                const row: u64 = if (e.last_only) 0 else r;
                const values = words[e.offset_words + row * e.stride ..][0..e.words];
                for (values) |v| if (!std.math.isFinite(v)) return error.NonFiniteCapture;
                try blob.writeAll(std.mem.sliceAsBytes(values));
                var name_buf: [64]u8 = undefined;
                const name = if (e.layer < 0) model.Capture.entryName(e) else try std.fmt.bufPrint(&name_buf, "{s}-{d}", .{ model.Capture.entryName(e), e.layer });
                try index.print("{{\"token\":{d},\"name\":\"{s}\",\"op\":\"zerv\",\"ne\":[{d},1,1,1],\"offset\":{d},\"count\":{d}}}\n", .{ start + r, name, e.words, offset, e.words });
                offset += @as(u64, e.words) * 4;
            }
            if (r == count - 1) {
                try logits_out.writeAll(std.mem.sliceAsBytes(logits));
                try logit_positions.append(a, start + r);
            } else try logits_out.writeAll(std.mem.sliceAsBytes(nan_row));
        }
        start += count;
    }
    try blob.flush();
    try index.flush();
    try logits_out.flush();

    // Determinism: plain prefill (no capture) twice from reset equals the capture run.
    var mismatches: [2]usize = .{ 0, 0 };
    var prefill_ns: [2]i96 = undefined;
    for (0..2) |pass| {
        try m.reset();
        const t0 = std.Io.Clock.awake.now(io);
        const logits = try splitPrefill(m, tokens, split);
        prefill_ns[pass] = t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(final_logits))) mismatches[pass] += 1;
    }
    // Serving pattern: prefill the prompt, then decode the rest one token at a time.
    try m.reset();
    const serving = try dir.createFile(io, "serving-logits.bin", .{ .exclusive = true });
    defer serving.close(io);
    var serving_buffer: [1 << 16]u8 = undefined;
    var serving_writer: std.Io.File.Writer = .init(serving, io, &serving_buffer);
    try serving_writer.interface.writeAll(std.mem.sliceAsBytes(try splitPrefill(m, tokens[0..n_prompt], if (split < n_prompt) split else 0)));
    for (tokens[n_prompt..]) |token| try serving_writer.interface.writeAll(std.mem.sliceAsBytes(try m.step(token)));
    try serving_writer.interface.flush();
    // Bounds: invalid token and context overflow rejected before GPU work.
    try m.reset();
    if (m.prefill(&.{model.config.vocab})) |_| return error.ExpectedInvalidToken else |e| if (e != error.InvalidToken) return e;
    const too_many = try a.alloc(u32, context + 1);
    @memset(too_many, 0);
    if (m.prefill(too_many)) |_| return error.ExpectedContextFull else |e| if (e != error.ContextFull) return e;
    if (m.position != 0) return error.PositionChangedOnRejection;

    const summary = try dir.createFile(io, "native-summary.json", .{ .exclusive = true });
    defer summary.close(io);
    var summary_buffer: [8192]u8 = undefined;
    var summary_writer: std.Io.File.Writer = .init(summary, io, &summary_buffer);
    try std.json.Stringify.value(.{
        .mode = "prefill",
        .chunk = chunk,
        .split = split,
        .precision = @tagName(m.options.prefill_precision),
        .tokens = tokens.len,
        .context = context,
        .banks = m.bank_count,
        .pipelines = m.pipeline_count,
        .gemm_pipelines = m.gemm_count,
        .load_ns = load_ns,
        .capture_entries_per_chunk = captures[0].count,
        .plans = m.plans[0..m.plan_count],
        .chunks_per_plan = plans_used[0..m.plan_count],
        .logit_positions = logit_positions.items,
        .serving_first_position = n_prompt - 1,
        .plain_vs_capture_logit_mismatches = mismatches[0],
        .reset_replay_logit_mismatches = mismatches[1],
        .plain_prefill_ns = prefill_ns,
        .bounds_checked = .{ .invalid_token = true, .context_overflow = true },
    }, .{}, &summary_writer.interface);
    try summary_writer.interface.flush();
    std.debug.print("prefill chunk={d} tokens={d} captures/chunk={d} mismatches plain={d} replay={d} prefill_ms={d}\n", .{ chunk, tokens.len, captures[0].count, mismatches[0], mismatches[1], @divTrunc(prefill_ns[0], std.time.ns_per_ms) });
    if (mismatches[0] != 0 or mismatches[1] != 0) return error.NondeterministicLogits;
}

/// `context` (0: one page) or a token count.
fn parsePage(text: []const u8) !u32 {
    return if (std.mem.eql(u8, text, "context")) 0 else try std.fmt.parseInt(u32, text, 10);
}
