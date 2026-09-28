//! zerv: native Qwen3.8-27B server, OpenAI Chat Completions v1 (`POST /v1/chat/completions`).
const std = @import("std");
const zerv = @import("zerv");

const usage =
    \\usage: zerv --model PATH [--host 127.0.0.1] [--port 8080] [--context 8192|max]
    \\            [--alias qwen3.8-27b] [--max-waiting 16] [--vram-budget-gib 23]
    \\            [--prefill-chunk 512]  (0 = token-by-token prefill)
    \\            [--drain-timeout 30]  (seconds; SIGINT/SIGTERM drain, a second one exits)
    \\            [--prefill-precision fp32]  (fp32 | f16: explicit f16 WMMA prompt projections)
    \\            [--prefill-attention fp32]  (fp32 | wmma: WMMA prompt attention with f16 Q and P; needs --prefill-precision f16, --kv-type f16, --parallel 1)
    \\            [--gemm-code native]  (f16 mode, Q4_0 prompt projections: native = our RDNA3 machine code for gemm_f16x, same values, ~12% lower TTFT; spirv = the compiled SPIR-V; native falls back to spirv on other drivers)
    \\            [--prefix-cache-slots 8, with --parallel N: 3N]  (recurrent-state snapshots, ~150 MiB each; 0 = no prefix cache)
    \\            [--prefix-cache radix]  (--parallel > 1: the prefix-cache policy: flat = checkpoint list; radix = prefix tree with deduplication of pages on insert)
    \\            [--prefix-cache-memory device, with --parallel: host]  (device: snapshots in VRAM; host: in system RAM, no VRAM, ~10 ms TTFT per save)
    \\            [--matvec-accumulation fma]  (decode/verify dot products: fma one rounding per step; separate: the pre-2026-09-24 multiply+add)
    \\            [--decode-fusion on]  (fused decode FFN input: same values, one dispatch fewer per layer; off: separate)
    \\            [--verify-fusion on]  (the same fusion in the speculative verify pass; off: separate)
    \\            [--delta-state-out on]  (DeltaNet kernels without the 128-VGPR scratch spill, same values; off: previous modules)
    \\            [--kv-type f32]  (attention KV cache: f32 exact, f16 half the bytes; docs/specs/model.md)
    \\            [--parallel 1]  (N > 1: up to N requests decode together in one batch, each bit-identical to decoding alone; speculative decoding and the prefix cache are off; --context is per request, KV for N of them)
    \\            [--f16-small-tile on]  (f16 prefill: short chunks whose 128-row tiles would leave the GPU idle use a 32-row tile; same values; off: previous kernels)
    \\            [--prefill-stall-ms 100]  (--parallel > 1: while a prompt prefills, running requests get a token at least every N ms plus one 4-layer segment; chunk: only between 512-token chunks)
    \\            [--prefill-order shortest]  (--parallel > 1: pending prompt with the fewest remaining tokens next; fifo: arrival order)
    \\            [--kv-pool shared]  (--parallel > 1: static = each request owns --context tokens of KV; shared = one pool (admission: --kv-admit), --context is the per-request maximum (max: the whole pool))
    \\            [--kv-pool-pages 0]  (shared pool size in 128-token pages; 0: all memory left)
    \\            [--kv-admit prompt]  (shared pool: reserve = admit prompt + max_tokens; prompt = admit the prompt, grow page by page, swap sequences to host memory when the pool runs out)
    \\            [--kv-swap-mib 8192 with --kv-admit prompt, else 0]  (host memory for swapped sequences)
    \\            [--kv-swap-slice-ms 10000]  (a sequence swapped out this long takes the place of the one that has run longest; 0: it waits for memory to be released)
    \\            [--prefill-pack 8]  (--parallel > 1, f16 prefill: up to N pending prompts prefill together in one chunk, each bit-identical to alone; 1: one at a time)
    \\            [--f16-split shape]  (f16 prefill: split-K of the small FP32 projections fixed per shape, as packing needs; plan: per chunk size, the rule before 2026-09-26)
    \\            [--kv-page-tokens 128]  (KV page size, a multiple of 128; context: one page, the layout before paging; same values)
    \\            [--vram-reserve-mib 1024]  (with --context max: VRAM left free for other processes)
    \\            [--embedding-memory host]  (host: token embedding in system RAM, 682 MiB less VRAM, no measured cost; device: in VRAM)
    \\            [--spec-draft 3]  (MTP speculative decoding: drafts per step, 0..4; 0 = off, no MTP memory; lossless)
    \\            [--spec-policy adaptive]  (adaptive: verify the drafts that maximize expected tokens/s; fixed: all)
    \\            [--spec-draft-vocab full]  (N: the draft head covers token ids 0..N-1 only, cheaper drafts; output unchanged)
    \\            [--sampler-order id]  (id: sums in token-id order, no vocabulary sort; sorted: the pre-2026-09-24 order, same distribution, its seeded draws)
    \\
;

/// Set by the first SIGINT/SIGTERM; `Server.run` polls it and drains.
var stop_requested: std.atomic.Value(bool) = .init(false);

fn onStopSignal(_: std.posix.SIG) callconv(.c) void {
    // Async-signal-safe: one atomic swap, or an immediate exit on the second signal.
    if (stop_requested.swap(true, .acq_rel)) std.os.linux.exit_group(130);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var model_path: ?[]const u8 = null;
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var context: u32 = 8192;
    var alias: []const u8 = "qwen3.8-27b";
    var max_waiting: u32 = 16;
    var budget_gib: u64 = 23;
    var prefill_chunk: u32 = 512;
    var drain_s: u32 = 30;
    var precision: zerv.model.gemm.Precision = .fp32;
    var prefill_attention: zerv.model.PrefillAttention = .fp32;
    var snapshot_slots_arg: ?u32 = null;
    var snapshot_memory_arg: ?zerv.gpu.Location = null;
    var prefix_cache_kind: zerv.session.kvcache.Kind = .radix;
    var embedding_memory: zerv.gpu.Location = .host;
    var reserve_mib: u64 = 1024;
    var kv_type: zerv.model.KvType = .f32;
    var kv_page: u32 = zerv.model.layout.default_kv_page;
    var parallel: u32 = 1;
    var f16_small_tile = true;
    var stall: zerv.serve.batcher.Stall = .{ .ns = 100 * std.time.ns_per_ms };
    var prefill_order: zerv.serve.batcher.Order = .shortest;
    var prefill_pack: u32 = 8;
    var kv_share = true;
    var kv_pool_pages: u32 = 0;
    var kv_admit_arg: ?zerv.serve.Admission = null;
    var kv_swap_mib: ?u64 = null;
    var kv_swap_slice_ms: u64 = 10000;
    var f16_split: zerv.model.F16Split = .shape;
    var decode_fusion = true;
    var verify_fusion = true;
    var delta_state_out = true;
    var gemm_code: zerv.model.gemm.Code = .native;
    var accumulation: zerv.matvec.Accumulation = .fma;
    var spec_draft: u32 = 3;
    var spec_adaptive = true;
    var sampler_order: zerv.session.sampler.Order = .id;
    var draft_vocab: u32 = 0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (i + 1 >= args.len) {
            std.debug.print("{s}", .{usage});
            return error.InvalidArguments;
        }
        const value = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, arg, "--model")) model_path = value //
        else if (std.mem.eql(u8, arg, "--host")) host = value //
        else if (std.mem.eql(u8, arg, "--port")) port = try std.fmt.parseInt(u16, value, 10) //
        else if (std.mem.eql(u8, arg, "--context")) context = try parseContext(value) //
        else if (std.mem.eql(u8, arg, "--alias")) alias = value //
        else if (std.mem.eql(u8, arg, "--max-waiting")) max_waiting = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--vram-budget-gib")) budget_gib = try std.fmt.parseInt(u64, value, 10) //
        else if (std.mem.eql(u8, arg, "--prefill-chunk")) prefill_chunk = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--drain-timeout")) drain_s = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--prefill-attention")) prefill_attention = std.meta.stringToEnum(zerv.model.PrefillAttention, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--prefill-precision")) precision = std.meta.stringToEnum(zerv.model.gemm.Precision, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--prefix-cache-slots")) snapshot_slots_arg = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--prefix-cache")) prefix_cache_kind = std.meta.stringToEnum(zerv.session.kvcache.Kind, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--prefix-cache-memory")) snapshot_memory_arg = std.meta.stringToEnum(zerv.gpu.Location, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--vram-reserve-mib")) reserve_mib = std.math.cast(u32, try std.fmt.parseInt(u64, value, 10)) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--decode-fusion")) decode_fusion = try parseSwitch(value) //
        else if (std.mem.eql(u8, arg, "--verify-fusion")) verify_fusion = try parseSwitch(value) //
        else if (std.mem.eql(u8, arg, "--delta-state-out")) delta_state_out = try parseSwitch(value) //
        else if (std.mem.eql(u8, arg, "--gemm-code")) gemm_code = std.meta.stringToEnum(zerv.model.gemm.Code, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--matvec-accumulation")) accumulation = std.meta.stringToEnum(zerv.matvec.Accumulation, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--kv-type")) kv_type = std.meta.stringToEnum(zerv.model.KvType, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--f16-small-tile")) f16_small_tile = try parseSwitch(value) //
        else if (std.mem.eql(u8, arg, "--parallel")) parallel = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--prefill-stall-ms")) stall = if (std.mem.eql(u8, value, "chunk")) .chunk else .{ .ns = try std.fmt.parseInt(u64, value, 10) * std.time.ns_per_ms } //
        else if (std.mem.eql(u8, arg, "--prefill-order")) prefill_order = std.meta.stringToEnum(zerv.serve.batcher.Order, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--prefill-pack")) prefill_pack = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--kv-pool")) kv_share = if (std.mem.eql(u8, value, "shared")) true else if (std.mem.eql(u8, value, "static")) false else return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--kv-pool-pages")) kv_pool_pages = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--kv-admit")) kv_admit_arg = std.meta.stringToEnum(zerv.serve.Admission, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--kv-swap-mib")) kv_swap_mib = try std.fmt.parseInt(u64, value, 10) //
        else if (std.mem.eql(u8, arg, "--kv-swap-slice-ms")) kv_swap_slice_ms = try std.fmt.parseInt(u64, value, 10) //
        else if (std.mem.eql(u8, arg, "--f16-split")) f16_split = std.meta.stringToEnum(zerv.model.F16Split, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--kv-page-tokens")) kv_page = if (std.mem.eql(u8, value, "context")) 0 else try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--embedding-memory")) embedding_memory = std.meta.stringToEnum(zerv.gpu.Location, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--spec-draft")) spec_draft = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--spec-draft-vocab")) draft_vocab = if (std.mem.eql(u8, value, "full")) 0 else try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--sampler-order")) sampler_order = std.meta.stringToEnum(zerv.session.sampler.Order, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--spec-policy")) spec_adaptive = if (std.mem.eql(u8, value, "adaptive")) true else if (std.mem.eql(u8, value, "fixed")) false else return error.InvalidArguments //
        else {
            std.debug.print("unknown option {s}\n{s}", .{ arg, usage });
            return error.InvalidArguments;
        }
    }
    const path = model_path orelse {
        std.debug.print("{s}", .{usage});
        return error.InvalidArguments;
    };
    if (parallel == 0 or parallel > zerv.serve.batcher.max_slots or parallel > zerv.model.layout.io.batch_max or (parallel > 1 and prefill_chunk < parallel)) {
        std.debug.print("zerv: --parallel must be 1..{d} (and needs --prefill-chunk >= it)\n", .{zerv.model.layout.io.batch_max});
        return error.InvalidArguments;
    }
    // One sequence: 8 snapshots of its history. Several: 3 checkpoints per slot (a system
    // prompt plus recent turns per conversation; docs/specs/concurrent.md "18d.4 design").
    var snapshot_slots: u32 = snapshot_slots_arg orelse if (parallel > 1) @min(3 * parallel, zerv.session.prefix.max_slots) else 8;
    if (parallel > 1) {
        // Per-slot speculation comes with block 18d. The prefix cache becomes checkpoints of
        // the shared pool (18d.4); a static pool has none.
        if (spec_draft > 0) std.debug.print("zerv: --parallel {d}: speculative decoding is off (per-slot version: block 18d)\n", .{parallel});
        spec_draft = 0;
        if (!kv_share and snapshot_slots > 0) {
            std.debug.print("zerv: --parallel {d} --kv-pool static: the prefix cache is off (it needs the shared pool)\n", .{parallel});
            snapshot_slots = 0;
        }
    }
    // Snapshots in VRAM for one sequence (0.5 ms copies); with several the pool needs the
    // VRAM, so they go to host memory (10-15 ms copies) unless given.
    const snapshot_memory: zerv.gpu.Location = snapshot_memory_arg orelse if (parallel > 1) .host else .device;
    if (spec_draft >= zerv.matvec.max_rows or (spec_draft > 0 and prefill_chunk < spec_draft + 1)) {
        std.debug.print("zerv: --spec-draft must be 0..{d} (and needs --prefill-chunk >= drafts + 1)\n", .{zerv.matvec.max_rows - 1});
        return error.InvalidArguments;
    }

    // Own the port before the (long) model load, so a busy port fails immediately.
    // Connections that arrive while loading wait in the kernel backlog and are served
    // once the model is ready (docs/specs/serving.md, "Startup").
    const address = try std.Io.net.IpAddress.parse(host, port);
    var listener = zerv.serve.listen.listen(address, 128) catch |e| {
        if (e == error.AddressInUse) std.debug.print("zerv: {s}:{d} is already in use (another server is listening there)\n", .{ host, port });
        return e;
    };
    var listener_owned = true; // until `Server.run` takes it
    defer if (listener_owned) listener.deinit(io);

    // The mapping is needed only until the weights are resident and the tokenizer has
    // copied its tables; it is released before serving (otherwise ~15.5 GB stays in RSS).
    var file = try zerv.artifact.MappedFile.open(io, path, 64 * 1024 * 1024 * 1024);
    var file_live = true;
    defer if (file_live) file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(gpa, file.bytes, .{});
    var container_live = true;
    defer if (container_live) container.deinit();
    var tokenizer = try zerv.tokenizer.fromGGUF(gpa, &container, .{});
    defer tokenizer.deinit();
    // The allocation cap counts every buffer; host snapshots and a host embedding are not
    // VRAM, so they add to it.
    const snapshot_host: u64 = if (snapshot_memory == .host) @as(u64, snapshot_slots) * (zerv.model.snapshot_bytes + @as(u64, if (spec_draft > 0) zerv.model.config.hidden * 4 else 0)) else 0;
    // Prompt admission (the default) needs the shared pool; a static pool reserves anyway.
    const kv_admission: zerv.serve.Admission = kv_admit_arg orelse if (kv_share and parallel > 1) .prompt else .reserve;
    if (prefill_attention == .wmma and (precision != .f16 or kv_type != .f16 or parallel > 1)) {
        std.debug.print("zerv: --prefill-attention wmma needs --prefill-precision f16, --kv-type f16 and --parallel 1\n", .{});
        return error.InvalidArguments;
    }
    if (kv_admission == .prompt and !(kv_share and parallel > 1)) {
        std.debug.print("zerv: --kv-admit prompt needs --kv-pool shared and --parallel > 1\n", .{});
        return error.InvalidArguments;
    }
    const swap_bytes: u64 = (kv_swap_mib orelse if (kv_admission == .prompt) @as(u64, 8192) else 0) * 1024 * 1024;
    if (swap_bytes > 0 and !(kv_share and parallel > 1)) {
        std.debug.print("zerv: --kv-swap-mib needs --kv-pool shared and --parallel > 1\n", .{});
        return error.InvalidArguments;
    }
    const embed_host: u64 = if (embedding_memory == .host) (container.findTensor("token_embd.weight") orelse return error.MissingTensor).data.len + 65536 else 0;
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = budget_gib * 1024 * 1024 * 1024 + snapshot_host + embed_host + swap_bytes, .cooperative_matrix = zerv.model.gemm.deviceNeeds(precision).cooperative_matrix, .subgroup_size_control = zerv.model.gemm.deviceNeeds(precision).subgroup_size_control, .storage16 = kv_type == .f16, .pipeline_binaries = zerv.model.gemm.needsPipelineBinaries(precision, gemm_code) });
    defer device.deinit() catch @panic("device resources still live");
    var context_text: [16]u8 = undefined;
    const context_name = if (context == zerv.model.context_max) "max" else std.fmt.bufPrint(&context_text, "{d}", .{context}) catch unreachable;
    std.debug.print("zerv: loading {s} on {s} (context {s}, prefill chunk {d}, prefill precision {s}, KV {s}, prefix cache slots {d} = {d} MiB of {s} memory, speculative drafts {d}, {s})\n", .{ path, device.name(), context_name, prefill_chunk, @tagName(precision), @tagName(kv_type), snapshot_slots, snapshot_slots * zerv.model.snapshot_bytes / (1024 * 1024), @tagName(snapshot_memory), spec_draft, if (spec_adaptive) "adaptive" else "fixed" });
    var model: zerv.model.Model = undefined;
    model.init(&device, &container, .{ .context = context, .prefill_rows = prefill_chunk, .prefill_precision = precision, .prefill_attention = prefill_attention, .snapshots = snapshot_slots, .snapshot_memory = snapshot_memory, .embedding_memory = embedding_memory, .context_reserve = reserve_mib * 1024 * 1024, .kv_type = kv_type, .kv_page_tokens = kv_page, .decode_fusion = decode_fusion, .verify_fusion = verify_fusion, .delta_state_out = delta_state_out, .gemm_code = gemm_code, .matvec_accumulation = accumulation, .verify_rows = if (spec_draft > 0) spec_draft + 1 else 0, .mtp = spec_draft > 0, .draft_vocab = draft_vocab, .slots = parallel, .batch_rows = if (parallel > 1) parallel else 0, .f16_small_tile = f16_small_tile, .f16_split = f16_split, .kv_share = kv_share and parallel > 1, .kv_pages = kv_pool_pages, .swap_bytes = swap_bytes }) catch |e| {
        if (e == error.InvalidKvPage) std.debug.print("zerv: --kv-page-tokens must be context or a positive multiple of {d} up to {d}\n", .{ zerv.model.layout.kv_page_quantum, zerv.model.layout.max_kv_page });
        if (e == error.InvalidDraftVocab) std.debug.print("zerv: --spec-draft-vocab must be full or 1..{d}\n", .{zerv.model.config.vocab});
        if (e == error.InvalidContext) std.debug.print("zerv: --context must be positive and even (the attention kernels read key pairs), and with --prefill-chunk > 0 a multiple of 32 (e.g. {d})\n", .{if (prefill_chunk > 0) context / 32 * 32 else context / 2 * 2});
        if (e == error.ContextTooLarge) std.debug.print("zerv: --context {d} is too large: the model was trained on {d} tokens, one KV cache holds at most {d}, and the activation arena (24 x verify rows x context scores) must stay below 4 GiB\n", .{ context, model.hyper.context_length, zerv.model.layout.maxContext(0xf000_0000, kv_type, kv_page) });
        if (e == error.VramBudgetUnknown) std.debug.print("zerv: --context max needs the driver's memory budget (VK_EXT_memory_budget); give --context explicitly\n", .{});
        if (e == error.InsufficientVram) {
            if (model.vram.before_context and model.vram.cap_limited) std.debug.print("zerv: --vram-budget-gib {d} leaves no room for any context: the model needs {d} MiB before the KV cache (weights, snapshots, {d} MiB headroom, --vram-reserve-mib {d})\n", .{ budget_gib, model.vram.needed >> 20, zerv.model.vram_headroom >> 20, reserve_mib }) //
            else if (model.vram.before_context) std.debug.print("zerv: not enough free VRAM for any context: the model needs {d} MiB before the KV cache (weights, snapshots, {d} MiB headroom, --vram-reserve-mib {d}), {d} MiB are free ({s})\n", .{ model.vram.needed >> 20, zerv.model.vram_headroom >> 20, reserve_mib, model.vram.free.? >> 20, if (model.vram.needed - reserve_mib * 1024 * 1024 < model.vram.free.?) "lower --vram-reserve-mib" else "another process is using the GPU?" }) //
            else std.debug.print("zerv: not enough free VRAM: the model needs {d} MiB (including {d} MiB headroom), {d} MiB are free (another process is using the GPU?)\n", .{ model.vram.needed >> 20, zerv.model.vram_headroom >> 20, model.vram.free.? >> 20 });
        }
        return e;
    };
    if (context == zerv.model.context_max) std.debug.print("zerv: context max = {d} tokens\n", .{model.state_layout.context});
    std.debug.print("zerv: gemm_f16x: {s}\n", .{model.gemmCodeStatus()});
    std.debug.print("zerv: KV cache: {d} pages of {d} tokens{s}\n", .{ model.state_layout.pages, model.state_layout.page, if (!model.options.kv_share) "" else if (kv_admission == .prompt) " (shared pool, prompts admitted, pages grown per step)" else " (shared pool, prompt + max_tokens admitted per request)" });
    if (model.swap_pages > 0) std.debug.print("zerv: swap store: {d} MiB of host memory, {d} KV pages\n", .{ swap_bytes >> 20, model.swap_pages });
    if (parallel > 1) {
        std.debug.print("zerv: parallel: {d} sequences of up to {d} tokens, batched decode, prefill order {s}, ", .{ parallel, model.state_layout.context, @tagName(prefill_order) });
        switch (stall) {
            .chunk => std.debug.print("decode steps between prefill chunks\n", .{}),
            .ns => |ns| std.debug.print("decode steps at least every {d} ms during prefill ({d} segments per chunk)\n", .{ ns / std.time.ns_per_ms, zerv.model.prefill_segments }),
        }
        const pack = if (model.packable()) @min(prefill_pack, model.pack_seqs) else 1;
        std.debug.print("zerv: prefill packing: up to {d} prompts per chunk{s}\n", .{ pack, if (model.packable()) "" else " (needs --prefill-precision f16 and --f16-split shape)" });
    }
    std.debug.print("zerv: FFN input fused in {d} (decode) and {d} (verify) of {d} layers\n", .{ model.fused_ffn_layers, model.fused_verify_layers, zerv.model.config.layers });
    if (model.vram.free) |free| std.debug.print("zerv: VRAM: needed {d} MiB of {d} MiB free\n", .{ model.vram.needed >> 20, free >> 20 }) //
    else std.debug.print("zerv: VRAM: needed {d} MiB (the driver reports no budget; not checked)\n", .{model.vram.needed >> 20});
    defer model.deinit();
    var native = try zerv.serve.Native.init(io, gpa, &model, &tokenizer);
    defer native.deinit(gpa);
    if (spec_draft > 0 and spec_adaptive) native.spec_policy = .init();
    native.sampler_order = sampler_order;
    native.admission = kv_admission;
    var model_backend_store = false;
    // --parallel N > 1: the batcher owns the model on its scheduler task.
    var model_backend: zerv.serve.ModelBackend = .{ .m = &model };
    var batch = try zerv.serve.Batcher.init(io, &model_backend, .{ .slots = parallel, .vocab = zerv.model.config.vocab, .stall = stall, .order = prefill_order, .swap_slice = if (model.swap_pages > 0 and kv_swap_slice_ms > 0) std.Io.Duration.fromMilliseconds(@intCast(kv_swap_slice_ms)) else null, .pack = if (model.packable()) @max(1, @min(prefill_pack, model.pack_seqs)) else 1 });
    var scheduler: ?std.Io.Future(void) = null;
    if (parallel > 1) {
        if (model.options.kv_share and model.snapshot_slots > 0) {
            try model_backend.initCache(gpa, prefix_cache_kind, native.boundary);
            model_backend_store = true;
            std.debug.print("zerv: prefix checkpoints: {d} ({s} memory, {s} policy)\n", .{ model.snapshot_slots, @tagName(snapshot_memory), @tagName(prefix_cache_kind) });
        }
        try native.attachBatcher(&batch, &model_backend);
        scheduler = try io.concurrent(zerv.serve.Batcher.run, .{&batch});
    }
    defer if (scheduler) |*task| {
        batch.stop();
        task.await(io);
        const st = batch.stats;
        const span: f64 = @floatFromInt(@max(st.last_ns - st.first_ns, 1));
        std.debug.print("zerv: batcher: {d} batches ({d} rows, {d} partial, {d} inside prefill chunks), {d} prefill chunks ({d} packed, {d} units, {d} aborted, {d} admission waits, {d} swaps out ({d} by time slice), {d} in, {d:.1} ms swapping, {d} rows failed for memory); GPU busy {d:.1}% (decode {d:.1}%, prefill {d:.1}%) over {d:.1} s; batches by rows:", .{ st.batches, st.batch_rows, st.partial_batches, st.batches_in_chunk, st.prefill_chunks, st.packed_chunks, st.prefill_units, st.aborted_chunks, st.admission_waits, st.swap_outs, st.slice_swaps, st.swap_ins, @as(f64, @floatFromInt(st.swap_ns)) / 1e6, st.swap_failures, 100 * @as(f64, @floatFromInt(st.batch_ns + st.prefill_ns)) / span, 100 * @as(f64, @floatFromInt(st.batch_ns)) / span, 100 * @as(f64, @floatFromInt(st.prefill_ns)) / span, span / 1e9 });
        for (st.sizes[1 .. parallel + 1], 1..) |n, rows| std.debug.print(" {d}:{d}", .{ rows, n });
        std.debug.print("\n", .{});
        if (model_backend_store) {
            const cs = model_backend.cache.?.stats();
            std.debug.print("zerv: prefix checkpoints: {d} taken, {d} of {d} lookups restored ({d} prompt tokens reused), {d} dropped for new ones, {d} for memory, {d} pages deduplicated\n", .{ cs.inserts, cs.restores, cs.lookups, cs.restored_tokens, cs.capacity_drops, cs.pressure_drops, cs.dedup_pages });
            model_backend.deinitCache(gpa);
        }
    };
    const ids = [_][]const u8{alias};
    const defaults = try samplingDefaults(&container);
    container.deinit();
    container_live = false;
    file.deinit();
    file_live = false;
    var server = zerv.serve.http.Server.init(io, gpa, native.engine(&ids, defaults), .{ .max_waiting = max_waiting, .drain_timeout = .fromSeconds(drain_s), .parallel = parallel });
    const action: std.posix.Sigaction = .{ .handler = .{ .handler = onStopSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &action, null);
    std.posix.sigaction(.TERM, &action, null);
    std.debug.print("zerv: serving {s} at http://{s}:{d}/v1/chat/completions\n", .{ alias, host, port });
    listener_owned = false;
    server.run(&listener, &stop_requested) catch |e| {
        if (e != error.EngineFailed) return e;
        // The device is gone: its objects cannot be torn down normally (commands may
        // still be pending). Exiting releases them; a supervisor can restart the server.
        std.debug.print("zerv: stopped: the GPU device was lost\n", .{});
        std.process.exit(3);
    };
    std.debug.print("zerv: stopped\n", .{});
}

/// The artifact's recommended sampling (general.sampling.*), else the model card values.
/// `on` or `off`.
fn parseSwitch(value: []const u8) !bool {
    if (std.mem.eql(u8, value, "on")) return true;
    if (std.mem.eql(u8, value, "off")) return false;
    return error.InvalidArguments;
}

/// `--context N` (N > 0) or `max` (the largest that fits; docs/specs/model.md).
fn parseContext(value: []const u8) !u32 {
    if (std.mem.eql(u8, value, "max")) return zerv.model.context_max;
    const n = try std.fmt.parseInt(u32, value, 10);
    if (n == 0) return error.InvalidArguments;
    return n;
}

fn samplingDefaults(container: *const zerv.artifact.gguf.Container) !zerv.serve.api.Defaults {
    var d: zerv.serve.api.Defaults = .{};
    if (container.findMetadata("general.sampling.temp")) |v| d.temperature = try v.scalar(f32);
    if (container.findMetadata("general.sampling.top_p")) |v| d.top_p = try v.scalar(f32);
    if (container.findMetadata("general.sampling.top_k")) |v| d.top_k = @intCast(try v.scalar(i32));
    return d;
}
