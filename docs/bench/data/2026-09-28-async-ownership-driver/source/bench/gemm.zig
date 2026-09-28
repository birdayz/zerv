//! Prefill GEMM component benchmark on actual model weights: GPU timestamps per shape.
//! Usage: zerv-gemm-bench MODEL [--variant V] [--samples N] [--x-zero 1]
//!                              [--spv FILE --grid MxN] [--dump DIR] ROWS [ROWS...]
//! --coopmat 1 opens the device with cooperative matrices enabled (WMMA experiments).
//! --wave 32|64 requires that subgroup size for the kernel (VK_EXT_subgroup_size_control).
//! --x-f16 1 uploads X rounded to f16 (RNE, the same values an in-kernel conversion gives),
//! packed row-major with the same x_rs (in halves), for kernels that read f16 X directly.
//! --tile narrow|wide selects the shipped 256x32 or 256x64 module (and its grid).
//! --tensor NAME benchmarks only that tensor (e.g. output.weight for Q6_K).
//! --k-chunk C forces the split-K chunk (0 = no split) instead of the tile-count rule,
//! so kernel variants with different tiles can be compared bit for bit.
//! --dump writes each result Y (rows x M f32, after any split-K reduce) to
//! DIR/<tensor>-<rows>.bin for bitwise comparison of kernel variants.
//! One JSON line per shape/rows. X is deterministic N(0,1) (or zero with --x-zero 1);
//! use large --samples for sustained-clock results. --warmup-ms T runs the dispatch for
//! at least T ms before sampling (steady clocks; docs/design/speed.md measurement rule).
//! --spv (research only) times an
//! experimental module with the same bindings (A, act, io, act) and tile grid MxN.
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const gemm = zerv.model.gemm;
const w = @import("matvec_workload");

const shapes = [_][]const u8{
    "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",   "blk.0.ssm_alpha.weight", "blk.0.ssm_out.weight",
    "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",    "blk.8.ffn_down.weight",  "blk.3.attn_q.weight",
    "blk.3.attn_k.weight",   "blk.3.attn_output.weight",
};

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) return error.Usage;
    var experiment: ?[]align(4) u8 = null;
    var only_variant: ?gemm.Variant = null;
    var grid: [2]u32 = .{ gemm.tile_m, gemm.Tile.narrow.rows() };
    var tile: gemm.Tile = .narrow;
    var sample_count: usize = 9;
    var x_zero = false;
    var dump: ?[]const u8 = null;
    var forced_chunk: ?u32 = null;
    var only_tensor: ?[]const u8 = null;
    var coopmat = false;
    var warmup_ms: u64 = 0;
    var wave: ?u32 = null;
    var x_f16 = false;
    var first_row_arg: usize = 2;
    while (first_row_arg + 1 < args.len and std.mem.startsWith(u8, args[first_row_arg], "--")) : (first_row_arg += 2) {
        const flag = args[first_row_arg];
        const value = args[first_row_arg + 1];
        if (std.mem.eql(u8, flag, "--spv")) {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, value, a, .limited(16 * 1024 * 1024));
            const aligned = try a.alignedAlloc(u8, .@"4", bytes.len);
            @memcpy(aligned, bytes);
            experiment = aligned;
        } else if (std.mem.eql(u8, flag, "--variant")) {
            only_variant = std.meta.stringToEnum(gemm.Variant, value) orelse return error.Usage;
        } else if (std.mem.eql(u8, flag, "--grid")) {
            var it = std.mem.splitScalar(u8, value, 'x');
            grid = .{ try std.fmt.parseInt(u32, it.next() orelse return error.Usage, 10), try std.fmt.parseInt(u32, it.next() orelse return error.Usage, 10) };
            if (grid[0] == 0 or grid[1] == 0) return error.Usage;
        } else if (std.mem.eql(u8, flag, "--samples")) {
            sample_count = try std.fmt.parseInt(usize, value, 10);
            if (sample_count == 0 or sample_count > 100000) return error.Usage;
        } else if (std.mem.eql(u8, flag, "--x-zero")) {
            x_zero = std.mem.eql(u8, value, "1");
        } else if (std.mem.eql(u8, flag, "--dump")) {
            dump = value;
        } else if (std.mem.eql(u8, flag, "--tile")) {
            tile = std.meta.stringToEnum(gemm.Tile, value) orelse return error.Usage;
            grid[1] = tile.rows();
        } else if (std.mem.eql(u8, flag, "--coopmat")) {
            coopmat = std.mem.eql(u8, value, "1");
        } else if (std.mem.eql(u8, flag, "--tensor")) {
            only_tensor = value;
        } else if (std.mem.eql(u8, flag, "--x-f16")) {
            x_f16 = std.mem.eql(u8, value, "1");
        } else if (std.mem.eql(u8, flag, "--wave")) {
            wave = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--warmup-ms")) {
            warmup_ms = try std.fmt.parseInt(u64, value, 10);
            if (warmup_ms > 600_000) return error.Usage;
        } else if (std.mem.eql(u8, flag, "--k-chunk")) {
            forced_chunk = try std.fmt.parseInt(u32, value, 10);
        } else return error.Usage;
    }
    if (experiment != null and only_variant == null) return error.Usage;
    const row_args = args[first_row_arg..];
    if (row_args.len == 0) return error.Usage;
    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 8 * 1024 * 1024 * 1024, .cooperative_matrix = coopmat, .subgroup_size_control = wave != null });
    defer device.deinit() catch @panic("live");
    if (!device.subgroup.computeBallotWithin(64)) return error.UnsupportedDevice;
    var max_rows: u32 = 0;
    for (row_args) |arg| max_rows = @max(max_rows, try std.fmt.parseInt(u32, arg, 10));
    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const selected: []const []const u8 = if (only_tensor) |name| &.{name} else &shapes;
    for (selected) |name| {
        const tensor = container.findTensor(name) orelse return error.MissingTensor;
        const format = try zerv.model.config.matrixFormat(tensor.kind);
        const K: u32 = @intCast(tensor.dims[0]);
        const M: u32 = @intCast(tensor.dims[1]);
        const v = try gemm.variant(format);
        if (only_variant) |ov| if (ov != v) continue;
        var weights = try gpu.Buffer.init(&device, tensor.data.len + 64, .device);
        defer weights.deinit() catch @panic("w");
        var staging = try gpu.Buffer.init(&device, tensor.data.len + 64, .host);
        defer staging.deinit() catch @panic("s");
        @memcpy((try staging.mapped())[0..tensor.data.len], tensor.data);
        const x_words = @as(u64, max_rows) * K;
        // Split-K partials: 64 splits at most (K / 256 for these shapes); none for
        // shapes that never split (the output head would otherwise need 32 GB).
        const part_words: u64 = if (forced_chunk == null and (std.math.divCeil(u32, M, grid[0]) catch unreachable) >= 384) 0 else @as(u64, max_rows) * M * 64;
        const act_words = x_words + @as(u64, max_rows) * M + part_words + 64;
        var act = try gpu.Buffer.init(&device, act_words * 4, .device);
        defer act.deinit() catch @panic("act");
        var io_buf = try gpu.Buffer.init(&device, 1024, .host);
        defer io_buf.deinit() catch @panic("io");
        // Deterministic X (normal, seed per shape): GEMM speed depends on data through
        // power/clock limits, so uninitialized or zero X overstates throughput.
        var x_host = try gpu.Buffer.init(&device, x_words * 4, .host);
        defer x_host.deinit() catch @panic("xh");
        {
            var prng = std.Random.DefaultPrng.init(0x9e3779b9 ^ K ^ (@as(u64, M) << 20));
            const xs = std.mem.bytesAsSlice(f32, try x_host.mapped());
            for (xs) |*x| x.* = if (x_zero) 0 else prng.random().floatNorm(f32);
            if (x_f16) {
                // Ascending in place: half i (bytes 2i..2i+1) never overwrites an unread f32.
                const hs = std.mem.bytesAsSlice(f16, try x_host.mapped());
                for (0..xs.len) |i| {
                    const half: f16 = @floatCast(xs[i]);
                    hs[i] = half;
                }
            }
        }
        var kernel = try gpu.Kernel.initWith(&device, experiment orelse try gemm.module(v, tile), &.{ &weights, &act, &io_buf, &act }, @sizeOf(gemm.Push), .{ .subgroup_size = wave });
        defer kernel.deinit() catch @panic("k");
        var reduce = try gpu.Kernel.init(&device, gemm.reduceModule(), &.{ &weights, &act, &act, &io_buf }, @sizeOf(gemm.ReducePush));
        defer reduce.deinit() catch @panic("r");
        var upload = try gpu.Commands.init(&device);
        defer upload.deinit() catch @panic("u");
        try upload.begin();
        try upload.copy(&staging, 0, &weights, 0, tensor.data.len + 64);
        try upload.copy(&x_host, 0, &act, 0, x_words * 4);
        try upload.barrier(.transfer, .compute);
        try upload.end();
        try upload.run(w.timeout_ns);
        const push: gemm.Push = if (format == .f32)
            .{ .a_base = 0, .a_rs = K, .a_cs = 1, .x_base = 0, .x_rs = K, .y_base = @intCast(x_words), .y_rs = M, .m = M, .k = K }
        else
            .{ .a_base = 0, .a_rs = @intCast(tensor.data.len / M), .x_base = 0, .x_rs = K, .y_base = @intCast(x_words), .y_rs = M, .m = M, .k = K };
        for (row_args) |arg| {
            const rows = try std.fmt.parseInt(u32, arg, 10);
            var run_push = push;
            // Same rule as gemm.splitChunk, for the (possibly experimental) grid.
            run_push.k_chunk = if (forced_chunk) |c| c else blk: {
                const tiles = (std.math.divCeil(u32, M, grid[0]) catch unreachable) * (std.math.divCeil(u32, rows, grid[1]) catch unreachable);
                if (tiles >= 384 or K < 512) break :blk 0;
                const want = @min(std.math.divCeil(u32, 384, tiles) catch unreachable, K / 256);
                if (want <= 1) break :blk 0;
                break :blk std.mem.alignForward(u32, std.math.divCeil(u32, K, want) catch unreachable, 256);
            };
            const splits = gemm.splitCount(K, run_push.k_chunk);
            const part_base: u32 = @intCast(x_words + @as(u64, max_rows) * M + 16);
            if (splits > 1) {
                run_push.y_base = part_base;
                run_push.y_bs = rows * M;
            }
            var groups = try gemm.validate(v, tile, run_push, .{ .rows = rows, .batches = splits }, weights.size, act.size);
            groups[0] = std.math.divCeil(u32, M, grid[0]) catch unreachable;
            groups[1] = std.math.divCeil(u32, rows, grid[1]) catch unreachable;
            const words = std.mem.bytesAsSlice(u32, try io_buf.mapped());
            words[2] = rows;
            words[3] = 0;
            var timing = try w.Timing.init(&device);
            defer timing.deinit() catch @panic("timing");
            var cmd = try gpu.Commands.init(&device);
            defer cmd.deinit() catch @panic("cmd");
            try cmd.begin();
            try timing.begin(&cmd);
            try cmd.dispatch(&kernel, std.mem.asBytes(&run_push), groups);
            if (splits > 1) {
                try cmd.barrier(.compute, .compute);
                const rp: gemm.ReducePush = .{ .part = part_base, .splits = splits, .part_bs = rows * M, .y = @intCast(x_words), .y_rs = M, .m = M };
                try cmd.dispatch(&reduce, std.mem.asBytes(&rp), .{ 1024, 1, 1 });
            }
            try timing.end();
            try cmd.end();
            for (0..3) |_| try cmd.run(w.timeout_ns);
            const warm_start = std.Io.Clock.awake.now(io);
            while (@as(u64, @intCast(warm_start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) < warmup_ms * std.time.ns_per_ms) try cmd.run(w.timeout_ns);
            const samples = try a.alloc(f64, sample_count);
            for (samples) |*s| {
                const start = std.Io.Clock.awake.now(io);
                try cmd.run(w.timeout_ns);
                s.* = try timing.read(@floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds));
            }
            std.mem.sort(f64, samples, {}, std.sort.asc(f64));
            const median = samples[samples.len / 2];
            if (dump) |dir| {
                const y_bytes = @as(u64, rows) * M * 4;
                var readback = try gpu.Buffer.init(&device, y_bytes, .host);
                defer readback.deinit() catch @panic("rb");
                var copy = try gpu.Commands.init(&device);
                defer copy.deinit() catch @panic("copy");
                try copy.begin();
                try copy.copy(&act, x_words * 4, &readback, 0, y_bytes);
                try copy.barrier(.transfer, .host);
                try copy.end();
                try copy.run(w.timeout_ns);
                const path = try std.fmt.allocPrint(a, "{s}/{s}-{d}.bin", .{ dir, name, rows });
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = try readback.mapped() });
            }
            const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(rows));
            try std.json.Stringify.value(.{ .tensor = name, .format = @tagName(format), .k = K, .m = M, .rows = rows, .splits = splits, .samples = sample_count, .warmup_ms = warmup_ms, .wave = wave, .x_f16 = x_f16, .x_zero = x_zero, .gpu_ns_median = median, .gpu_ns_min = samples[0], .tflops = flops / median / 1e3 }, .{}, &stdout.interface);
            try stdout.interface.writeByte('\n');
            try stdout.interface.flush();
        }
    }
}
