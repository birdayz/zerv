//! f16 batched-decode projection benchmark on actual model weights (block 18e): for one
//! projection role (e.g. ffn_gate), every layer's copy is resident, and one command runs the
//! role over all layers with the model's split chunk (`gemm.f16nChunk`) and reduce, so the
//! weights stream from DRAM as in a decode step (the set is far larger than the 96 MB
//! Infinity Cache). Kernels: v1 (gemm_f16n), v2 (gemm_f16d), and optionally an experimental
//! SPIR-V module with v2's bindings, grid and push (`--spv FILE`). After timing, every
//! kernel's output of the last layer is compared bitwise with v1's.
//! Usage: zerv-decode-f16-bench MODEL [--roles a,b] [--samples N] [--warmup-ms T] [--spv FILE [--spv-x16 1]] [--chunk-align A] R [R...]
//! Prints one JSON line per (role, rows, kernel): GPU median over all layers, and the
//! effective weight read rate.
const std = @import("std");
const zerv = @import("zerv");
const w = @import("matvec_workload");
const gpu = zerv.gpu;
const gemm = zerv.model.gemm;

const roles = [_][]const u8{ "ffn_gate", "ffn_up", "ffn_down", "attn_qkv", "attn_gate", "attn_q", "attn_output" };

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) return error.Usage;
    var samples_n: usize = 21;
    var warmup_ms: u64 = 300;
    var role_list: []const u8 = "";
    var spv_path: ?[]const u8 = null;
    var spv_x16 = false;
    var chunk_align: u32 = 256;
    var rows_list: std.ArrayList(u32) = .empty;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--")) {
            if (i + 1 >= args.len) return error.Usage;
            const value = args[i + 1];
            i += 1;
            if (std.mem.eql(u8, arg, "--samples")) samples_n = try std.fmt.parseInt(usize, value, 10) //
            else if (std.mem.eql(u8, arg, "--warmup-ms")) warmup_ms = try std.fmt.parseInt(u64, value, 10) //
            else if (std.mem.eql(u8, arg, "--roles")) role_list = value //
            else if (std.mem.eql(u8, arg, "--spv")) spv_path = value //
            else if (std.mem.eql(u8, arg, "--spv-x16")) spv_x16 = std.mem.eql(u8, value, "1") //
            else if (std.mem.eql(u8, arg, "--chunk-align")) chunk_align = try std.fmt.parseInt(u32, value, 10) //
            else return error.Usage;
        } else try rows_list.append(a, try std.fmt.parseInt(u32, arg, 10));
    }
    if (rows_list.items.len == 0) return error.Usage;
    var max_rows: u32 = 0;
    for (rows_list.items) |r| max_rows = @max(max_rows, r);
    const span_max = std.mem.alignForward(u32, max_rows, gemm.f16n_tile_n);

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 20 * 1024 * 1024 * 1024, .cooperative_matrix = true, .subgroup_size_control = true });
    defer device.deinit() catch @panic("live");
    const spv: ?[]align(4) const u8 = if (spv_path) |p| try std.Io.Dir.cwd().readFileAllocOptions(io, p, a, .limited(1 << 24), .@"4", null) else null;

    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    for (roles) |role| {
        if (role_list.len != 0 and std.mem.indexOf(u8, role_list, role) == null) continue;
        // Every layer's tensor of this role, Q4_0 only (the v2 formats).
        var tensors: std.ArrayList(*const zerv.artifact.gguf.Tensor) = .empty;
        for (0..64) |l| {
            const name = try std.fmt.allocPrint(a, "blk.{d}.{s}.weight", .{ l, role });
            const t = container.findTensor(name) orelse continue;
            if (try zerv.model.config.matrixFormat(t.kind) != .q4_0) continue;
            try tensors.append(a, t);
        }
        if (tensors.items.len == 0) continue;
        const K: u32 = @intCast(tensors.items[0].dims[0]);
        const M: u32 = @intCast(tensors.items[0].dims[1]);
        const bytes = tensors.items[0].data.len;
        const L: u32 = @intCast(tensors.items.len);
        var weights = try gpu.Buffer.init(&device, bytes * L + 64, .device);
        defer weights.deinit() catch @panic("w");
        {
            var staging = try gpu.Buffer.init(&device, bytes * L + 64, .host);
            defer staging.deinit() catch @panic("s");
            const m = try staging.mapped();
            for (tensors.items, 0..) |t, l| {
                if (t.data.len != bytes) return error.InvalidShape;
                @memcpy(m[l * bytes ..][0..bytes], t.data);
            }
            var up = try gpu.Commands.init(&device);
            defer up.deinit() catch @panic("u");
            try up.begin();
            try up.copy(&staging, 0, &weights, 0, bytes * L + 64);
            try up.barrier(.transfer, .compute);
            try up.end();
            try up.run(w.timeout_ns);
        }
        // The model's rule (`gemm.f16nChunk`) with the part size aligned to `--chunk-align`
        // (256 = the shipped rule; larger: research on longer contiguous reads).
        const chunk: u32 = blk: {
            const parts_want = std.math.divCeil(u32, 384, M / 128) catch unreachable;
            if (parts_want <= 1) break :blk 0;
            const c = std.mem.alignForward(u32, std.math.divCeil(u32, K, parts_want) catch unreachable, chunk_align);
            break :blk if (c >= K) 0 else c;
        };
        if (chunk_align == 256 and chunk != gemm.f16nChunk(M, K, 384)) return error.RuleMismatch;
        const parts = gemm.splitCount(K, chunk);
        const x_words: u32 = span_max * K;
        const part_base: u32 = x_words + 64;
        // f16 copy of X (row-major, x_rs = K halves) for `--spv-x16 1` kernels.
        const x16_word: u32 = part_base + parts * span_max * M + 64;
        const y_base: u32 = x16_word + x_words / 2 + 64;
        const act_words: u64 = @as(u64, y_base) + @as(u64, L) * span_max * M + 64;
        var act = try gpu.Buffer.init(&device, act_words * 4, .device);
        defer act.deinit() catch @panic("act");
        // Host mirror: X upload and output readback (the benchmark reads device-local memory).
        var host = try gpu.Buffer.init(&device, act_words * 4, .host);
        defer host.deinit() catch @panic("host");
        var io_buf = try gpu.Buffer.init(&device, 1024, .host);
        defer io_buf.deinit() catch @panic("io");
        {
            var prng = std.Random.DefaultPrng.init(0x18e0 ^ K ^ @as(u64, M) << 20);
            const xs = std.mem.bytesAsSlice(f32, try host.mapped());
            for (xs[0..x_words]) |*x| x.* = prng.random().floatNorm(f32);
            const hs = std.mem.bytesAsSlice(f16, std.mem.sliceAsBytes(xs[x16_word..][0 .. x_words / 2]));
            for (hs, xs[0..x_words]) |*h, x| h.* = @floatCast(x);
            var up = try gpu.Commands.init(&device);
            defer up.deinit() catch @panic("u");
            try up.begin();
            try up.copy(&host, 0, &act, 0, @as(u64, y_base) * 4);
            try up.barrier(.transfer, .compute);
            try up.end();
            try up.run(w.timeout_ns);
        }
        const buffers = [_]*gpu.Buffer{ &weights, &act, &io_buf, &act };
        var k1 = try gpu.Kernel.init(&device, try gemm.moduleF16n(.q4_0), &buffers, @sizeOf(gemm.Push));
        defer k1.deinit() catch @panic("k1");
        const opts: gpu.Kernel.Options = .{ .subgroup_size = gemm.f16d_subgroup, .full_subgroups = true };
        var k2 = try gpu.Kernel.initWith(&device, try gemm.moduleF16d(.q4_0), &buffers, @sizeOf(gemm.Push), opts);
        defer k2.deinit() catch @panic("k2");
        var k3: ?gpu.Kernel = if (spv) |s| try gpu.Kernel.initWith(&device, s, &buffers, @sizeOf(gemm.Push), opts) else null;
        defer if (k3) |*k| k.deinit() catch @panic("k3");
        var reduce = try gpu.Kernel.init(&device, gemm.reduceModule(), &.{ &weights, &act, &act, &io_buf }, @sizeOf(gemm.ReducePush));
        defer reduce.deinit() catch @panic("r");
        const row_bytes: u32 = @intCast(bytes / M);
        for (rows_list.items) |rows| {
            const span = std.mem.alignForward(u32, rows, gemm.f16n_tile_n);
            std.mem.bytesAsSlice(u32, try io_buf.mapped())[2] = rows;
            const names = [_][]const u8{ "v1", "v2", "spv" };
            var reference: []u32 = &.{};
            _ = &reference;
            for (names, 0..) |kname, ki| {
                const kernel: *gpu.Kernel = switch (ki) {
                    0 => &k1,
                    1 => &k2,
                    else => if (k3) |*k| k else break,
                };
                var timing = try w.Timing.init(&device);
                defer timing.deinit() catch @panic("t");
                var cmd = try gpu.Commands.init(&device);
                defer cmd.deinit() catch @panic("c");
                try cmd.begin();
                try timing.begin(&cmd);
                for (0..L) |l| {
                    const y: u32 = y_base + @as(u32, @intCast(l)) * span_max * M;
                    var push: gemm.Push = .{ .a_base = @intCast(l * bytes), .a_rs = row_bytes, .x_base = 0, .x_rs = K, .y_base = y, .y_rs = M, .m = M, .k = K, .k_chunk = chunk };
                    const x16_push = ki == 2 and spv_x16;
                    if (x16_push) push.x_base = 2 * x16_word;
                    if (parts > 1) {
                        push.y_base = part_base;
                        push.y_bs = span * M;
                    }
                    const lim: gemm.Limits = .{ .rows = span, .batches = parts };
                    var vpush = push;
                    vpush.x_base = 0; // the f16 copy lies inside the validated f32 extents' buffer
                    const groups = if (ki == 0) try gemm.validateF16n(.q4_0, vpush, lim, weights.size, act.size) else try gemm.validateF16d(.q4_0, vpush, lim, weights.size, act.size);
                    try cmd.dispatch(kernel, std.mem.asBytes(&push), groups);
                    if (parts > 1) {
                        try cmd.barrier(.compute, .compute);
                        const rp: gemm.ReducePush = .{ .part = part_base, .splits = parts, .part_bs = span * M, .y = y, .y_rs = M, .m = M };
                        try cmd.dispatch(&reduce, std.mem.asBytes(&rp), .{ 1024, 1, 1 });
                    }
                    try cmd.barrier(.compute, .compute);
                }
                try timing.end();
                try cmd.barrier(.compute, .host);
                try cmd.end();
                for (0..2) |_| try cmd.run(w.timeout_ns);
                const warm_start = std.Io.Clock.awake.now(io);
                while (@as(u64, @intCast(warm_start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) < warmup_ms * std.time.ns_per_ms) try cmd.run(w.timeout_ns);
                const samples = try a.alloc(f64, samples_n);
                for (samples) |*s| {
                    const start = std.Io.Clock.awake.now(io);
                    try cmd.run(w.timeout_ns);
                    s.* = try timing.read(@floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds));
                }
                std.mem.sort(f64, samples, {}, std.sort.asc(f64));
                {
                    var rb = try gpu.Commands.init(&device);
                    defer rb.deinit() catch @panic("rb");
                    try rb.begin();
                    try rb.copy(&act, @as(u64, y_base) * 4, &host, @as(u64, y_base) * 4, @as(u64, L) * span_max * M * 4);
                    try rb.barrier(.transfer, .host);
                    try rb.end();
                    try rb.run(w.timeout_ns);
                }
                // Bitwise check of every layer's output rows against v1.
                const out = std.mem.bytesAsSlice(u32, try host.mapped())[y_base..][0 .. L * span_max * M];
                var equal = true;
                if (ki == 0) {
                    reference = try a.alloc(u32, out.len);
                    @memcpy(reference, out);
                } else for (0..L) |l| {
                    const o = l * span_max * M;
                    for (out[o..][0 .. rows * M], reference[o..][0 .. rows * M]) |x, y| if (x != y) {
                        equal = false;
                    };
                }
                const median = samples[samples.len / 2];
                try std.json.Stringify.value(.{ .role = role, .kernel = kname, .rows = rows, .layers = L, .m = M, .k = K, .parts = parts, .gpu_ms_median = median / 1e6, .gpu_ms_min = samples[0] / 1e6, .weight_gb_s = @as(f64, @floatFromInt(bytes * L)) / median, .bitwise_equal_v1 = equal }, .{}, &stdout.interface);
                try stdout.interface.writeByte('\n');
                try stdout.interface.flush();
            }
        }
    }
}
