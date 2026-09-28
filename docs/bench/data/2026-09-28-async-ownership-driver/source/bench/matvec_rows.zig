//! Multi-row decode projection benchmark on actual model weights (block 17b): for one
//! projection role (e.g. ffn_gate), every layer's copy is resident, and one command runs
//! the role over all layers, so the weights stream from DRAM as in a decode step (the set is
//! far larger than the 96 MB Infinity Cache). Compares R single-row dispatches per layer
//! (the shipped module) against one multi-row dispatch per layer (count = R), checks every
//! row bitwise, and prints one JSON line per (role, R) with GPU medians.
//! Usage: zerv-matvec-rows-bench MODEL --spv-dir DIR [--group G] [--samples N] [--warmup-ms T] R [R...]
//! DIR holds <format>.spv modules built from src/matvec/matvec_rows.comp (ROWS >= max R,
//! GROUP = G weight rows per workgroup). Inputs and outputs are device-local, as in the
//! runtime; every layer writes its own output rows (checked after one run).
//! Diagnostic `--same-x 1`: every multi-row input row reads row 0 (input stride 0), so X
//! occupies one row's cache footprint; rows r > 0 then mismatch by design.
//! `--wave S` requires subgroup size S for the multi-row kernel (default: driver choice).
//! `--roles a,b` runs only these roles (default: all). R may go up to 16 here (batch-size
//! research, block 18a), beyond the runtime's `matvec.max_rows`; DIR's module must then be
//! built with ROWS = R.
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const matvec = zerv.matvec;
const w = @import("matvec_workload");

const roles = [_][]const u8{ "ffn_gate.weight", "ffn_down.weight", "attn_qkv.weight", "ssm_out.weight", "ssm_alpha.weight", "attn_q.weight", "attn_k.weight", "output.weight" };
const RowsPush = matvec.RowsPush;
/// Largest row count this research tool accepts (the runtime's limit is `matvec.max_rows`).
const bench_max_rows = 16;

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) return error.Usage;
    var spv_dir: ?[]const u8 = null;
    var sample_count: usize = 21;
    var warmup_ms: u64 = 0;
    var group: u32 = 1;
    var same_x = false;
    var wave: ?u32 = null;
    var role_filter: ?[]const u8 = null;
    var first: usize = 2;
    while (first + 1 < args.len and std.mem.startsWith(u8, args[first], "--")) : (first += 2) {
        const flag = args[first];
        const value = args[first + 1];
        if (std.mem.eql(u8, flag, "--spv-dir")) spv_dir = value //
        else if (std.mem.eql(u8, flag, "--samples")) sample_count = try std.fmt.parseInt(usize, value, 10) //
        else if (std.mem.eql(u8, flag, "--warmup-ms")) warmup_ms = try std.fmt.parseInt(u64, value, 10) //
        else if (std.mem.eql(u8, flag, "--group")) group = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, flag, "--same-x")) same_x = std.mem.eql(u8, value, "1") //
        else if (std.mem.eql(u8, flag, "--wave")) wave = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, flag, "--roles")) role_filter = value //
        else return error.Usage;
    }
    if (spv_dir == null or first >= args.len or sample_count == 0 or group == 0 or group > 64) return error.Usage;
    var counts: [16]u32 = undefined;
    var count_n: usize = 0;
    var max_rows: u32 = 0;
    for (args[first..]) |arg| {
        if (count_n == counts.len) return error.Usage;
        counts[count_n] = try std.fmt.parseInt(u32, arg, 10);
        if (counts[count_n] == 0 or counts[count_n] > bench_max_rows) return error.Usage;
        max_rows = @max(max_rows, counts[count_n]);
        count_n += 1;
    }
    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 8 * 1024 * 1024 * 1024, .subgroup_size_control = wave != null });
    defer device.deinit() catch @panic("live");
    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    for (roles) |role| {
        if (role_filter) |filter| {
            var it = std.mem.splitScalar(u8, filter, ',');
            const wanted = while (it.next()) |name| {
                if (std.mem.eql(u8, name, role)) break true;
            } else false;
            if (!wanted) continue;
        }
        // Every tensor of this role (all layers, or output.weight alone).
        var tensors: [64]*const zerv.artifact.gguf.Tensor = undefined;
        var nt: usize = 0;
        for (0..64) |il| {
            const name = try std.fmt.allocPrint(a, "blk.{d}.{s}", .{ il, role });
            if (container.findTensor(name)) |t| {
                tensors[nt] = t;
                nt += 1;
            }
        }
        if (container.findTensor(role)) |t| {
            tensors[0] = t;
            nt = 1;
        }
        if (nt == 0) return error.MissingTensor;
        const format = try zerv.model.config.matrixFormat(tensors[0].kind);
        const K: u32 = @intCast(tensors[0].dims[0]);
        const M: u32 = @intCast(tensors[0].dims[1]);
        // Q4_1 layers mixed into ffn_down: keep only the first tensor's format.
        var keep: usize = 0;
        for (tensors[0..nt]) |t| if (t.kind == tensors[0].kind) {
            tensors[keep] = t;
            keep += 1;
        };
        nt = keep;
        const bytes = std.mem.alignForward(u64, tensors[0].data.len, 256);
        const shape: matvec.Shape = .{ .format = format, .columns = K, .rows = M };
        const spv_path = try std.fmt.allocPrint(a, "{s}/{s}.spv", .{ spv_dir.?, @tagName(format) });
        const code_bytes = try std.Io.Dir.cwd().readFileAlloc(io, spv_path, a, .limited(1 << 20));
        const code = try a.alignedAlloc(u8, .@"4", code_bytes.len);
        @memcpy(code, code_bytes);

        var weights = try gpu.Buffer.init(&device, bytes * nt + 64, .device);
        defer weights.deinit() catch @panic("w");
        {
            var staging = try gpu.Buffer.init(&device, bytes, .host);
            defer staging.deinit() catch @panic("s");
            var upload = try gpu.Commands.init(&device);
            defer upload.deinit() catch @panic("u");
            for (tensors[0..nt], 0..) |t, i| {
                @memcpy((try staging.mapped())[0..t.data.len], t.data);
                try upload.reset();
                try upload.begin();
                try upload.copy(&staging, 0, &weights, bytes * i, t.data.len);
                try upload.barrier(.transfer, .compute);
                try upload.end();
                try upload.run(w.timeout_ns);
            }
        }
        const x_words = @as(u64, max_rows) * K;
        var input = try gpu.Buffer.init(&device, x_words * 4, .device);
        defer input.deinit() catch @panic("x");
        {
            var staging = try gpu.Buffer.init(&device, x_words * 4, .host);
            defer staging.deinit() catch @panic("s");
            var prng = std.Random.DefaultPrng.init(0x17b ^ K ^ (@as(u64, M) << 20));
            for (std.mem.bytesAsSlice(f32, try staging.mapped())) |*x| x.* = prng.random().floatNorm(f32);
            var upload = try gpu.Commands.init(&device);
            defer upload.deinit() catch @panic("u");
            try upload.begin();
            try upload.copy(&staging, 0, &input, 0, x_words * 4);
            try upload.barrier(.transfer, .compute);
            try upload.end();
            try upload.run(w.timeout_ns);
        }
        // Per layer: single-row results in rows 0..R-1, multi-row results in rows
        // max_rows..max_rows+R-1 (no two dispatches write the same word).
        const layer_words = @as(u64, 2 * max_rows) * M;
        var output = try gpu.Buffer.init(&device, layer_words * nt * 4, .device);
        defer output.deinit() catch @panic("y");
        var readback = try gpu.Buffer.init(&device, layer_words * nt * 4, .host);
        defer readback.deinit() catch @panic("r");
        var single = try matvec.Pipeline.init(format, false, &weights, &input, &output);
        defer single.deinit() catch @panic("single");
        var multi = try gpu.Kernel.initWith(&device, code, &.{ &weights, &input, &output }, @sizeOf(RowsPush), .{ .subgroup_size = wave });
        defer multi.deinit() catch @panic("multi");

        for (counts[0..count_n]) |R| {
            var t_single = try w.Timing.init(&device);
            defer t_single.deinit() catch @panic("t");
            var c_single = try gpu.Commands.init(&device);
            defer c_single.deinit() catch @panic("c");
            try c_single.begin();
            try t_single.begin(&c_single);
            for (0..nt) |layer| for (0..R) |r| try single.record(&c_single, try single.projection(shape, bytes * layer, @as(u64, r) * K * 4, (layer * layer_words + @as(u64, r) * M) * 4));
            try t_single.end();
            try c_single.barrier(.compute, .host);
            try c_single.end();

            var t_multi = try w.Timing.init(&device);
            defer t_multi.deinit() catch @panic("t");
            var c_multi = try gpu.Commands.init(&device);
            defer c_multi.deinit() catch @panic("c");
            try c_multi.begin();
            try t_multi.begin(&c_multi);
            for (0..nt) |layer| {
                var base = try single.projection(shape, bytes * layer, 0, (layer * layer_words + @as(u64, max_rows) * M) * 4);
                // GROUP weight rows per workgroup: ceil(M / G) groups.
                const total = (M + group - 1) / group;
                const gy = (total + 65534) / 65535;
                const gx = (total + gy - 1) / gy;
                base.push.groups_x = gx;
                const push: RowsPush = .{ .base = base.push, .input_stride = if (same_x) 0 else K, .output_stride = M, .count = R };
                try c_multi.dispatch(&multi, std.mem.asBytes(&push), .{ gx, gy, 1 });
            }
            try t_multi.end();
            try c_multi.barrier(.compute, .host);
            try c_multi.end();

            {
                var fill = try gpu.Commands.init(&device);
                defer fill.deinit() catch @panic("f");
                @memset(try readback.mapped(), 0xff);
                try fill.begin();
                try fill.copy(&readback, 0, &output, 0, layer_words * nt * 4);
                try fill.barrier(.transfer, .compute);
                try fill.end();
                try fill.run(w.timeout_ns);
            }
            try c_single.run(w.timeout_ns);
            try c_multi.run(w.timeout_ns);
            {
                var copy = try gpu.Commands.init(&device);
                defer copy.deinit() catch @panic("c");
                try copy.begin();
                try copy.barrier(.compute, .transfer);
                try copy.copy(&output, 0, &readback, 0, layer_words * nt * 4);
                try copy.barrier(.transfer, .host);
                try copy.end();
                try copy.run(w.timeout_ns);
            }
            const ys = std.mem.bytesAsSlice(u32, try readback.mapped());
            var mismatches: u64 = 0;
            for (0..nt) |layer| for (0..R) |r| for (0..M) |m| {
                const o = layer * layer_words;
                if (ys[o + r * M + m] != ys[o + (max_rows + r) * M + m] or ys[o + r * M + m] == 0xffffffff) mismatches += 1;
            };
            const start = std.Io.Clock.awake.now(io);
            while (@as(u64, @intCast(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) < warmup_ms * std.time.ns_per_ms) {
                try c_single.run(w.timeout_ns);
                try c_multi.run(w.timeout_ns);
            }
            const s1 = try a.alloc(f64, sample_count);
            const s2 = try a.alloc(f64, sample_count);
            for (s1, s2) |*x, *y| {
                var t0 = std.Io.Clock.awake.now(io);
                try c_single.run(w.timeout_ns);
                x.* = try t_single.read(@floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds));
                t0 = std.Io.Clock.awake.now(io);
                try c_multi.run(w.timeout_ns);
                y.* = try t_multi.read(@floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds));
            }
            std.mem.sort(f64, s1, {}, std.sort.asc(f64));
            std.mem.sort(f64, s2, {}, std.sort.asc(f64));
            const gb = @as(f64, @floatFromInt(tensors[0].data.len * nt)) / 1e9;
            try std.json.Stringify.value(.{
                .role = role,
                .format = @tagName(format),
                .k = K,
                .m = M,
                .layers = nt,
                .rows = R,
                .group = group,
                .same_x = same_x,
                .wave = wave,
                .samples = sample_count,
                .single_ns_median = s1[s1.len / 2],
                .multi_ns_median = s2[s2.len / 2],
                .single_one_row_gbs = gb * @as(f64, @floatFromInt(R)) / (s1[s1.len / 2] / 1e9),
                .multi_gbs = gb / (s2[s2.len / 2] / 1e9),
                .bitwise_mismatches = mismatches,
            }, .{}, &stdout.interface);
            try stdout.interface.writeByte('\n');
            try stdout.interface.flush();
        }
    }
}
