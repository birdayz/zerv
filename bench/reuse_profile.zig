const std = @import("std");
const zerv = @import("zerv");
const Mode = enum { checkpointed, split_no_save, joined };
const Times = struct { begin_ns: i96 = 0, segment_ns: i96 = 0, checkpoint_ns: i96 = 0, final_ns: i96 = 0, restored: u32 = 0, checkpoint_tokens: u32 = 0 };
fn elapsed(io: std.Io, start: std.Io.Timestamp) i96 {
    return start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
}
fn run(b: *zerv.serve.ModelBackend, io: std.Io, tokens: []const u32, boundary: u32, times: *Times, mode: Mode) ![]const f32 {
    const m = b.m;
    var start = std.Io.Clock.awake.now(io);
    const restored = try b.begin(0, tokens);
    times.begin_ns = elapsed(io, start);
    times.restored = restored;
    if (!b.admit(0, tokens.len)) return error.AdmissionFailed;
    var points: [zerv.session.checkpoint.max_points]u32 = undefined;
    var at = restored;
    const candidates = zerv.session.checkpoint.candidatePoints(tokens, restored, boundary, &points);
    for (if (mode == .joined) candidates[0..0] else candidates) |point| {
        start = std.Io.Clock.awake.now(io);
        _ = try m.prefill(tokens[at..point]);
        times.segment_ns += elapsed(io, start);
        start = std.Io.Clock.awake.now(io);
        if (mode == .checkpointed) try b.checkpoint(0, tokens[0..point]);
        times.checkpoint_ns += elapsed(io, start);
        times.checkpoint_tokens = point;
        at = point;
    }
    start = std.Io.Clock.awake.now(io);
    const logits = try m.prefill(tokens[at..]);
    times.final_ns = elapsed(io, start);
    return logits;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 4) return error.Usage;
    const counterfactual = args.len == 4 and std.mem.eql(u8, args[3], "counterfactual");
    if (args.len == 4 and !counterfactual) return error.Usage;
    var f = try zerv.artifact.MappedFile.open(init.io, args[1], 64 << 30);
    defer f.deinit();
    var gguf = try zerv.artifact.gguf.Container.parse(a, f.bytes, .{});
    defer gguf.deinit();
    var input = try zerv.artifact.MappedFile.open(init.io, args[2], 1 << 20);
    defer input.deinit();
    const data = try std.json.parseFromSlice(struct { first: []u32, reuse: []u32, boundary: u32 }, a, input.bytes, .{});
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 24 << 30, .cooperative_matrix = true, .subgroup_size_control = true, .storage16 = true, .pipeline_binaries = true, .host_import = true });
    defer device.deinit() catch @panic("live device");
    const m = try a.create(zerv.model.Model);
    try m.init(&device, &gguf, .{ .context = 12288, .prefill_rows = 512, .prefill_precision = .f16, .snapshots = 8, .snapshot_memory = .host, .embedding_memory = .host, .kv_type = .f16, .slots = 2, .batch_rows = 2, .swap_bytes = 4096 << 20, .kv_share = true, .kv_pages = 192 });
    defer m.deinit();
    var b: zerv.serve.ModelBackend = .{ .m = m };
    const golden = try a.alloc(f32, 3 * zerv.model.config.vocab);
    const modes: usize = if (counterfactual) 3 else 1;
    for (0..6) |trial| {
        for (0..modes) |index| {
            const mode_index = if (trial % 2 == 0) index else modes - 1 - index;
            const mode: Mode = @enumFromInt(mode_index);
            const expected = golden[mode_index * zerv.model.config.vocab ..][0..zerv.model.config.vocab];
            try b.initCache(a, .radix, data.value.boundary, true);
            var seed: Times = .{};
            _ = try run(&b, init.io, data.value.first, data.value.boundary, &seed, .checkpointed);
            try b.reset(0);
            var times: Times = .{};
            const got = try run(&b, init.io, data.value.reuse, data.value.boundary, &times, mode);
            if (trial == 0) @memcpy(expected, got) else if (!std.mem.eql(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(got))) return error.WrongRepeatedLogits;
            const baseline = golden[0..zerv.model.config.vocab];
            var max_abs: f32 = 0;
            var argmax_a: usize = 0;
            var argmax_b: usize = 0;
            for (baseline, got, 0..) |x, y, i| {
                if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonfiniteLogits;
                max_abs = @max(max_abs, @abs(x - y));
                if (x > baseline[argmax_a]) argmax_a = i;
                if (y > got[argmax_b]) argmax_b = i;
            }
            const equal = std.mem.eql(u8, std.mem.sliceAsBytes(baseline), std.mem.sliceAsBytes(got));
            if (mode == .split_no_save and !equal) return error.CheckpointChangedLogits;
            if (trial > 0) {
                const row = try std.json.Stringify.valueAlloc(a, .{ .trial = trial - 1, .mode = @tagName(mode), .times = times, .exact = true, .equal_baseline = equal, .max_abs = max_abs, .argmax_equal = argmax_a == argmax_b }, .{});
                std.debug.print("{s}\n", .{row});
            }
            try b.reset(0);
            while (b.cache.?.evict(b.device()) or b.cache.?.evictHost(b.device())) {}
            b.deinitCache(a);
        }
    }
}
