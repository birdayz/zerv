const std = @import("std");
const zerv = @import("zerv");
const Times = struct { begin_ns: i96 = 0, segment_ns: i96 = 0, checkpoint_ns: i96 = 0, final_ns: i96 = 0, restored: u32 = 0, checkpoint_tokens: u32 = 0 };
fn elapsed(io: std.Io, start: std.Io.Timestamp) i96 {
    return start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
}
fn run(b: *zerv.serve.ModelBackend, io: std.Io, tokens: []const u32, boundary: u32, times: *Times) ![]const f32 {
    const m = b.m;
    var start = std.Io.Clock.awake.now(io);
    const restored = try b.begin(0, tokens);
    times.begin_ns = elapsed(io, start);
    times.restored = restored;
    if (!b.admit(0, tokens.len)) return error.AdmissionFailed;
    var points: [zerv.session.checkpoint.max_points]u32 = undefined;
    var at = restored;
    for (zerv.session.checkpoint.candidatePoints(tokens, restored, boundary, &points)) |point| {
        start = std.Io.Clock.awake.now(io);
        _ = try m.prefill(tokens[at..point]);
        times.segment_ns += elapsed(io, start);
        start = std.Io.Clock.awake.now(io);
        try b.checkpoint(0, tokens[0..point]);
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
    if (args.len != 3) return error.Usage;
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
    const golden = try a.alloc(f32, zerv.model.config.vocab);
    for (0..6) |trial| {
        try b.initCache(a, .radix, data.value.boundary, true);
        var seed: Times = .{};
        _ = try run(&b, init.io, data.value.first, data.value.boundary, &seed);
        try b.reset(0);
        var times: Times = .{};
        const got = try run(&b, init.io, data.value.reuse, data.value.boundary, &times);
        if (trial == 0) @memcpy(golden, got) else if (!std.mem.eql(u8, std.mem.sliceAsBytes(golden), std.mem.sliceAsBytes(got))) return error.WrongRepeatedLogits;
        if (trial > 0) {
            const row = try std.json.Stringify.valueAlloc(a, .{ .trial = trial - 1, .times = times, .exact = true }, .{});
            std.debug.print("{s}\n", .{row});
        }
        try b.reset(0);
        while (b.cache.?.evict(b.device()) or b.cache.?.evictHost(b.device())) {}
        b.deinitCache(a);
    }
}
