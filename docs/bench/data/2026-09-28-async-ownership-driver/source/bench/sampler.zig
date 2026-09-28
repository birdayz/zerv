//! Token sampler component benchmark (docs/performance.md: every performance-sensitive
//! component): host time per sampled token on real next-token logits rows, for greedy and
//! the sampling settings in use, with the single-pass top-k path (default) and the
//! full-array reference path (`fast_top_k = false`, draw-for-draw equal; tests/session.zig),
//! and `--sampler-order sorted` (the previous definition: its own draws).
//! Each configuration samples every row of LOGITS (FP32 rows of VOCAB), ROUNDS times,
//! accepting each sampled token (penalty history grows as in a generation); the first
//! round is warmup. Prints one JSON line per configuration and path: median, p10, p90 µs.
//! Usage: zerv-sampler-bench LOGITS.bin VOCAB ROUNDS
const std = @import("std");
const zerv = @import("zerv");
const sampler = zerv.session.sampler;

const Config = struct { name: []const u8, params: sampler.Params };
const configs = [_]Config{
    .{ .name = "greedy", .params = .{ .temperature = 0 } },
    .{ .name = "artifact-default (T1, top-k 20, top-p 0.95)", .params = .{ .temperature = 1.0, .top_k = 20, .top_p = 0.95, .seed = 1 } },
    .{ .name = "T0.8, top-k 40, top-p 0.9, min-p 0.05", .params = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.9, .min_p = 0.05, .seed = 2 } },
    .{ .name = "T0.7, top-k 20, penalties", .params = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8, .presence_penalty = 1.5, .repetition_penalty = 1.05, .seed = 3 } },
    .{ .name = "T1, top-p 0.95 only (no top-k)", .params = .{ .temperature = 1.0, .top_p = 0.95, .seed = 4 } },
    .{ .name = "T1, min-p 0.05 only (no top-k)", .params = .{ .temperature = 1.0, .min_p = 0.05, .seed = 5 } },
    .{ .name = "T1, no truncation", .params = .{ .temperature = 1.0, .seed = 6 } },
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) return error.Usage;
    const vocab = try std.fmt.parseInt(usize, args[2], 10);
    const rounds = try std.fmt.parseInt(usize, args[3], 10);
    if (vocab == 0 or rounds < 2) return error.Usage;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], a, .limited(1 << 31));
    if (bytes.len % (vocab * 4) != 0) return error.Usage;
    const all: []align(1) const f32 = std.mem.bytesAsSlice(f32, bytes);
    const logits = try a.alloc(f32, all.len);
    @memcpy(logits, all);
    const rows = logits.len / vocab;
    const samples = try a.alloc(f64, rows * (rounds - 1));
    var out_buf: [1024]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const Path = enum { fast, reference, sorted };
    for (configs) |c| for ([_]Path{ .fast, .reference, .sorted }) |path| {
        if (c.params.temperature == 0 and path != .fast) continue; // greedy has one path
        var params = c.params;
        if (path == .sorted) params.order = .sorted;
        var s = try sampler.Sampler.init(a, vocab, params);
        defer s.deinit(a);
        s.fast_top_k = path != .reference;
        var n: usize = 0;
        var checksum: u64 = 0;
        for (0..rounds) |round| for (0..rows) |r| {
            const row = logits[r * vocab ..][0..vocab];
            const t0 = std.Io.Clock.awake.now(io);
            const token = try s.sample(row);
            const dt = t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
            try s.accept(a, token);
            checksum +%= token;
            if (round == 0) continue;
            samples[n] = @as(f64, @floatFromInt(dt)) / 1e3;
            n += 1;
        };
        std.mem.sort(f64, samples[0..n], {}, std.sort.asc(f64));
        try std.json.Stringify.value(.{ .config = c.name, .path = if (c.params.temperature == 0) "greedy" else switch (path) {
            .fast => "single-pass top-k",
            .reference => "reference (full array)",
            .sorted => "order sorted",
        }, .vocab = vocab, .rows = rows, .samples = n, .us_median = samples[n / 2], .us_p10 = samples[n / 10], .us_p90 = samples[n * 9 / 10], .checksum = checksum }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
    };
}
