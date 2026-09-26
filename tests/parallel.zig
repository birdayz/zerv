//! Independent rounds of one test, run concurrently on `std.testing.io` (a threaded Io
//! with one thread per CPU). The Zig 0.16 test runner runs the tests of a binary one at a
//! time (ziglang/zig#15953); `zig build test` runs the binaries in parallel
//! (docs/development.md, "Tests in parallel").
const std = @import("std");
const t = std.testing;

/// Calls `f(ctx, round)` for every round in 0..n, concurrently where the Io allows (else
/// inline), and returns the first error in round order. Rounds must not share mutable
/// state; a round seeds its own random generator from `round` (`seed`).
pub fn rounds(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) anyerror!void) !void {
    const Ctx = @TypeOf(ctx);
    const Run = struct {
        fn run(c: Ctx, i: usize, out: *anyerror!void) void {
            out.* = f(c, i);
        }
    };
    const results = try t.allocator.alloc(anyerror!void, n);
    defer t.allocator.free(results);
    var group: std.Io.Group = .init;
    defer group.cancel(t.io);
    for (results, 0..) |*r, i| group.async(t.io, Run.run, .{ ctx, i, r });
    try group.await(t.io);
    for (results) |r| try r;
}

/// A per-round random generator: independent streams for `base` and each round.
pub fn seed(base: u64, round: usize) std.Random.DefaultPrng {
    return .init(base ^ (@as(u64, round) +% 1) *% 0x9e37_79b9_7f4a_7c15);
}

/// Feeds `hasher` the bytes of chunks 0..n in order, producing them concurrently: waves of
/// one chunk per CPU, each chunk appended by `produce(ctx, chunk, out)`. The digest equals a
/// sequential run; memory is one wave of chunks.
pub fn hashChunks(hasher: anytype, n: usize, ctx: anytype, comptime produce: fn (@TypeOf(ctx), usize, *std.ArrayList(u8)) anyerror!void) !void {
    const Ctx = @TypeOf(ctx);
    const wave = @max(1, std.Thread.getCpuCount() catch 1);
    const bufs = try t.allocator.alloc(std.ArrayList(u8), wave);
    defer t.allocator.free(bufs);
    @memset(bufs, .empty);
    defer for (bufs) |*b| b.deinit(t.allocator);
    const Wave = struct {
        ctx: Ctx,
        first: usize,
        bufs: []std.ArrayList(u8),
        fn run(w: @This(), i: usize) !void {
            try produce(w.ctx, w.first + i, &w.bufs[i]);
        }
    };
    var first: usize = 0;
    while (first < n) : (first += wave) {
        const m = @min(wave, n - first);
        for (bufs[0..m]) |*b| b.clearRetainingCapacity();
        try rounds(m, Wave{ .ctx = ctx, .first = first, .bufs = bufs }, Wave.run);
        for (bufs[0..m]) |b| hasher.update(b.items);
    }
}
