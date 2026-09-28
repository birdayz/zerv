//! Pure preparation bookkeeping, including reset/promotion for repetition; no DMA.
const std = @import("std");
const pages = @import("model").pages;
const iterations = 20000;
const Mode = enum { abort, commit, hit };
pub fn main(init: std.process.Init) !void {
    const pool = try init.gpa.create(pages.Pool);
    defer init.gpa.destroy(pool);
    for ([_]struct { gpu: u32, source: u32, host: u32 }{
        .{ .gpu = 192, .source = 128, .host = 512 },
        .{ .gpu = 1024, .source = 626, .host = 1024 },
        .{ .gpu = 4096, .source = 2048, .host = 4096 },
    }) |shape| {
        const list = try init.gpa.alloc(u32, shape.source);
        defer init.gpa.free(list);
        for ([_]u32{ 1, 2, 4 }) |window| for ([_]Mode{ .abort, .commit, .hit }) |mode| {
            for (0..6) |trial| {
                try pool.init(shape.gpu, 2, shape.source);
                try pool.setHost(shape.host);
                for (list, 0..) |*q, i| {
                    q.* = @intCast(i);
                    pool.pins[i] = 1;
                    pool.logical[i] = @intCast(i);
                }
                var checksum: u64 = 0;
                const start = std.Io.Clock.awake.now(init.io);
                for (0..iterations) |_| {
                    const g = (try pool.prepare(list, shape.source / 2, window, 1)) orelse return error.MissingPreparation;
                    const pending = try pool.preparedMoves(g);
                    if (pending.len != window) return error.WrongWindow;
                    for (pending, 0..) |m, j| if (m.index != shape.source / 2 + j or m.from != m.index or m.to != j) return error.WrongMove;
                    if (mode == .hit) {
                        if (try pool.attachPlan(0, list, null) != null) return error.WrongAttach;
                        pool.commitMap(0, list);
                    }
                    try pool.ackPreparation(g);
                    if (mode == .abort) {
                        try pool.abortPreparation(g);
                    } else {
                        const result = try pool.commitPreparation(g, list);
                        if (result.copied != window or result.freed != (if (mode == .hit) @as(u32, 0) else window)) return error.WrongOwnership;
                        checksum += result.copied + result.freed;
                        if (mode == .hit) try pool.release(0);
                        var moves: [4]pages.Move = undefined;
                        const restore = pool.promotePlan(list, &moves) orelse return error.MissingPromotion;
                        if (restore.len != window) return error.WrongPromotion;
                        pool.promoteCommit(list, restore); // simulated bytes; metadata only
                    }
                    if (pool.hostFree() != shape.host) return error.LeakedReservation;
                    checksum +%= g;
                }
                const ns = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
                try pool.checkCheckpoint(list);
                if (!pool.check()) return error.WrongPool;
                for (list, 0..) |q, i| if (q != i) return error.WrongRestoredId;
                if (trial > 0) std.debug.print("{{\"trial\":{d},\"gpu_pages\":{d},\"source_pages\":{d},\"host_pages\":{d},\"window\":{d},\"mode\":\"{s}\",\"iterations\":{d},\"elapsed_ns\":{d},\"checksum\":{d},\"exact\":true}}\n", .{ trial - 1, shape.gpu, shape.source, shape.host, window, @tagName(mode), iterations, ns, checksum });
            }
        };
    }
}
