//! Pressure selection component, not disk I/O or serving; exact rotating-LRU result.
const std = @import("std");
const p = @import("session").pressure;
const iterations = 100000;
pub fn main(init: std.process.Init) !void {
    for ([_]u32{ 1, 8, 64, 256 }) |count| {
        for ([_]bool{ false, true }) |host| {
            const usage: p.Usage = .{ .slots = count + 1, .free_slots = if (host) 1 else 0, .slot_headroom = 0, .host_pages = count, .free_host_pages = 0, .host_headroom = 0 };
            try usage.validate();
            var candidates: [256]p.Candidate = undefined;
            for (0..6) |trial| {
                for (candidates[0..count], 0..) |*c, i| c.* = .{ .handle = .{ .index = @intCast(i), .generation = 1 }, .used = i, .has_host = true, .backed = i % 2 == 0, .fits = true };
                var checksum: u64 = 0;
                const start = std.Io.Clock.awake.now(init.io);
                for (0..iterations) |i| {
                    candidates[i % count].used = count + i;
                    var select: p.Select = .{ .usage = usage };
                    for (candidates[0..count]) |c| select.consider(c);
                    const d = select.decision() orelse return error.MissingDecision;
                    const expected = (i + 1) % count;
                    const expected_action: @FieldType(p.Decision, "action") = if (expected % 2 == 0) .discard else .preserve;
                    if (d.handle.index != expected or d.handle.generation != 1 or d.action != expected_action) return error.WrongDecision;
                    checksum +%= d.handle.index + d.handle.generation;
                }
                const ns = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
                if (trial > 0) std.debug.print("{{\"trial\":{d},\"candidates\":{d},\"host_only\":{},\"iterations\":{d},\"elapsed_ns\":{d},\"checksum\":{d},\"exact\":true}}\n", .{ trial - 1, count, host, iterations, ns, checksum });
            }
        }
    }
}
