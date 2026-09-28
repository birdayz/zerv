//! Snapshot residency metadata ONLY: no disk/GPU copies or serving latency.
const std = @import("std");
const r = @import("session").residency;
const cycles = 64;
const Timing = struct { save_ns: u64 = 0, restore_ns: u64 = 0, drop_ns: u64 = 0, checksum: u64 = 0 };
fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}
fn run(a: std.mem.Allocator, io: std.Io, count: u32, hot: u32) !Timing {
    var table = try r.Table.init(a, count, hot, count);
    defer table.deinit() catch @panic("pending benchmark metadata");
    const handles = try a.alloc(r.Handle, count);
    defer a.free(handles);
    const moves = try a.alloc(r.Transfer, hot);
    defer a.free(moves);
    var timing: Timing = .{};
    for (0..cycles) |cycle| {
        var start = now(io);
        var first: usize = 0;
        while (first < count) : (first += hot) {
            for (0..hot) |j| {
                const i = first + j;
                handles[i] = try table.reserve();
                try table.saved(handles[i], true);
                moves[j] = (try table.spill(handles[i])).?;
            }
            // Model a completed transfer, not its time: no I/O submitted in this harness.
            for (0..hot) |j| try table.complete(moves[hot - 1 - j], true);
        }
        timing.save_ns += @intCast(now(io) - start);
        start = now(io);
        first = 0;
        while (first < count) : (first += hot) {
            for (0..hot) |j| moves[j] = (try table.restore(handles[first + j])).?;
            for (0..hot) |j| {
                const k = hot - 1 - j;
                const h = handles[first + k];
                try table.complete(moves[k], true);
                timing.checksum += try table.acquireResident(h);
                try table.releaseResident(h);
                if (try table.spill(h) != null) return error.UnexpectedWrite;
            }
        }
        timing.restore_ns += @intCast(now(io) - start);
        // Check every location/identity outside timing, not only a plausible aggregate.
        for (handles, 0..) |h, i| {
            const e = try table.inspect(h);
            if (h.index != i or h.generation != cycle + 1 or e.phase != .disk or e.serial != 2 or
                e.hot != null or e.disk != @as(u32, @intCast(i)) or e.leases != 0 or table.disk_owners[i] != @as(u32, @intCast(i))) return error.WrongResidency;
        }
        for (table.hot_owners) |owner| if (owner != null) return error.LeakedHotSlot;
        start = now(io);
        for (handles) |h| try table.drop(h);
        timing.drop_ns += @intCast(now(io) - start);
        for (table.entries) |e| if (e.phase != .free or e.hot != null or e.disk != null) return error.LeakedEntry;
        for (table.disk_owners) |owner| if (owner != null) return error.LeakedDiskSlot;
    }
    const expected: u64 = @as(u64, cycles) * count * (hot - 1) / 2;
    if (timing.checksum != expected) return error.WrongChecksum;
    return timing;
}
pub fn main(init: std.process.Init) !void {
    const counts = [_]u32{ 64, 256, 1024 };
    const depths = [_]u32{ 1, 8 };
    for (0..6) |round| {
        for (0..counts.len) |ci| {
            const count = counts[if (round % 2 == 0) ci else counts.len - 1 - ci];
            for (0..depths.len) |di| {
                const hot = depths[if (round % 2 == 0) di else depths.len - 1 - di];
                const result = try run(init.gpa, init.io, count, hot);
                if (round > 0) std.debug.print("{{\"trial\":{d},\"entries\":{d},\"hot\":{d},\"cycles\":{d},\"save_spill_ns\":{d},\"restore_lease_demote_ns\":{d},\"drop_ns\":{d},\"metadata_bytes\":{d},\"checksum\":{d},\"exact\":true}}\n", .{ round - 1, count, hot, cycles, result.save_ns, result.restore_ns, result.drop_ns, count * @sizeOf(r.Entry) + (count + hot) * @sizeOf(?u32), result.checksum });
            }
        }
    }
}
