//! Source ownership metadata only: fixed prefix chains, no tensor copies or I/O.
const std = @import("std");
const k = @import("session").kvcache;
const iterations = 100000;
pub fn main(init: std.process.Init) !void {
    for ([_]u32{ 1, 8, 64, 256 }) |depth| {
        const r = try k.Radix.create(init.gpa, depth, depth * 4, depth, 0);
        const c = r.cache();
        defer c.deinit(init.gpa);
        const tokens = try init.gpa.alloc(u32, depth * 4);
        defer init.gpa.free(tokens);
        const pages = try init.gpa.alloc(u32, depth);
        defer init.gpa.free(pages);
        for (tokens, 0..) |*v, i| v.* = @intCast(i + 1);
        for (pages, 0..) |*v, i| v.* = @intCast(i);
        // Construct the known chain outside timing. This benchmark has no device
        // adapter: it measures acquire/release, not checkpoint creation or serving.
        for (0..depth) |i| {
            try r.store.fill(i, tokens[0 .. (i + 1) * 4], pages[0 .. i + 1]);
            r.parent[i] = if (i == 0) null else @intCast(i - 1);
            r.children[i] = @intFromBool(i + 1 < depth);
            r.own[i] = @intCast(i);
            r.sources[i].generation = 1;
        }
        if (!r.checkTree() or r.ownViolation() != null) return error.InvalidFixture;
        const h = c.coldSource() orelse return error.NoSource;
        var checksum: u64 = 0;
        for (0..6) |trial| {
            const start = std.Io.Clock.awake.now(init.io);
            for (0..iterations) |_| {
                const s = try c.acquireSource(h);
                checksum +%= s.lease.serial + s.snapshot + s.pages.len + s.tokens.len;
                try c.releaseSource(s.lease);
            }
            const ns = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            for (r.sources, 0..) |s, i| {
                if (s.refs != 0 or s.active or s.generation != 1 or s.serial != (if (i + 1 == depth) (trial + 1) * iterations else @as(usize, 0))) return error.WrongOwnership;
            }
            const n: u64 = (trial + 1) * iterations;
            if (checksum != n * (n + 1) / 2 + n * (6 * depth - 1)) return error.WrongChecksum;
            if (trial > 0) std.debug.print("{{\"trial\":{d},\"depth\":{d},\"iterations\":{d},\"elapsed_ns\":{d},\"checksum\":{d},\"ownership_bytes\":{d},\"exact\":true}}\n", .{ trial - 1, depth, iterations, ns, checksum, depth * @sizeOf(k.SourceState) });
        }
    }
}
