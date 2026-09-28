//! Golden draws for `sampler.Order.sorted`: runs tests/sampler_cases.zig against the
//! sampler source of commit 3c03b07 (the definition before 2026-09-24) and prints the
//! fixture JSON (tests/fixtures/session/sampler-sorted.json). Command:
//!   git show 3c03b07:src/session/sampler.zig > third_party/sampler-head-3c03b07/sampler.zig
//!   zig run --dep sampler --dep cases -Mroot=tests/reference/generate_sampler_sorted.zig \
//!     -Msampler=third_party/sampler-head-3c03b07/sampler.zig -Mcases=tests/sampler_cases.zig \
//!     -- third_party/sampler-head-3c03b07/sampler.zig > tests/fixtures/session/sampler-sorted.json
const std = @import("std");
const sampler = @import("sampler");
const cases = @import("cases");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.InvalidArguments;
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], allocator, .limited(1 << 20));
    defer allocator.free(source);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source, &digest, .{});

    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(init.io, &buffer);
    const out = &writer.interface;
    try out.print("{{\"sampler_commit\":\"3c03b07cfb4acc634b852e5da396abdc02d9cacb\",\"sampler_sha256\":\"{x}\",\"draws\":[", .{&digest});
    for (cases.cases, 0..) |c, i| {
        const draws = try allocator.alloc(u32, c.steps);
        defer allocator.free(draws);
        try cases.run(sampler.Sampler, sampler.Params, allocator, c, draws);
        try out.print("{s}[", .{if (i > 0) "," else ""});
        for (draws, 0..) |d, j| try out.print("{s}{d}", .{ if (j > 0) "," else "", d });
        try out.writeAll("]");
    }
    try out.writeAll("]}\n");
    try out.flush();
}
