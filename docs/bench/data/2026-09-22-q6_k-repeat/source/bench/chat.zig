const std = @import("std");
const builtin = @import("builtin");
const zerv = @import("zerv");
const chat = zerv.chat.qwen38;
const Corpus = struct { cases: []const struct {
    messages: []const chat.Message,
    options: chat.Options,
    official: struct { output: ?[]const u8 = null },
} };

pub fn main(init: std.process.Init) !void {
    if (builtin.mode != .ReleaseFast) return error.BenchmarkRequiresReleaseFast;
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2) return error.ExpectedFixturePath;
    var file = try zerv.artifact.MappedFile.open(init.io, args[1], 1024 * 1024);
    defer file.deinit();
    const parsed = try std.json.parseFromSlice(Corpus, allocator, file.bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var buffer: [8192]u8 = undefined;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var renders: usize = 0;
    var bytes: usize = 0;
    for (parsed.value.cases) |case| {
        const expected = case.official.output orelse continue;
        var writer: std.Io.Writer = .fixed(&buffer);
        try chat.render(case.messages, case.options, &writer);
        if (!std.mem.eql(u8, expected, writer.buffered())) return error.GoldenMismatch;
        hash.update(writer.buffered());
        renders += 1;
        bytes += writer.buffered().len;
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    var stdout_buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    for (0..10) |trial| {
        const iterations: usize = if (trial < 3) 1 else 100;
        const start = std.Io.Clock.awake.now(init.io);
        for (0..iterations) |_| {
            for (parsed.value.cases) |case| {
                if (case.official.output == null) continue;
                var writer: std.Io.Writer = .fixed(&buffer);
                try chat.render(case.messages, case.options, &writer);
                std.mem.doNotOptimizeAway(writer.buffered());
            }
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
        if (elapsed <= 0) return error.InvalidElapsedTime;
        if (trial >= 3) {
            try std.json.Stringify.value(.{ .trial = trial - 3, .iterations = iterations, .renders = renders, .bytes = bytes, .elapsed_ns = elapsed, .output_sha256 = hex }, .{}, &stdout.interface);
            try stdout.interface.writeByte('\n');
        }
    }
    try stdout.interface.flush();
}
