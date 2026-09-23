const std = @import("std");
const builtin = @import("builtin");
const zerv = @import("zerv");
const split = zerv.tokenizer.qwen_split;
const Case = struct { name: []const u8, text: []const u8, ends: []const u32 };
const Corpus = struct { workloads: []const Case };

fn validate(case: Case) ![64]u8 {
    var iterator = try split.Iterator.init(case.text, .{});
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    var at: usize = 0;
    for (case.ends) |expected| {
        const piece = iterator.next() orelse return error.MissingPiece;
        at += piece.len;
        if (at != expected) return error.GoldenMismatch;
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, @intCast(at), .little);
        digest.update(&bytes);
    }
    if (at != case.text.len or iterator.next() != null) return error.GoldenMismatch;
    var hash: [32]u8 = undefined;
    digest.final(&hash);
    return std.fmt.bytesToHex(hash, .lower);
}

pub fn main(init: std.process.Init) !void {
    if (builtin.mode != .ReleaseFast) return error.BenchmarkRequiresReleaseFast;
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2) return error.ExpectedFixturePath;
    var file = try zerv.artifact.MappedFile.open(init.io, args[1], 1024 * 1024);
    defer file.deinit();
    const parsed = try std.json.parseFromSlice(Corpus, allocator, file.bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var stdout_buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    for (parsed.value.workloads) |case| {
        const hash = try validate(case);
        for (0..10) |trial| {
            const iterations: usize = if (trial < 3) 1 else 100;
            const start = std.Io.Clock.awake.now(init.io);
            for (0..iterations) |_| {
                var iterator = try split.Iterator.init(case.text, .{});
                while (iterator.next()) |piece| std.mem.doNotOptimizeAway(piece);
            }
            const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            if (elapsed <= 0) return error.InvalidElapsedTime;
            const checked = try validate(case);
            if (!std.mem.eql(u8, &hash, &checked)) return error.GoldenMismatch;
            if (trial >= 3) {
                try std.json.Stringify.value(.{
                    .workload = case.name,
                    .trial = trial - 3,
                    .iterations = iterations,
                    .input_bytes = case.text.len,
                    .pieces = case.ends.len,
                    .elapsed_ns = elapsed,
                    .output_sha256 = hash,
                }, .{}, &stdout.interface);
                try stdout.interface.writeByte('\n');
            }
        }
    }
    try stdout.interface.flush();
}
