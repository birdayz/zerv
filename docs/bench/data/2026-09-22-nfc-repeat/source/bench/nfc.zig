const std = @import("std");
const builtin = @import("builtin");
const zerv = @import("zerv");
const nfc = zerv.text.nfc;
const Corpus = struct { workloads: []const struct { name: []const u8, text: []const u8, output: []const u8 } };

fn iterationsFor(name: []const u8) !usize {
    if (std.mem.eql(u8, name, "ascii")) return 1000;
    if (std.mem.eql(u8, name, "multilingual")) return 100;
    if (std.mem.eql(u8, name, "marks")) return 10;
    return error.UnknownWorkload;
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
        const output = try allocator.alloc(u8, case.output.len);
        defer allocator.free(output);
        const scratch = try allocator.alloc(u21, try nfc.scratchSize(case.text));
        defer allocator.free(scratch);
        std.debug.print("workspace {s}: output_bytes={d} scratch_bytes={d}\n", .{ case.name, output.len, scratch.len * @sizeOf(u21) });
        const size = try nfc.normalize(case.text, output, scratch);
        if (!std.mem.eql(u8, case.output, output[0..size])) return error.GoldenMismatch;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(output[0..size], &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        const iterations = try iterationsFor(case.name);
        for (0..10) |trial| {
            const count: usize = if (trial < 3) 1 else iterations;
            const start = std.Io.Clock.awake.now(init.io);
            for (0..count) |_| {
                const length = try nfc.normalize(case.text, output, scratch);
                std.mem.doNotOptimizeAway(output[0..length]);
            }
            const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            if (elapsed <= 0) return error.InvalidElapsedTime;
            if (!std.mem.eql(u8, case.output, output)) return error.GoldenMismatch;
            if (trial >= 3) {
                try std.json.Stringify.value(.{
                    .workload = case.name,
                    .trial = trial - 3,
                    .iterations = iterations,
                    .input_bytes = case.text.len,
                    .output_bytes = size,
                    .elapsed_ns = elapsed,
                    .output_sha256 = hex,
                }, .{}, &stdout.interface);
                try stdout.interface.writeByte('\n');
            }
        }
    }
    try stdout.interface.flush();
}
