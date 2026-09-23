const std = @import("std");
const builtin = @import("builtin");
const artifact = @import("zerv").artifact;

pub fn main(init: std.process.Init) !void {
    if (builtin.mode != .ReleaseFast) return error.BenchmarkRequiresReleaseFast;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModelPath;
    var file = try artifact.MappedFile.open(init.io, args[1], 1024 * 1024 * 1024 * 1024);
    defer file.deinit();
    const allocator = std.heap.page_allocator;
    for (0..3) |_| {
        var parsed = try artifact.gguf.Container.parse(allocator, file.bytes, .{});
        parsed.deinit();
    }
    var buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    for (0..7) |trial| {
        var metadata: usize = 0;
        var tensors: usize = 0;
        var data_offset: usize = 0;
        const start = std.Io.Clock.awake.now(init.io);
        for (0..10) |_| {
            var parsed = try artifact.gguf.Container.parse(allocator, file.bytes, .{});
            metadata = parsed.metadata.len;
            tensors = parsed.tensors.len;
            data_offset = parsed.data_offset;
            std.mem.doNotOptimizeAway(&parsed);
            parsed.deinit();
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
        if (elapsed <= 0) return error.InvalidElapsedTime;
        try std.json.Stringify.value(.{ .trial = trial, .iterations = 10, .elapsed_ns = elapsed, .metadata = metadata, .tensors = tensors, .data_offset = data_offset }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
    }
    try stdout.interface.flush();
}
