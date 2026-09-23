const std = @import("std");
const builtin = @import("builtin");
const quant = @import("zerv").quant;
const Sha256 = std.crypto.hash.sha2.Sha256;

const values_per_call = 5120 * 2048;
const iterations = 16;
const trials = 5;

pub fn main(init: std.process.Init) !void {
    if (builtin.mode != .ReleaseFast) return error.BenchmarkRequiresReleaseFast;
    if (builtin.cpu.arch.endian() != .little) return error.BenchmarkRequiresLittleEndian;
    var buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const writer = &stdout.interface;
    const allocator = std.heap.page_allocator;

    inline for (.{ quant.Format.q4_0, quant.Format.q8_0 }) |format| {
        const width = comptime quant.blockBytes(format);
        const encoded = try allocator.alloc(u8, values_per_call / 32 * width);
        defer allocator.free(encoded);
        const output = try allocator.alloc(f32, values_per_call);
        defer allocator.free(output);
        for (0..values_per_call / 32) |block| {
            const offset = block * width;
            std.mem.writeInt(u16, encoded[offset..][0..2], @intCast(0x3000 + block % 0x1000), .little);
            for (0..width - 2) |j| encoded[offset + 2 + j] = @truncate(block * 37 + j * 19);
        }
        var input_hash: [32]u8 = undefined;
        Sha256.hash(encoded, &input_hash, .{});
        const input_hex = std.fmt.bytesToHex(input_hash, .lower);
        for (0..3) |_| {
            try quant.decode(format, encoded, output);
            std.mem.doNotOptimizeAway(output);
        }
        for (0..trials) |trial| {
            const start = std.Io.Clock.awake.now(init.io);
            for (0..iterations) |_| {
                try quant.decode(format, encoded, output);
                std.mem.doNotOptimizeAway(output);
            }
            const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            if (elapsed <= 0) return error.InvalidElapsedTime;
            var output_hash: [32]u8 = undefined;
            Sha256.hash(std.mem.sliceAsBytes(output), &output_hash, .{});
            const output_hex = std.fmt.bytesToHex(output_hash, .lower);
            try writer.print(
                "{{\"format\":\"{s}\",\"trial\":{d},\"iterations\":{d},\"values_per_call\":{d},\"elapsed_ns\":{d},\"input_sha256\":\"{s}\",\"output_sha256\":\"{s}\"}}\n",
                .{ @tagName(format), trial, iterations, values_per_call, elapsed, input_hex, output_hex },
            );
        }
    }
    try writer.flush();
}
