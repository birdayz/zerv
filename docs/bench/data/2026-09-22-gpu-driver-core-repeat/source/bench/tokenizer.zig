const std = @import("std");
const builtin = @import("builtin");
const zerv = @import("zerv");
const Tokenizer = zerv.tokenizer.bpe.Tokenizer;
const Case = struct { text: []const u8, ids: []const u32, raw_hex: []const u8 };
const Workload = struct { name: []const u8, text: []const u8, ids: []const u32, raw_hex: []const u8 };
const Corpus = struct { raw_piece_count: u32, raw_pieces_sha256: []const u8, cases: []const Case, decode_cases: []const struct { ids: []const u32, raw_hex: []const u8 }, workloads: []const Workload };
const Sha256 = std.crypto.hash.sha2.Sha256;
fn rawHex(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const bytes = try allocator.alloc(u8, text.len / 2);
    errdefer allocator.free(bytes);
    return std.fmt.hexToBytes(bytes, text);
}
fn checkDecode(allocator: std.mem.Allocator, tokenizer: *const Tokenizer, ids: []const u32, raw: []const u8) !void {
    const expected = try rawHex(allocator, raw);
    defer allocator.free(expected);
    const output = try allocator.alloc(u8, expected.len);
    defer allocator.free(output);
    var writer: std.Io.Writer = .fixed(output);
    try tokenizer.decode(ids, &writer);
    if (!std.mem.eql(u8, expected, writer.buffered())) return error.DecodeMismatch;
}
fn idHash(ids: []const u32) [64]u8 {
    var hash = Sha256.init(.{});
    var bytes: [4]u8 = undefined;
    for (ids) |id| {
        std.mem.writeInt(u32, &bytes, id, .little);
        hash.update(&bytes);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 4) return error.ExpectedModelFixtureAndOptionalBench;
    const benchmark = args.len == 4;
    if (benchmark and (!std.mem.eql(u8, args[3], "--bench") or builtin.mode != .ReleaseFast)) return error.InvalidBenchmarkMode;
    var file = try zerv.artifact.MappedFile.open(init.io, args[1], 32 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(allocator, file.bytes, .{});
    defer container.deinit();
    const start = std.Io.Clock.awake.now(init.io);
    var tokenizer = try zerv.tokenizer.fromGGUF(allocator, &container, .{});
    defer tokenizer.deinit();
    std.debug.print("vocabulary_init_ns={d}\n", .{start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds});
    var fixture = try zerv.artifact.MappedFile.open(init.io, args[2], 16 * 1024 * 1024);
    defer fixture.deinit();
    const parsed = try std.json.parseFromSlice(Corpus, allocator, fixture.bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const corpus = parsed.value;
    if (tokenizer.pieces.len != corpus.raw_piece_count) return error.VocabularyMismatch;
    var hash = Sha256.init(.{});
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, corpus.raw_piece_count, .little);
    hash.update(&bytes);
    for (0..corpus.raw_piece_count) |id| {
        const piece = try tokenizer.piece(@intCast(id));
        std.mem.writeInt(u32, &bytes, @intCast(piece.len), .little);
        hash.update(&bytes);
        hash.update(piece);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    const pieces_hash = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, corpus.raw_pieces_sha256, &pieces_hash)) return error.PieceMismatch;
    for (corpus.cases) |case| {
        const ids = try tokenizer.encode(allocator, case.text, .{});
        defer allocator.free(ids);
        if (!std.mem.eql(u32, ids, case.ids)) return error.TokenMismatch;
        try checkDecode(allocator, &tokenizer, ids, case.raw_hex);
    }
    for (corpus.decode_cases) |case| try checkDecode(allocator, &tokenizer, case.ids, case.raw_hex);
    var stdout_buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    if (!benchmark) {
        try std.json.Stringify.value(.{ .cases = corpus.cases.len, .decode_cases = corpus.decode_cases.len, .pieces = corpus.raw_piece_count, .raw_pieces_sha256 = pieces_hash }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
    } else for (corpus.workloads) |case| {
        const expected = try rawHex(allocator, case.raw_hex);
        defer allocator.free(expected);
        const output = try allocator.alloc(u8, expected.len);
        defer allocator.free(output);
        inline for (.{ "encode", "decode" }) |operation| {
            var check: [32]u8 = undefined;
            Sha256.hash(expected, &check, .{});
            const result_hash = if (comptime std.mem.eql(u8, operation, "encode")) idHash(case.ids) else std.fmt.bytesToHex(check, .lower);
            for (0..10) |trial| {
                const iterations: usize = if (trial < 3) 1 else if (comptime std.mem.eql(u8, operation, "encode")) 100 else 1000;
                const begin = std.Io.Clock.awake.now(init.io);
                for (0..iterations) |_| {
                    if (comptime std.mem.eql(u8, operation, "encode")) {
                        const ids = try tokenizer.encode(allocator, case.text, .{});
                        std.mem.doNotOptimizeAway(ids);
                        allocator.free(ids);
                    } else {
                        var writer: std.Io.Writer = .fixed(output);
                        try tokenizer.decode(case.ids, &writer);
                        std.mem.doNotOptimizeAway(writer.buffered());
                    }
                }
                const elapsed = begin.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
                if (elapsed <= 0) return error.InvalidElapsed;
                const ids = try tokenizer.encode(allocator, case.text, .{});
                defer allocator.free(ids);
                if (!std.mem.eql(u32, ids, case.ids)) return error.TokenMismatch;
                try checkDecode(allocator, &tokenizer, case.ids, case.raw_hex);
                if (trial >= 3) {
                    try std.json.Stringify.value(.{ .workload = case.name, .operation = operation, .trial = trial - 3, .iterations = iterations, .elapsed_ns = elapsed, .output_sha256 = result_hash }, .{}, &stdout.interface);
                    try stdout.interface.writeByte('\n');
                }
            }
        }
    }
    try stdout.interface.flush();
}
