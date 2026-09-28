//! Long-context KV precision (docs/specs/model.md, "KV precision", gate 4): next-token
//! logits on real text after a long prefix, for one KV cache type, to compare f32 and f16
//! (tools/kv_quality.py computes the KL divergences; llama's side runs
//! tests/reference/llama_batch_capture.c on the tokens written here).
//!
//! Tokenizes TEXT (raw, no special tokens) with the artifact's tokenizer, keeps the first
//! PREFIX + STEPS tokens, prefills PREFIX tokens (logits at position PREFIX - 1), then
//! decodes the next STEPS tokens one at a time, teacher-forced (logits at every position
//! through PREFIX + STEPS - 1). Writes OUT_DIR/tokens.json and OUT_DIR/logits.bin
//! (STEPS + 1 rows of FP32 logits, little-endian).
//! Usage: zerv-kv-quality MODEL TEXT OUT_DIR PREFIX STEPS f32|f16 [fp32|f16[@wmma]]
//! (the last: the prefill mode; f16 = f16 projections, @wmma = WMMA prefill attention,
//! docs/specs/prefill.md; default fp32)
const std = @import("std");
const zerv = @import("zerv");
const model = zerv.model;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 7 and args.len != 8) return error.Usage;
    var precision: model.gemm.Precision = .fp32;
    var attention: model.PrefillAttention = .fp32;
    if (args.len == 8) {
        var pp = std.mem.splitScalar(u8, args[7], '@');
        precision = std.meta.stringToEnum(model.gemm.Precision, pp.first()) orelse return error.Usage;
        while (pp.next()) |o| {
            if (std.mem.eql(u8, o, "wmma")) attention = .wmma else return error.Usage;
        }
    }
    const prefix = try std.fmt.parseInt(u32, args[4], 10);
    const steps = try std.fmt.parseInt(u32, args[5], 10);
    const kv_type = std.meta.stringToEnum(model.KvType, args[6]) orelse return error.Usage;
    if (prefix == 0 or steps == 0) return error.Usage;

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var tokenizer = try zerv.tokenizer.fromGGUF(a, &container, .{});
    defer tokenizer.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(16 * 1024 * 1024));
    const all = try tokenizer.encode(a, text, .{ .max_input_bytes = text.len + 1 });
    const n = prefix + steps;
    if (all.len < n) {
        std.debug.print("zerv-kv-quality: the text has {d} tokens, {d} needed\n", .{ all.len, n });
        return error.Usage;
    }
    const tokens = all[0..n];
    try std.Io.Dir.cwd().createDirPath(io, args[3]);
    var dir = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer dir.close(io);
    var json: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(.{ .prefix = prefix, .steps = steps, .tokens = tokens }, .{}, &json.writer);
    try dir.writeFile(io, .{ .sub_path = "tokens.json", .data = json.written() });

    const needs = model.gemm.deviceNeeds(precision);
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .storage16 = kv_type == .f16, .cooperative_matrix = needs.cooperative_matrix, .subgroup_size_control = needs.subgroup_size_control });
    defer device.deinit() catch @panic("live device resources");
    var m: model.Model = undefined;
    const context = std.mem.alignForward(u32, n + 1, 32);
    try m.init(&device, &container, .{ .context = context, .prefill_rows = 512, .kv_type = kv_type, .prefill_precision = precision, .prefill_attention = attention });
    defer m.deinit();
    const vocab = model.config.vocab;
    const logits = try a.alloc(f32, (@as(usize, steps) + 1) * vocab);
    try m.reset();
    const t0 = std.Io.Clock.awake.now(io);
    @memcpy(logits[0..vocab], try m.prefill(tokens[0..prefix]));
    const t1 = std.Io.Clock.awake.now(io);
    for (0..steps) |i| @memcpy(logits[(i + 1) * vocab ..][0..vocab], try m.step(tokens[prefix + i]));
    const t2 = std.Io.Clock.awake.now(io);
    try dir.writeFile(io, .{ .sub_path = "logits.bin", .data = std.mem.sliceAsBytes(logits) });
    var out_buf: [512]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    try std.json.Stringify.value(.{ .kv = @tagName(kv_type), .prefill = @tagName(precision), .attention = @tagName(attention), .text_tokens = all.len, .prefix = prefix, .steps = steps, .context = context, .prefill_ms = @as(f64, @floatFromInt(t0.durationTo(t1).nanoseconds)) / 1e6, .decode_ms_per_step = @as(f64, @floatFromInt(t1.durationTo(t2).nanoseconds)) / 1e6 / @as(f64, @floatFromInt(steps)) }, .{}, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}
