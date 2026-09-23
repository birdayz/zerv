//! Real-model gate for prefix caching (docs/specs/prefix-cache.md): does resuming from a
//! restored snapshot give exactly the logits of a cold run?
//!
//! For a fixed token sequence T (n tokens) and decode continuation D:
//!   cold:     reset; prefill(T); step(D...)                      -> reference logits
//!   split p:  reset; prefill(T[0..p]); prefill(T[p..]); step(D...)
//!   restore p: reset; prefill(T[0..p]); save; prefill(U); step(U'); load(p);
//!             prefill(T[p..]); step(D...)   (U overwrites KV beyond p and the state)
//!   extend:   cold, then continue decoding after a snapshot round trip at the live position
//! Every comparison is bitwise over all vocabulary logits.
//! Usage: zerv-prefix-check MODEL [fp32|f16] > report.jsonl
const std = @import("std");
const zerv = @import("zerv");
const model = zerv.model;

const vocab = model.config.vocab;
const n_prompt = 1500;
const n_decode = 6;

const Diff = struct { equal: bool, max_abs: f32, argmax_equal: bool };

fn compare(a: []const f32, b: []const f32) Diff {
    var max_abs: f32 = 0;
    var equal = true;
    var ia: usize = 0;
    var ib: usize = 0;
    for (a, b, 0..) |x, y, i| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) equal = false;
        max_abs = @max(max_abs, @abs(x - y));
        if (x > a[ia]) ia = i;
        if (y > b[ib]) ib = i;
    }
    return .{ .equal = equal, .max_abs = max_abs, .argmax_equal = ia == ib };
}

/// Runs the decode continuation, storing each step's logits.
fn decode(m: *model.Model, tokens: []const u32, out: [][]f32) !void {
    for (tokens, out) |t, o| @memcpy(o, try m.step(t));
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2 and args.len != 3) return error.Usage;
    const precision: model.gemm.Precision = if (args.len == 3) std.meta.stringToEnum(model.gemm.Precision, args[2]) orelse return error.Usage else .fp32;

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .cooperative_matrix = model.gemm.deviceNeeds(precision).cooperative_matrix, .subgroup_size_control = model.gemm.deviceNeeds(precision).subgroup_size_control });
    defer device.deinit() catch @panic("live device resources");
    var m: model.Model = undefined;
    try m.init(&device, &container, .{ .context = 4096, .prefill_rows = 512, .prefill_precision = precision, .snapshots = 2 });
    defer m.deinit();

    // Token ids from a fixed LCG over the ordinary-text range (values do not affect cost).
    const tokens = try a.alloc(u32, n_prompt);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 150000);
    const other = try a.alloc(u32, 700);
    for (other, 0..) |*t, i| t.* = @intCast((i * 104729 + 7) % 150000);
    const cont = [_]u32{ 11, 220, 1000, 42, 7, 99 };

    const ref_prefill = try a.alloc(f32, vocab);
    const ref_decode = try a.alloc([]f32, n_decode);
    for (ref_decode) |*d| d.* = try a.alloc(f32, vocab);
    const got_prefill = try a.alloc(f32, vocab);
    const got_decode = try a.alloc([]f32, n_decode);
    for (got_decode) |*d| d.* = try a.alloc(f32, vocab);

    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const w = &stdout.interface;

    // Cold reference, twice (run-to-run determinism).
    try m.reset();
    @memcpy(ref_prefill, try m.prefill(tokens));
    try decode(&m, &cont, ref_decode);
    try m.reset();
    @memcpy(got_prefill, try m.prefill(tokens));
    try decode(&m, &cont, got_decode);
    var all_equal = true;
    try report(w, "repeat", 0, ref_prefill, got_prefill, ref_decode, got_decode, &all_equal);

    const points = [_]u32{ 1, 5, 100, 511, 512, 513, 700, 1024, 1300, 1495, 1499 };
    for (points) |p| {
        // Split: two prefill calls with a boundary at p.
        try m.reset();
        _ = try m.prefill(tokens[0..p]);
        @memcpy(got_prefill, try m.prefill(tokens[p..]));
        try decode(&m, &cont, got_decode);
        try report(w, "split", p, ref_prefill, got_prefill, ref_decode, got_decode, &all_equal);

        // Restore: save at p, diverge (overwrites KV beyond p and the recurrent state),
        // restore, finish the prompt.
        try m.reset();
        _ = try m.prefill(tokens[0..p]);
        try m.saveSnapshot(1);
        _ = try m.prefill(other[0..600]);
        _ = try m.step(3);
        try m.loadSnapshot(1, p);
        @memcpy(got_prefill, try m.prefill(tokens[p..]));
        try decode(&m, &cont, got_decode);
        try report(w, "restore", p, ref_prefill, got_prefill, ref_decode, got_decode, &all_equal);
    }

    // Extend: the live state after decoding survives a save/garbage/load round trip at the
    // live position (the "continue from the previous turn" case).
    try m.reset();
    _ = try m.prefill(tokens);
    try decode(&m, cont[0..3], got_decode[0..3]);
    const live = m.position;
    try m.saveSnapshot(0);
    _ = try m.prefill(other[0..200]);
    try m.loadSnapshot(0, live);
    try decode(&m, cont[3..], got_decode[3..]);
    const d = compare(ref_decode[n_decode - 1], got_decode[n_decode - 1]);
    all_equal = all_equal and d.equal;
    try w.print("{{\"case\":\"extend\",\"position\":{d},\"decode_equal\":{},\"max_abs\":{e}}}\n", .{ live, d.equal, d.max_abs });
    try w.print("{{\"summary\":true,\"precision\":\"{s}\",\"prompt\":{d},\"all_bit_identical\":{}}}\n", .{ @tagName(precision), n_prompt, all_equal });
    try w.flush();
    if (!all_equal) std.process.exit(1);
}

fn report(w: *std.Io.Writer, case: []const u8, p: u32, ref_p: []const f32, got_p: []const f32, ref_d: []const []f32, got_d: []const []f32, all_equal: *bool) !void {
    const pd = compare(ref_p, got_p);
    var decode_equal = true;
    var decode_max: f32 = 0;
    var decode_argmax = true;
    for (ref_d, got_d) |r, g| {
        const x = compare(r, g);
        decode_equal = decode_equal and x.equal;
        decode_argmax = decode_argmax and x.argmax_equal;
        decode_max = @max(decode_max, x.max_abs);
    }
    all_equal.* = all_equal.* and pd.equal and decode_equal;
    try w.print("{{\"case\":\"{s}\",\"point\":{d},\"prefill_equal\":{},\"prefill_max_abs\":{e},\"prefill_argmax_equal\":{},\"decode_equal\":{},\"decode_max_abs\":{e},\"decode_argmax_equal\":{}}}\n", .{ case, p, pd.equal, pd.max_abs, pd.argmax_equal, decode_equal, decode_max, decode_argmax });
    try w.flush();
}
