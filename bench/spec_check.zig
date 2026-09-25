//! Real-model gate 1 for speculative verification (docs/specs/speculative.md): is an
//! n-row verify, committed at m rows, bit for bit the same as m (and n) decode steps?
//!
//! For each case (prompt length p, rows n, commit m) and tokens X[0..n] (fixed pseudo-
//! random, the check is about arithmetic, not about which tokens are plausible):
//!   decode: restore p; step(X[i]) for i < n, logits D[i]; state after m steps -> S_d;
//!           then from a second restore, m steps, then k follow-up steps F_d.
//!   verify: restore p; verify(X) -> V[0..n]; commit(m) -> state S_v; k follow-up steps F_v.
//! Gate: V[i] == D[i] (all vocabulary logits, all i < n), S_v == S_d (every byte of the
//! recurrent and conv state) and F_v == F_d. The attention K/V written for rejected rows
//! must not matter; the follow-up steps overwrite and read them.
//! Usage: zerv-spec-check MODEL [f32|f16 [fused|separate [fma|separate [fused|separate [state-out|legacy]]]]] > report.jsonl
//! (KV cache type, decode FFN fusion, matvec accumulation, verify FFN fusion, DeltaNet
//! kernels; exit status 1 on any mismatch)
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const model = zerv.model;

const vocab = model.config.vocab;
const rows_max = zerv.matvec.max_rows;
const follow = 4;
const Case = struct { p: u32, n: u32, m: u32 };
// Positions across the 64-key attention chunk edge (62..66), inside one chunk, and long.
const cases = [_]Case{
    .{ .p = 40, .n = 1, .m = 1 },  .{ .p = 40, .n = 2, .m = 1 },  .{ .p = 40, .n = 2, .m = 2 },
    .{ .p = 40, .n = 5, .m = 5 },  .{ .p = 40, .n = 5, .m = 3 },  .{ .p = 61, .n = 5, .m = 1 },
    .{ .p = 61, .n = 5, .m = 4 },  .{ .p = 62, .n = 4, .m = 4 },  .{ .p = 127, .n = 3, .m = 2 },
    .{ .p = 700, .n = 5, .m = 5 }, .{ .p = 700, .n = 5, .m = 2 },
};

fn equalBits(a: []const f32, b: []const f32) bool {
    for (a, b) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return false;
    return true;
}

/// Reads the model's recurrent + conv state (the snapshot region) into `out`.
fn readState(m: *model.Model, staging: *gpu.Buffer, cmd: *gpu.Commands, out: []u8) !void {
    try cmd.reset();
    try cmd.begin();
    try cmd.barrier(.compute, .transfer);
    try cmd.copy(&m.state, @as(u64, m.state_layout.ssm) * 4, staging, 0, model.snapshot_bytes);
    try cmd.barrier(.transfer, .host);
    try cmd.end();
    try cmd.run(60 * std.time.ns_per_s);
    @memcpy(out, (try staging.mapped())[0..model.snapshot_bytes]);
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or args.len > 7) return error.Usage;
    const kv_type: model.KvType = if (args.len >= 3) std.meta.stringToEnum(model.KvType, args[2]) orelse return error.Usage else .f32;
    const fusion = if (args.len >= 4) (if (std.mem.eql(u8, args[3], "fused")) true else if (std.mem.eql(u8, args[3], "separate")) false else return error.Usage) else true;
    const accumulation: zerv.matvec.Accumulation = if (args.len >= 5) std.meta.stringToEnum(zerv.matvec.Accumulation, args[4]) orelse return error.Usage else .fma;
    const delta_state_out = if (args.len == 7) (if (std.mem.eql(u8, args[6], "state-out")) true else if (std.mem.eql(u8, args[6], "legacy")) false else return error.Usage) else true;
    const verify_fusion = if (args.len >= 6) (if (std.mem.eql(u8, args[5], "fused")) true else if (std.mem.eql(u8, args[5], "separate")) false else return error.Usage) else true;
    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .storage16 = kv_type == .f16 });
    defer device.deinit() catch @panic("live device resources");
    var m: model.Model = undefined;
    try m.init(&device, &container, .{ .context = 4096, .prefill_rows = 512, .snapshots = 1, .verify_rows = rows_max, .kv_type = kv_type, .decode_fusion = fusion, .matvec_accumulation = accumulation, .verify_fusion = verify_fusion, .delta_state_out = delta_state_out });
    defer m.deinit();
    std.debug.print("FFN input fused in {d} (decode) and {d} (verify) of {d} layers\n", .{ m.fused_ffn_layers, m.fused_verify_layers, model.config.layers });
    var staging = try gpu.Buffer.init(&device, model.snapshot_bytes, .host);
    defer staging.deinit() catch @panic("staging");
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");

    var prng = std.Random.DefaultPrng.init(0x17b1);
    const random = prng.random();
    var prompt: [1024]u32 = undefined;
    for (&prompt) |*token| token.* = random.intRangeLessThan(u32, 0, 150000);
    const d_logits = try a.alloc(f32, rows_max * vocab);
    const f_d = try a.alloc(f32, follow * vocab);
    const state_d = try a.alloc(u8, model.snapshot_bytes);
    const state_v = try a.alloc(u8, model.snapshot_bytes);
    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    var failures: usize = 0;
    for (cases) |c| {
        var x: [rows_max + follow]u32 = undefined;
        for (&x) |*token| token.* = random.intRangeLessThan(u32, 0, 150000);
        try m.reset();
        _ = try m.prefill(prompt[0..c.p]);
        try m.saveSnapshot(0);
        // Decode reference: n steps (logits); the state after m of them.
        for (0..c.n) |i| {
            @memcpy(d_logits[i * vocab ..][0..vocab], try m.step(x[i]));
            if (i + 1 == c.m) try readState(&m, &staging, &cmd, state_d);
        }
        // Follow-up steps from the state after m decode steps (row m's token onward).
        try m.loadSnapshot(0, c.p);
        for (0..c.m) |i| _ = try m.step(x[i]);
        for (0..follow) |i| @memcpy(f_d[i * vocab ..][0..vocab], try m.step(x[c.m + i]));
        // Speculative path.
        try m.loadSnapshot(0, c.p);
        const v = try m.verify(x[0..c.n]);
        var rows_equal: u32 = 0;
        for (0..c.n) |i| {
            if (equalBits(v[i * vocab ..][0..vocab], d_logits[i * vocab ..][0..vocab])) rows_equal += 1;
        }
        try m.commit(c.m);
        try readState(&m, &staging, &cmd, state_v);
        const state_equal = std.mem.eql(u8, state_d, state_v);
        var follow_equal: u32 = 0;
        for (0..follow) |i| {
            if (equalBits(try m.step(x[c.m + i]), f_d[i * vocab ..][0..vocab])) follow_equal += 1;
        }
        const ok = rows_equal == c.n and state_equal and follow_equal == follow;
        if (!ok) failures += 1;
        try std.json.Stringify.value(.{ .p = c.p, .n = c.n, .m = c.m, .rows_equal = rows_equal, .state_equal = state_equal, .follow_equal = follow_equal, .ok = ok }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
    }
    // Cost (wall clock per call, as the engine sees it): one decode step, and verify of n
    // rows plus commit of n rows, at position ~300, median of 41 after warm-up.
    try m.reset();
    _ = try m.prefill(prompt[0..300]);
    try m.saveSnapshot(0);
    const samples = try a.alloc(f64, 41);
    for (0..rows_max + 1) |n| {
        for (0..samples.len + 8) |s| {
            try m.loadSnapshot(0, 300);
            const t0 = std.Io.Clock.awake.now(io);
            if (n == 0) {
                _ = try m.step(prompt[s % 300]);
            } else {
                _ = try m.verify(prompt[0..n]);
                try m.commit(@intCast(n));
            }
            const dt: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
            if (s >= 8) samples[s - 8] = dt / 1e6;
        }
        std.mem.sort(f64, samples, {}, std.sort.asc(f64));
        try std.json.Stringify.value(.{ .timing = if (n == 0) "step" else "verify+commit", .rows = @max(n, 1), .ms_median = samples[samples.len / 2], .ms_min = samples[0], .ms_max = samples[samples.len - 1] }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
    }
    if (failures != 0) std.process.exit(1);
}
