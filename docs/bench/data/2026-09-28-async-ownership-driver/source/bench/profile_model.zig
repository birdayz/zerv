//! Per-phase GPU time of the resident model: prefill chunks and decode steps, from
//! timestamps written between recorded phases (instrumented commands only; the
//! production commands carry no timestamps). Tokens are a fixed synthetic sequence:
//! the kernels' cost does not depend on token values.
//! Usage: zerv-model-profile MODEL CONTEXT CHUNK PROMPT_TOKENS DECODE_STEPS [fp32|f16[@native|@spirv][@small=on|off][@wmma] [f32|f16[@page=N|@page=context]]]
//! (prefill precision, gemm_f16x code and `Options.f16_small_tile`, KV cache type and page tokens)
//! Prints one JSON line per prefill chunk and one summary line for decode (median step).
const std = @import("std");
const zerv = @import("zerv");
const model = zerv.model;
const w = @import("matvec_workload");

const phase_count = @typeInfo(model.Phase).@"enum".fields.len;
const max_marks = 2048;

const Recorder = struct {
    marks: w.Marks,
    phases: [max_marks]model.Phase = undefined,

    fn mark(context: *anyopaque, commands: *zerv.gpu.Commands, phase: model.Phase, layer: i32) error{ProbeFailed}!void {
        _ = layer;
        const self: *Recorder = @ptrCast(@alignCast(context));
        const index = self.marks.mark(commands) catch return error.ProbeFailed;
        self.phases[index] = phase;
    }
    fn totals(self: *Recorder, out: *[phase_count]f64) !f64 {
        var gaps: [max_marks]f64 = undefined;
        try self.marks.read(gaps[0 .. self.marks.count - 1]);
        out.* = @splat(0);
        var sum: f64 = 0;
        for (gaps[0 .. self.marks.count - 1], 1..) |gap, i| {
            out[@intFromEnum(self.phases[i])] += gap;
            sum += gap;
        }
        return sum;
    }
};

fn emit(writer: *std.Io.Writer, kind: []const u8, rows: usize, position: u32, total: f64, by_phase: *const [phase_count]f64, wall_ns: f64) !void {
    try writer.print("{{\"kind\":\"{s}\",\"rows\":{d},\"position\":{d},\"gpu_ms\":{d:.4},\"wall_ms\":{d:.4},\"phases_ms\":{{", .{ kind, rows, position, total / 1e6, wall_ns / 1e6 });
    var first = true;
    inline for (@typeInfo(model.Phase).@"enum".fields, 0..) |f, i| {
        if (by_phase[i] > 0) {
            try writer.print("{s}\"{s}\":{d:.4}", .{ if (first) "" else ",", f.name, by_phase[i] / 1e6 });
            first = false;
        }
    }
    try writer.writeAll("}}\n");
    try writer.flush();
}

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").mode != .ReleaseFast) return error.ExpectedReleaseFast;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 6 or args.len > 8) return error.Usage;
    // PRECISION[@native|@spirv][@small=on|off]: gemm_f16x machine code, 32-row tile (`Options`).
    var precision: zerv.model.gemm.Precision = .fp32;
    var gemm_code: zerv.model.gemm.Code = .spirv;
    var small_tile = true;
    var attention: zerv.model.PrefillAttention = .fp32;
    if (args.len >= 7) {
        var pp = std.mem.splitScalar(u8, args[6], '@');
        precision = std.meta.stringToEnum(zerv.model.gemm.Precision, pp.first()) orelse return error.Usage;
        while (pp.next()) |option| {
            if (std.mem.eql(u8, option, "small=on")) {
                small_tile = true;
            } else if (std.mem.eql(u8, option, "small=off")) {
                small_tile = false;
            } else if (std.mem.eql(u8, option, "wmma")) {
                attention = .wmma;
            } else gemm_code = std.meta.stringToEnum(zerv.model.gemm.Code, option) orelse return error.Usage;
        }
    }
    var kv_type: model.KvType = .f32;
    var kv_page: u32 = model.layout.default_kv_page;
    if (args.len == 8) {
        var kp = std.mem.splitScalar(u8, args[7], '@');
        kv_type = std.meta.stringToEnum(model.KvType, kp.first()) orelse return error.Usage;
        while (kp.next()) |option| {
            if (!std.mem.startsWith(u8, option, "page=")) return error.Usage;
            const v = option["page=".len..];
            kv_page = if (std.mem.eql(u8, v, "context")) 0 else try std.fmt.parseInt(u32, v, 10);
        }
    }
    const context = try std.fmt.parseInt(u32, args[2], 10);
    const chunk = try std.fmt.parseInt(u32, args[3], 10);
    const prompt_len = try std.fmt.parseInt(u32, args[4], 10);
    const steps = try std.fmt.parseInt(u32, args[5], 10);
    if (chunk == 0 or prompt_len == 0 or steps == 0 or steps > 1024 or prompt_len + steps + 1 > context) return error.Usage;

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024, .cooperative_matrix = model.gemm.deviceNeeds(precision).cooperative_matrix, .subgroup_size_control = model.gemm.deviceNeeds(precision).subgroup_size_control, .storage16 = kv_type == .f16, .pipeline_binaries = model.gemm.needsPipelineBinaries(precision, gemm_code) });
    defer device.deinit() catch @panic("live device resources");
    var m: model.Model = undefined;
    try m.init(&device, &container, .{ .context = context, .prefill_rows = chunk, .prefill_precision = precision, .kv_type = kv_type, .kv_page_tokens = kv_page, .gemm_code = gemm_code, .f16_small_tile = small_tile, .prefill_attention = attention });
    std.debug.print("gemm_f16x: {s}\n", .{m.gemmCodeStatus()});
    defer m.deinit();

    const tokens = try a.alloc(u32, prompt_len + steps);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 150000);

    var pres: [model.max_plans]Recorder = undefined;
    var pre_cmds: [model.max_plans]zerv.gpu.Commands = undefined;
    for (0..m.plan_count) |p| {
        pres[p] = .{ .marks = try w.Marks.init(&device, max_marks) };
        pre_cmds[p] = try zerv.gpu.Commands.init(&device);
        try pre_cmds[p].begin();
        try pres[p].marks.begin(&pre_cmds[p]);
        try m.recordPrefill(&pre_cmds[p], p, .{ .probe = .{ .context = &pres[p], .mark = Recorder.mark } });
        try pre_cmds[p].end();
    }
    defer for (pre_cmds[0..m.plan_count], pres[0..m.plan_count]) |*c, *r| {
        c.deinit() catch @panic("pending");
        r.marks.deinit();
    };
    var dec = Recorder{ .marks = try w.Marks.init(&device, max_marks) };
    defer dec.marks.deinit();
    var dec_cmd = try zerv.gpu.Commands.init(&device);
    defer dec_cmd.deinit() catch @panic("pending");
    try dec_cmd.begin();
    try dec.marks.begin(&dec_cmd);
    try m.record(&dec_cmd, .{ .probe = .{ .context = &dec, .mark = Recorder.mark } });
    try dec_cmd.end();

    var out_buf: [8192]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const writer = &stdout.interface;
    // Warmup pass (pipelines, caches), then the measured pass from a reset state.
    for (0..2) |pass| {
        try m.reset();
        var by_phase: [phase_count]f64 = undefined;
        var i: usize = 0;
        while (i < prompt_len) {
            const next = m.nextChunk(@intCast(prompt_len - i));
            const n = next.rows;
            const plan = next.plan;
            const position = m.position;
            const start = std.Io.Clock.awake.now(io);
            _ = try m.runChunk(&pre_cmds[plan], plan, tokens[i..][0..n]);
            const wall: f64 = @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
            const total = try pres[plan].totals(&by_phase);
            if (pass == 1) try emit(writer, "prefill", n, position, total, &by_phase, wall);
            i += n;
        }
        var samples = try a.alloc(f64, steps);
        var phase_samples = try a.alloc([phase_count]f64, steps);
        var walls = try a.alloc(f64, steps);
        const decode_start = m.position;
        for (0..steps) |s| {
            const start = std.Io.Clock.awake.now(io);
            _ = try m.run(&dec_cmd, tokens[prompt_len + s]);
            walls[s] = @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
            samples[s] = try dec.totals(&phase_samples[s]);
        }
        if (pass == 1) {
            // Median step by GPU time; its phase breakdown and wall time.
            const order = try a.alloc(usize, steps);
            for (order, 0..) |*o, k| o.* = k;
            std.mem.sort(usize, order, samples, struct {
                fn lt(ctx: []f64, x: usize, y: usize) bool {
                    return ctx[x] < ctx[y];
                }
            }.lt);
            const mid = order[steps / 2];
            try emit(writer, "decode", 1, decode_start + @as(u32, @intCast(mid)), samples[mid], &phase_samples[mid], walls[mid]);
        }
    }
}
