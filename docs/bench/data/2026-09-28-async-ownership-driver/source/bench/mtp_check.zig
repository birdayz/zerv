//! Gate 2 input for the MTP draft layer (docs/specs/speculative.md): run zerv's MTP on a
//! teacher-forced token sequence and dump exactly what it consumed and produced, for the
//! FP64 reference (`tests/reference/mtp_reference.py`) to recompute from the same inputs.
//!
//! Scenario A: prefill(prompt) (one chunk, catch-up of every prompt row), then draft(t, j)
//! for j = 1..N: the chain step j's h' (`mo` row 0) and draft logits.
//! Scenario B: verify([t, c1..cN]), commit(M), then draft(c_M, j) for j = 1..N: a first
//! pass of M rows with h = the verify's final-norm rows.
//! Dumps: `hrows.bin` (prompt final-norm rows), `verify_hn.bin` (the verify's rows), per
//! scenario and j `<s>-<j>-mo.bin` (the h' of the row that drafted step j) and
//! `<s>-<j>-logits.bin` (FP32 little-endian), and
//! `index.json` (tokens, drafts, M, N).
//! Scenario C (no dump; bitwise, exit status 1 on a mismatch): a plain `step` with the
//! MTP layer runs the pending MTP rows and its own row, so the drafts after steps equal
//! the drafts after the equivalent draft/verify/commit path, bit for bit (probabilities
//! included): a step right after the prefill, a step after a 2-row commit, two steps.
//! Then the draft cost: wall-clock `draft(t, k)` per pending-row count and k (JSON lines).
//! Usage: zerv-mtp-check MODEL TOKENS.json OUTDIR [DRAFT_VOCAB]   (TOKENS: {"prompt": [..], "next": [..]})
//! DRAFT_VOCAB (default full): `Options.draft_vocab`; the logits dumps then hold that
//! many values (the draft head's ids 0..DRAFT_VOCAB-1).
const std = @import("std");
const zerv = @import("zerv");
const gpu = zerv.gpu;
const model = zerv.model;

const H = model.config.hidden;
const drafts_max = zerv.matvec.max_rows - 1;
const commit_rows = 3;

fn readAct(m: *model.Model, staging: *gpu.Buffer, cmd: *gpu.Commands, word: u64, words: u64, out: []f32) !void {
    try cmd.reset();
    try cmd.begin();
    try cmd.barrier(.compute, .transfer);
    try cmd.copy(&m.act, word * 4, staging, 0, words * 4);
    try cmd.barrier(.transfer, .host);
    try cmd.end();
    try cmd.run(60 * std.time.ns_per_s);
    @memcpy(std.mem.sliceAsBytes(out[0..words]), (try staging.mapped())[0 .. words * 4]);
}

fn write(io: std.Io, dir: std.Io.Dir, name: []const u8, data: []const f32) !void {
    try dir.writeFile(io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(data) });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4 and args.len != 5) return error.Usage;
    const draft_vocab: u32 = if (args.len == 5) try std.fmt.parseInt(u32, args[4], 10) else 0;
    const Tokens = struct { prompt: []const u32, next: []const u32 };
    const json = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 24));
    const tokens = try std.json.parseFromSliceLeaky(Tokens, a, json, .{});
    if (tokens.prompt.len == 0 or tokens.prompt.len > 512 or tokens.next.len < drafts_max + 1) return error.Usage;
    try std.Io.Dir.cwd().createDirPath(io, args[3]);
    var dir = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer dir.close(io);

    var file = try zerv.artifact.MappedFile.open(io, args[1], 64 * 1024 * 1024 * 1024);
    defer file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(a, file.bytes, .{});
    defer container.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 23 * 1024 * 1024 * 1024 });
    defer device.deinit() catch @panic("live device resources");
    var m: model.Model = undefined;
    try m.init(&device, &container, .{ .context = 4096, .prefill_rows = 512, .verify_rows = drafts_max + 1, .mtp = true, .draft_vocab = draft_vocab });
    const nv = m.draftVocab();
    defer m.deinit();
    const M = m.act_layout.mtp.?;
    var staging = try gpu.Buffer.init(&device, @as(u64, 512) * H * 4, .host);
    defer staging.deinit() catch @panic("staging");
    var cmd = try gpu.Commands.init(&device);
    defer cmd.deinit() catch @panic("cmd");
    const buf = try a.alloc(f32, 512 * H);

    // Scenario A.
    const P: u32 = @intCast(tokens.prompt.len);
    _ = try m.prefill(tokens.prompt);
    try readAct(&m, &staging, &cmd, M.hrows + H, @as(u64, P) * H, buf); // row 0 is the pending h
    try write(io, dir, "hrows.bin", buf[0 .. P * H]);
    const t0 = tokens.next[0];
    var drafts_a: [drafts_max]u32 = undefined;
    var probs_a: [drafts_max]f32 = undefined;
    for (1..drafts_max + 1) |j| {
        const d = try m.draft(t0, @intCast(j));
        drafts_a[j - 1] = d[j - 1];
        probs_a[j - 1] = m.draftProbs(@intCast(j))[j - 1];
        if (j > 1) if (!std.mem.eql(u32, d[0 .. j - 1], drafts_a[0 .. j - 1])) return error.DraftNotRepeatable;
        try readAct(&m, &staging, &cmd, M.mo, H, buf);
        try write(io, dir, try std.fmt.allocPrint(a, "A-{d}-mo.bin", .{j}), buf[0..H]);
        try readAct(&m, &staging, &cmd, M.logits, nv, buf);
        try write(io, dir, try std.fmt.allocPrint(a, "A-{d}-logits.bin", .{j}), buf[0..nv]);
    }

    // Scenario B: verify the true continuation, keep `commit_rows` rows.
    const verify_tokens = tokens.next[0 .. drafts_max + 1];
    _ = try m.verify(verify_tokens);
    try readAct(&m, &staging, &cmd, m.act_layout.hn, (drafts_max + 1) * H, buf);
    try write(io, dir, "verify_hn.bin", buf[0 .. (drafts_max + 1) * H]);
    try m.commit(commit_rows);
    const t1 = tokens.next[commit_rows];
    var drafts_b: [drafts_max]u32 = undefined;
    var probs_b: [drafts_max]f32 = undefined;
    for (1..drafts_max + 1) |j| {
        const d = try m.draft(t1, @intCast(j));
        drafts_b[j - 1] = d[j - 1];
        probs_b[j - 1] = m.draftProbs(@intCast(j))[j - 1];
        if (j > 1) if (!std.mem.eql(u32, d[0 .. j - 1], drafts_b[0 .. j - 1])) return error.DraftNotRepeatable;
        // Step 1's h' is the first pass's last row (commit_rows rows); chained passes use row 0.
        try readAct(&m, &staging, &cmd, M.mo + @as(u64, if (j == 1) commit_rows - 1 else 0) * H, H, buf);
        try write(io, dir, try std.fmt.allocPrint(a, "B-{d}-mo.bin", .{j}), buf[0..H]);
        try readAct(&m, &staging, &cmd, M.logits, nv, buf);
        try write(io, dir, try std.fmt.allocPrint(a, "B-{d}-logits.bin", .{j}), buf[0..nv]);
    }
    var out: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(.{ .prompt = tokens.prompt, .next = tokens.next, .commit_rows = commit_rows, .drafts = drafts_max, .drafts_a = drafts_a, .drafts_b = drafts_b, .probs_a = probs_a, .probs_b = probs_b }, .{}, &out.writer);
    try dir.writeFile(io, .{ .sub_path = "index.json", .data = out.written() });
    var stdout_buf: [256]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &stdout_buf);
    try stdout.interface.print("{s}\n", .{out.written()});
    try stdout.interface.flush();

    // Scenario C. `x` = t0..t4; each case ends with draft(x[e], N) on both paths.
    const x = tokens.next[0..5];
    const Path = enum { steps, verified };
    var ok = true;
    for (0..3) |case| {
        var got: [2][drafts_max]u32 = undefined;
        var got_p: [2][drafts_max]f32 = undefined;
        for ([_]Path{ .steps, .verified }, 0..) |path, pi| {
            try m.reset();
            _ = try m.prefill(tokens.prompt);
            var e: usize = 0;
            switch (case) {
                // step(t0) | draft(t0,1) verify(t0,t1) commit(1); then draft(t1)
                0 => {
                    if (path == .steps) _ = try m.step(x[0]) else try verified(&m, x[0..2], 1);
                    e = 1;
                },
                // commit(2) of (t0,t1); then step(t2) | draft(t2,1) verify(t2,t3) commit(1); then draft(t3)
                1 => {
                    try verified(&m, x[0..2], 2);
                    if (path == .steps) _ = try m.step(x[2]) else try verified(&m, x[2..4], 1);
                    e = 3;
                },
                // step(t0) step(t1) | draft(t0,1) verify(t0,t1,t2) commit(2); then draft(t2)
                else => {
                    if (path == .steps) {
                        _ = try m.step(x[0]);
                        _ = try m.step(x[1]);
                    } else try verified(&m, x[0..3], 2);
                    e = 2;
                },
            }
            const d = try m.draft(x[e], drafts_max);
            got[pi] = d[0..drafts_max].*;
            got_p[pi] = m.draftProbs(drafts_max)[0..drafts_max].*;
        }
        const equal = std.mem.eql(u32, &got[0], &got[1]) and std.mem.eql(u32, @ptrCast(&got_p[0]), @ptrCast(&got_p[1]));
        ok = ok and equal;
        try std.json.Stringify.value(.{ .scenario = "C", .case = case, .drafts_steps = got[0], .drafts_verified = got[1], .equal = equal }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
    }
    // Cost (wall clock per call, as the engine sees it; no dump): draft(t, k) with m
    // pending rows, m = 1 (after a step) and m = 3 (after a 3-row commit), k = 1..N; a
    // draft is idempotent, so the same call repeats. Median of 41 after 8 warm-up calls.
    const samples = try a.alloc(f64, 41);
    for ([_]u32{ 1, 3 }) |pending| {
        try m.reset();
        _ = try m.prefill(tokens.prompt);
        if (pending == 1) _ = try m.step(tokens.next[0]) else try verified(&m, tokens.next[0..3], 3);
        const token = tokens.next[pending];
        for (1..drafts_max + 1) |k| {
            for (0..samples.len + 8) |s| {
                const start = std.Io.Clock.awake.now(io);
                _ = try m.draft(token, @intCast(k));
                const dt: f64 = @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
                if (s >= 8) samples[s - 8] = dt / 1e6;
            }
            std.mem.sort(f64, samples, {}, std.sort.asc(f64));
            try std.json.Stringify.value(.{ .timing = "draft", .pending_rows = pending, .k = k, .ms_median = samples[samples.len / 2], .ms_min = samples[0], .ms_max = samples[samples.len - 1] }, .{}, &stdout.interface);
            try stdout.interface.writeByte('\n');
            try stdout.interface.flush();
        }
    }
    if (!ok) std.process.exit(1);
}

/// draft(tokens[0], 1), then verify `tokens` and commit `keep` rows (the drafts' values do
/// not matter: the verify processes `tokens`).
fn verified(m: *model.Model, tokens: []const u32, keep: u32) !void {
    _ = try m.draft(tokens[0], 1);
    _ = try m.verify(tokens);
    try m.commit(keep);
}
