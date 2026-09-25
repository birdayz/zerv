// Throwaway: largest context (multiple of 32) whose buffers fit `free` MiB, with the
// runtime's own layout functions and its `needed` formula.
const std = @import("std");
const zerv = @import("zerv");
const layout = zerv.model.layout;
const config = zerv.model.config;
const gemm = zerv.model.gemm;
fn splitSlot(name: []const u8) usize {
    const suffixes = [_]struct { []const u8, usize }{
        .{ "attn_k.weight", 1 },    .{ "attn_v.weight", 2 },   .{ "attn_gate.weight", 1 },
        .{ "ssm_alpha.weight", 2 }, .{ "ssm_beta.weight", 3 }, .{ "ffn_up.weight", 1 },
    };
    for (suffixes) |e| if (std.mem.endsWith(u8, name, e[0])) return e[1];
    return 0;
}
const weights_trunk: u64 = 15049 * 1024 * 1024 + 700 * 1024; // from container.json (32-byte aligned)
const weights_mtp: u64 = 253 * 1024 * 1024;
fn needed(ctx: u32, rows: u32, verify: u32, mtp: bool, slots: u64) !u64 {
    var plans: [zerv.model.max_plans]zerv.model.Plan = undefined;
    const n = zerv.model.makePlans(rows, &plans);
    var slot_words: [4]u64 = @splat(0);
    for (plans[0..n]) |plan| for (&config.tensors()) |*s| {
        if (s.role != .matrix or std.mem.eql(u8, config.specName(s), "output.weight")) continue;
        const chunk = gemm.splitChunk(@intCast(s.rows), plan.rows, @intCast(s.k), 384, gemm.tileFor(@intCast(s.rows), plan.rows));
        if (chunk == 0) continue;
        const w = @as(u64, gemm.splitCount(@intCast(s.k), chunk)) * plan.rows * s.rows;
        const sl = splitSlot(config.specName(s));
        slot_words[sl] = @max(slot_words[sl], std.mem.alignForward(u64, w, 64));
    };
    var part: u64 = 0;
    for (slot_words) |w| part += w;
    const a = try layout.act(ctx, rows, part, false, @max(verify, 1), mtp);
    const s = try layout.state(ctx, 0xf000_0000, mtp);
    const snap = zerv.model.snapshot_bytes + @as(u64, if (mtp) config.hidden * 4 else 0);
    return a.words * 4 + s.words * 4 + s.kvBytes() + snap * slots + zerv.model.vram_headroom + weights_trunk + (if (mtp) weights_mtp else 0);
}
pub fn main() !void {
    const free: u64 = 23759 * 1024 * 1024;
    const cfgs = [_]struct { name: []const u8, rows: u32, verify: u32, mtp: bool, slots: u64 }{
        .{ .name = "no spec, 8 slots", .rows = 512, .verify = 0, .mtp = false, .slots = 8 },
        .{ .name = "no spec, 2 slots", .rows = 512, .verify = 0, .mtp = false, .slots = 2 },
        .{ .name = "spec 3, 8 slots", .rows = 512, .verify = 4, .mtp = true, .slots = 8 },
        .{ .name = "spec 3, 2 slots", .rows = 512, .verify = 4, .mtp = true, .slots = 2 },
        .{ .name = "spec 3, 2 slots, chunk 256", .rows = 256, .verify = 4, .mtp = true, .slots = 2 },
        .{ .name = "spec 3, 0 slots, chunk 256", .rows = 256, .verify = 4, .mtp = true, .slots = 0 },
    };
    for (cfgs) |c| {
        var ctx: u32 = 32;
        var best: u32 = 0;
        var why: []const u8 = "vram";
        while (ctx <= 400000) : (ctx += 32) {
            const b = needed(ctx, c.rows, c.verify, c.mtp, c.slots) catch {
                why = "activation arena < 4 GiB";
                break;
            };
            if (b > free) break;
            best = ctx;
        }
        const b = try needed(best, c.rows, c.verify, c.mtp, c.slots);
        const s = try layout.state(best, 0xf000_0000, c.mtp);
        std.debug.print("{s:28}: max context {d} ({s}); needs {d} MiB, KV buffers {d}\n", .{ c.name, best, why, b >> 20, s.kv_buffers });
    }
}
