//! Split-K decode attention (docs/specs/decode-attention.md): push layouts, modules and
//! dispatch geometry of the three passes. Kernels bind (params, act, state, io) and
//! read the position from io; keys are split into chunks of `layout.attn_chunk`.
const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");

pub const chunk = layout.attn_chunk;
pub const heads_per_kv = config.heads / config.kv_heads;

/// `pos`: io word holding the position of row 0 (decode and verify: `layout.io.position`).
/// `slots, slot_rs`: the slot entry of the rows (docs/specs/concurrent.md, "18b.2 design";
/// 0, 0 = none): with `slot_rs` 4, row r's position and page table come from its entry.
pub const ScoresPush = extern struct { qr: u32, scores: u32, amax: u32, kcache: u32, ctx: u32, chunks: u32, scale: f32, pos: u32, ptab: u32, pstride: u32, slots: u32 = 0, slot_rs: u32 = 0 };
pub const PvPush = extern struct { scores: u32, amax: u32, apart: u32, asum: u32, vcache: u32, ctx: u32, chunks: u32, pos: u32, gmax: u32, ptab: u32, pstride: u32, slots: u32 = 0, slot_rs: u32 = 0 };
/// Between scores and P·V (block 17c): per (row, head) maximum of the live chunk maxima.
pub const GmaxPush = extern struct { amax: u32, gmax: u32, chunks: u32, pos: u32, slots: u32 = 0, slot_rs: u32 = 0 };
pub const CombinePush = extern struct { bpart: u32, bsum: u32, qf: u32, pregate: u32, gates: u32, gated: u32, blocks: u32, pos: u32, slots: u32 = 0, slot_rs: u32 = 0 };
/// Pass 3a (block 17c): 8-chunk block sums of the partials, one workgroup per block.
pub const BlockPush = extern struct { apart: u32, asum: u32, bpart: u32, bsum: u32, chunks: u32, blocks: u32, pos: u32, slots: u32 = 0, slot_rs: u32 = 0 };

pub const Pass = enum { scores, gmax, pv, block, combine };

/// The pass's module for KV cache elements of type `kv` (combine reads no KV).
pub fn module(pass: Pass, kv: layout.KvType) []align(4) const u8 {
    const M = struct {
        const scores align(4) = @embedFile("shaders/attn_scores.spv").*;
        const pv align(4) = @embedFile("shaders/attn_pv.spv").*;
        const combine align(4) = @embedFile("shaders/attn_combine.spv").*;
        const gmax align(4) = @embedFile("shaders/attn_gmax.spv").*;
        const block align(4) = @embedFile("shaders/attn_cblock.spv").*;
        const scores16 align(4) = @embedFile("shaders/attn_scores_kv16.spv").*;
        const pv16 align(4) = @embedFile("shaders/attn_pv_kv16.spv").*;
    };
    return switch (pass) {
        .scores => if (kv == .f16) &M.scores16 else &M.scores,
        .pv => if (kv == .f16) &M.pv16 else &M.pv,
        .combine => &M.combine,
        .gmax => &M.gmax,
        .block => &M.block,
    };
}

pub fn pushBytes(pass: Pass) u32 {
    return switch (pass) {
        .scores => @sizeOf(ScoresPush),
        .pv => @sizeOf(PvPush),
        .combine => @sizeOf(CombinePush),
        .gmax => @sizeOf(GmaxPush),
        .block => @sizeOf(BlockPush),
    };
}

/// Grids sized for the whole context; chunks past the live key count exit early. `rows`
/// query rows (decode 1, speculative verification up to 5) at positions io position + r,
/// each with its own scratch rows (docs/specs/speculative.md). Scores and P V run
/// (row, chunk, KV head) with rows fastest, so the rows of a key chunk read its K/V back
/// to back (from L2, not DRAM, for every row but the first); combine runs (head, 1, row).
pub fn groups(pass: Pass, chunks: u32, rows: u32) [3]u32 {
    return switch (pass) {
        .scores => .{ rows, std.math.divCeil(u32, chunks, 2) catch unreachable, config.kv_heads },
        .pv => .{ rows, chunks, config.kv_heads },
        .gmax => .{ rows, 1, config.kv_heads },
        .block => .{ std.math.divCeil(u32, chunks, 8) catch unreachable, config.heads, rows },
        .combine => .{ config.heads, 1, rows },
    };
}
