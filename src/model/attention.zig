//! Split-K decode attention (docs/specs/decode-attention.md): push layouts, modules and
//! dispatch geometry of the three passes. Kernels bind (params, act, state, io) and
//! read the position from io; keys are split into chunks of `layout.attn_chunk`.
const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");

pub const chunk = layout.attn_chunk;
pub const heads_per_kv = config.heads / config.kv_heads;

pub const ScoresPush = extern struct { qr: u32, scores: u32, amax: u32, kcache: u32, ctx: u32, chunks: u32, scale: f32 };
pub const PvPush = extern struct { scores: u32, amax: u32, apart: u32, asum: u32, vcache: u32, ctx: u32, chunks: u32 };
pub const CombinePush = extern struct { apart: u32, asum: u32, qf: u32, pregate: u32, gates: u32, gated: u32, chunks: u32 };

pub const Pass = enum { scores, pv, combine };

pub fn module(pass: Pass) []align(4) const u8 {
    const M = struct {
        const scores align(4) = @embedFile("shaders/attn_scores.spv").*;
        const pv align(4) = @embedFile("shaders/attn_pv.spv").*;
        const combine align(4) = @embedFile("shaders/attn_combine.spv").*;
    };
    return switch (pass) {
        .scores => &M.scores,
        .pv => &M.pv,
        .combine => &M.combine,
    };
}

pub fn pushBytes(pass: Pass) u32 {
    return switch (pass) {
        .scores => @sizeOf(ScoresPush),
        .pv => @sizeOf(PvPush),
        .combine => @sizeOf(CombinePush),
    };
}

/// Grids sized for the whole context; chunks past the live key count exit early.
pub fn groups(pass: Pass, chunks: u32) [3]u32 {
    return switch (pass) {
        .scores, .pv => .{ chunks, config.kv_heads, 1 },
        .combine => .{ config.heads, 1, 1 },
    };
}
