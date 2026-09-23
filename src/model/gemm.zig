//! Scalar-X FP32 GEMM kernels for batched prefill (see gemm.comp). Driver-free geometry
//! and extent validation; the caller owns kernels, buffers and barriers.
const std = @import("std");
const matvec = @import("../matvec/root.zig");

pub const Variant = enum { f32_k, f32_m, q4_0, q4_1, q5_k, q6_k };
pub const Push = extern struct {
    a_base: u32,
    a_rs: u32,
    a_cs: u32 = 0,
    a_bs: u32 = 0,
    a_group: u32 = 1,
    x_base: u32,
    x_rs: u32,
    x_bs: u32 = 0,
    y_base: u32,
    y_rs: u32,
    y_bs: u32 = 0,
    m: u32,
    k: u32,
    flags: u32 = 0,
    /// >0: split-K; batch index z becomes the split, partial z at y_base + z*y_bs.
    k_chunk: u32 = 0,
};
/// Split-K reduction kernel (model.comp K_REDUCE; bindings: bank0, act, state, io).
pub const ReducePush = extern struct { part: u32, splits: u32, part_bs: u32, y: u32, y_rs: u32, m: u32 };
pub fn reduceModule() []align(4) const u8 {
    const M = struct {
        const code align(4) = @embedFile("shaders/reduce.spv").*;
    };
    return &M.code;
}
pub const m_from_keys: u32 = 1;
pub const k_from_keys: u32 = 2;
pub const Error = error{ InvalidShape, InvalidRange };

pub fn variant(format: matvec.Format) Variant {
    return switch (format) {
        .f32 => .f32_k,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
    };
}

/// Output tile of the scalar-X kernel (gemm.comp): 256 M rows x `Tile.rows()` chunk rows
/// per workgroup. `wide` (16 X rows per 64-thread group, XW=16) has the same arithmetic
/// and summation order per element as `narrow`; outputs are bit-identical at equal
/// split-K. Block 13h: docs/specs/prefill.md.
pub const Tile = enum {
    narrow,
    wide,
    pub fn rows(self: Tile) u32 {
        return switch (self) {
            .narrow => 32,
            .wide => 64,
        };
    }
};
pub const tile_m = 256;

/// Measured rule (docs/bench/2026-09-23-gemm-efficiency.md): the wide tile is faster for
/// full 512-row (or larger) chunks of projections with M >= 4096, slower for small row
/// counts (padding) and small M (too few workgroups).
pub fn tileFor(M: u32, rows: u32) Tile {
    return if (rows >= 512 and M >= 4096) .wide else .narrow;
}

/// Kernels bind (A, act, io, act): the fourth binding is a read-only view of the
/// activation buffer for X. They need compute subgroups of at most 64 with ballot.
/// The wide tile exists for the quantized formats only.
pub fn module(v: Variant, tile: Tile) Error![]align(4) const u8 {
    const M = struct {
        const f32_k align(4) = @embedFile("shaders/gemm_f32_k.spv").*;
        const f32_m align(4) = @embedFile("shaders/gemm_f32_m.spv").*;
        const q4_0 align(4) = @embedFile("shaders/gemm_q4_0.spv").*;
        const q4_1 align(4) = @embedFile("shaders/gemm_q4_1.spv").*;
        const q5_k align(4) = @embedFile("shaders/gemm_q5_k.spv").*;
        const q6_k align(4) = @embedFile("shaders/gemm_q6_k.spv").*;
        const q4_0_w align(4) = @embedFile("shaders/gemm_q4_0_w.spv").*;
        const q4_1_w align(4) = @embedFile("shaders/gemm_q4_1_w.spv").*;
        const q5_k_w align(4) = @embedFile("shaders/gemm_q5_k_w.spv").*;
        const q6_k_w align(4) = @embedFile("shaders/gemm_q6_k_w.spv").*;
    };
    return switch (tile) {
        .narrow => switch (v) {
            .f32_k => &M.f32_k,
            .f32_m => &M.f32_m,
            .q4_0 => &M.q4_0,
            .q4_1 => &M.q4_1,
            .q5_k => &M.q5_k,
            .q6_k => &M.q6_k,
        },
        .wide => switch (v) {
            .f32_k, .f32_m => error.InvalidShape,
            .q4_0 => &M.q4_0_w,
            .q4_1 => &M.q4_1_w,
            .q5_k => &M.q5_k_w,
            .q6_k => &M.q6_k_w,
        },
    };
}

/// Prefill arithmetic (block 14, docs/specs/prefill.md). `fp32` is the default; `f16`
/// runs in-scope projections on the f16 x f16 -> f32 WMMA kernel (gemm_f16.comp).
pub const Precision = enum { fp32, f16 };

/// f16 WMMA tile: 128 M rows x 128 chunk rows per workgroup (4 subgroups of 64).
pub const f16_tile_m = 128;
pub const f16_tile_n = 128;

/// Projections the f16 mode runs on the WMMA kernel; everything else stays FP32.
pub fn f16Eligible(v: Variant, M: u32, plan_rows: u32) bool {
    const format_ok = v == .q4_0 or v == .q4_1 or v == .q5_k;
    return format_ok and M >= 4096 and M % f16_tile_m == 0 and plan_rows % f16_tile_n == 0;
}

/// Device features a prefill precision needs (pass them to `gpu.Device.open`).
pub const DeviceNeeds = struct { cooperative_matrix: bool, subgroup_size_control: bool };
pub fn deviceNeeds(p: Precision) DeviceNeeds {
    return .{ .cooperative_matrix = p == .f16, .subgroup_size_control = p == .f16 };
}

/// f16 WMMA kernels (docs/specs/prefill.md).
pub const F16Kernel = enum {
    /// gemm_f16.comp (block 14): wave64, 128 x 128 tile, converts f32 X in the kernel.
    wave64,
    /// gemm_f16x.comp (block 16b): wave32 (required size, full subgroups), 128 x 256
    /// tile, BK 64; reads the producer's f16 copy of X (x_base / x_rs in halves).
    x16,
};
pub const f16x_tile_m = 128;
pub const f16x_tile_n = 256;
pub const f16x_subgroup = 32;

/// Kernel of an f16-mode projection, or null when it stays FP32. Measured rule for the
/// RDNA3 target (docs/bench/2026-09-24-gemm-f16-lab.md): the wave32 kernel for Q4_0 with
/// whole 256-row tiles, else the block-14 kernel.
pub fn f16Kernel(v: Variant, M: u32, K: u32, plan_rows: u32) ?F16Kernel {
    if (!f16Eligible(v, M, plan_rows)) return null;
    if (v == .q4_0 and K % 64 == 0 and plan_rows % f16x_tile_n == 0) return .x16;
    return .wave64;
}

/// gemm_f16x modules (Q4_0 only). Need cooperative matrices, a required subgroup size of
/// `f16x_subgroup` and full subgroups.
pub fn moduleF16x(v: Variant) Error![]align(4) const u8 {
    const M = struct {
        const q4_0 align(4) = @embedFile("shaders/gemm_f16x_q4_0.spv").*;
    };
    return switch (v) {
        .q4_0 => &M.q4_0,
        else => error.InvalidShape,
    };
}

/// Checks a gemm_f16x dispatch: no split-K, whole 256-row tiles (every row of every tile,
/// up to `lim.rows`, may be read and written), f16 X rows 16-byte aligned (x_base and x_rs
/// in halves, multiples of 8). Returns the grid.
pub fn validateF16x(v: Variant, p: Push, lim: Limits, a_bytes: u64, act_bytes: u64) Error![3]u32 {
    if (lim.batches != 1 or p.k_chunk != 0 or p.flags != 0 or p.a_group != 1 or p.k == 0) return error.InvalidShape;
    if (f16Kernel(v, p.m, p.k, lim.rows) != .x16) return error.InvalidShape;
    try checkA(v, p, 1, p.m, p.k, a_bytes);
    if (p.x_base % 8 != 0 or p.x_rs % 8 != 0 or p.x_rs < p.k) return error.InvalidShape;
    const x_end = last(p.x_base, lim.rows, p.x_rs) + p.k; // halves
    const y_end = last(p.y_base, lim.rows, p.y_rs) + p.m;
    if (x_end * 2 > act_bytes or y_end * 4 > act_bytes) return error.InvalidRange;
    return .{ p.m / f16x_tile_m, lim.rows / f16x_tile_n, 1 };
}

/// f16 WMMA modules (Q4_0, Q4_1, Q5_K). Need a device opened with cooperative matrices
/// and subgroups of exactly 64.
pub fn moduleF16(v: Variant) Error![]align(4) const u8 {
    const M = struct {
        const q4_0 align(4) = @embedFile("shaders/gemm_f16_q4_0.spv").*;
        const q4_1 align(4) = @embedFile("shaders/gemm_f16_q4_1.spv").*;
        const q5_k align(4) = @embedFile("shaders/gemm_f16_q5_k.spv").*;
    };
    return switch (v) {
        .q4_0 => &M.q4_0,
        .q4_1 => &M.q4_1,
        .q5_k => &M.q5_k,
        else => error.InvalidShape,
    };
}

/// Checks an f16 WMMA dispatch (no split-K, no runtime M/K, whole tiles: every row of
/// every tile, up to `lim.rows`, may be read and written). Returns the grid.
pub fn validateF16(v: Variant, p: Push, lim: Limits, a_bytes: u64, act_bytes: u64) Error![3]u32 {
    if (lim.batches != 1 or p.k_chunk != 0 or p.flags != 0 or p.a_group != 1) return error.InvalidShape;
    if (!f16Eligible(v, p.m, lim.rows) or p.k == 0) return error.InvalidShape;
    const grid = try validate(v, .narrow, p, lim, a_bytes, act_bytes); // extents and format rules
    // Stores cover whole 128-row tiles; lim.rows is a multiple of 128 (checked above).
    const y_end = last(p.y_base, lim.rows, p.y_rs) + p.m;
    if (y_end * 4 > act_bytes) return error.InvalidRange;
    if (v == .q5_k and p.a_bs % 16 != 0) return error.InvalidShape;
    return .{ p.m / f16_tile_m, lim.rows / f16_tile_n, grid[2] };
}

/// Worst-case extents for validation before recording. Sizes are in elements (M, K,
/// rows); `max_keys` bounds M/K when taken from the io block at run time.
pub const Limits = struct { rows: u32, batches: u32 = 1, max_keys: u32 = 0 };

fn last(base: u64, count: u64, stride: u64) u64 {
    return if (count == 0) base else base + (count - 1) * stride;
}

/// Checks every address the kernel can touch lies inside the given buffer sizes
/// (bytes). Returns the dispatch grid.
pub fn validate(v: Variant, tile: Tile, p: Push, lim: Limits, a_bytes: u64, act_bytes: u64) Error![3]u32 {
    if (lim.rows == 0 or lim.batches == 0 or p.a_group == 0) return error.InvalidShape;
    if (tile == .wide and (v == .f32_k or v == .f32_m)) return error.InvalidShape;
    const M: u64 = if (p.flags & m_from_keys != 0) lim.max_keys else p.m;
    const K: u64 = if (p.flags & k_from_keys != 0) lim.max_keys else p.k;
    if (M == 0 or K == 0) return error.InvalidShape;
    const split = p.k_chunk != 0;
    if (split and (p.k_chunk % 256 != 0 or p.flags != 0 or lim.batches != std.math.divCeil(u64, K, p.k_chunk) catch unreachable)) return error.InvalidShape;
    const a_batches = if (split) 1 else (lim.batches + p.a_group - 1) / p.a_group;
    const x_batches: u32 = if (split) 1 else lim.batches;
    if ((v != .f32_k and v != .f32_m) and p.flags != 0) return error.InvalidShape;
    try checkA(v, p, a_batches, M, K, a_bytes);
    // vec4 X loads: 16-byte aligned rows; a row may be read to the end of its last 32-k block.
    if (p.x_base % 4 != 0 or p.x_rs % 4 != 0 or p.x_bs % 4 != 0) return error.InvalidShape;
    const x_end = last(last(p.x_base, x_batches, p.x_bs), lim.rows, p.x_rs) + std.mem.alignForward(u64, K, 32);
    const y_end = last(last(p.y_base, lim.batches, p.y_bs), lim.rows, p.y_rs) + M;
    if (x_end * 4 > act_bytes or y_end * 4 > act_bytes) return error.InvalidRange;
    const gx = std.math.divCeil(u64, M, tile_m) catch unreachable;
    const gy = std.math.divCeil(u64, lim.rows, tile.rows()) catch unreachable;
    if (gx > 65535 or gy > 65535 or lim.batches > 65535) return error.InvalidShape;
    return .{ @intCast(gx), @intCast(gy), lim.batches };
}

/// Every weight address of `a_batches` batches of M rows by K lies inside `a_bytes`, and
/// the format's layout rules hold.
fn checkA(v: Variant, p: Push, a_batches: u64, M: u64, K: u64, a_bytes: u64) Error!void {
    switch (v) {
        .f32_k, .f32_m => {
            const end = last(last(last(p.a_base, a_batches, p.a_bs), M, p.a_rs), K, p.a_cs) + 1;
            if (end * 4 > a_bytes) return error.InvalidRange;
        },
        else => {
            const format: matvec.Format = switch (v) {
                .q4_0 => .q4_0,
                .q4_1 => .q4_1,
                .q5_k => .q5_k,
                .q6_k => .q6_k,
                else => unreachable,
            };
            if (K % format.blockElements() != 0) return error.InvalidShape;
            if (p.a_rs != K / format.blockElements() * format.blockBytes() or p.a_base % 2 != 0) return error.InvalidShape;
            // Q5_K rows are decoded with aligned 16-byte loads.
            if (v == .q5_k and (p.a_base % 16 != 0 or p.a_rs % 16 != 0 or p.a_bs % 16 != 0)) return error.InvalidShape;
            // Two-byte-aligned fields are read through the words containing their bytes.
            const end = std.mem.alignForward(u64, last(p.a_base, a_batches, p.a_bs) + M * p.a_rs, 4);
            if (end > a_bytes) return error.InvalidRange;
        },
    }
}

/// Split-K chunk (multiple of 256) so that at least `target` workgroups of `tile` run;
/// 0 = no split.
pub fn splitChunk(M: u32, rows: u32, K: u32, target: u32, tile: Tile) u32 {
    const tiles = (std.math.divCeil(u32, M, tile_m) catch unreachable) * (std.math.divCeil(u32, rows, tile.rows()) catch unreachable);
    if (tiles >= target or K < 512) return 0;
    const splits = @min(std.math.divCeil(u32, target, tiles) catch unreachable, K / 256);
    if (splits <= 1) return 0;
    return std.mem.alignForward(u32, std.math.divCeil(u32, K, splits) catch unreachable, 256);
}
pub fn splitCount(K: u32, k_chunk: u32) u32 {
    return if (k_chunk == 0) 1 else std.math.divCeil(u32, K, k_chunk) catch unreachable;
}
