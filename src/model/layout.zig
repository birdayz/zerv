//! Driver-free placement of weights into banks and word layouts of the device arenas.
const std = @import("std");
const config = @import("config.zig");

pub const max_banks = 8;
pub const Error = error{ BankOverflow, ContextTooLarge, InvalidContext };

pub const Item = struct { role: config.Role, bytes: u64 };
pub const Placement = struct { bank: u8, offset: u64 };
pub const Banks = struct { count: u8, bytes: [max_banks]u64 };

/// Params and the embedding go first into bank 0 (operator kernels bind only bank 0);
/// matrices then fill banks in inventory order. Every offset is 32-byte aligned.
/// `capacity` must itself be a multiple of 32 and below 4 GiB (uint32 addressing).
pub fn place(items: []const Item, capacity: u64, out: []Placement) Error!Banks {
    std.debug.assert(out.len == items.len and capacity % 32 == 0 and capacity <= 0xffffffe0);
    var banks: Banks = .{ .count = 1, .bytes = @splat(0) };
    for ([_]bool{ true, false }) |bank0_pass| {
        for (items, out) |item, *slot| {
            if ((item.role != .matrix) != bank0_pass) continue;
            const size = std.mem.alignForward(u64, item.bytes, 32);
            if (size > capacity) return error.BankOverflow;
            var bank = banks.count - 1;
            if (banks.bytes[bank] + size > capacity) {
                if (bank0_pass) return error.BankOverflow; // bank 0 must hold every param
                if (banks.count == max_banks) return error.BankOverflow;
                banks.count += 1;
                bank += 1;
            }
            slot.* = .{ .bank = bank, .offset = banks.bytes[bank] };
            banks.bytes[bank] += size;
        }
    }
    return banks;
}

/// Activation arena, FP32 word offsets. `scores` holds heads * context words.
pub const Act = struct {
    x: u32,
    r: u32,
    h: u32,
    a: u32,
    f: u32,
    hn: u32,
    mixed: u32,
    z: u32,
    beta_raw: u32,
    alpha: u32,
    beta: u32,
    softplus: u32,
    gate: u32,
    conv_raw: u32,
    conv_silu: u32,
    conv_out: u32,
    o: u32,
    fo: u32,
    qf: u32,
    kc: u32,
    vc: u32,
    qn: u32,
    qr: u32,
    kn: u32,
    kr: u32,
    pregate: u32,
    gates: u32,
    gated: u32,
    fg: u32,
    fu: u32,
    sw: u32,
    scores: u32,
    part: u32,
    /// Split-K decode attention scratch: per (head, key chunk) maximum, partial P·V
    /// (head_dim words) and partial exp-sum; `chunks` = ceil(context / attn_chunk).
    amax: u32,
    apart: u32,
    asum: u32,
    chunks: u32,
    /// f16 copy of one GEMM input at a time (f16 mode, block 16b): `rows` rows of up to
    /// `config.ffn` halves; null when not requested.
    x16: ?u32,
    rows: u32,
    words: u64,
};

/// Keys per split-K decode attention chunk (fixed in the shaders).
pub const attn_chunk = 64;

pub const State = struct {
    context: u32,
    ssm: u32, // linear layers × v_heads × 128 × 128, zeroed by reset
    conv: u32, // linear layers × channels × 3, zeroed by reset (contiguous after ssm)
    kv: u32, // attention layers × (K [kvh][dim][ctx] + V [kvh][ctx][dim])
    words: u64,

    pub const ssm_words: u32 = config.linear_layers * config.v_heads * config.linear_head * config.linear_head;
    pub const conv_words: u32 = config.linear_layers * config.conv_channels * (config.conv_kernel - 1);
    pub fn kcache(self: State, attention: u32) u32 {
        return self.kv + attention * 2 * config.kv_heads * config.head_dim * self.context;
    }
    pub fn vcache(self: State, attention: u32) u32 {
        return self.kcache(attention) + config.kv_heads * config.head_dim * self.context;
    }
    pub fn ssmLayer(self: State, linear: u32) u32 {
        return self.ssm + linear * config.v_heads * config.linear_head * config.linear_head;
    }
    pub fn convLayer(self: State, linear: u32) u32 {
        return self.conv + linear * config.conv_channels * (config.conv_kernel - 1);
    }
};

/// Largest context whose state arena fits `capacity` bytes.
pub fn maxContext(capacity: u64) u32 {
    const fixed = @as(u64, State.ssm_words) + State.conv_words;
    const per_token: u64 = config.attention_layers * 2 * config.kv_heads * config.head_dim;
    if (capacity / 4 <= fixed) return 0;
    return @intCast(@min((capacity / 4 - fixed) / per_token, std.math.maxInt(u32)));
}

pub fn state(context: u32, capacity: u64) Error!State {
    if (context == 0) return error.InvalidContext;
    if (context > maxContext(capacity)) return error.ContextTooLarge;
    const kv = State.ssm_words + State.conv_words;
    const words = @as(u64, kv) + @as(u64, config.attention_layers) * 2 * config.kv_heads * config.head_dim * context;
    return .{ .context = context, .ssm = 0, .conv = State.ssm_words, .kv = kv, .words = words };
}

/// Row stride (words) of each activation region; regions hold `rows` rows.
pub const widths = .{
    .{ "x", config.hidden },                       .{ "r", config.hidden },                        .{ "h", config.hidden },
    .{ "a", config.hidden },                       .{ "f", config.hidden },                        .{ "hn", config.hidden },
    .{ "mixed", config.conv_channels },            .{ "z", config.value_dim },                     .{ "beta_raw", config.v_heads },
    .{ "alpha", config.v_heads },                  .{ "beta", config.v_heads },                    .{ "softplus", config.v_heads },
    .{ "gate", config.v_heads },                   .{ "conv_raw", config.conv_channels },          .{ "conv_silu", config.conv_channels },
    .{ "conv_out", config.conv_channels },         .{ "o", config.value_dim },                     .{ "fo", config.value_dim },
    .{ "qf", 2 * config.heads * config.head_dim }, .{ "kc", config.kv_heads * config.head_dim },   .{ "vc", config.kv_heads * config.head_dim },
    .{ "qn", config.heads * config.head_dim },     .{ "qr", config.heads * config.head_dim },      .{ "kn", config.kv_heads * config.head_dim },
    .{ "kr", config.kv_heads * config.head_dim },  .{ "pregate", config.heads * config.head_dim }, .{ "gates", config.heads * config.head_dim },
    .{ "gated", config.heads * config.head_dim },  .{ "fg", config.ffn },                          .{ "fu", config.ffn },
    .{ "sw", config.ffn },
};

/// `rows` >= 1 activation rows per region (1 = decode only); `scores` holds
/// heads * rows * context words; `part` holds split-K partial slots; `x16` adds the f16
/// input-copy region last.
pub fn act(context: u32, rows: u32, part_words: u64, x16: bool) Error!Act {
    if (context == 0 or rows == 0) return error.InvalidContext;
    var result: Act = undefined;
    var at: u64 = 0;
    inline for (widths) |entry| {
        @field(result, entry[0]) = @intCast(at);
        at += std.mem.alignForward(u64, @as(u64, entry[1]) * rows, 64);
    }
    result.scores = @intCast(at);
    at += std.mem.alignForward(u64, @as(u64, config.heads) * rows * context, 64);
    result.part = @intCast(at);
    at += std.mem.alignForward(u64, part_words, 64);
    const chunks = std.math.divCeil(u32, context, attn_chunk) catch unreachable;
    const per_head: u64 = @as(u64, config.heads) * chunks;
    if (at * 4 > 0xffffffe0) return error.ContextTooLarge;
    result.amax = @intCast(at);
    at += std.mem.alignForward(u64, per_head, 64);
    result.apart = @intCast(at);
    at += std.mem.alignForward(u64, per_head * config.head_dim, 64);
    result.asum = @intCast(at);
    at += std.mem.alignForward(u64, per_head, 64);
    result.chunks = chunks;
    result.x16 = null;
    if (x16) {
        result.x16 = @intCast(at);
        at += std.mem.alignForward(u64, @as(u64, config.ffn / 2) * rows, 64);
    }
    if (at * 4 > 0xffffffe0) return error.ContextTooLarge;
    result.rows = rows;
    result.words = at;
    return result;
}

/// Host io block (32-bit words): control, host-computed RoPE tables, logits, prefill
/// token ids and per-row RoPE rows (cos[32] then sin[32]).
pub const io = struct {
    pub const token = 0;
    pub const position = 1;
    pub const count = 2;
    pub const p0 = 3;
    pub const cos = 4;
    pub const sin = 36;
    pub const logits = 128;
    pub const tokens = logits + config.vocab;
    pub fn rope(rows: u32) u32 {
        return tokens + rows;
    }
    pub fn words(rows: u32) u64 {
        return @as(u64, rope(rows)) + @as(u64, rows) * 64;
    }
};
