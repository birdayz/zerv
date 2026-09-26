//! Qwen3.8 (GGUF `qwen35`) hyperparameters and strict tensor inventory. Driver-free.
const std = @import("std");
const gguf = @import("artifact").gguf;
const matvec = @import("matvec");

pub const hidden = 5120;
pub const ffn = 17408;
pub const vocab = 248320;
pub const layers = 64;
pub const heads = 24;
pub const kv_heads = 4;
pub const head_dim = 256;
pub const rope_dims = 64;
pub const k_heads = 16;
pub const v_heads = 48;
pub const linear_head = 128;
pub const conv_kernel = 4;
pub const key_dim = k_heads * linear_head; // 2048
pub const value_dim = v_heads * linear_head; // 6144
pub const conv_channels = 2 * key_dim + value_dim; // 10240
pub const attention_layers = layers / 4;
pub const linear_layers = layers - attention_layers;

pub fn isAttention(layer: u32) bool {
    return layer % 4 == 3;
}
/// Index among attention (resp. linear) layers.
pub fn attentionIndex(layer: u32) u32 {
    return layer / 4;
}
pub fn linearIndex(layer: u32) u32 {
    return layer - layer / 4;
}

pub const Error = error{ UnsupportedModel, MissingTensor, WrongTensorShape, WrongTensorType } || gguf.ParseError;
/// `context_length`: the trained context (`qwen35.context_length`); longer contexts are
/// rejected.
pub const Hyper = struct { eps: f32, rope_base: f64, context_length: u32 };

const Expect = union(enum) { int: u64, string: []const u8 };
const required = [_]struct { []const u8, Expect }{
    .{ "general.architecture", .{ .string = "qwen35" } },
    .{ "qwen35.block_count", .{ .int = 65 } },
    .{ "qwen35.nextn_predict_layers", .{ .int = 1 } },
    .{ "qwen35.embedding_length", .{ .int = hidden } },
    .{ "qwen35.feed_forward_length", .{ .int = ffn } },
    .{ "qwen35.attention.head_count", .{ .int = heads } },
    .{ "qwen35.attention.head_count_kv", .{ .int = kv_heads } },
    .{ "qwen35.attention.key_length", .{ .int = head_dim } },
    .{ "qwen35.attention.value_length", .{ .int = head_dim } },
    .{ "qwen35.rope.dimension_count", .{ .int = rope_dims } },
    .{ "qwen35.full_attention_interval", .{ .int = 4 } },
    .{ "qwen35.ssm.conv_kernel", .{ .int = conv_kernel } },
    .{ "qwen35.ssm.state_size", .{ .int = linear_head } },
    .{ "qwen35.ssm.group_count", .{ .int = k_heads } },
    .{ "qwen35.ssm.time_step_rank", .{ .int = v_heads } },
    .{ "qwen35.ssm.inner_size", .{ .int = value_dim } },
};

fn integer(value: gguf.Value) gguf.ParseError!u64 {
    return switch (value.kind) {
        .uint32 => try value.scalar(u32),
        .int32 => std.math.cast(u64, try value.scalar(i32)) orelse error.WrongValueType,
        .uint64 => try value.scalar(u64),
        else => error.WrongValueType,
    };
}

/// Reject every artifact whose architecture or dimensions differ from the verified model.
pub fn hyper(container: *const gguf.Container) Error!Hyper {
    for (required) |entry| {
        const value = container.findMetadata(entry[0]) orelse return error.UnsupportedModel;
        switch (entry[1]) {
            .int => |expected| if ((integer(value) catch return error.UnsupportedModel) != expected) return error.UnsupportedModel,
            .string => |expected| if (!std.mem.eql(u8, value.string() catch return error.UnsupportedModel, expected)) return error.UnsupportedModel,
        }
    }
    const eps = (container.findMetadata("qwen35.attention.layer_norm_rms_epsilon") orelse return error.UnsupportedModel).scalar(f32) catch return error.UnsupportedModel;
    const base = (container.findMetadata("qwen35.rope.freq_base") orelse return error.UnsupportedModel).scalar(f32) catch return error.UnsupportedModel;
    // Text-only positions: all MRoPE sections share one position (research note).
    const sections = (container.findMetadata("qwen35.rope.dimension_sections") orelse return error.UnsupportedModel).array() catch return error.UnsupportedModel;
    const expected_sections = [4]i32{ 11, 11, 10, 0 };
    if (sections.count != 4) return error.UnsupportedModel;
    var it = sections.iterator();
    for (expected_sections) |want| {
        const got = (it.next() catch return error.UnsupportedModel) orelse return error.UnsupportedModel;
        if ((got.scalar(i32) catch return error.UnsupportedModel) != want) return error.UnsupportedModel;
    }
    if (!(std.math.isFinite(eps) and eps > 0 and eps < 1e-3 and std.math.isFinite(base) and base > 1)) return error.UnsupportedModel;
    const trained = integer((container.findMetadata("qwen35.context_length") orelse return error.UnsupportedModel)) catch return error.UnsupportedModel;
    if (trained == 0 or trained > std.math.maxInt(u32)) return error.UnsupportedModel;
    return .{ .eps = eps, .rope_base = base, .context_length = @intCast(trained) };
}

/// How the runtime consumes a tensor. Params are FP32 vectors read by operator kernels.
pub const Role = enum { param, embedding, matrix };
/// `rows_only`: a matrix used only through multi-row projections (may be Q8_0).
pub const Spec = struct { name: [64]u8 = undefined, len: u8 = 0, role: Role, k: u64, rows: u64, rows_only: bool = false };

fn spec(role: Role, k: u64, rows: u64, comptime fmt: []const u8, args: anytype) Spec {
    var s: Spec = .{ .role = role, .k = k, .rows = rows };
    s.len = @intCast((std.fmt.bufPrint(&s.name, fmt, args) catch unreachable).len);
    return s;
}
pub fn specName(s: *const Spec) []const u8 {
    return s.name[0..s.len];
}

pub const tensor_count = 3 + attention_layers * 11 + linear_layers * 14;

/// Complete trunk inventory in a fixed order. Dimensions are GGUF [k (ne0), rows (ne1)].
pub fn tensors() [tensor_count]Spec {
    var out: [tensor_count]Spec = undefined;
    var n: usize = 0;
    out[n] = spec(.embedding, hidden, vocab, "token_embd.weight", .{});
    n += 1;
    out[n] = spec(.param, hidden, 1, "output_norm.weight", .{});
    n += 1;
    out[n] = spec(.matrix, hidden, vocab, "output.weight", .{});
    n += 1;
    for (0..layers) |il| {
        const add = struct {
            fn f(o: *[tensor_count]Spec, i: *usize, s: Spec) void {
                o[i.*] = s;
                i.* += 1;
            }
        }.f;
        add(&out, &n, spec(.param, hidden, 1, "blk.{d}.attn_norm.weight", .{il}));
        add(&out, &n, spec(.param, hidden, 1, "blk.{d}.post_attention_norm.weight", .{il}));
        add(&out, &n, spec(.matrix, hidden, ffn, "blk.{d}.ffn_gate.weight", .{il}));
        add(&out, &n, spec(.matrix, hidden, ffn, "blk.{d}.ffn_up.weight", .{il}));
        add(&out, &n, spec(.matrix, ffn, hidden, "blk.{d}.ffn_down.weight", .{il}));
        if (isAttention(@intCast(il))) {
            add(&out, &n, spec(.matrix, hidden, 2 * heads * head_dim, "blk.{d}.attn_q.weight", .{il}));
            add(&out, &n, spec(.matrix, hidden, kv_heads * head_dim, "blk.{d}.attn_k.weight", .{il}));
            add(&out, &n, spec(.matrix, hidden, kv_heads * head_dim, "blk.{d}.attn_v.weight", .{il}));
            add(&out, &n, spec(.param, head_dim, 1, "blk.{d}.attn_q_norm.weight", .{il}));
            add(&out, &n, spec(.param, head_dim, 1, "blk.{d}.attn_k_norm.weight", .{il}));
            add(&out, &n, spec(.matrix, heads * head_dim, hidden, "blk.{d}.attn_output.weight", .{il}));
        } else {
            add(&out, &n, spec(.matrix, hidden, conv_channels, "blk.{d}.attn_qkv.weight", .{il}));
            add(&out, &n, spec(.matrix, hidden, value_dim, "blk.{d}.attn_gate.weight", .{il}));
            add(&out, &n, spec(.matrix, hidden, v_heads, "blk.{d}.ssm_alpha.weight", .{il}));
            add(&out, &n, spec(.matrix, hidden, v_heads, "blk.{d}.ssm_beta.weight", .{il}));
            add(&out, &n, spec(.param, v_heads, 1, "blk.{d}.ssm_a", .{il}));
            add(&out, &n, spec(.param, v_heads, 1, "blk.{d}.ssm_dt.bias", .{il}));
            add(&out, &n, spec(.param, conv_kernel, conv_channels, "blk.{d}.ssm_conv1d.weight", .{il}));
            add(&out, &n, spec(.param, linear_head, 1, "blk.{d}.ssm_norm.weight", .{il}));
            add(&out, &n, spec(.matrix, value_dim, hidden, "blk.{d}.ssm_out.weight", .{il}));
        }
    }
    std.debug.assert(n == tensor_count);
    return out;
}

/// The MTP (nextn) layer's block index and inventory (block 17b, loaded only for
/// speculative decoding; docs/specs/speculative.md).
pub const mtp_layer = layers;
pub const mtp_tensor_count = 15;
pub fn mtpTensors() [mtp_tensor_count]Spec {
    const il = mtp_layer;
    return .{
        spec(.param, hidden, 1, "blk.{d}.attn_norm.weight", .{il}),
        spec(.param, hidden, 1, "blk.{d}.post_attention_norm.weight", .{il}),
        spec(.matrix, hidden, ffn, "blk.{d}.ffn_gate.weight", .{il}),
        spec(.matrix, hidden, ffn, "blk.{d}.ffn_up.weight", .{il}),
        spec(.matrix, ffn, hidden, "blk.{d}.ffn_down.weight", .{il}),
        spec(.matrix, hidden, 2 * heads * head_dim, "blk.{d}.attn_q.weight", .{il}),
        spec(.matrix, hidden, kv_heads * head_dim, "blk.{d}.attn_k.weight", .{il}),
        spec(.matrix, hidden, kv_heads * head_dim, "blk.{d}.attn_v.weight", .{il}),
        spec(.param, head_dim, 1, "blk.{d}.attn_q_norm.weight", .{il}),
        spec(.param, head_dim, 1, "blk.{d}.attn_k_norm.weight", .{il}),
        spec(.matrix, heads * head_dim, hidden, "blk.{d}.attn_output.weight", .{il}),
        rowsOnly(spec(.matrix, 2 * hidden, hidden, "blk.{d}.nextn.eh_proj.weight", .{il})),
        spec(.param, hidden, 1, "blk.{d}.nextn.enorm.weight", .{il}),
        spec(.param, hidden, 1, "blk.{d}.nextn.hnorm.weight", .{il}),
        spec(.param, hidden, 1, "blk.{d}.nextn.shared_head_norm.weight", .{il}),
    };
}

fn rowsOnly(s: Spec) Spec {
    var r = s;
    r.rows_only = true;
    return r;
}

/// Map a GGUF tensor type to a projection format, rejecting types the runtime lacks.
/// Q8_0 has only the multi-row module (the MTP's eh_proj).
pub fn matrixFormat(kind: gguf.TensorType) Error!matvec.Format {
    return switch (kind) {
        .f32 => .f32,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_0 => .q8_0,
        else => error.WrongTensorType,
    };
}

/// Check one tensor against its expected role/shape (rank 1 tensors have rows 1).
pub fn check(s: *const Spec, tensor: *const gguf.Tensor) Error!void {
    const rows_dim: u64 = if (tensor.rank >= 2) tensor.dims[1] else 1;
    if (tensor.dims[0] != s.k or rows_dim != s.rows or tensor.dims[2] != 1 or tensor.dims[3] != 1) return error.WrongTensorShape;
    switch (s.role) {
        .param => if (tensor.kind != .f32) return error.WrongTensorType,
        .embedding => if (tensor.kind != .q4_0) return error.WrongTensorType,
        .matrix => if (try matrixFormat(tensor.kind) == .q8_0 and !s.rows_only) return error.WrongTensorType,
    }
}
