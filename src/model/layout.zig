//! Driver-free placement of weights into banks and word layouts of the device arenas.
const std = @import("std");
const config = @import("config.zig");

pub const max_banks = 8;
pub const Error = error{ BankOverflow, ContextTooLarge, InvalidContext, InvalidKvPage, InvalidSlots };

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

/// Activation arena, FP32 word offsets. `scores` holds heads * decode_rows * context words.
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
    /// Per (row, head) global score maximum (block 17c): heads x decode_rows words.
    gmax: u32,
    /// Per (row, head, 8-chunk block) partial P·V (head_dim words) and exp-sum (block 17c).
    bpart: u32,
    bsum: u32,
    chunks: u32,
    /// ceil(chunks / 8).
    blocks: u32,
    /// f16 copy of one GEMM input at a time (f16 mode, block 16b): `rows` rows of up to
    /// `config.ffn` halves; null when not requested.
    x16: ?u32,
    /// KV page tables of the slots: `ptab_words` = `ptabWords(context)` u32 words each, slot s
    /// at `ptab + s * ptab_words` (docs/specs/concurrent.md).
    ptab: u32,
    ptab_words: u32,
    /// With more than one slot: the batched decode's own residual rows (`r` then `f`,
    /// `decode_rows` rows of `hidden` words each), so a decode step between the segments of
    /// a prefill chunk never writes the chunk's live `r`/`f` rows (docs/specs/concurrent.md,
    /// "18c.2 design"). Null with one slot.
    brf: ?u32,
    /// Speculative verification (block 17b, docs/specs/speculative.md): rows of the decode
    /// attention scratch (`amax/apart/asum`, 1 without speculation) and the per-linear-layer
    /// input slots (`spec`, null without speculation; see `specSlot`).
    decode_rows: u32,
    spec: ?u32,
    /// MTP draft regions (block 17b; null without the MTP layer).
    mtp: ?Mtp,
    rows: u32,
    words: u64,

    /// Linear layer `li`'s verify slot: lin_in outputs and conv outputs for `decode_rows`
    /// rows, kept until the commit pass (row strides as the decode kernels use them).
    /// The layout the batched decode records against: this one, with `r` and `f` in `brf`
    /// when it exists.
    pub fn batch(self: Act) Act {
        var b = self;
        if (self.brf) |base| {
            b.r = base;
            b.f = base + self.decode_rows * config.hidden;
        }
        return b;
    }

    pub fn specSlot(self: Act, li: u32) SpecSlot {
        const r: u32 = self.decode_rows;
        const base = self.spec.? + li * r * spec_slot_row_words;
        const cc = config.conv_channels;
        return .{ .mixed = base, .conv_out = base + r * cc, .z = base + 2 * r * cc, .alpha = base + 2 * r * cc + r * config.value_dim, .beta = base + 2 * r * cc + r * config.value_dim + r * config.v_heads };
    }
};
pub const SpecSlot = struct { mixed: u32, conv_out: u32, z: u32, alpha: u32, beta: u32 };
/// MTP regions (docs/specs/speculative.md, "MTP runtime design"), `decode_rows` = R:
/// `cat` rows x 2H (e then g per row; prefill catch-up uses every row), `mh` R x H (h
/// inputs of a pass), `mo` R x H (the MTP's h' outputs), `hrows` (rows + 1) x H (row 0:
/// the pending h, rows 1..: the final norm of a prefill chunk's rows), `logits`
/// vocab (draft head), `part` argmax partials (maxima, indices, exp sums; `argmax_groups`
/// each).
pub const Mtp = struct { cat: u32, mh: u32, mo: u32, hrows: u32, logits: u32, part: u32 };
/// Workgroups of the first argmax phase (each reduces a strided share of the vocabulary).
pub const argmax_groups = 256;
/// Words per row of one verify slot: mixed and conv_out (conv_channels each), z (value_dim),
/// alpha and beta_raw (v_heads each).
pub const spec_slot_row_words: u32 = 2 * config.conv_channels + config.value_dim + 2 * config.v_heads;

/// Keys per split-K decode attention chunk (fixed in the shaders).
pub const attn_chunk = 64;
/// Tokens per KV page (docs/specs/concurrent.md, "Addressing"; `--kv-page-tokens`), the
/// specialization constant `KV_PAGE` of the KV kernels. A multiple of `kv_page_quantum`, so
/// no scores chunk pair, P·V chunk or flash key tile straddles a page; 0 = one page holding
/// the whole context (the layout before paging).
pub const default_kv_page: u32 = 128;
pub const kv_page_quantum: u32 = 128;
pub const max_kv_page: u32 = 1 << 20;
/// Elements per token of one attention layer's K and V.
pub const kv_token_elements: u32 = 2 * config.kv_heads * config.head_dim;

/// The state arena (recurrent and conv state, the MTP's pending h) and the placement of
/// the attention KV caches in their own KV buffers (docs/specs/model.md, "KV buffers").
/// Element type of the attention KV caches (docs/specs/model.md, "KV precision").
pub const KvType = enum {
    f32,
    f16,
    pub fn bytes(self: KvType) u32 {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
        };
    }
};

pub const State = struct {
    context: u32,
    /// KV cache element type; every KV size and offset below is in its elements.
    kv: KvType,
    ssm: u32, // linear layers × v_heads × 128 × 128, zeroed by reset
    conv: u32, // linear layers × channels × 3, zeroed by reset (contiguous after ssm)
    /// MTP only: the pending h (hidden words) after conv, zeroed by reset and part of
    /// snapshots; null without the MTP layer.
    hp: ?u32,
    /// State arena words: `slots` slot states, slot s (ssm, conv, hp) at s * `slot_words`
    /// (docs/specs/concurrent.md, "18b.2 design"); slot 0 is the single-sequence layout.
    words: u64,
    slots: u32,
    slot_words: u32,
    /// KV caches (trunk attention layers, then the MTP's at `mtp_attention`), `per_buffer`
    /// consecutive caches per KV buffer, `kv_buffers` buffers (the last may hold fewer).
    caches: u32,
    per_buffer: u32,
    kv_buffers: u32,
    /// Physical KV pages of `page` tokens (docs/specs/concurrent.md, "Addressing"): each KV
    /// buffer holds, for every page, one `piece()` per attention layer it carries. `pages` is
    /// the pool shared by the slots; a sequence uses at most `seq_pages` of them, through its
    /// slot's page table (default: slot s owns pages s * seq_pages ..).
    page: u32,
    pages: u32,
    seq_pages: u32,

    pub const ssm_words: u32 = config.linear_layers * config.v_heads * config.linear_head * config.linear_head;
    pub const conv_words: u32 = config.linear_layers * config.conv_channels * (config.conv_kernel - 1);
    /// Elements of one attention layer's K and V for one page: K [kvh][dim][page], then V
    /// [kvh][page][dim].
    pub fn piece(self: State) u32 {
        return kv_token_elements * self.page;
    }
    /// Elements of one attention layer's cache over all pages.
    pub fn cacheElements(self: State) u64 {
        return @as(u64, self.piece()) * self.pages;
    }
    /// KV buffer holding attention cache `attention`.
    pub fn kvBuffer(self: State, attention: u32) u32 {
        return attention / self.per_buffer;
    }
    /// Attention layers carried by KV buffer `buffer`.
    pub fn layersIn(self: State, buffer: u32) u32 {
        return @min(self.per_buffer, self.caches - buffer * self.per_buffer);
    }
    /// Element offsets of cache `attention`'s K and V inside each page of its KV buffer.
    pub fn kcache(self: State, attention: u32) u32 {
        return (attention % self.per_buffer) * self.piece();
    }
    pub fn vcache(self: State, attention: u32) u32 {
        return self.kcache(attention) + config.kv_heads * config.head_dim * self.page;
    }
    /// Elements between consecutive pages in cache `attention`'s KV buffer.
    pub fn pstride(self: State, attention: u32) u32 {
        return self.layersIn(self.kvBuffer(attention)) * self.piece();
    }
    /// Elements of KV buffer `buffer`.
    pub fn kvElements(self: State, buffer: u32) u64 {
        return @as(u64, self.layersIn(buffer)) * self.cacheElements();
    }
    /// Bytes of KV buffer `buffer`, and of all KV buffers.
    pub fn kvBufferBytes(self: State, buffer: u32) u64 {
        return self.kvElements(buffer) * self.kv.bytes();
    }
    pub fn kvBytes(self: State) u64 {
        return @as(u64, self.caches) * self.cacheElements() * self.kv.bytes();
    }
    /// Canonicalize only unused last-page KV bytes in a completed archive window.
    /// `self` comes from stateWith; no conversion of valid elements (including f16 pairs).
    pub fn clearArchiveTail(self: State, snapshot_bytes: u64, tokens: u32, offset: u64, bytes: []u8) error{ InvalidToken, InvalidRange }!void {
        if (tokens == 0 or tokens > self.context) return error.InvalidToken;
        const npages = std.math.divCeil(u32, tokens, self.page) catch unreachable;
        const piece_bytes = @as(u64, self.piece()) * self.kv.bytes();
        const total = std.math.add(u64, snapshot_bytes, @as(u64, npages) * self.caches * piece_bytes) catch return error.InvalidRange;
        if (offset > total or bytes.len > total - offset) return error.InvalidRange;
        const tail = tokens % self.page;
        if (tail == 0 or bytes.len == 0) return;
        const element_bytes = self.kv.bytes();
        var group_base = snapshot_bytes;
        for (0..self.kv_buffers) |g| {
            const layers = self.layersIn(@intCast(g));
            const page_bytes = layers * piece_bytes;
            const last = group_base + (npages - 1) * page_bytes;
            group_base += npages * page_bytes;
            if (offset + bytes.len <= last or offset >= group_base) continue;
            for (0..layers) |layer| {
                const base = last + layer * piece_bytes;
                for (0..config.kv_heads * config.head_dim) |row| {
                    const start = base + (row * self.page + tail) * element_bytes;
                    clearIntersection(bytes, offset, start, @as(u64, self.page - tail) * element_bytes);
                }
                for (0..config.kv_heads) |head| {
                    const start = base + piece_bytes / 2 + (head * self.page + tail) * config.head_dim * element_bytes;
                    clearIntersection(bytes, offset, start, @as(u64, self.page - tail) * config.head_dim * element_bytes);
                }
            }
        }
    }
    fn clearIntersection(bytes: []u8, offset: u64, start: u64, size: u64) void {
        const lo = @max(offset, start);
        const hi = @min(offset + bytes.len, start + size);
        if (hi > lo) @memset(bytes[@intCast(lo - offset)..@intCast(hi - offset)], 0);
    }

    pub fn ssmLayer(self: State, linear: u32) u32 {
        return self.ssm + linear * config.v_heads * config.linear_head * config.linear_head;
    }
    pub fn convLayer(self: State, linear: u32) u32 {
        return self.conv + linear * config.conv_channels * (config.conv_kernel - 1);
    }
};

/// Attention cache index of the MTP layer (after the trunk's).
pub const mtp_attention = config.attention_layers;
/// KV buffers at most (one cache each).
pub const max_kv_buffers = config.attention_layers + 1;

/// The largest context whose attention cache (one layer's K and V, all pages) fits one KV
/// buffer of `capacity` bytes, with pages of `page_tokens` (0: one page of the context).
/// The activation arena and free VRAM bound it further.
pub fn maxContext(capacity: u64, kv: KvType, page_tokens: u32) u32 {
    const per_token = @as(u64, kv_token_elements) * kv.bytes();
    if (page_tokens == 0) return @intCast(@min(capacity / per_token, std.math.maxInt(u32)));
    const pages = capacity / (per_token * page_tokens);
    return @intCast(@min(pages * page_tokens, std.math.maxInt(u32) / page_tokens * page_tokens));
}

/// Words of the page-table region `Act.ptab` for `context`: one per page at the smallest
/// page size, so the activation arena does not depend on `--kv-page-tokens`.
pub fn ptabWords(context: u32) u32 {
    return std.math.divCeil(u32, context, kv_page_quantum) catch unreachable;
}

/// One sequence (slot 0, its identity page table): `stateWith(.., 1, 0)`.
pub fn state(context: u32, capacity: u64, mtp: bool, kv: KvType, page_tokens: u32) Error!State {
    return stateWith(context, capacity, mtp, kv, page_tokens, 1, 0);
}

/// `capacity`: the most bytes of one KV buffer (below 4 GiB: element offsets are
/// 32-bit). The context is even: the attention kernels read key pairs as one load.
/// `page_tokens`: a multiple of `kv_page_quantum` up to `max_kv_page`, or 0 for one page of
/// `context` tokens. `slots` >= 1 sequence states; `kv_pages` pool pages (0: `slots` x the
/// pages of `context`, at least the pages of one `context`).
pub fn stateWith(context: u32, capacity: u64, mtp: bool, kv: KvType, page_tokens: u32, slots: u32, kv_pages: u32) Error!State {
    std.debug.assert(capacity <= 0xffffffe0);
    if (context == 0 or context % 2 != 0) return error.InvalidContext;
    if (page_tokens % kv_page_quantum != 0 or page_tokens > max_kv_page) return error.InvalidKvPage;
    if (context > maxContext(capacity, kv, page_tokens)) return error.ContextTooLarge;
    if (slots == 0 or slots > max_slots) return error.InvalidSlots;
    const page = if (page_tokens == 0) context else page_tokens;
    const seq_pages = std.math.divCeil(u32, context, page) catch unreachable;
    const pages64: u64 = if (kv_pages == 0) @as(u64, slots) * seq_pages else kv_pages;
    if (pages64 < seq_pages) return error.InvalidSlots;
    const caches: u32 = config.attention_layers + @as(u32, @intFromBool(mtp));
    const cache_bytes = @as(u64, kv_token_elements) * page * pages64 * kv.bytes();
    if (cache_bytes > capacity) return error.ContextTooLarge;
    const pages: u32 = @intCast(pages64);
    const per_buffer: u32 = @intCast(@min(capacity / cache_bytes, caches));
    const hp: ?u32 = if (mtp) State.ssm_words + State.conv_words else null;
    const one = @as(u64, State.ssm_words) + State.conv_words + @as(u64, if (mtp) config.hidden else 0);
    const slot_words = std.mem.alignForward(u64, one, 64);
    // (The state arena's own bound, one storage buffer, is the caller's: `Model.init`.)
    const words = @as(u64, slots - 1) * slot_words + one;
    return .{ .context = context, .kv = kv, .ssm = 0, .conv = State.ssm_words, .hp = hp, .words = words, .slots = slots, .slot_words = @intCast(slot_words), .caches = caches, .per_buffer = per_buffer, .kv_buffers = (caches + per_buffer - 1) / per_buffer, .page = page, .pages = pages, .seq_pages = seq_pages };
}

/// Sequence slots at most (a bound for the host tables; the state arena bounds it too).
pub const max_slots = 64;

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
/// input-copy region last; `decode_rows` (1..rows) > 1 sizes the decode attention scratch
/// for speculative verification and adds the verify slots; `mtp` (needs decode_rows > 1)
/// adds the MTP regions before `x16`. `scores` holds decode (and verify) attention's
/// scores, heads x decode_rows x context words (prefill attention is fused).
pub fn act(context: u32, rows: u32, part_words: u64, x16: bool, decode_rows: u32, mtp: bool) Error!Act {
    return actWith(context, rows, part_words, x16, decode_rows, mtp, decode_rows > 1, 1);
}

/// `verify`: add the verify slots (needs decode_rows > 1); `slots`: page tables of that many
/// sequences (docs/specs/concurrent.md, "18b.2 design"). Otherwise as `act`; `decode_rows`
/// also sizes the batched decode's attention scratch.
pub fn actWith(context: u32, rows: u32, part_words: u64, x16: bool, decode_rows: u32, mtp: bool, verify: bool, slots: u32) Error!Act {
    if (context == 0 or rows == 0 or decode_rows == 0 or decode_rows > rows or (mtp and decode_rows < 2) or (verify and decode_rows < 2)) return error.InvalidContext;
    if (slots == 0 or slots > max_slots) return error.InvalidSlots;
    var result: Act = undefined;
    var at: u64 = 0;
    inline for (widths) |entry| {
        @field(result, entry[0]) = @intCast(at);
        at += std.mem.alignForward(u64, @as(u64, entry[1]) * rows, 64);
    }
    result.scores = @intCast(at);
    at += std.mem.alignForward(u64, @as(u64, config.heads) * decode_rows * context, 64);
    result.part = @intCast(at);
    at += std.mem.alignForward(u64, part_words, 64);
    const chunks = std.math.divCeil(u32, context, attn_chunk) catch unreachable;
    const per_head: u64 = @as(u64, config.heads) * chunks * decode_rows;
    if (at * 4 > 0xffffffe0) return error.ContextTooLarge;
    result.amax = @intCast(at);
    at += std.mem.alignForward(u64, per_head, 64);
    result.apart = @intCast(at);
    at += std.mem.alignForward(u64, per_head * config.head_dim, 64);
    result.asum = @intCast(at);
    at += std.mem.alignForward(u64, per_head, 64);
    result.gmax = @intCast(at);
    at += std.mem.alignForward(u64, @as(u64, config.heads) * decode_rows, 64);
    const blocks = std.math.divCeil(u32, chunks, 8) catch unreachable;
    const per_block: u64 = @as(u64, config.heads) * blocks * decode_rows;
    result.bpart = @intCast(at);
    at += std.mem.alignForward(u64, per_block * config.head_dim, 64);
    result.bsum = @intCast(at);
    at += std.mem.alignForward(u64, per_block, 64);
    result.blocks = blocks;
    result.chunks = chunks;
    result.decode_rows = decode_rows;
    result.spec = null;
    if (verify) {
        result.spec = @intCast(at);
        at += std.mem.alignForward(u64, @as(u64, config.linear_layers) * decode_rows * spec_slot_row_words, 64);
    }
    result.mtp = null;
    if (mtp) {
        var m: Mtp = undefined;
        const sizes = [_]u64{ @as(u64, rows) * 2 * config.hidden, @as(u64, decode_rows) * config.hidden, @as(u64, decode_rows) * config.hidden, (@as(u64, rows) + 1) * config.hidden, config.vocab, 3 * argmax_groups };
        const fields = [_]*u32{ &m.cat, &m.mh, &m.mo, &m.hrows, &m.logits, &m.part };
        for (fields, sizes) |field, size| {
            if (at * 4 > 0xffffffe0) return error.ContextTooLarge;
            field.* = @intCast(at);
            at += std.mem.alignForward(u64, size, 64);
        }
        result.mtp = m;
    }
    // The slots' page tables (u32 physical page per logical page), `ptabWords(context)` each.
    if (at * 4 > 0xffffffe0) return error.ContextTooLarge;
    result.ptab = @intCast(at);
    result.ptab_words = ptabWords(context);
    at += std.mem.alignForward(u64, @as(u64, ptabWords(context)) * slots, 64);
    result.brf = null;
    if (slots > 1) {
        result.brf = @intCast(at);
        at += std.mem.alignForward(u64, 2 * @as(u64, decode_rows) * config.hidden, 64);
    }
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
/// token ids and per-row RoPE rows (cos[32] then sin[32]), then the MTP words (block 17b):
/// positions, pass tokens, drafts, row-copy sources and RoPE rows of MTP rows.
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
    /// MTP rows per command at most: a pass of up to `spec_rows` rows, then up to
    /// `spec_drafts - 1` chained rows.
    pub const spec_rows = 5;
    pub const spec_drafts = spec_rows - 1;
    pub const spec_positions = spec_rows + spec_drafts;
    pub fn spec(rows: u32) Spec {
        const base = rope(rows) + rows * 64;
        const draft = base + spec_positions + spec_rows;
        return .{ .pos = base, .tok = base + spec_positions, .draft = draft, .prob = draft + spec_drafts, .src = draft + 2 * spec_drafts, .rope = base + 32 };
    }
    /// `pos[i]`: position of MTP row i; `tok[r]`: a pass's tokens; `draft[k]`: argmax of
    /// chain step k; `prob[k]`: its softmax probability (f32 bits; `prob = draft +
    /// spec_drafts`); `src[0..3]`: row-copy source words (bit 31: state arena); `rope`:
    /// RoPE rows of MTP row i at `rope + 64 i`.
    pub const Spec = struct { pos: u32, tok: u32, draft: u32, prob: u32, src: u32, rope: u32 };
    pub const spec_words = 32 + spec_positions * 64;
    /// Slot entries (docs/specs/concurrent.md, "18b.2 design"), `slot_entry` words each:
    /// position, page-table offset (words from `Act.ptab`), state offset (words), reserved.
    /// `slot`: the entry of single-slot commands (in the control words before the logits);
    /// `batch(rows)`: the batched decode's per-row entries, `batch_max` at most.
    pub const slot = 68;
    pub const slot_entry = 4;
    pub const batch_max = 32;
    pub fn batch(rows: u32) u32 {
        return rope(rows) + rows * 64 + spec_words;
    }
    /// Packed multi-sequence prefill (docs/specs/concurrent.md, "18d.1 design"): per-row
    /// entries (position, page-table offset, state offset, padding flag; `rows` of them) and
    /// the sequence table (`max_seqs` entries of `seq_words`: first row, rows, first position,
    /// page-table offset, state offset, word of the sequence's last final-norm row; rows 0:
    /// unused).
    pub const max_seqs = 8;
    pub const seq_words = 8;
    pub fn packRows(rows: u32) u32 {
        return batch(rows) + batch_max * slot_entry;
    }
    pub fn seqs(rows: u32) u32 {
        return packRows(rows) + rows * 4;
    }
    pub fn words(rows: u32) u64 {
        return @as(u64, seqs(rows)) + max_seqs * seq_words;
    }
};
