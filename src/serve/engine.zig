//! Native Qwen3.8 engine: official template -> tokenizer -> resident model -> session.
const std = @import("std");
const api = @import("api.zig");
const http = @import("http.zig");
const chat = @import("chat").qwen38;
const bpe = @import("tokenizer").bpe;
const model = @import("model");
const session = @import("session");
const batcher = @import("batcher.zig");

pub const eos = [_][]const u8{ "<|im_end|>", "<|endoftext|>" };

pub const Native = struct {
    io: std.Io,
    model: *model.Model,
    tokenizer: *const bpe.Tokenizer,
    eos_ids: [eos.len]u32,
    tool_call_id: u32,
    /// Tokens whose output text is only spaces, tabs and newlines (between tool calls).
    whitespace_ids: []const u32,
    /// Present when the model has snapshot slots (docs/specs/prefix-cache.md).
    cache: ?session.prefix.Cache = null,
    seed_counter: std.atomic.Value(u64) = .init(0),
    /// Speculative decoding: the adaptive verify-count policy (learns across requests);
    /// null = verify every draft (`--spec-policy fixed`).
    spec_policy: ?session.spec.Policy = null,
    /// Sampler summation/draw order for every request (`--sampler-order`).
    sampler_order: session.sampler.Order = .id,
    /// Concurrent sequences (`--parallel N` > 1, docs/specs/concurrent.md "18c design"): the
    /// batcher owning the model; generations run through it. Null: one generation at a time.
    batch: ?*Batcher = null,
    batch_backend: ?*ModelBackend = null,
    /// Shared pool admission (docs/specs/concurrent.md, "18d.3 design"): `.reserve` admits
    /// prompt + output limit; `.prompt` admits the prompt and grows page by page (swapping
    /// sequences to host memory under pressure).
    admission: Admission = .reserve,

    /// Resolve and verify profile token ids against the loaded tokenizer.
    /// `allocator` owns `whitespace_ids` until `deinit`.
    pub fn init(io: std.Io, allocator: std.mem.Allocator, m: *model.Model, tokenizer: *const bpe.Tokenizer) !Native {
        var result: Native = .{ .io = io, .model = m, .tokenizer = tokenizer, .eos_ids = undefined, .tool_call_id = undefined, .whitespace_ids = &.{} };
        for (eos, &result.eos_ids) |piece, *id| id.* = try single(allocator, tokenizer, piece);
        // The output parsers find these delimiters in rendered text, so they must render.
        for ([_][]const u8{ "</think>", "<tool_call>", "</tool_call>" }) |delimiter| {
            const id = try single(allocator, tokenizer, delimiter);
            if (!std.mem.eql(u8, try tokenizer.outputPiece(id), delimiter)) return error.UnsupportedTokenizer;
            if (delimiter[1] == 't') result.tool_call_id = id;
        }
        var whitespace: std.ArrayList(u32) = .empty;
        errdefer whitespace.deinit(allocator);
        for (0..model.config.vocab) |i| {
            const id: u32 = @intCast(i);
            const piece = tokenizer.outputPiece(id) catch continue;
            if (piece.len == 0 or piece.len > 23) continue;
            for (piece) |c| {
                if (c != ' ' and c != '\t' and c != '\n') break;
            } else try whitespace.append(allocator, id);
        }
        result.whitespace_ids = try whitespace.toOwnedSlice(allocator);
        errdefer allocator.free(result.whitespace_ids);
        if (m.snapshot_slots > 0) {
            const boundary = try single(allocator, tokenizer, "<|im_start|>");
            result.cache = try session.prefix.Cache.init(allocator, m.state_layout.context, .{ .slots = m.snapshot_slots, .boundary = boundary });
        }
        return result;
    }

    pub fn deinit(self: *Native, allocator: std.mem.Allocator) void {
        allocator.free(self.whitespace_ids);
        if (self.cache) |*c| c.deinit(allocator);
    }

    fn single(allocator: std.mem.Allocator, tokenizer: *const bpe.Tokenizer, piece: []const u8) !u32 {
        const ids = try tokenizer.encode(allocator, piece, .{});
        defer allocator.free(ids);
        if (ids.len != 1 or !std.mem.eql(u8, try tokenizer.piece(ids[0]), piece)) return error.UnsupportedTokenizer;
        return ids[0];
    }

    /// Serve through `b` (--parallel > 1). The prefix cache and the speculation policy
    /// describe the single model sequence; with several sequences they would mix users'
    /// state, so they must be off (`error.SharedStateWithBatching`).
    pub fn attachBatcher(self: *Native, b: *Batcher, mb: *ModelBackend) error{SharedStateWithBatching}!void {
        if (self.cache != null or self.spec_policy != null) return error.SharedStateWithBatching;
        self.batch = b;
        self.batch_backend = mb;
    }

    pub fn engine(self: *Native, ids: []const []const u8, defaults: api.Defaults) http.Engine {
        return .{ .context = self, .prepareFn = prepare, .generateFn = generate, .usableFn = usable, .ids = ids, .defaults = defaults };
    }

    /// The model can run again: the device is not lost and no command is left pending
    /// (a timed-out command can be neither waited for again nor resubmitted).
    fn usable(ctx: *anyopaque) bool {
        const self: *Native = @ptrCast(@alignCast(ctx));
        const device = self.model.device;
        // With batching the scheduler task owns the device (another sequence's command may be
        // pending right now); it publishes a failure it cannot recover from in `fatal`, the
        // only thing read here.
        if (self.batch_backend) |mb| return !mb.fatal.load(.acquire);
        return !device.lost and device.pending == 0;
    }

    fn prepare(ctx: *anyopaque, arena: std.mem.Allocator, request: *const api.ChatRequest, rejection: *api.ApiError) anyerror!http.Prepared {
        const self: *Native = @ptrCast(@alignCast(ctx));
        var prompt: std.Io.Writer.Allocating = .init(arena);
        chat.render(request.messages, request.template, &prompt.writer) catch |e| {
            rejection.* = .{ .message = switch (e) {
                error.EmptyMessages => "No messages provided.",
                error.MissingUserQuery => "No user query found in messages.",
                error.MisplacedSystem => "System message must be at the beginning.",
                error.UnsupportedRole => "Unexpected message role.",
                error.InvalidUtf8 => "message content is not valid UTF-8",
                error.LimitExceeded => "messages exceed the template limits",
                else => return e,
            }, .param = "messages" };
            return error.Rejected;
        };
        const text = prompt.written();
        const tokens = self.tokenizer.encode(arena, text, .{}) catch |e| {
            rejection.* = .{ .message = "prompt could not be tokenized", .param = "messages" };
            return e;
        };
        if (tokens.len >= self.model.state_layout.context) {
            rejection.* = .{ .message = "prompt exceeds the context length", .param = "messages", .code = "context_length_exceeded" };
            return error.Rejected;
        }
        return .{ .tokens = tokens, .thinking = std.mem.endsWith(u8, text, "<think>\n") };
    }

    const Backend = struct {
        m: *model.Model,
        pub fn context(b: Backend) u32 {
            return b.m.state_layout.context;
        }
        pub fn vocab(_: Backend) usize {
            return model.config.vocab;
        }
        pub fn reset(b: Backend) !void {
            try b.m.reset();
        }
        pub fn step(b: Backend, token: u32) ![]const f32 {
            return b.m.step(token);
        }
        pub fn saveSnapshot(b: Backend, slot: u32) !void {
            try b.m.saveSnapshot(slot);
        }
        pub fn loadSnapshot(b: Backend, slot: u32, position: u32) !void {
            try b.m.loadSnapshot(slot, position);
        }
        /// Speculative decoding (docs/specs/speculative.md): drafts per step, 0 = off.
        pub fn speculative(b: Backend) u32 {
            return if (b.m.options.mtp) b.m.options.verify_rows - 1 else 0;
        }
        pub fn draft(b: Backend, token: u32, k: u32) ![]const u32 {
            return b.m.draft(token, k);
        }
        pub fn draftProbs(b: Backend, k: u32) []const f32 {
            return b.m.draftProbs(k);
        }
        pub fn verify(b: Backend, tokens: []const u32) ![]const f32 {
            return b.m.verify(tokens);
        }
        pub fn commit(b: Backend, m: u32) !void {
            try b.m.commit(m);
        }
        pub fn prefill(b: Backend, tokens: []const u32) ![]const f32 {
            if (b.m.rows == 0) {
                var logits: []const f32 = undefined;
                for (tokens) |token| logits = try b.m.step(token);
                return logits;
            }
            return b.m.prefill(tokens);
        }
    };
    const Generation = session.Generation(Backend, *const bpe.Tokenizer, http.Sink);

    /// A generation's view of the batched model: its slot's operations through the batcher.
    const SlotBackend = struct {
        b: *Batcher,
        slot: u32,
        m: *model.Model,
        pub fn context(s: SlotBackend) u32 {
            return s.m.state_layout.context;
        }
        pub fn vocab(_: SlotBackend) usize {
            return model.config.vocab;
        }
        pub fn reset(s: SlotBackend) !void {
            try s.b.reset(s.slot);
        }
        pub fn prefill(s: SlotBackend, tokens: []const u32) ![]const f32 {
            return s.b.prefill(s.slot, tokens);
        }
        pub fn step(s: SlotBackend, token: u32) ![]const f32 {
            return s.b.step(s.slot, token);
        }
        pub fn sampled(s: SlotBackend) void {
            s.b.sampled(s.slot);
        }
    };
    const SlotGeneration = session.Generation(SlotBackend, *const bpe.Tokenizer, http.Sink);

    fn ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    fn generate(ctx: *anyopaque, arena: std.mem.Allocator, request: *const api.ChatRequest, prepared: http.Prepared, sink: http.Sink) anyerror!http.Completion {
        const self: *Native = @ptrCast(@alignCast(ctx));
        var params = request.params;
        params.order = self.sampler_order;
        if (!request.seed_given) {
            const now: u64 = @bitCast(@as(i64, @truncate(std.Io.Clock.real.now(self.io).nanoseconds)));
            params.seed = now ^ (self.seed_counter.fetchAdd(1, .monotonic) *% 0x9e3779b97f4a7c15);
        }
        const special: session.Special = .{ .eos = &self.eos_ids, .whitespace = self.whitespace_ids, .tool_call = self.tool_call_id };
        const r: session.Request = .{
            .prompt = prepared.tokens,
            .max_tokens = request.max_tokens orelse self.model.state_layout.context,
            .params = params,
            .stops = request.stops,
            .thinking = prepared.thinking,
            .tools = request.toolMode(),
            .cache = if (self.cache) |*c| c else null,
            .spec_policy = if (self.spec_policy) |*p| p else null,
        };
        const result = if (self.batch) |b| slot: {
            // `attachBatcher` guarantees no shared prefix cache or speculation policy here:
            // they track one model sequence and must never see another user's tokens.
            var sr = r;
            sr.cache = null;
            sr.spec_policy = null;
            const slot = try b.join();
            defer b.leave(slot);
            // Shared KV pool (docs/specs/concurrent.md, "18d.2 design"): memory for the prompt
            // plus the output limit (the generation's own `limit`), admitted before it starts.
            const context = self.model.state_layout.context;
            if (prepared.tokens.len < context and self.admission == .reserve) b.reserve(slot, prepared.tokens.len + @min(r.max_tokens, context - @as(u32, @intCast(prepared.tokens.len))));
            break :slot try SlotGeneration.run(self.io, arena, .{ .b = b, .slot = slot, .m = self.model }, self.tokenizer, special, sr, sink);
        } else try Generation.run(self.io, arena, .{ .m = self.model }, self.tokenizer, special, r, sink);
        if (self.cache != null) std.log.info("prefix cache: {s}, reused {d} of {d} prompt tokens", .{ @tagName(result.cache_outcome), result.cached_tokens, result.prompt_tokens });
        if (result.spec.verifies > 0) std.log.info("speculative: {d} of {d} verified drafts accepted ({d} drafted, {d} verifies, {d} tokens)", .{ result.spec.accepted, result.spec.verified, result.spec.drafted, result.spec.verifies, result.completion_tokens });
        std.log.info("decode time: {d:.3} ms total, {d:.3} ms in backend calls, {d:.3} ms sampling, {d:.3} ms other host work", .{ ms(result.decode_ns), ms(result.backend_ns), ms(result.sample_ns), ms(result.decode_ns -| result.backend_ns -| result.sample_ns) });
        return .{ .finish = result.finish, .prompt_tokens = result.prompt_tokens, .completion_tokens = result.completion_tokens, .prefill_ns = result.prefill_ns, .decode_ns = result.decode_ns, .tool_calls = result.tool_calls, .tool_call_failed = result.tool_call_failed, .cached_tokens = result.cached_tokens, .cache_outcome = result.cache_outcome, .spec = result.spec };
    }
};

pub const Batcher = batcher.Batcher(*ModelBackend);
pub const Admission = enum { reserve, prompt };

/// The resident model as the batcher's backend (docs/specs/concurrent.md, "18c design"):
/// called by the scheduler task only. A failed call that leaves a command pending (a
/// timeout) or loses the device sets `fatal`.
pub const ModelBackend = struct {
    m: *model.Model,
    rows: [batcher.max_slots]model.BatchRow = undefined,
    fatal: std.atomic.Value(bool) = .init(false),

    pub fn checkRow(self: *ModelBackend, row: batcher.Row) !void {
        try self.m.checkRow(.{ .slot = row.slot, .token = row.token });
    }
    fn check(self: *ModelBackend, e: anyerror) anyerror {
        self.noteFailure();
        return e;
    }
    /// After a failed call: a pending command or a lost device makes the engine unusable.
    fn noteFailure(self: *ModelBackend) void {
        if (self.m.device.lost or self.m.device.pending != 0) self.fatal.store(true, .release);
    }
    pub fn reset(self: *ModelBackend, slot: u32) !void {
        self.m.select(slot) catch |e| return self.check(e);
        self.m.reset() catch |e| return self.check(e);
        // Shared pool: a new sequence starts without memory (`admit` maps it).
        if (self.m.options.kv_share) self.m.releasePages(slot) catch |e| return self.check(e);
    }
    /// Shared pool: map pages for positions below `tokens` (false: too few free pages now;
    /// a static pool always holds them).
    pub fn admit(self: *ModelBackend, slot: u32, tokens: usize) bool {
        if (!self.m.options.kv_share) return true;
        const want: u32 = @intCast(@min(tokens, self.m.state_layout.context));
        self.m.ensurePages(slot, want) catch |e| {
            if (e != error.PoolExhausted) self.noteFailure();
            return false;
        };
        return true;
    }
    pub fn release(self: *ModelBackend, slot: u32) void {
        if (!self.m.options.kv_share) return;
        self.m.releasePages(slot) catch self.noteFailure();
    }
    /// Shared pool: a page for the slot's next decode position (false: none free).
    pub fn grow(self: *ModelBackend, slot: u32) !bool {
        if (!self.m.options.kv_share) return true;
        self.m.ensurePages(slot, self.m.slotPosition(slot) + 1) catch |e| {
            if (e == error.PoolExhausted) return false;
            return self.check(e);
        };
        return true;
    }
    pub fn swapOut(self: *ModelBackend, slot: u32) !bool {
        if (self.m.swap_pages == 0) return false;
        self.m.swapOut(slot) catch |e| {
            if (e == error.SwapFull) return false;
            return self.check(e);
        };
        return true;
    }
    pub fn swapIn(self: *ModelBackend, slot: u32, spare: u32) !bool {
        return self.m.swapIn(slot, spare) catch |e| return self.check(e);
    }
    /// Whether these prompts' next chunks fit one packed chunk (docs/specs/concurrent.md,
    /// "18d.1 design"): always for one; several need the model's packed commands.
    pub fn packFits(self: *ModelBackend, remaining: []const usize) bool {
        if (remaining.len <= 1) return true;
        if (!self.m.packable() or remaining.len > self.m.pack_seqs) return false;
        var left: [batcher.max_pack]u32 = undefined;
        for (remaining, left[0..remaining.len]) |r, *l| l.* = @intCast(@min(r, std.math.maxInt(u32)));
        return self.m.packSpan(left[0..remaining.len]) <= self.m.rows;
    }
    /// One segment of a (packed) chunk when the model records them (several slots; docs/specs/
    /// concurrent.md "18c.2 design"), else a whole chunk of the one item.
    pub fn prefillUnit(self: *ModelBackend, items: []const batcher.Item) !batcher.Unit {
        if (self.m.live_seg > 0) {
            var pack: [batcher.max_pack]model.PackItem = undefined;
            for (items, pack[0..items.len]) |item, *pi| pi.* = .{ .slot = item.slot, .tokens = item.tokens };
            const seg = self.m.prefillPackedSegment(pack[0..items.len]) catch |e| return self.check(e);
            if (!seg.done) return .{ .done = false };
            var unit: batcher.Unit = .{ .done = true, .logits = seg.logits };
            for (seg.consumed[0..batcher.max_pack], &unit.consumed) |c, *u| u.* = c;
            return unit;
        }
        if (items.len != 1) return error.InvalidBatch;
        self.m.select(items[0].slot) catch |e| return self.check(e);
        const tokens = items[0].tokens;
        const c = self.m.nextChunk(@intCast(tokens.len));
        const logits = self.m.runChunk(&self.m.prefill_commands[c.plan], c.plan, tokens[0..c.rows]) catch |e| return self.check(e);
        var unit: batcher.Unit = .{ .done = true, .logits = logits };
        unit.consumed[0] = c.rows;
        return unit;
    }
    pub fn abortChunk(self: *ModelBackend) void {
        self.m.abortChunk();
    }
    pub fn decodeBatch(self: *ModelBackend, rows: []const batcher.Row) ![]const f32 {
        for (rows, self.rows[0..rows.len]) |r, *m| m.* = .{ .slot = r.slot, .token = r.token };
        return self.m.decodeBatch(self.rows[0..rows.len]) catch |e| return self.check(e);
    }
};
