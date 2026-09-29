//! Native Qwen3.8 engine: official template -> tokenizer -> resident model -> session.
const std = @import("std");
const api = @import("api.zig");
const http = @import("http.zig");
const chat = @import("chat").qwen38;
const bpe = @import("tokenizer").bpe;
const model = @import("model");
const session = @import("session");
const batcher = @import("batcher.zig");
const disk = @import("disk.zig");
const preparation = @import("preparation.zig");
const DemandKey = session.readahead.Key;
pub const DemandMode = enum { off, protect, prefetch };

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
    /// `<|im_start|>`: where prefix checkpoints go (single-slot cache and shared pool).
    boundary: u32 = 0,

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
        result.boundary = try single(allocator, tokenizer, "<|im_start|>");
        // One sequence: the single-history prefix cache. Several share the snapshots through
        // the batcher backend's checkpoint store instead (`ModelBackend.initStore`).
        if (m.snapshot_slots > 0 and m.state_layout.slots == 1) {
            result.cache = try session.prefix.Cache.init(allocator, m.state_layout.context, .{ .slots = m.snapshot_slots, .boundary = result.boundary });
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
        pub fn begin(s: SlotBackend, prompt: []const u32) !u32 {
            return s.b.begin(s.slot, prompt);
        }
        pub fn checkpoint(s: SlotBackend, prefix: []const u32) !void {
            try s.b.checkpoint(s.slot, prefix);
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
            if (self.batch_backend) |mb| if (mb.cache != null or mb.disk_archive != null) {
                sr.checkpoints = self.boundary;
            };
            const slot = try b.join();
            defer b.leave(slot);
            // Shared KV pool (docs/specs/concurrent.md, "18d.2 design"): memory for the prompt
            // plus the output limit (the generation's own `limit`), admitted before it starts.
            const context = self.model.state_layout.context;
            if (prepared.tokens.len < context and self.admission == .reserve) b.reserve(slot, prepared.tokens.len + @min(r.max_tokens, context - @as(u32, @intCast(prepared.tokens.len))));
            break :slot try SlotGeneration.run(self.io, arena, .{ .b = b, .slot = slot, .m = self.model }, self.tokenizer, special, sr, sink);
        } else try Generation.run(self.io, arena, .{ .m = self.model }, self.tokenizer, special, r, sink);
        if (self.cache != null or (self.batch_backend != null and self.batch_backend.?.cache != null)) std.log.info("prefix cache: {s}, reused {d} of {d} prompt tokens", .{ @tagName(result.cache_outcome), result.cached_tokens, result.prompt_tokens });
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
    /// Prefix cache of the shared pool (docs/specs/concurrent.md, "18d.4 design"): a policy
    /// behind `session.kvcache.Cache`; this backend is its `Device`. Scheduler thread only.
    cache: ?session.kvcache.Cache = null,
    disk_archive: ?*disk.Disk = null,
    pressure_reclaim: bool = false,
    preparation_owner: ?preparation.Preparation = null,
    demand_mode: DemandMode = .off,
    prefetch_chunks: u32 = 1,
    demand_key: ?DemandKey = null,
    suppressed_prefetch: ?DemandKey = null,
    demand_overrides: u64 = 0,
    demand_waits: u64 = 0,

    fn sameDemand(a: ?DemandKey, b: ?DemandKey) bool {
        return if (a) |x| if (b) |y| x.eql(y) else false else b == null;
    }
    /// Tokens are borrowed only for this callback. Longer ownership uses key + record pin.
    pub fn pollDemand(self: *ModelBackend, requested: ?batcher.Demand, stopping: bool, reads_pending: bool) !batcher.MaintenancePoll {
        if (self.demand_mode == .off) return .{};
        const demand = if (stopping) null else requested;
        const key: ?DemandKey = if (demand) |q| .{ .slot = q.slot, .order = q.order } else null;
        self.demand_key = key;
        const hot = if (self.cache) |c| c.setDemand(if (demand) |q| q.tokens else &.{}) else null;
        if (hot) |h| if (h.has_host and h.source_held) {
            if (self.disk_archive) |d| d.source_cancel_requested = true;
            if (self.preparation_owner) |*p| if (p.held != null) {
                p.canceled = true;
            };
        };
        const d = self.disk_archive orelse return .{};
        if (d.read_ahead.held) |held| {
            if (reads_pending or (hot != null and hot.?.tokens >= d.catalog.entries[held.record].len)) d.read_ahead.cancel();
            const p = d.read_ahead.poll(&d.catalog, key, !stopping and !reads_pending) catch |e| {
                self.suppressed_prefetch = held.key;
                self.noteFailure();
                return e;
            };
            return .{ .pending = !p.done, .progressed = p.progressed, .reclaimed = p.done };
        }
        if (self.demand_mode != .prefetch or stopping or reads_pending or key == null or sameDemand(key, self.suppressed_prefetch)) return .{};
        const record = d.catalog.lookup(demand.?.tokens) orelse return .{};
        if (hot != null and hot.?.tokens >= d.catalog.entries[record].len) return .{};
        try d.read_ahead.start(&d.catalog, key.?, record, self.prefetch_chunks);
        const p = try d.read_ahead.poll(&d.catalog, key, true);
        return .{ .pending = true, .progressed = p.progressed };
    }

    pub fn initPreparation(self: *ModelBackend, io: std.Io, options: preparation.Options) !void {
        if (self.preparation_owner != null) return error.InvalidOptions;
        self.preparation_owner = try preparation.Preparation.init(self.m, self.cache orelse return error.InvalidOptions, io, options);
    }
    pub fn deinitPreparation(self: *ModelBackend) !void {
        if (self.preparation_owner) |*p| try p.deinit();
        self.preparation_owner = null;
    }

    pub fn initDisk(self: *ModelBackend, allocator: std.mem.Allocator, io: std.Io, options: disk.Options) !void {
        if (self.disk_archive != null) return error.InvalidOptions;
        if (self.cache) |c| _ = try c.sourceCapacity();
        self.disk_archive = try disk.Disk.create(allocator, io, self.m, options);
    }
    pub fn deinitDisk(self: *ModelBackend) void {
        if (self.disk_archive) |d| d.destroy();
        self.disk_archive = null;
    }
    pub fn pollCache(self: *ModelBackend, slot: u32, cancel: bool) !batcher.CachePoll {
        const d = self.disk_archive orelse return error.InvalidState;
        const p = d.poll(slot, cancel) catch |e| return self.check(e);
        return .{ .done = p.done, .progressed = p.progressed, .position = p.position };
    }

    /// Only an already-owned immutable source may progress between packed prefill units.
    /// No slot selection, cache admission, discard or publication in this lane.
    pub fn pollMaintenanceInChunk(self: *ModelBackend, stopping: bool, reads_pending: bool) !batcher.MaintenancePoll {
        if (self.preparation_owner) |*owner| if (owner.held != null) {
            const p = owner.poll(stopping, false) catch |e| return self.check(e);
            return .{ .pending = p.pending or (p.reclaimed and !stopping), .progressed = p.progressed, .reclaimed = p.reclaimed };
        };
        const d = self.disk_archive orelse return .{};
        if (d.source) |held| {
            const h = held.view.lease.handle;
            const p = d.pollSourceWith(stopping or self.pressure_reclaim or d.source_cancel_requested, !reads_pending) catch |e| {
                d.failed_generation[h.index] = h.generation;
                if (e != error.Canceled) self.noteFailure();
                return .{ .progressed = true, .reclaimed = true };
            };
            if (p.done) {
                // Disk failures are an optional-cache miss; never immediately retry forever.
                if (!d.catalog.containsReady(held.view.tokens)) d.failed_generation[h.index] = h.generation;
                return .{ .pending = !stopping, .progressed = true, .reclaimed = true };
            }
            return .{ .pending = true, .progressed = p.progressed };
        }
        return .{};
    }

    /// One optional ownership/selection quantum, independent of request-slot lifetimes.
    pub fn pollMaintenance(self: *ModelBackend, stopping: bool, reads_pending: bool) !batcher.MaintenancePoll {
        if (self.preparation_owner) |*owner| {
            if (owner.held != null) {
                const p = owner.poll(stopping, true) catch |e| return self.check(e);
                return .{ .pending = p.pending or (p.reclaimed and !stopping), .progressed = p.progressed, .reclaimed = p.reclaimed };
            }
            const archive_held = if (self.disk_archive) |d| d.source != null else false;
            if (!stopping and !reads_pending and !archive_held) {
                const started = owner.start() catch |e| return self.check(e);
                if (started) return .{ .pending = true, .progressed = true };
            }
        }
        const d = self.disk_archive orelse return .{};
        if (d.source != null) return self.pollMaintenanceInChunk(stopping, reads_pending);
        if (stopping or self.pressure_reclaim or reads_pending) return .{};
        const c = self.cache orelse return .{};
        const capacity = try c.sourceCapacity();
        if (capacity.slots != self.m.snapshot_slots) return error.InvalidOptions;
        var select: session.pressure.Select = .{ .usage = .{ .slots = capacity.slots, .free_slots = capacity.free_slots, .slot_headroom = d.headroom_slots, .host_pages = self.m.swap_pages, .free_host_pages = self.m.hostFreePages(), .host_headroom = d.headroom_pages } };
        try select.usage.validate();
        if (!select.usage.pressured()) return .{};
        for (0..capacity.slots) |i| if (c.sourceCandidate(@intCast(i))) |candidate| {
            const bytes = try self.m.archiveBytes(@intCast(candidate.tokens.len));
            select.consider(.{ .handle = candidate.handle, .used = candidate.used, .has_host = candidate.has_host, .backed = d.catalog.containsReady(candidate.tokens), .fits = bytes <= d.store.file_bytes and d.failed_generation[i] != candidate.handle.generation });
        };
        const decision = select.decision() orelse return .{};
        switch (decision.action) {
            .discard => {
                try c.discardSource(self.device(), decision.handle);
                return .{ .pending = true, .progressed = true, .reclaimed = true };
            },
            .preserve => {
                const started = d.startSourceHandle(c, decision.handle) catch |e| {
                    d.failed_generation[decision.handle.index] = decision.handle.generation;
                    return self.check(e);
                };
                if (!started) d.failed_generation[decision.handle.index] = decision.handle.generation;
                return .{ .pending = started, .progressed = started };
            },
        }
    }

    pub fn initCache(self: *ModelBackend, allocator: std.mem.Allocator, kind: session.kvcache.Kind, boundary: u32, tier: bool) !void {
        const m = self.m;
        if (!m.options.kv_share or m.snapshot_slots == 0 or (self.disk_archive != null and kind != .radix)) return error.InvalidOptions;
        self.cache = try session.kvcache.createTiered(allocator, kind, m.snapshot_slots, m.state_layout.context, m.state_layout.seq_pages, boundary, tier and m.swap_pages > 0);
    }
    pub fn deinitCache(self: *ModelBackend, allocator: std.mem.Allocator) void {
        if (self.cache) |c| c.deinit(allocator);
        self.cache = null;
    }
    pub fn device(self: *ModelBackend) session.kvcache.Device {
        return .{ .ctx = self, .vtable = &device_vtable };
    }
    const device_vtable: session.kvcache.Device.VTable = .{ .pageTokens = devPageTokens, .save = devSave, .load = devLoad, .pin = devPin, .unpin = devUnpin, .attach = devAttach, .rebind = devRebind, .demote = devDemote, .promote = devPromote, .reclaimable = devReclaimable };
    fn backendOf(ctx: *anyopaque) *ModelBackend {
        return @ptrCast(@alignCast(ctx));
    }
    fn devPageTokens(ctx: *anyopaque) u32 {
        return backendOf(ctx).m.state_layout.page;
    }
    fn devSave(ctx: *anyopaque, slot: u32, snapshot: u32) anyerror!void {
        const self = backendOf(ctx);
        self.m.select(slot) catch |e| return self.check(e);
        self.m.saveSnapshot(snapshot) catch |e| return self.check(e);
    }
    fn devLoad(ctx: *anyopaque, slot: u32, snapshot: u32, position: u32) anyerror!void {
        const self = backendOf(ctx);
        self.m.select(slot) catch |e| return self.check(e);
        self.m.loadSnapshot(snapshot, position) catch |e| return self.check(e);
    }
    fn devPin(ctx: *anyopaque, slot: u32, tokens: u32, out: []u32) anyerror![]const u32 {
        const self = backendOf(ctx);
        return self.m.pinPrefix(slot, tokens, out) catch |e| return self.check(e);
    }
    fn devUnpin(ctx: *anyopaque, pages: []const u32) anyerror!void {
        const self = backendOf(ctx);
        self.m.unpinPages(pages) catch |e| return self.check(e);
    }
    fn devAttach(ctx: *anyopaque, slot: u32, full: []const u32, partial: ?u32) anyerror!void {
        const self = backendOf(ctx);
        self.m.attachPrefix(slot, full, partial) catch |e| {
            if (e == error.PoolExhausted) return e;
            return self.check(e);
        };
    }
    fn devReclaimable(ctx: *anyopaque, pages: []const u32) u32 {
        return backendOf(ctx).m.pool.reclaimable(pages);
    }
    fn devDemote(ctx: *anyopaque, pages: []u32) anyerror!u32 {
        const self = backendOf(ctx);
        return self.m.demotePages(pages) catch |e| return self.check(e);
    }
    fn devPromote(ctx: *anyopaque, pages: []u32) anyerror!bool {
        const self = backendOf(ctx);
        return self.m.promotePages(pages) catch |e| return self.check(e);
    }
    fn devRebind(ctx: *anyopaque, slot: u32, pages: []const u32) anyerror!u32 {
        const self = backendOf(ctx);
        return self.m.rebindPages(slot, pages) catch |e| return self.check(e);
    }
    /// Start `prompt` in `slot`: reset, then the cache's best checkpoint (or cold).
    pub fn begin(self: *ModelBackend, slot: u32, prompt: []const u32) !u32 {
        return self.beginKey(slot, null, prompt);
    }
    pub fn beginWithDemand(self: *ModelBackend, slot: u32, order: u64, prompt: []const u32) !u32 {
        return self.beginKey(slot, .{ .slot = slot, .order = order }, prompt);
    }
    fn beginKey(self: *ModelBackend, slot: u32, key: ?DemandKey, prompt: []const u32) !u32 {
        if (self.demand_mode != .off) {
            if (self.cache) |c| if (c.setDemand(prompt)) |h| if (h.has_host and h.source_held) {
                if (self.disk_archive) |d| d.source_cancel_requested = true;
                if (self.preparation_owner) |*p| if (p.held != null) {
                    p.canceled = true;
                };
                self.demand_waits += 1;
                return error.CacheReclaimPending;
            };
            if (self.disk_archive) |d| if (d.read_ahead.held) |held| if (held.key.slot == slot and (key == null or !d.read_ahead.matches(key.?))) {
                d.read_ahead.cancel();
                return error.CacheReclaimPending;
            };
        }
        const m = self.m;
        m.select(slot) catch |e| return self.check(e);
        m.reset() catch |e| return self.check(e);
        if (m.options.kv_share) m.releasePages(slot) catch |e| return self.check(e);
        if (self.disk_archive) |d| if (key) |k| if (d.read_ahead.matches(k)) {
            const n = d.catalog.entries[d.read_ahead.held.?.record].len;
            if (!self.admit(slot, n)) {
                d.read_ahead.cancel();
                self.suppressed_prefetch = k;
                return error.CacheReclaimPending;
            }
            d.takeReadAhead(k) catch |e| return self.check(e);
            return error.PendingIo;
        };
        const hot = if (self.cache) |c| try c.restore(self.device(), slot, prompt) else 0;
        if (self.disk_archive) |d| if (d.catalog.lookup(prompt)) |record| {
            const n = d.catalog.entries[record].len;
            if (n > hot) {
                try self.reset(slot);
                if (!self.admit(slot, n)) return 0;
                d.startRead(slot, record) catch |e| return self.check(e);
                return error.PendingIo;
            }
        };
        return hot;
    }
    /// Keep `slot`'s state after `prefix` (its current position) as a checkpoint.
    pub fn checkpoint(self: *ModelBackend, slot: u32, prefix: []const u32) !void {
        if (self.m.slotPosition(slot) != prefix.len) return error.InvalidToken;
        if (self.cache) |c| try c.checkpoint(self.device(), slot, prefix);
    }
    /// Memory pressure: demote or drop one cached checkpoint; when none frees a page, a
    /// swapped sequence other than `except` gives up the pool pages it still holds (a deep
    /// swap-out: they were shared or pinned, so checkpoints could not free them either).
    /// False: nothing left to free.
    fn makeRoom(self: *ModelBackend, except: ?u32) bool {
        if (self.preparation_owner) |*p| if (p.held != null) return false;
        if (self.cache) |c| {
            if (c.evict(self.device())) return true;
            if (self.demand_mode != .off) {
                c.clearDemand();
                self.demand_overrides += 1;
                if (c.evict(self.device())) return true;
            }
        }
        if (self.disk_archive) |d| if (d.source != null) {
            self.pressure_reclaim = true;
            d.source_cancel_requested = true;
            return false;
        };
        if (self.m.swap_pages == 0) return false;
        for (0..self.m.state_layout.slots) |i| {
            const slot: u32 = @intCast(i);
            if (except != null and except.? == slot or !self.m.holdsResident(slot)) continue;
            while (true) {
                const moved = self.m.swapShared(slot) catch |e| {
                    if (e == error.SwapFull) if (self.evictHostForLive()) continue;
                    if (e != error.SwapFull) self.noteFailure();
                    break;
                };
                if (moved) return true;
                break;
            }
        }
        return false;
    }
    pub fn checkRow(self: *ModelBackend, row: batcher.Row) !void {
        try self.m.checkRow(.{ .slot = row.slot, .token = row.token });
    }
    fn check(self: *ModelBackend, e: anyerror) anyerror {
        self.noteFailure();
        return e;
    }
    /// After a failed call: a pending command or a lost device makes the engine unusable.
    fn noteFailure(self: *ModelBackend) void {
        var owned_pending: u32 = if (self.disk_archive) |d| @intFromBool(d.commands.state == .pending) else 0;
        if (self.preparation_owner) |*p| owned_pending += @intFromBool(p.commands.state == .pending);
        if (self.m.device.lost or self.m.device.pending != owned_pending) self.fatal.store(true, .release);
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
        while (true) {
            self.m.ensurePages(slot, want) catch |e| {
                if (e != error.PoolExhausted) self.noteFailure();
                // Unused checkpoints go before a prompt waits.
                if (e == error.PoolExhausted and self.makeRoom(null)) continue;
                return false;
            };
            self.pressure_reclaim = false;
            return true;
        }
    }
    pub fn release(self: *ModelBackend, slot: u32) void {
        if (self.disk_archive) |d| if (d.read_ahead.held) |held| if (held.key.slot == slot) d.read_ahead.cancel();
        self.pressure_reclaim = false;
        if (!self.m.options.kv_share) return;
        self.m.releasePages(slot) catch self.noteFailure();
    }
    /// Shared pool: a page for the slot's next decode position (false: none free).
    pub fn grow(self: *ModelBackend, slot: u32) !bool {
        if (!self.m.options.kv_share) return true;
        while (true) {
            self.m.ensurePages(slot, self.m.slotPosition(slot) + 1) catch |e| {
                // Unused checkpoints go before any sequence is swapped out.
                if (e == error.PoolExhausted and self.makeRoom(null)) continue;
                if (e == error.PoolExhausted) {
                    if (self.preparation_owner) |*p| if (p.held != null) return error.CacheReclaimPending;
                    if (self.disk_archive) |d| if (d.source != null and d.source_cancel_requested) return error.CacheReclaimPending;
                    return false;
                }
                return self.check(e);
            };
            self.pressure_reclaim = false;
            return true;
        }
    }
    fn evictHostForLive(self: *ModelBackend) bool {
        const c = self.cache orelse return false;
        if (c.evictHost(self.device())) return true;
        if (self.demand_mode == .off) return false;
        c.clearDemand();
        self.demand_overrides += 1;
        return c.evictHost(self.device());
    }
    pub fn swapOut(self: *ModelBackend, slot: u32) !bool {
        if (self.m.swap_pages == 0) return false;
        while (true) {
            self.m.swapOut(slot) catch |e| {
                // A running sequence goes before demoted checkpoints in the host store.
                if (e == error.SwapFull) {
                    if (self.evictHostForLive()) continue;
                    return false;
                }
                return self.check(e);
            };
            return true;
        }
    }
    pub fn swapIn(self: *ModelBackend, slot: u32, spare: u32) !bool {
        while (true) {
            if (self.m.swapIn(slot, spare) catch |e| return self.check(e)) return true;
            // Unused checkpoints go before a swapped sequence waits (otherwise they could hold
            // the pool while every sequence is swapped out).
            if (!self.makeRoom(slot)) return false;
        }
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
