//! One-sequence generation: prefill, sampling, EOS/stop/length, and streamed
//! reasoning/content text. Backend-agnostic (tests drive it with CPU logits).
const std = @import("std");
pub const sampler = @import("sampler.zig");
pub const text = @import("text.zig");
pub const tools = @import("tools.zig");
pub const prefix = @import("prefix.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const residency = @import("residency.zig");
pub const archive = @import("archive.zig");
pub const readahead = @import("readahead.zig");
pub const kvcache = @import("kvcache.zig");
pub const pressure = @import("pressure.zig");
pub const spec = @import("spec.zig");
pub const media = @import("media.zig");
pub const Media = media.Media;

/// Model-profile token ids, resolved from the tokenizer at load time.
pub const Special = struct {
    eos: []const u32,
    /// Tokens whose output text is only spaces, tabs and newlines (tool mode).
    whitespace: []const u32 = &.{},
    /// The `<tool_call>` token (tool mode).
    tool_call: ?u32 = null,
};

pub const Request = struct {
    prompt: []const u32,
    /// Prompt rows fed by media (image encoder outputs) instead of their placeholder
    /// tokens (docs/specs/prefix-cache.md, "Media spans"). The backend must then provide
    /// `prefillMedia(tokens, spans, first)`: `spans` with prompt rows, `first` the prompt
    /// row of `tokens[0]`, every span whole inside the tokens.
    media: []const Media = &.{},
    max_tokens: u32,
    params: sampler.Params = .{},
    stops: []const []const u8 = &.{},
    /// Prompt ends inside an open `<think>` block: output starts as reasoning.
    thinking: bool = false,
    /// Parse tool calls (tools given and tool_choice auto). docs/specs/tool-calling.md.
    tools: ?tools.Mode = null,
    /// Reuse processed tokens across requests (docs/specs/prefix-cache.md). The backend
    /// must then provide `saveSnapshot(slot)` and `loadSnapshot(slot, position)`.
    cache: ?*prefix.Cache = null,
    /// Prefix checkpoints of a shared pool (docs/specs/concurrent.md, "18d.4 design"): the
    /// message boundary token. The backend must then provide `begin(prompt) !u32` (start
    /// position) and `checkpoint(prefix)`. Exclusive with `cache`; media prompts start cold.
    checkpoints: ?u32 = null,
    /// Opt-in maximum cache-hit suffix that skips new intermediate checkpoints.
    reuse_join: u32 = 0,
    /// Speculative decoding (backends with `speculative() > 0`): the verify-count policy
    /// (null: verify every draft). Owned by the caller; it learns across requests.
    spec_policy: ?*spec.Policy = null,
};
pub const Event = text.Event;
pub const Finish = enum { stop, length };
/// Speculative decoding counters of one generation.
pub const SpecStats = struct {
    /// Verify passes.
    verifies: u32 = 0,
    /// Drafts the drafter produced (the policy may verify fewer).
    drafted: u32 = 0,
    /// Verified drafts that were decided, and those accepted (sampled equal to the draft).
    /// Drafts left over when a generation ends inside verified rows are neither.
    verified: u32 = 0,
    accepted: u32 = 0,
};
pub const Result = struct {
    finish: Finish,
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
    /// Tool calls reported (complete NAME), and whether one was malformed.
    tool_calls: u32 = 0,
    tool_call_failed: bool = false,
    /// Prompt tokens reused from the prefix cache, and how the prompt was started.
    cached_tokens: u32 = 0,
    cache_outcome: prefix.Outcome = .reset,
    spec: SpecStats = .{},
    /// Of `decode_ns`: time inside backend calls (step, draft, verify, commit: submission,
    /// GPU and fence wait) and inside the sampler. The rest is host token handling
    /// (detokenizing, stop strings, streaming to the sink).
    backend_ns: u64 = 0,
    sample_ns: u64 = 0,
};
pub const Error = sampler.Error || error{ InvalidStop, EmptyPrompt, ContextExceeded, InvalidMaxTokens, PieceTooLong, NoAllowedToken, SnapshotsUnsupported, InvalidMedia, MediaUnsupported };

const max_piece = 16384;

pub fn Generation(comptime Backend: type, comptime Tokenizer: type, comptime Sink: type) type {
    return struct {
        pub fn run(io: std.Io, allocator: std.mem.Allocator, backend: Backend, tokenizer: Tokenizer, special: Special, request: Request, sink: Sink) !Result {
            if (request.prompt.len == 0) return error.EmptyPrompt;
            if (request.max_tokens == 0) return error.InvalidMaxTokens;
            const context = backend.context();
            if (request.prompt.len >= context) return error.ContextExceeded;
            try media.validate(request.media, request.prompt.len);
            if (request.media.len > 0 and !comptime mediaInput(Backend)) return error.MediaUnsupported;
            const limit: u32 = @intCast(@min(request.max_tokens, context - request.prompt.len));
            var s = try sampler.Sampler.init(allocator, backend.vocab(), request.params);
            defer s.deinit(allocator);
            // Scratch for one token: UTF-8 output, stop holdback + text, split holdback + text.
            const utf8_len = max_piece * 3 + 64;
            const stop_len = utf8_len + text.max_stop_bytes;
            var buffers = try allocator.alloc(u8, utf8_len + stop_len + stop_len + text.Splitter.max_held);
            defer allocator.free(buffers);
            const utf8_out = buffers[0..utf8_len];
            const stop_out = buffers[utf8_len..][0..stop_len];
            const split_out = buffers[utf8_len + stop_len ..];
            var utf8: text.Utf8 = .{};
            var stops = try text.Stops.init(request.stops);
            // Tool mode: the call parser's value buffer and the between-call token set.
            var calls: tools.Calls = undefined;
            var value_buf: []u8 = &.{};
            defer allocator.free(value_buf);
            var allowed: []u32 = &.{};
            defer allocator.free(allowed);
            if (request.tools) |mode| {
                value_buf = try allocator.alloc(u8, tools.max_value);
                allowed = try allocator.alloc(u32, special.eos.len + special.whitespace.len + 1);
                calls = tools.Calls.init(mode, value_buf);
            }
            var splitter = text.Splitter.init(request.thinking, if (request.tools != null) &calls else null);

            const start = std.Io.Clock.awake.now(io);
            var begin: prefix.Begin = .{ .outcome = .reset, .start = 0 };
            var logits: []const f32 = undefined;
            if (request.cache) |cache| {
                if (comptime snapshots(Backend)) {
                    logits = try prefillCached(backend, cache, request.prompt, request.media, &begin);
                } else return error.SnapshotsUnsupported;
            } else if (request.checkpoints != null and request.media.len == 0) {
                if (comptime checkpoints(Backend)) {
                    logits = try prefillCheckpointed(backend, request.prompt, request.checkpoints.?, request.reuse_join, &begin);
                } else return error.SnapshotsUnsupported;
            } else {
                try backend.reset();
                logits = try prefillSegment(backend, request.prompt, request.media, 0);
            }
            const prefill_end = std.Io.Clock.awake.now(io);
            var generated: u32 = 0;
            var finish: Finish = .length;
            // Speculative decoding (docs/specs/speculative.md): `sv.logits` holds verified
            // rows (row i = the logits after verify token i = `sv.tokens[i]`, the drafts
            // follow the sampled token). Sampling stays one token at a time with the same
            // sampler calls, and row i+1 is used only when the token sampled from row i
            // equals its draft, so the output is the non-speculative output (sample matching).
            const spec_drafts: u32 = if (comptime speculative(Backend)) backend.speculative() else 0;
            var sv: Spec = .{ .row = 0, .rows = 1, .logits = logits, .vocab = backend.vocab() };
            var stats: SpecStats = .{};
            var backend_ns: u64 = 0;
            var sample_ns: u64 = 0;
            // An error with an uncommitted verify leaves the model ahead of the recorded
            // tokens: forget the cache (the next request resets).
            errdefer if (sv.pending) if (request.cache) |cache| cache.invalidate();
            while (generated < limit) {
                // Cancelation point between steps (server drain deadline); the device is idle here.
                try io.checkCancel();
                const row_logits = sv.current();
                const sample_start = std.Io.Clock.awake.now(io);
                const token = if (splitter.afterCall()) |space|
                    try s.sampleFrom(row_logits, try afterCallTokens(tokenizer, special, request.tools.?.parallel, space, allowed))
                else
                    try s.sample(row_logits);
                sample_ns += elapsed(io, sample_start);
                // A batching backend may reuse the logits memory from here on
                // (docs/specs/concurrent.md, "18c design").
                if (comptime releases(Backend)) backend.sampled();
                generated += 1;
                if (std.mem.indexOfScalar(u32, special.eos, token) != null) {
                    finish = .stop;
                    break;
                }
                try s.accept(allocator, token);
                const piece = try tokenizer.outputPiece(token);
                if (piece.len > max_piece) return error.PieceTooLong;
                try splitter.push(stops.push(utf8.push(piece, utf8_out), stop_out), split_out, sink);
                if (stops.stopped or splitter.done()) {
                    finish = .stop;
                    break;
                }
                if (generated == limit) break;
                if (comptime speculative(Backend)) {
                    if (sv.pending) {
                        // The next verified row is valid only if its draft was sampled.
                        if (sv.row + 1 < sv.rows and token == sv.tokens[sv.row + 1]) {
                            sv.row += 1;
                            continue;
                        }
                        const m = sv.row + 1;
                        sv.pending = false;
                        stats.verified += sv.rows - 1;
                        stats.accepted += m - 1;
                        errdefer if (request.cache) |cache| cache.invalidate();
                        const t0 = std.Io.Clock.awake.now(io);
                        try backend.commit(m);
                        const commit_ns = elapsed(io, t0);
                        backend_ns += commit_ns;
                        if (request.spec_policy) |policy| {
                            policy.timeCommit(commit_ns);
                            policy.observe(sv.probs[0 .. sv.rows - 1], m - 1);
                        }
                        if (request.cache) |cache| cache.record(sv.tokens[0..m], &.{});
                    }
                    // Drafts: at most one fewer than the tokens still to sample (each verified
                    // row yields at most one) and inside the context (the verify's last row).
                    const room = backend.context() - @as(u32, @intCast(request.prompt.len)) - generated;
                    const k = @min(spec_drafts, limit - generated - 1, room);
                    if (k > 0) {
                        errdefer if (request.cache) |cache| cache.invalidate();
                        var t0 = std.Io.Clock.awake.now(io);
                        const drafts = try backend.draft(token, k);
                        const draft_ns = elapsed(io, t0);
                        backend_ns += draft_ns;
                        stats.drafted += k;
                        sv.tokens[0] = token;
                        @memcpy(sv.tokens[1..][0..k], drafts);
                        // The policy may verify fewer drafts (never more): speed only.
                        var verified = k;
                        if (request.spec_policy) |policy| {
                            policy.timeDraft(draft_ns);
                            const probs = backend.draftProbs(k);
                            @memcpy(sv.probs[0..k], probs);
                            verified = policy.choose(probs);
                        }
                        t0 = std.Io.Clock.awake.now(io);
                        sv.logits = try backend.verify(sv.tokens[0 .. verified + 1]);
                        const verify_ns = elapsed(io, t0);
                        backend_ns += verify_ns;
                        if (request.spec_policy) |policy| policy.timeVerify(verified + 1, verify_ns);
                        sv.rows = verified + 1;
                        sv.row = 0;
                        sv.pending = true;
                        stats.verifies += 1;
                        continue;
                    }
                }
                sv.rows = 1;
                sv.row = 0;
                const step_start = std.Io.Clock.awake.now(io);
                if (request.cache) |cache| {
                    errdefer cache.invalidate();
                    sv.logits = try backend.step(token);
                    cache.record(&.{token}, &.{});
                } else sv.logits = try backend.step(token);
                backend_ns += elapsed(io, step_start);
            }
            if (comptime speculative(Backend)) if (sv.pending) {
                // The generation ended inside verified rows: keep the rows whose tokens were
                // consumed (the last sampled token stays unprocessed, as without speculation).
                errdefer if (request.cache) |cache| cache.invalidate();
                const t0 = std.Io.Clock.awake.now(io);
                try backend.commit(sv.row + 1);
                backend_ns += elapsed(io, t0);
                if (request.cache) |cache| cache.record(sv.tokens[0 .. sv.row + 1], &.{});
                sv.pending = false;
                stats.verified += sv.row;
                stats.accepted += sv.row;
            };
            // An incomplete UTF-8 tail and a held partial delimiter are dropped;
            // bytes held for a stop string that never completed are released.
            utf8.finish();
            if (!stops.stopped and !splitter.done()) try splitter.push(stops.finish(stop_out), split_out, sink);
            try splitter.finish(sink);
            const end = std.Io.Clock.awake.now(io);
            var result: Result = .{ .finish = finish, .prompt_tokens = @intCast(request.prompt.len), .completion_tokens = generated, .prefill_ns = @intCast(start.durationTo(prefill_end).nanoseconds), .decode_ns = @intCast(prefill_end.durationTo(end).nanoseconds) };
            if (request.tools != null) {
                result.tool_calls = calls.count;
                result.tool_call_failed = calls.failed;
            }
            result.cached_tokens = begin.start;
            result.cache_outcome = begin.outcome;
            result.spec = stats;
            result.backend_ns = backend_ns;
            result.sample_ns = sample_ns;
            return result;
        }
    };
}

/// Speculative decoding state of a generation: verified logits rows and their tokens.
const Spec = struct {
    const max_rows = 8;
    logits: []const f32,
    vocab: usize,
    rows: u32,
    row: u32,
    /// A verify is uncommitted (rows 0..row consumed so far).
    pending: bool = false,
    /// Verify tokens: the sampled token, then the drafts; the drafter's probabilities.
    tokens: [max_rows]u32 = @splat(0),
    probs: [max_rows]f32 = @splat(0),

    fn current(self: *const Spec) []const f32 {
        return self.logits[self.row * self.vocab ..][0..self.vocab];
    }
};

/// Whether `Backend` offers speculative decoding (`speculative() u32` drafts per step,
/// `draft`, `draftProbs`, `verify`, `commit`).
/// Whether the backend wants to know when the last logits have been sampled.
fn releases(comptime Backend: type) bool {
    const T = switch (@typeInfo(Backend)) {
        .pointer => |p| p.child,
        else => Backend,
    };
    return @hasDecl(T, "sampled");
}

fn speculative(comptime Backend: type) bool {
    const T = switch (@typeInfo(Backend)) {
        .pointer => |p| p.child,
        else => Backend,
    };
    return @hasDecl(T, "speculative") and @hasDecl(T, "draft") and @hasDecl(T, "draftProbs") and @hasDecl(T, "verify") and @hasDecl(T, "commit");
}

fn elapsed(io: std.Io, since: std.Io.Timestamp) u64 {
    return @intCast(@max(since.durationTo(std.Io.Clock.awake.now(io)).nanoseconds, 0));
}

/// Starts `prompt` from the prefix cache: reset, restore or keep, then prefill the rest
/// in segments that end at the snapshot points, saving a snapshot after each.
fn prefillCached(backend: anytype, cache: *prefix.Cache, prompt: []const u32, spans: []const Media, begin: *prefix.Begin) ![]const f32 {
    // Any backend failure leaves the model state unknown: forget the cache.
    errdefer cache.invalidate();
    begin.* = cache.begin(prompt, spans);
    switch (begin.outcome) {
        .reset => try backend.reset(),
        .restore => try backend.loadSnapshot(begin.slot, begin.start),
        .keep => {},
    }
    var points: [prefix.max_points]u32 = undefined;
    var at = begin.start;
    for (cache.points(prompt, spans, begin.start, &points)) |point| {
        _ = try prefillSegment(backend, prompt[at..point], spans, at);
        cache.record(prompt[at..point], media.within(spans, at, point));
        try backend.saveSnapshot(cache.claim());
        at = point;
    }
    const n: u32 = @intCast(prompt.len);
    const logits = try prefillSegment(backend, prompt[at..], spans, at);
    cache.record(prompt[at..], media.within(spans, at, n));
    return logits;
}

/// Starts `prompt` from the backend's longest prefix checkpoint (or cold), then prefills the
/// rest in segments ending at the checkpoint positions, taking a checkpoint after each.
fn prefillCheckpointed(backend: anytype, prompt: []const u32, boundary: u32, reuse_join: u32, begin: *prefix.Begin) ![]const f32 {
    const start = try backend.begin(prompt);
    begin.* = .{ .outcome = if (start > 0) .restore else .reset, .start = start };
    var points: [checkpoint.max_points]u32 = undefined;
    var at = start;
    for (checkpoint.reusePoints(prompt, start, boundary, reuse_join, &points)) |point| {
        _ = try backend.prefill(prompt[at..point]);
        try backend.checkpoint(prompt[0..point]);
        at = point;
    }
    return backend.prefill(prompt[at..]);
}

/// Whether `Backend` (a type or a pointer to one) keeps prefix checkpoints.
fn checkpoints(comptime Backend: type) bool {
    const T = switch (@typeInfo(Backend)) {
        .pointer => |p| p.child,
        else => Backend,
    };
    return @hasDecl(T, "begin") and @hasDecl(T, "checkpoint");
}

/// Prefills prompt rows `first..first + tokens.len` (never cutting a span): through
/// `prefillMedia` when spans lie among them, else `prefill`.
fn prefillSegment(backend: anytype, tokens: []const u32, spans: []const Media, first: u32) ![]const f32 {
    const inner = media.within(spans, first, first + @as(u32, @intCast(tokens.len)));
    if (inner.len == 0) return backend.prefill(tokens);
    if (comptime mediaInput(@TypeOf(backend))) return backend.prefillMedia(tokens, inner, first);
    unreachable; // rejected in `run`
}

/// Whether `Backend` (a type or a pointer to one) takes media rows (`prefillMedia`).
fn mediaInput(comptime Backend: type) bool {
    const T = switch (@typeInfo(Backend)) {
        .pointer => |p| p.child,
        else => Backend,
    };
    return @hasDecl(T, "prefillMedia");
}

/// Whether `Backend` (a type or a pointer to one) supports prefix-cache snapshots.
fn snapshots(comptime Backend: type) bool {
    const T = switch (@typeInfo(Backend)) {
        .pointer => |p| p.child,
        else => Backend,
    };
    return @hasDecl(T, "saveSnapshot") and @hasDecl(T, "loadSnapshot");
}

/// The tokens llama-server's grammar allows right after a complete tool call
/// (docs/specs/tool-calling.md, "Constrained decoding"): EOS; whitespace tokens that
/// keep `space` a SPACE_RULE prefix; `<tool_call>` when parallel calls are allowed.
fn afterCallTokens(tokenizer: anytype, special: Special, parallel: bool, space: []const u8, out: []u32) ![]const u32 {
    var n: usize = 0;
    for (special.eos) |id| {
        out[n] = id;
        n += 1;
    }
    for (special.whitespace) |id| {
        if (tools.spaceRuleExtends(space, try tokenizer.outputPiece(id))) {
            out[n] = id;
            n += 1;
        }
    }
    if (parallel) if (special.tool_call) |id| {
        out[n] = id;
        n += 1;
    };
    if (n == 0) return error.NoAllowedToken;
    return out[0..n];
}
