//! One-sequence generation: prefill, sampling, EOS/stop/length, and streamed
//! reasoning/content text. Backend-agnostic (tests drive it with CPU logits).
const std = @import("std");
pub const sampler = @import("sampler.zig");
pub const text = @import("text.zig");
pub const tools = @import("tools.zig");
pub const prefix = @import("prefix.zig");

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
};
pub const Event = text.Event;
pub const Finish = enum { stop, length };
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
};
pub const Error = sampler.Error || error{ InvalidStop, EmptyPrompt, ContextExceeded, InvalidMaxTokens, PieceTooLong, NoAllowedToken, SnapshotsUnsupported };

const max_piece = 16384;

pub fn Generation(comptime Backend: type, comptime Tokenizer: type, comptime Sink: type) type {
    return struct {
        pub fn run(io: std.Io, allocator: std.mem.Allocator, backend: Backend, tokenizer: Tokenizer, special: Special, request: Request, sink: Sink) !Result {
            if (request.prompt.len == 0) return error.EmptyPrompt;
            if (request.max_tokens == 0) return error.InvalidMaxTokens;
            const context = backend.context();
            if (request.prompt.len >= context) return error.ContextExceeded;
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
                    logits = try prefillCached(backend, cache, request.prompt, &begin);
                } else return error.SnapshotsUnsupported;
            } else {
                try backend.reset();
                logits = try backend.prefill(request.prompt);
            }
            const prefill_end = std.Io.Clock.awake.now(io);
            var generated: u32 = 0;
            var finish: Finish = .length;
            while (generated < limit) {
                // Cancelation point between steps (server drain deadline); the device is idle here.
                try io.checkCancel();
                const token = if (splitter.afterCall()) |space|
                    try s.sampleFrom(logits, try afterCallTokens(tokenizer, special, request.tools.?.parallel, space, allowed))
                else
                    try s.sample(logits);
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
                if (request.cache) |cache| {
                    errdefer cache.invalidate();
                    logits = try backend.step(token);
                    cache.record(&.{token});
                } else logits = try backend.step(token);
            }
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
            return result;
        }
    };
}

/// Starts `prompt` from the prefix cache: reset, restore or keep, then prefill the rest
/// in segments that end at the snapshot points, saving a snapshot after each.
fn prefillCached(backend: anytype, cache: *prefix.Cache, prompt: []const u32, begin: *prefix.Begin) ![]const f32 {
    // Any backend failure leaves the model state unknown: forget the cache.
    errdefer cache.invalidate();
    begin.* = cache.begin(prompt);
    switch (begin.outcome) {
        .reset => try backend.reset(),
        .restore => try backend.loadSnapshot(begin.slot, begin.start),
        .keep => {},
    }
    var points: [prefix.max_points]u32 = undefined;
    var at = begin.start;
    for (cache.points(prompt, begin.start, &points)) |point| {
        _ = try backend.prefill(prompt[at..point]);
        cache.record(prompt[at..point]);
        try backend.saveSnapshot(cache.claim());
        at = point;
    }
    const logits = try backend.prefill(prompt[at..]);
    cache.record(prompt[at..]);
    return logits;
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
