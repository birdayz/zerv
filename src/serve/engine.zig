//! Native Qwen3.8 engine: official template -> tokenizer -> resident model -> session.
const std = @import("std");
const api = @import("api.zig");
const http = @import("http.zig");
const chat = @import("../chat/qwen38.zig");
const bpe = @import("../tokenizer/bpe.zig");
const model = @import("../model/root.zig");
const session = @import("../session/root.zig");

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

    pub fn engine(self: *Native, ids: []const []const u8, defaults: api.Defaults) http.Engine {
        return .{ .context = self, .prepareFn = prepare, .generateFn = generate, .usableFn = usable, .ids = ids, .defaults = defaults };
    }

    /// The model can run again: the device is not lost and no command is left pending
    /// (a timed-out command can be neither waited for again nor resubmitted).
    fn usable(ctx: *anyopaque) bool {
        const self: *Native = @ptrCast(@alignCast(ctx));
        const device = self.model.device;
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

    fn generate(ctx: *anyopaque, arena: std.mem.Allocator, request: *const api.ChatRequest, prepared: http.Prepared, sink: http.Sink) anyerror!http.Completion {
        const self: *Native = @ptrCast(@alignCast(ctx));
        var params = request.params;
        if (!request.seed_given) {
            const now: u64 = @bitCast(@as(i64, @truncate(std.Io.Clock.real.now(self.io).nanoseconds)));
            params.seed = now ^ (self.seed_counter.fetchAdd(1, .monotonic) *% 0x9e3779b97f4a7c15);
        }
        const result = try Generation.run(self.io, arena, .{ .m = self.model }, self.tokenizer, .{ .eos = &self.eos_ids, .whitespace = self.whitespace_ids, .tool_call = self.tool_call_id }, .{
            .prompt = prepared.tokens,
            .max_tokens = request.max_tokens orelse self.model.state_layout.context,
            .params = params,
            .stops = request.stops,
            .thinking = prepared.thinking,
            .tools = request.toolMode(),
            .cache = if (self.cache) |*c| c else null,
        }, sink);
        if (self.cache != null) std.log.info("prefix cache: {s}, reused {d} of {d} prompt tokens", .{ @tagName(result.cache_outcome), result.cached_tokens, result.prompt_tokens });
        return .{ .finish = result.finish, .prompt_tokens = result.prompt_tokens, .completion_tokens = result.completion_tokens, .prefill_ns = result.prefill_ns, .decode_ns = result.decode_ns, .tool_calls = result.tool_calls, .tool_call_failed = result.tool_call_failed, .cached_tokens = result.cached_tokens, .cache_outcome = result.cache_outcome };
    }
};
