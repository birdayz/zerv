//! Bounded HTTP/1.1 server for Chat Completions v1 over an abstract Engine.
//! Up to `Options.parallel` generations at a time (1: the single-sequence model); a
//! bounded wait queue; other requests beyond it get 503. Streaming uses SSE over chunked
//! transfer encoding.
const std = @import("std");
const api = @import("api.zig");
const session = @import("../session/root.zig");

pub const Event = session.Event;
pub const Finish = session.Finish;

pub const Sink = struct {
    context: *anyopaque,
    emitFn: *const fn (*anyopaque, Event) anyerror!void,
    pub fn emit(self: Sink, event: Event) anyerror!void {
        return self.emitFn(self.context, event);
    }
};

pub const Completion = struct {
    finish: Finish,
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
    tool_calls: u32 = 0,
    tool_call_failed: bool = false,
    cached_tokens: u32 = 0,
    cache_outcome: session.prefix.Outcome = .reset,
    spec: session.SpecStats = .{},
};
pub const Prepared = struct { tokens: []const u32, thinking: bool };

/// Engine errors are mapped to OpenAI errors; `Rejected` carries a message in `detail`.
pub const Engine = struct {
    context: *anyopaque,
    /// Render + tokenize + validate. May allocate from `arena`. No model access.
    prepareFn: *const fn (*anyopaque, std.mem.Allocator, *const api.ChatRequest, *api.ApiError) anyerror!Prepared,
    /// Generate; at most `Options.parallel` calls run at once.
    generateFn: *const fn (*anyopaque, std.mem.Allocator, *const api.ChatRequest, Prepared, Sink) anyerror!Completion,
    /// Asked after a failed generation: false when the engine can never generate again
    /// (e.g. the GPU device was lost). The server then fails every request and stops.
    /// Null: always usable.
    usableFn: ?*const fn (*anyopaque) bool = null,
    ids: []const []const u8,
    defaults: api.Defaults,
};

pub const Options = struct {
    max_connections: u32 = 64,
    max_waiting: u32 = 16,
    /// Generations at once (`--parallel`; the engine must support that many).
    parallel: u32 = 1,
    /// How long `run` lets admitted chat requests finish after a stop request.
    drain_timeout: std.Io.Duration = .fromSeconds(30),
    limits: api.Limits = .{},
};

pub const Server = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    engine: Engine,
    options: Options,
    /// Generation permits (`options.parallel`).
    busy: std.Io.Semaphore,
    waiting: std.atomic.Value(u32) = .init(0),
    connections: std.atomic.Value(u32) = .init(0),
    /// Connection tasks. Spawned only by the accept loop; canceled by `serve`/`run`.
    group: std.Io.Group = .init,
    /// Set once shutdown starts: no new chat admissions, connections close after use.
    draining: std.atomic.Value(bool) = .init(false),
    /// Chat requests past the drain check (queued, generating or responding).
    admitted: std.atomic.Value(u32) = .init(0),
    /// Cleared when the accept loop exits (canceled or listener failure).
    accepting: std.atomic.Value(bool) = .init(true),
    /// Set when the engine became unusable: health checks and new requests get 503,
    /// and `run` shuts down as if stopped, then returns `error.EngineFailed`.
    failed: std.atomic.Value(bool) = .init(false),
    counter: std.atomic.Value(u64) = .init(0),
    created: i64,
    metrics: Metrics = .{},

    pub const Metrics = struct {
        requests: std.atomic.Value(u64) = .init(0),
        rejected: std.atomic.Value(u64) = .init(0),
        overloaded: std.atomic.Value(u64) = .init(0),
        failed: std.atomic.Value(u64) = .init(0),
        prompt_tokens: std.atomic.Value(u64) = .init(0),
        completion_tokens: std.atomic.Value(u64) = .init(0),
        prefill_ns: std.atomic.Value(u64) = .init(0),
        decode_ns: std.atomic.Value(u64) = .init(0),
        tool_call_parse_failures: std.atomic.Value(u64) = .init(0),
        prompt_cached_tokens: std.atomic.Value(u64) = .init(0),
        /// Requests by prefix-cache outcome (reset, restore, keep).
        cache_outcomes: [3]std.atomic.Value(u64) = @splat(.init(0)),
        /// Speculative decoding (session.SpecStats, summed).
        spec_verifies: std.atomic.Value(u64) = .init(0),
        spec_drafted: std.atomic.Value(u64) = .init(0),
        spec_verified: std.atomic.Value(u64) = .init(0),
        spec_accepted: std.atomic.Value(u64) = .init(0),
    };

    pub fn init(io: std.Io, allocator: std.mem.Allocator, engine: Engine, options: Options) Server {
        return .{ .io = io, .allocator = allocator, .engine = engine, .options = options, .busy = .{ .permits = @max(options.parallel, 1) }, .created = std.Io.Clock.real.now(io).toSeconds() };
    }

    /// Accept until the listener fails or this task is canceled; then cancel every
    /// connection immediately (no drain). Call at most once, and not with `run`.
    pub fn serve(self: *Server, listener: *std.Io.net.Server) !void {
        defer self.group.cancel(self.io);
        return self.acceptLoop(listener);
    }

    /// Serve until `stop` becomes true or the listener fails, then shut down gracefully:
    /// stop accepting; answer new chat requests 503 `shutting_down` (`/ready` → 503);
    /// let admitted requests finish for up to `options.drain_timeout`; then cancel the
    /// remaining connections (idle keep-alive ones, or a generation past the deadline,
    /// which stops at its next token). `stop` is polled every 20 ms, so it may be set
    /// from a signal handler. Takes ownership of `listener` and closes it as soon as
    /// draining starts, so new connections are refused rather than left in the backlog.
    /// Returns the listener error, if that ended serving, or `error.EngineFailed` if the
    /// engine became unusable. Call at most once, and not with `serve`.
    pub fn run(self: *Server, listener: *std.Io.net.Server, stop: *const std.atomic.Value(bool)) !void {
        var listening = true;
        defer if (listening) listener.deinit(self.io);
        var acceptor = try self.io.concurrent(acceptLoop, .{ self, listener });
        defer self.group.cancel(self.io);
        while (!stop.load(.acquire) and self.accepting.load(.acquire) and !self.failed.load(.acquire)) {
            self.io.sleep(.fromMilliseconds(20), .awake) catch break;
        }
        self.draining.store(true, .seq_cst);
        const accepted = acceptor.cancel(self.io);
        listener.deinit(self.io);
        listening = false;
        std.log.info("shutting down: draining {d} admitted request(s)", .{self.admitted.load(.seq_cst)});
        const deadline = std.Io.Clock.awake.now(self.io).addDuration(self.options.drain_timeout);
        while (self.admitted.load(.seq_cst) != 0 and std.Io.Clock.awake.now(self.io).nanoseconds < deadline.nanoseconds) {
            self.io.sleep(.fromMilliseconds(10), .awake) catch break;
        }
        accepted catch |e| switch (e) {
            error.Canceled => {},
            else => return e,
        };
        if (self.failed.load(.acquire)) return error.EngineFailed;
    }

    /// After a failed generation: if the engine can no longer generate, fail the server.
    fn checkEngine(self: *Server) void {
        const usable = self.engine.usableFn orelse return;
        if (usable(self.engine.context)) return;
        if (!self.failed.swap(true, .acq_rel)) std.log.warn("engine unusable (GPU device lost?): failing all requests and shutting down", .{});
    }

    fn acceptLoop(self: *Server, listener: *std.Io.net.Server) std.Io.net.Server.AcceptError!void {
        defer self.accepting.store(false, .release);
        while (true) {
            const stream = listener.accept(self.io) catch |e| switch (e) {
                // Transient: keep serving the connections we have.
                error.ConnectionAborted, error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded, error.SystemResources => {
                    std.log.warn("accept failed: {s}", .{@errorName(e)});
                    try self.io.sleep(.fromMilliseconds(100), .awake);
                    continue;
                },
                else => return e,
            };
            if (self.connections.load(.acquire) >= self.options.max_connections) {
                stream.close(self.io);
                continue;
            }
            _ = self.connections.fetchAdd(1, .acq_rel);
            self.group.concurrent(self.io, connection, .{ self, stream }) catch {
                _ = self.connections.fetchSub(1, .acq_rel);
                stream.close(self.io);
            };
        }
    }

    fn connection(self: *Server, stream: std.Io.net.Stream) void {
        defer {
            stream.close(self.io);
            _ = self.connections.fetchSub(1, .acq_rel);
        }
        var in_buf: [64 * 1024]u8 = undefined;
        var out_buf: [16 * 1024]u8 = undefined;
        var reader = stream.reader(self.io, &in_buf);
        var writer = stream.writer(self.io, &out_buf);
        var http: std.http.Server = .init(&reader.interface, &writer.interface);
        while (true) {
            var request = http.receiveHead() catch return;
            var arena_state: std.heap.ArenaAllocator = .init(self.allocator);
            defer arena_state.deinit();
            const keep = self.handle(arena_state.allocator(), &request) catch return;
            if (!keep or self.draining.load(.acquire)) return;
        }
    }

    fn respondJson(request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
        try request.respond(body, .{ .status = status, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
    }

    fn respondError(arena: std.mem.Allocator, request: *std.http.Server.Request, e: api.ApiError) !void {
        var out: std.Io.Writer.Allocating = .init(arena);
        try api.writeError(&out.writer, e);
        try respondJson(request, e.status, out.written());
    }

    /// Returns whether the connection may be reused.
    fn handle(self: *Server, arena: std.mem.Allocator, request: *std.http.Server.Request) !bool {
        // While draining every response carries `connection: close`.
        if (self.draining.load(.acquire)) request.head.keep_alive = false;
        const target = request.head.target;
        const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
        const keep = request.head.keep_alive;
        if (std.mem.eql(u8, path, "/v1/models")) {
            if (request.head.method != .GET) {
                try respondError(arena, request, .{ .status = .method_not_allowed, .message = "method not allowed" });
                return keep;
            }
            var out: std.Io.Writer.Allocating = .init(arena);
            try api.writeModels(&out.writer, self.engine.ids, self.created);
            try respondJson(request, .ok, out.written());
            return keep;
        }
        if (std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/ready")) {
            // Connections are accepted only after the model is loaded, validated and uploaded.
            if (self.failed.load(.acquire)) {
                request.head.keep_alive = false;
                try respondJson(request, .service_unavailable, "{\"status\":\"failed\"}");
                return false;
            }
            if (path[1] == 'r' and self.draining.load(.acquire)) {
                try respondJson(request, .service_unavailable, "{\"status\":\"draining\"}");
                return false;
            }
            try respondJson(request, .ok, "{\"status\":\"ok\"}");
            return keep;
        }
        if (std.mem.eql(u8, path, "/metrics")) {
            var out: std.Io.Writer.Allocating = .init(arena);
            const m = &self.metrics;
            try out.writer.print(
                \\# TYPE zerv_requests_total counter
                \\zerv_requests_total {d}
                \\# TYPE zerv_rejected_total counter
                \\zerv_rejected_total {d}
                \\# TYPE zerv_overloaded_total counter
                \\zerv_overloaded_total {d}
                \\# TYPE zerv_failed_total counter
                \\zerv_failed_total {d}
                \\# TYPE zerv_prompt_tokens_total counter
                \\zerv_prompt_tokens_total {d}
                \\# TYPE zerv_completion_tokens_total counter
                \\zerv_completion_tokens_total {d}
                \\# TYPE zerv_prefill_seconds_total counter
                \\zerv_prefill_seconds_total {d:.6}
                \\# TYPE zerv_decode_seconds_total counter
                \\zerv_decode_seconds_total {d:.6}
                \\# TYPE zerv_waiting gauge
                \\zerv_waiting {d}
                \\# TYPE zerv_tool_call_parse_failures_total counter
                \\zerv_tool_call_parse_failures_total {d}
                \\# TYPE zerv_prompt_cached_tokens_total counter
                \\zerv_prompt_cached_tokens_total {d}
                \\# TYPE zerv_prefix_cache_requests_total counter
                \\zerv_prefix_cache_requests_total{{outcome="reset"}} {d}
                \\zerv_prefix_cache_requests_total{{outcome="restore"}} {d}
                \\zerv_prefix_cache_requests_total{{outcome="keep"}} {d}
                \\# TYPE zerv_spec_verifies_total counter
                \\zerv_spec_verifies_total {d}
                \\# TYPE zerv_spec_draft_tokens_total counter
                \\zerv_spec_draft_tokens_total{{stage="drafted"}} {d}
                \\zerv_spec_draft_tokens_total{{stage="verified"}} {d}
                \\zerv_spec_draft_tokens_total{{stage="accepted"}} {d}
                \\
            , .{ m.requests.load(.monotonic), m.rejected.load(.monotonic), m.overloaded.load(.monotonic), m.failed.load(.monotonic), m.prompt_tokens.load(.monotonic), m.completion_tokens.load(.monotonic), @as(f64, @floatFromInt(m.prefill_ns.load(.monotonic))) / 1e9, @as(f64, @floatFromInt(m.decode_ns.load(.monotonic))) / 1e9, self.waiting.load(.monotonic), m.tool_call_parse_failures.load(.monotonic), m.prompt_cached_tokens.load(.monotonic), m.cache_outcomes[0].load(.monotonic), m.cache_outcomes[1].load(.monotonic), m.cache_outcomes[2].load(.monotonic), m.spec_verifies.load(.monotonic), m.spec_drafted.load(.monotonic), m.spec_verified.load(.monotonic), m.spec_accepted.load(.monotonic) });
            try request.respond(out.written(), .{ .extra_headers = &.{.{ .name = "content-type", .value = "text/plain; version=0.0.4" }} });
            return keep;
        }
        if (!std.mem.eql(u8, path, "/v1/chat/completions")) {
            try respondError(arena, request, .{ .status = .not_found, .message = "unknown endpoint", .code = "not_found" });
            return keep;
        }
        if (request.head.method != .POST) {
            try respondError(arena, request, .{ .status = .method_not_allowed, .message = "use POST" });
            return keep;
        }
        const length = request.head.content_length orelse {
            try respondError(arena, request, .{ .status = .length_required, .message = "content-length is required" });
            return false;
        };
        if (length > self.options.limits.max_body_bytes) {
            try respondError(arena, request, .{ .status = .payload_too_large, .message = "request body too large" });
            return false;
        }
        var body_buf: [4096]u8 = undefined;
        const body_reader = request.readerExpectContinue(&body_buf) catch return false;
        const body = body_reader.readAlloc(arena, @intCast(length)) catch return false;
        // Counted before the drain check (both seq_cst), so `run` never sees zero
        // admitted requests while one is past the check.
        _ = self.admitted.fetchAdd(1, .seq_cst);
        defer _ = self.admitted.fetchSub(1, .seq_cst);
        if (self.failed.load(.seq_cst)) {
            request.head.keep_alive = false;
            try respondError(arena, request, .{ .status = .service_unavailable, .message = "the model is unusable (GPU device lost); the server is shutting down", .kind = "server_error", .code = "engine_failed" });
            return false;
        }
        if (self.draining.load(.seq_cst)) {
            request.head.keep_alive = false;
            try respondError(arena, request, .{ .status = .service_unavailable, .message = "server is shutting down", .kind = "server_error", .code = "shutting_down" });
            return false;
        }
        _ = self.metrics.requests.fetchAdd(1, .monotonic);
        const parsed = try api.parseChat(arena, body, self.engine.ids, self.engine.defaults, self.options.limits);
        const chat = switch (parsed) {
            .err => |e| {
                _ = self.metrics.rejected.fetchAdd(1, .monotonic);
                try respondError(arena, request, e);
                return keep;
            },
            .ok => |c| c,
        };
        var rejection: api.ApiError = .{ .message = "request rejected" };
        const prepared = self.engine.prepareFn(self.engine.context, arena, &chat, &rejection) catch {
            _ = self.metrics.rejected.fetchAdd(1, .monotonic);
            try respondError(arena, request, rejection);
            return keep;
        };
        // Bounded queue for the generation permits.
        if (self.waiting.fetchAdd(1, .acq_rel) >= self.options.max_waiting) {
            _ = self.waiting.fetchSub(1, .acq_rel);
            _ = self.metrics.overloaded.fetchAdd(1, .monotonic);
            try respondError(arena, request, .{ .status = .service_unavailable, .message = "server is at capacity; retry later", .kind = "server_error", .code = "overloaded" });
            return keep;
        }
        self.busy.wait(self.io) catch {
            _ = self.waiting.fetchSub(1, .acq_rel);
            return false;
        };
        _ = self.waiting.fetchSub(1, .acq_rel);
        defer self.busy.post(self.io);

        var id_buf: [40]u8 = undefined;
        const id = try std.fmt.bufPrint(&id_buf, "chatcmpl-{x:0>16}{x:0>8}", .{ self.counter.fetchAdd(1, .monotonic), @as(u32, @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(self.io).nanoseconds)))) });
        const meta: api.Meta = .{ .id = id, .created = std.Io.Clock.real.now(self.io).toSeconds(), .model = chat.model };
        if (chat.stream) return self.streamResponse(arena, request, &chat, prepared, meta);

        var collector: Collector = .{ .arena = arena, .meta = meta };
        const done = self.engine.generateFn(self.engine.context, arena, &chat, prepared, collector.sink()) catch |e| {
            _ = self.metrics.failed.fetchAdd(1, .monotonic);
            self.checkEngine();
            if (e == error.Canceled) return false; // drain deadline: the connection is closing
            std.log.warn("generation failed: {s}", .{@errorName(e)});
            try respondError(arena, request, .{ .status = .internal_server_error, .message = "generation failed", .kind = "server_error" });
            return false;
        };
        self.record(done);
        if (self.draining.load(.acquire)) request.head.keep_alive = false;
        var out: std.Io.Writer.Allocating = .init(arena);
        try api.writeCompletion(&out.writer, meta, if (collector.reasoning.items.len > 0) collector.reasoning.items else null, collector.content.items, if (collector.calls.items.len > 0) collector.calls.items else null, finishName(done), .{ .prompt_tokens = done.prompt_tokens, .completion_tokens = done.completion_tokens, .cached_tokens = done.cached_tokens });
        try respondJson(request, .ok, out.written());
        return request.head.keep_alive;
    }

    fn record(self: *Server, done: Completion) void {
        _ = self.metrics.prompt_cached_tokens.fetchAdd(done.cached_tokens, .monotonic);
        _ = self.metrics.cache_outcomes[@intFromEnum(done.cache_outcome)].fetchAdd(1, .monotonic);
        _ = self.metrics.spec_verifies.fetchAdd(done.spec.verifies, .monotonic);
        _ = self.metrics.spec_drafted.fetchAdd(done.spec.drafted, .monotonic);
        _ = self.metrics.spec_verified.fetchAdd(done.spec.verified, .monotonic);
        _ = self.metrics.spec_accepted.fetchAdd(done.spec.accepted, .monotonic);
        if (done.tool_call_failed) {
            _ = self.metrics.tool_call_parse_failures.fetchAdd(1, .monotonic);
            std.log.warn("a generated tool call was malformed; generation stopped (see docs/specs/tool-calling.md)", .{});
        }
        _ = self.metrics.prompt_tokens.fetchAdd(done.prompt_tokens, .monotonic);
        _ = self.metrics.completion_tokens.fetchAdd(done.completion_tokens, .monotonic);
        _ = self.metrics.prefill_ns.fetchAdd(done.prefill_ns, .monotonic);
        _ = self.metrics.decode_ns.fetchAdd(done.decode_ns, .monotonic);
    }

    fn streamResponse(self: *Server, arena: std.mem.Allocator, request: *std.http.Server.Request, chat: *const api.ChatRequest, prepared: Prepared, meta: api.Meta) !bool {
        var buf: [16 * 1024]u8 = undefined;
        var body = try request.respondStreaming(&buf, .{ .respond_options = .{ .extra_headers = &.{
            .{ .name = "content-type", .value = "text/event-stream" },
            .{ .name = "cache-control", .value = "no-cache" },
        } } });
        var streamer: Streamer = .{ .body = &body, .meta = meta };
        try api.writeChunk(&body.writer, meta, .role);
        try push(&body);
        const done = self.engine.generateFn(self.engine.context, arena, chat, prepared, streamer.sink()) catch |e| {
            _ = self.metrics.failed.fetchAdd(1, .monotonic);
            self.checkEngine();
            if (streamer.failed) return false; // client went away: generation was cancelled
            if (e == error.Canceled) return false; // drain deadline: the connection is closing
            std.log.warn("generation failed: {s}", .{@errorName(e)});
            try body.writer.writeAll("data: ");
            try api.writeError(&body.writer, .{ .status = .internal_server_error, .message = "generation failed", .kind = "server_error" });
            try body.writer.writeAll("\n\n");
            try body.end();
            return false;
        };
        self.record(done);
        try api.writeChunk(&body.writer, meta, .{ .finish = finishName(done) });
        if (chat.include_usage) try api.writeUsageChunk(&body.writer, meta, .{ .prompt_tokens = done.prompt_tokens, .completion_tokens = done.completion_tokens, .cached_tokens = done.cached_tokens });
        try body.writer.writeAll("data: [DONE]\n\n");
        try body.end();
        return request.head.keep_alive;
    }
};

/// Send buffered SSE bytes to the client now: `BodyWriter.flush` alone only flushes
/// the connection, not the body writer's own buffer.
fn push(body: *std.http.BodyWriter) !void {
    try body.writer.flush();
    try body.flush();
}

fn finishName(done: Completion) []const u8 {
    return switch (done.finish) {
        .stop => if (done.tool_calls > 0) "tool_calls" else "stop",
        .length => "length",
    };
}

const Collector = struct {
    arena: std.mem.Allocator,
    meta: api.Meta,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(api.CallOut) = .empty,
    arguments: std.ArrayList(std.ArrayList(u8)) = .empty,
    fn emit(ctx: *anyopaque, event: Event) anyerror!void {
        const self: *Collector = @ptrCast(@alignCast(ctx));
        switch (event) {
            .reasoning => |bytes| try self.reasoning.appendSlice(self.arena, bytes),
            .content => |bytes| try self.content.appendSlice(self.arena, bytes),
            .call_begin => |c| {
                std.debug.assert(c.index == self.calls.items.len);
                var id_buf: [40]u8 = undefined;
                const id = try self.arena.dupe(u8, api.callId(&id_buf, self.meta, c.index));
                var arguments: std.ArrayList(u8) = .empty;
                try arguments.append(self.arena, '{'); // implied by call_begin
                try self.calls.append(self.arena, .{ .id = id, .function = .{ .name = try self.arena.dupe(u8, c.name), .arguments = arguments.items } });
                try self.arguments.append(self.arena, arguments);
            },
            .call_arguments => |c| {
                const list = &self.arguments.items[c.index];
                try list.appendSlice(self.arena, c.text);
                self.calls.items[c.index].function.arguments = list.items;
            },
        }
    }
    fn sink(self: *Collector) Sink {
        return .{ .context = self, .emitFn = emit };
    }
};

const Streamer = struct {
    body: *std.http.BodyWriter,
    meta: api.Meta,
    failed: bool = false,
    fn emit(ctx: *anyopaque, event: Event) anyerror!void {
        const self: *Streamer = @ptrCast(@alignCast(ctx));
        var id_buf: [40]u8 = undefined;
        const delta: api.Delta = switch (event) {
            .reasoning => |bytes| .{ .reasoning = bytes },
            .content => |bytes| .{ .content = bytes },
            .call_begin => |c| .{ .call_begin = .{ .index = c.index, .id = api.callId(&id_buf, self.meta, c.index), .name = c.name } },
            .call_arguments => |c| .{ .call_arguments = .{ .index = c.index, .text = c.text } },
        };
        api.writeChunk(&self.body.writer, self.meta, delta) catch |e| {
            self.failed = true;
            return e;
        };
        push(self.body) catch |e| {
            self.failed = true;
            return e;
        };
    }
    fn sink(self: *Streamer) Sink {
        return .{ .context = self, .emitFn = emit };
    }
};
