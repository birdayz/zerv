//! Continuous batching of concurrent sequences (docs/specs/concurrent.md, "18c design").
//!
//! Each generation holds a slot and submits reset, prefill and decode-step operations; one
//! scheduler task (`run`) executes them on the backend. Prompts are prefilled in chunks,
//! first come first served; the decode steps of every waiting slot run as one batch. The
//! logits handed to a generation stay valid until it calls `sampled` (or its next operation,
//! or `leave`): no later batch or prefill chunk overwrites them.
//!
//! Scheduling: a batch runs when steps are pending, no row of the previous batch is still
//! held, and every decoding slot has submitted or `gather` has passed since the last hold
//! was released (a slow client never holds the GPU). When a prefill chunk and a batch are
//! both ready they alternate.
const std = @import("std");

pub const max_slots = 64;
pub const Row = struct { slot: u32, token: u32 };
/// A prefill chunk: `consumed` (>= 1) tokens processed; the last token's logits when it
/// consumed all of them.
pub const Chunk = struct { consumed: usize, logits: ?[]const f32 };

pub const Stats = struct {
    batches: u64 = 0,
    batch_rows: u64 = 0,
    prefill_chunks: u64 = 0,
    /// Batches that ran before every decoding slot had submitted (`gather` expired).
    partial_batches: u64 = 0,
    /// Time inside backend calls (decode batches, prefill chunks and resets).
    batch_ns: u64 = 0,
    prefill_ns: u64 = 0,
    /// From the first operation to the last completion.
    first_ns: i96 = 0,
    last_ns: i96 = 0,
    /// Batches by row count (index = rows, up to `max_slots`).
    sizes: [max_slots + 1]u64 = @splat(0),
};

/// `Backend` is called by the scheduler task only, never concurrently:
///   reset(slot: u32) !void
///   prefillChunk(slot: u32, tokens: []const u32) !Chunk
///   decodeBatch(rows: []const Row) ![]const f32   (rows.len logits rows of `vocab` floats)
pub fn Batcher(comptime Backend: type) type {
    return struct {
        const Self = @This();
        pub const Options = struct {
            slots: u32,
            vocab: usize,
            gather: std.Io.Duration = .fromMilliseconds(2),
        };
        const Op = enum { none, reset, prefill, step };
        const Hold = enum { none, row, prefill };
        const Slot = struct {
            used: bool = false,
            /// Left while its operation ran: freed when that completes.
            closing: bool = false,
            /// After a completed prefill, until reset or leave: counted by the gather rule.
            decoding: bool = false,
            op: Op = .none,
            running: bool = false,
            order: u64 = 0,
            tokens: []const u32 = &.{},
            done: usize = 0,
            token: u32 = 0,
            /// Futex word: bumped when the slot's operation completes.
            status: std.atomic.Value(u32) = .init(0),
            result: []const f32 = &.{},
            err: ?anyerror = null,
            hold: Hold = .none,
        };

        io: std.Io,
        backend: Backend,
        options: Options,
        mutex: std.Io.Mutex = .init,
        /// Scheduler wake word: bumped on every submission, release and leave.
        wake: std.atomic.Value(u32) = .init(0),
        slot: [max_slots]Slot = @splat(.{}),
        stopping: bool = false,
        held_rows: u32 = 0,
        prefill_held: bool = false,
        /// When the last held row was released (the gather window starts).
        released_ns: i96 = 0,
        arrivals: u64 = 0,
        last_batch: bool = false,
        stats: Stats = .{},

        pub fn init(io: std.Io, backend: Backend, options: Options) error{InvalidSlots}!Self {
            if (options.slots == 0 or options.slots > max_slots) return error.InvalidSlots;
            return .{ .io = io, .backend = backend, .options = options };
        }

        /// A free slot for a new generation; `leave` returns it.
        pub fn join(self: *Self) error{ NoSlot, Canceled }!u32 {
            try self.mutex.lock(self.io);
            defer self.mutex.unlock(self.io);
            for (self.slot[0..self.options.slots], 0..) |*s, i| if (!s.used) {
                s.used = true;
                s.closing = false;
                s.decoding = false;
                s.op = .none;
                s.hold = .none;
                return @intCast(i);
            };
            return error.NoSlot;
        }

        /// End a generation's use of `slot` (after an error too): releases its hold and
        /// withdraws a waiting operation; a running one completes first.
        pub fn leave(self: *Self, slot: u32) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            const s = &self.slot[slot];
            self.release(s);
            s.decoding = false;
            if (s.running) s.closing = true else self.free(s);
            self.poke();
        }

        pub fn reset(self: *Self, slot: u32) anyerror!void {
            _ = try self.submit(slot, .reset, &.{}, 0);
        }
        /// Borrowed logits of the last prompt token (until `sampled`).
        pub fn prefill(self: *Self, slot: u32, tokens: []const u32) anyerror![]const f32 {
            if (tokens.len == 0) return error.InvalidToken;
            return self.submit(slot, .prefill, tokens, 0);
        }
        /// Borrowed logits after `token` (until `sampled`).
        pub fn step(self: *Self, slot: u32, token: u32) anyerror![]const f32 {
            return self.submit(slot, .step, &.{}, token);
        }
        /// The generation is done with the logits last returned to `slot`.
        pub fn sampled(self: *Self, slot: u32) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            self.release(&self.slot[slot]);
            self.poke();
        }

        /// Ask `run` to return (after the operation it is running, if any).
        pub fn stop(self: *Self) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            self.stopping = true;
            self.poke();
        }

        fn submit(self: *Self, slot: u32, op: Op, tokens: []const u32, token: u32) anyerror![]const f32 {
            const s = &self.slot[slot];
            try self.mutex.lock(self.io);
            if (!s.used or s.op != .none or self.stopping) {
                self.mutex.unlock(self.io);
                return error.InvalidState;
            }
            self.release(s); // a new operation implies the previous logits were consumed
            s.op = op;
            s.tokens = tokens;
            s.done = 0;
            s.token = token;
            s.err = null;
            s.result = &.{};
            if (op != .step) {
                s.decoding = false;
                s.order = self.arrivals;
                self.arrivals += 1;
            }
            const ticket = s.status.load(.acquire);
            self.poke();
            self.mutex.unlock(self.io);
            while (s.status.load(.acquire) == ticket) {
                self.io.futexWait(u32, &s.status.raw, ticket) catch |e| {
                    self.withdraw(s);
                    return e;
                };
            }
            if (s.err) |e| return e;
            return s.result;
        }

        /// A canceled waiter: an operation not yet taken is dropped.
        fn withdraw(self: *Self, s: *Slot) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (!s.running) s.op = .none;
        }

        fn release(self: *Self, s: *Slot) void {
            switch (s.hold) {
                .none => return,
                .row => {
                    self.held_rows -= 1;
                    if (self.held_rows == 0) self.released_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
                },
                .prefill => self.prefill_held = false,
            }
            s.hold = .none;
        }

        fn free(_: *Self, s: *Slot) void {
            s.used = false;
            s.closing = false;
            s.decoding = false;
            s.op = .none;
            s.hold = .none;
            s.tokens = &.{};
        }

        fn now(self: *Self) i96 {
            return std.Io.Clock.awake.now(self.io).nanoseconds;
        }
        fn account(self: *Self, total: *u64, t0: i96, t1: i96) void {
            total.* += @intCast(t1 - t0);
            if (self.stats.first_ns == 0) self.stats.first_ns = t0;
            self.stats.last_ns = t1;
        }

        fn poke(self: *Self) void {
            _ = self.wake.fetchAdd(1, .release);
            self.io.futexWake(u32, &self.wake.raw, 1);
        }

        /// Called with the lock held, after the scheduler wrote the result.
        fn complete(self: *Self, s: *Slot) void {
            s.running = false;
            if (s.closing) {
                self.release(s);
                self.free(s);
            }
            _ = s.status.fetchAdd(1, .release);
            self.io.futexWake(u32, &s.status.raw, 1);
        }

        /// The scheduler task: runs operations until `stop`.
        pub fn run(self: *Self) void {
            var rows: [max_slots]Row = undefined;
            while (true) {
                self.mutex.lockUncancelable(self.io);
                if (self.stopping) {
                    self.mutex.unlock(self.io);
                    return;
                }
                const seen = self.wake.load(.acquire);
                var pre: ?u32 = null;
                var n_steps: u32 = 0;
                var n_decoding: u32 = 0;
                for (self.slot[0..self.options.slots], 0..) |*s, i| {
                    if (!s.used or s.closing) continue;
                    if (s.decoding) n_decoding += 1;
                    switch (s.op) {
                        .reset, .prefill => if (pre == null or s.order < self.slot[pre.?].order) {
                            pre = @intCast(i);
                        },
                        .step => n_steps += 1,
                        .none => {},
                    }
                }
                const prefill_ready = if (pre) |i| self.slot[i].op == .reset or !self.prefill_held else false;
                // Nanoseconds the batch still waits for missing slots (null: not ready yet).
                var wait_ns: ?i96 = null;
                if (n_steps > 0 and self.held_rows == 0) {
                    const left = self.options.gather.nanoseconds - (std.Io.Clock.awake.now(self.io).nanoseconds - self.released_ns);
                    wait_ns = if (n_steps >= n_decoding) 0 else @max(left, 0);
                }
                const batch_ready = if (wait_ns) |w| w == 0 else false;
                if (!prefill_ready and !batch_ready) {
                    self.mutex.unlock(self.io);
                    const timeout: std.Io.Timeout = if (wait_ns) |w| .{ .duration = .{ .raw = .{ .nanoseconds = w }, .clock = .awake } } else .none;
                    self.io.futexWaitTimeout(u32, &self.wake.raw, seen, timeout) catch {};
                    continue;
                }
                if (batch_ready and (!prefill_ready or !self.last_batch)) {
                    var n: usize = 0;
                    for (self.slot[0..self.options.slots], 0..) |*s, i| if (s.used and !s.closing and s.op == .step) {
                        rows[n] = .{ .slot = @intCast(i), .token = s.token };
                        s.running = true;
                        n += 1;
                    };
                    self.stats.batches += 1;
                    self.stats.batch_rows += n;
                    if (n < n_decoding) self.stats.partial_batches += 1;
                    self.last_batch = true;
                    self.stats.sizes[n] += 1;
                    self.mutex.unlock(self.io);
                    const t0 = self.now();
                    const out = self.backend.decodeBatch(rows[0..n]);
                    const t1 = self.now();
                    self.mutex.lockUncancelable(self.io);
                    self.account(&self.stats.batch_ns, t0, t1);
                    for (rows[0..n], 0..) |row, r| {
                        const s = &self.slot[row.slot];
                        s.op = .none;
                        if (out) |logits| {
                            s.result = logits[r * self.options.vocab ..][0..self.options.vocab];
                            s.hold = .row;
                            self.held_rows += 1;
                        } else |e| s.err = e;
                        self.complete(s);
                    }
                } else {
                    const s = &self.slot[pre.?];
                    s.running = true;
                    self.last_batch = false;
                    self.mutex.unlock(self.io);
                    const t0 = self.now();
                    if (s.op == .reset) {
                        const out = self.backend.reset(pre.?);
                        const t1 = self.now();
                        self.mutex.lockUncancelable(self.io);
                        self.account(&self.stats.prefill_ns, t0, t1);
                        s.op = .none;
                        if (out) |_| {} else |e| s.err = e;
                        self.complete(s);
                    } else {
                        const out = self.backend.prefillChunk(pre.?, s.tokens[s.done..]);
                        const t1 = self.now();
                        self.mutex.lockUncancelable(self.io);
                        self.account(&self.stats.prefill_ns, t0, t1);
                        self.stats.prefill_chunks += 1;
                        if (out) |chunk| {
                            s.done += chunk.consumed;
                            if (s.done == s.tokens.len) {
                                s.op = .none;
                                s.result = chunk.logits orelse &.{};
                                if (chunk.logits == null) s.err = error.MissingLogits else {
                                    s.hold = .prefill;
                                    self.prefill_held = true;
                                    s.decoding = true;
                                }
                                self.complete(s);
                            } else if (chunk.consumed == 0 or s.done > s.tokens.len) {
                                s.op = .none;
                                s.err = error.InvalidChunk;
                                self.complete(s);
                            } else {
                                // More chunks: the operation stays queued (still the oldest).
                                s.running = false;
                                if (s.closing) self.complete(s);
                            }
                        } else |e| {
                            s.op = .none;
                            s.err = e;
                            self.complete(s);
                        }
                    }
                }
                self.mutex.unlock(self.io);
            }
        }
    };
}

// Host tests with a fake backend (tests/batcher.zig).
