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
//! was released (a slow client never holds the GPU). Prefill runs in units the backend
//! defines (a chunk, or one segment of a chunk; docs/specs/concurrent.md "18c.2 design").
//! When both are ready, `stall` decides: a batch preempts once `stall.ns` have passed since
//! the last batch ended (at a unit boundary), or, with `.chunk`, only between chunks
//! (alternating). A chunk in flight is finished before another prompt starts; between
//! chunks the next prompt is the one with the fewest remaining tokens (`order`).
const std = @import("std");

pub const max_slots = 64;
pub const Row = struct { slot: u32, token: u32 };
/// A prefill unit: `consumed` tokens processed (0: a segment of a chunk still in flight);
/// the last token's logits when it consumed all of them.
pub const Chunk = struct { consumed: usize, logits: ?[]const f32 };
/// When a waiting decode step preempts a prefill: `ns` after the last batch ended, at the
/// next prefill unit boundary; `chunk`: only between chunks, alternating with them.
pub const Stall = union(enum) { chunk, ns: u64 };
pub const Order = enum { shortest, fifo };

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
    /// Prefill units (segments) run, batches run inside a chunk, aborted chunks.
    prefill_units: u64 = 0,
    batches_in_chunk: u64 = 0,
    aborted_chunks: u64 = 0,
};

/// `Backend` is called by the scheduler task only, never concurrently:
///   reset(slot: u32) !void
///   prefillChunk(slot: u32, tokens: []const u32) !Chunk   (one unit; while it returns
///       consumed 0 the next calls continue that chunk, with the remaining tokens)
///   abortChunk() void   (drop the chunk in flight; its slot is reset before reuse)
///   checkRow(row: Row) !void   (can this row run now? a failing row gets its own error,
///       the other rows of the batch run)
///   decodeBatch(rows: []const Row) ![]const f32   (rows.len logits rows of `vocab` floats)
///
/// Ownership: an operation's `tokens` belong to the generation that submitted it and are
/// read only while it waits in `submit`; a canceled waiter whose operation is running waits
/// until the running unit ends (the operation is then dropped), so nothing reads its tokens
/// after `submit` returns.
pub fn Batcher(comptime Backend: type) type {
    return struct {
        const Self = @This();
        pub const Options = struct {
            slots: u32,
            vocab: usize,
            gather: std.Io.Duration = .fromMilliseconds(2),
            stall: Stall = .chunk,
            order: Order = .fifo,
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
            /// Its waiter was canceled while the operation ran: drop it after the running unit.
            canceled: bool = false,
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
        last_batch_end: i96 = 0,
        /// Slot whose chunk is in flight (between prefill units).
        in_chunk: ?u32 = null,
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
            s.canceled = false;
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
                    if (self.withdraw(s)) return e;
                    // Running: the scheduler may be reading `tokens`; it drops the
                    // operation when the unit ends (one prefill segment or one batch).
                    while (s.status.load(.acquire) == ticket) self.io.futexWaitUncancelable(u32, &s.status.raw, ticket);
                    return e;
                };
            }
            if (s.err) |e| return e;
            return s.result;
        }

        /// A canceled waiter: an operation not yet taken is dropped (true); a running one is
        /// marked, and the scheduler completes it as canceled when the unit ends (false).
        fn withdraw(self: *Self, s: *Slot) bool {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (!s.running) {
                s.op = .none;
                return true;
            }
            s.canceled = true;
            return false;
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

        /// Prefill order between chunks: resets first (they are cheap), then the fewest
        /// remaining tokens (`.shortest`) or arrival (`.fifo`); ties by arrival.
        fn before(self: *const Self, a: *const Slot, b: *const Slot) bool {
            const ka: usize = if (a.op == .reset or self.options.order == .fifo) 0 else a.tokens.len - a.done;
            const kb: usize = if (b.op == .reset or self.options.order == .fifo) 0 else b.tokens.len - b.done;
            if (ka != kb) return ka < kb;
            return a.order < b.order;
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
                if (self.in_chunk) |ic| {
                    const s = &self.slot[ic];
                    if (!s.used or s.closing or s.op != .prefill) {
                        // Its generation went away between two units: drop the chunk.
                        self.in_chunk = null;
                        self.stats.aborted_chunks += 1;
                        self.mutex.unlock(self.io);
                        self.backend.abortChunk();
                        continue;
                    }
                }
                var pre: ?u32 = self.in_chunk;
                var n_steps: u32 = 0;
                var n_decoding: u32 = 0;
                for (self.slot[0..self.options.slots], 0..) |*s, i| {
                    if (!s.used or s.closing) continue;
                    if (s.decoding) n_decoding += 1;
                    switch (s.op) {
                        .reset, .prefill => if (self.in_chunk == null and (pre == null or self.before(s, &self.slot[pre.?]))) {
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
                const preempt = switch (self.options.stall) {
                    .chunk => self.in_chunk == null,
                    .ns => |ns| std.Io.Clock.awake.now(self.io).nanoseconds - self.last_batch_end >= ns,
                };
                if (batch_ready and (!prefill_ready or (!self.last_batch and preempt))) {
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
                    if (self.in_chunk != null) self.stats.batches_in_chunk += 1;
                    self.mutex.unlock(self.io);
                    // A row that cannot run fails alone; the others still run.
                    var good: [max_slots]Row = undefined;
                    var index: [max_slots]?usize = @splat(null);
                    var bad: [max_slots]anyerror = undefined;
                    var k: usize = 0;
                    for (rows[0..n], 0..) |row, r| {
                        if (self.backend.checkRow(row)) |_| {
                            good[k] = row;
                            index[r] = k;
                            k += 1;
                        } else |e| bad[r] = e;
                    }
                    const t0 = self.now();
                    const out: anyerror![]const f32 = if (k > 0) self.backend.decodeBatch(good[0..k]) else &[_]f32{};
                    const t1 = self.now();
                    self.mutex.lockUncancelable(self.io);
                    self.account(&self.stats.batch_ns, t0, t1);
                    self.last_batch_end = t1;
                    for (rows[0..n], 0..) |row, r| {
                        const s = &self.slot[row.slot];
                        s.op = .none;
                        if (index[r]) |g| {
                            if (out) |logits| {
                                s.result = logits[g * self.options.vocab ..][0..self.options.vocab];
                                s.hold = .row;
                                self.held_rows += 1;
                            } else |e| s.err = e;
                        } else s.err = bad[r];
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
                        self.stats.prefill_units += 1;
                        if (out) |chunk| {
                            if (chunk.consumed > 0) self.stats.prefill_chunks += 1;
                            self.in_chunk = if (chunk.consumed == 0) pre.? else null;
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
                            } else if (s.done > s.tokens.len) {
                                s.op = .none;
                                s.err = error.InvalidChunk;
                                self.complete(s);
                            } else if (s.canceled or s.closing) {
                                // Its generation is gone: drop the prompt (a chunk in flight is
                                // aborted by the next loop).
                                s.op = .none;
                                s.err = error.Canceled;
                                self.complete(s);
                            } else {
                                // More units: the operation stays queued.
                                s.running = false;
                            }
                        } else |e| {
                            // A failed unit leaves no chunk in flight (the backend drops it).
                            self.in_chunk = null;
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
