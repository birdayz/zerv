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
//! chunks the next prompt is the one with the fewest remaining tokens (`order`), and up to
//! `pack` pending prompts that the backend can pack with it run their next chunks together
//! (docs/specs/concurrent.md, "18d.1 design").
const std = @import("std");

pub const max_slots = 64;
pub const Row = struct { slot: u32, token: u32 };
/// Sequences in one packed chunk at most.
pub const max_pack = 8;
/// A sequence of a prefill unit: its slot and its prompt tokens not yet consumed.
pub const Item = struct { slot: u32, tokens: []const u32 };
/// A prefill unit's result: `done` when its chunk ended; then item i consumed `consumed[i]`
/// tokens, and `logits[i * vocab ..]` are its last token's logits (meaningful only for an
/// item whose prompt is complete).
pub const Unit = struct { done: bool, consumed: [max_pack]usize = @splat(0), logits: ?[]const f32 = null };
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
    /// Prefill units (segments) run, batches run inside a chunk, aborted chunks, chunks
    /// holding several prompts.
    packed_chunks: u64 = 0,
    prefill_units: u64 = 0,
    batches_in_chunk: u64 = 0,
    aborted_chunks: u64 = 0,
    /// Times a prompt had to wait for memory (`admit` false).
    admission_waits: u64 = 0,
    /// Sequences swapped out to host memory and back (docs/specs/concurrent.md, "18d.3
    /// design"), the time in those calls, and rows failed because nothing could be swapped.
    swap_outs: u64 = 0,
    swap_ins: u64 = 0,
    swap_ns: u64 = 0,
    swap_failures: u64 = 0,
    /// Sequences swapped out by the time slice (not by memory pressure).
    slice_swaps: u64 = 0,
};

/// `Backend` is called by the scheduler task only, never concurrently:
///   reset(slot: u32) !void
///   begin(slot: u32, prompt: []const u32) !u32   (start a prompt from the longest prefix
///       checkpoint or a reset; returns the start position; docs/specs/concurrent.md
///       "18d.4 design")
///   checkpoint(slot: u32, prefix: []const u32) !void   (keep the slot's state after
///       `prefix`, its current position, as a checkpoint)
///   packFits(remaining: []const usize) bool   (can these prompts' next chunks run as one
///       packed chunk? remaining.len >= 1)
///   prefillUnit(items: []const Item) !Unit   (starts a chunk holding each item's next chunk
///       and runs its first unit; with no items, runs the next unit of the chunk in flight;
///       the tokens are read only by the starting call)
///   abortChunk() void   (drop the chunk in flight; its slots are reset before reuse)
///   admit(slot: u32, tokens: usize) bool   (make room for `tokens` positions of the slot's
///       sequence before its prompt starts; false: not now, the prompt waits)
///   release(slot: u32) void   (a freed slot's memory goes back to the pool, host copy too)
///   grow(slot: u32) !bool   (memory for the slot's next decode position; false: the pool
///       is exhausted; docs/specs/concurrent.md "18d.3 design")
///   swapOut(slot: u32) !bool   (copy the sequence to host memory and free its pool memory;
///       false: no room there; its state is unchanged either way)
///   swapIn(slot: u32, spare: u32) !bool   (bring it back when the pool has its memory plus
///       `spare` pages more; false: not now)
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
            /// Prompts per packed chunk at most (1: one prompt at a time).
            pack: u32 = 1,
            /// Time slice of swapped-out sequences (docs/specs/concurrent.md, "18d.3
            /// design"): once the longest-waiting one has waited this long and cannot come
            /// back, the sequence that has run longest since it last came in is swapped out
            /// for it (null: a swapped sequence waits until memory is released).
            swap_slice: ?std.Io.Duration = null,
        };
        /// The chunk in flight: its slots (running until it ends) and which of them lost
        /// their generation meanwhile (completed as canceled; freed when the chunk ends).
        const Pack = struct { n: u32, slots: [max_pack]u32, gone: [max_pack]bool };
        const Op = enum { none, reset, begin, checkpoint, prefill, step };
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
            /// Positions the sequence may reach (prompt + output limit; 0: its prompt only),
            /// and whether the backend admitted it (`admit`); a failed admission waits for
            /// a release (`admit_epoch`).
            reserve: usize = 0,
            /// Positions processed before its current prompt op (a restored prefix and earlier
            /// segments): admission covers `base` + the op's tokens.
            base: usize = 0,
            /// A `begin`'s start position.
            value: u32 = 0,
            admitted: bool = false,
            failed_epoch: u64 = std.math.maxInt(u64),
            /// Swapped out to host memory (its step waits until `swapIn`); the release epoch
            /// of its last failed swap-in.
            swapped: bool = false,
            swap_epoch: u64 = std.math.maxInt(u64),
            /// When it was last swapped out, and when it last started running (admitted or
            /// swapped in): the time slice orders by these.
            swapped_ns: i96 = 0,
            resumed_ns: i96 = 0,
            /// When its current reset or prompt was submitted (the wait order of prompts).
            wait_ns: i96 = 0,
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
        /// Completed prompts whose logits are not yet sampled (a new chunk would overwrite them).
        prefill_holds: u32 = 0,
        /// When the last held row was released (the gather window starts).
        released_ns: i96 = 0,
        arrivals: u64 = 0,
        last_batch: bool = false,
        last_batch_end: i96 = 0,
        /// The chunk in flight (between prefill units).
        pack: ?Pack = null,
        /// Slots freed since the scheduler last ran `release` for them.
        to_release: u64 = 0,
        /// Bumped whenever memory is released: prompts that failed admission retry.
        admit_epoch: u64 = 0,
        /// Arrival order of the oldest prompt that failed admission: newer prompts are not
        /// admitted before it (no starvation of large prompts).
        blocked: ?u64 = null,
        /// Slots swapped out: while any is, no new prompt is admitted.
        swapped_slots: u32 = 0,
        /// When the time slice is next due (0: not waiting for one).
        slice_wait_ns: i96 = 0,
        /// Since when the longest-waiting swapped sequence waits (valid while any is
        /// swapped): prompts queued later are not admitted before it.
        first_swapped_ns: i96 = 0,
        stats: Stats = .{},

        pub fn init(io: std.Io, backend: Backend, options: Options) error{InvalidSlots}!Self {
            if (options.slots == 0 or options.slots > max_slots) return error.InvalidSlots;
            if (options.pack == 0 or options.pack > max_pack) return error.InvalidSlots;
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
        /// Start `prompt` (instead of `reset`) from the longest prefix checkpoint; the
        /// position its prefill continues at (0: reset). `prompt` is read during the call.
        pub fn begin(self: *Self, slot: u32, prompt: []const u32) anyerror!u32 {
            if (prompt.len == 0) return error.InvalidToken;
            _ = try self.submit(slot, .begin, prompt, 0);
            return self.slot[slot].value;
        }
        /// Keep the slot's state after `prefix` (all it has processed) as a checkpoint.
        pub fn checkpoint(self: *Self, slot: u32, prefix: []const u32) anyerror!void {
            if (prefix.len == 0) return error.InvalidToken;
            _ = try self.submit(slot, .checkpoint, prefix, 0);
        }
        /// Borrowed logits of the last prompt token (until `sampled`).
        pub fn prefill(self: *Self, slot: u32, tokens: []const u32) anyerror![]const f32 {
            if (tokens.len == 0) return error.InvalidToken;
            return self.submit(slot, .prefill, tokens, 0);
        }
        /// The positions `slot`'s sequence may reach (prompt plus output limit): memory the
        /// backend admits before the next prompt starts (`admit`).
        pub fn reserve(self: *Self, slot: u32, tokens: usize) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            self.slot[slot].reserve = tokens;
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

        /// Ask `run` to return (after the operation it is running, if any, and after returning
        /// the memory of slots already left).
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
            // A prompt segment needs memory for its own positions (a prompt split at checkpoint
            // points is several ops; each is admitted: `base` is where it starts).
            if (op == .prefill) s.admitted = false;
            if (op != .step) {
                s.decoding = false;
                s.wait_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
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
            self.poke(); // a chunk member is completed at the next unit boundary
            return false;
        }

        fn release(self: *Self, s: *Slot) void {
            switch (s.hold) {
                .none => return,
                .row => {
                    self.held_rows -= 1;
                    if (self.held_rows == 0) self.released_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
                },
                .prefill => self.prefill_holds -= 1,
            }
            s.hold = .none;
        }

        fn free(self: *Self, s: *Slot) void {
            const i = (@intFromPtr(s) - @intFromPtr(&self.slot[0])) / @sizeOf(Slot);
            self.to_release |= @as(u64, 1) << @intCast(i);
            if (s.admitted or (self.blocked != null and s.order == self.blocked.?)) self.blocked = null;
            if (s.swapped) self.swapped_slots -= 1; // `release` drops its host copy
            s.swapped = false;
            s.swap_epoch = std.math.maxInt(u64);
            s.admitted = false;
            s.reserve = 0;
            s.base = 0;
            s.failed_epoch = std.math.maxInt(u64);
            s.used = false;
            s.closing = false;
            s.decoding = false;
            s.op = .none;
            s.hold = .none;
            s.tokens = &.{};
        }

        /// Whether a pending reset or prompt may be chosen now: resets always; a prompt when
        /// admitted, or when it has not failed admission since the last release and no older
        /// prompt is waiting for memory.
        fn eligible(self: *const Self, s: *const Slot) bool {
            if (s.op == .reset or s.op == .begin or s.op == .checkpoint) return true;
            // A prompt swapped out while it waited comes back through `swapIn` first.
            if (s.swapped) return false;
            if (s.admitted) return true;
            if (s.failed_epoch == self.admit_epoch) return false;
            if (self.swapped_slots > 0 and self.first_swapped_ns <= s.wait_ns) return false;
            return self.blocked == null or s.order <= self.blocked.?;
        }

        /// Prefill order between chunks: resets first (they are cheap), then the fewest
        /// remaining tokens (`.shortest`) or arrival (`.fifo`); ties by arrival.
        fn before(self: *const Self, a: *const Slot, b: *const Slot) bool {
            const ka: usize = if (a.op != .prefill or self.options.order == .fifo) 0 else a.tokens.len - a.done;
            const kb: usize = if (b.op != .prefill or self.options.order == .fifo) 0 else b.tokens.len - b.done;
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

        /// A chunk member whose generation went away: its waiter is told (canceled); the slot
        /// stays running (so it is not reused) until the chunk ends.
        fn dropMember(self: *Self, s: *Slot) void {
            s.op = .none;
            s.err = error.Canceled;
            _ = s.status.fetchAdd(1, .release);
            self.io.futexWake(u32, &s.status.raw, 1);
        }

        /// The chunk ended (or failed): release its members.
        fn endPack(self: *Self, pk: Pack, unit: anyerror!Unit) void {
            for (pk.slots[0..pk.n], 0..) |slot, i| {
                const s = &self.slot[slot];
                if (pk.gone[i]) {
                    s.running = false;
                    if (s.closing) {
                        self.release(s);
                        self.free(s);
                    }
                    continue;
                }
                const u = unit catch |e| {
                    s.op = .none;
                    s.err = e;
                    self.complete(s);
                    continue;
                };
                s.done += u.consumed[i];
                if (s.done == s.tokens.len) {
                    s.op = .none;
                    s.base += s.tokens.len;
                    if (u.logits) |l| {
                        s.result = l[i * self.options.vocab ..][0..self.options.vocab];
                        s.hold = .prefill;
                        self.prefill_holds += 1;
                        s.decoding = true;
                    } else s.err = error.MissingLogits;
                    self.complete(s);
                } else if (s.done > s.tokens.len or u.consumed[i] == 0) {
                    s.op = .none;
                    s.err = error.InvalidChunk;
                    self.complete(s);
                } else if (s.canceled) {
                    // More chunks, but the waiter was canceled during this unit: complete it
                    // now; the rest of its prompt is not run (and its tokens not read).
                    s.op = .none;
                    s.err = error.Canceled;
                    self.complete(s);
                } else {
                    // More chunks: the operation stays queued.
                    s.running = false;
                }
            }
        }

        /// The scheduler task: runs operations until `stop`.
        pub fn run(self: *Self) void {
            var rows: [max_slots]Row = undefined;
            while (true) {
                self.mutex.lockUncancelable(self.io);
                const seen = self.wake.load(.acquire);
                if (self.to_release != 0) {
                    // Freed slots' memory back to the pool (before any new admission, and
                    // before stopping: a slot left just before `stop` returns its memory).
                    const bits = self.to_release;
                    self.to_release = 0;
                    self.admit_epoch += 1;
                    self.mutex.unlock(self.io);
                    for (0..self.options.slots) |i| if (bits & (@as(u64, 1) << @intCast(i)) != 0) self.backend.release(@intCast(i));
                    continue;
                }
                if (self.stopping) {
                    self.mutex.unlock(self.io);
                    return;
                }
                if (self.swapped_slots > 0 or (self.options.swap_slice != null and self.blocked != null)) swap: {
                    // Waiters in order of how long they wait: swapped sequences (since their
                    // swap-out) and prompts that failed admission (since they were queued).
                    // A swapped one comes back with a spare page per running decoding
                    // sequence (not evicted again at its next page).
                    var oldest: ?usize = null;
                    var first_swapped: ?i96 = null;
                    var first_prompt: ?i96 = null;
                    var spare: u32 = 0;
                    for (self.slot[0..self.options.slots], 0..) |*s, i| {
                        if (!s.used or s.closing) continue;
                        if (s.swapped and !s.running) {
                            if (first_swapped == null or s.swapped_ns < first_swapped.?) first_swapped = s.swapped_ns;
                            if (s.swap_epoch != self.admit_epoch and (oldest == null or s.swapped_ns < self.slot[oldest.?].swapped_ns)) oldest = i;
                        }
                        if (s.op == .prefill and !s.admitted and !s.running and !s.swapped and s.failed_epoch != std.math.maxInt(u64)) {
                            if (first_prompt == null or s.wait_ns < first_prompt.?) first_prompt = s.wait_ns;
                        }
                        if (s.decoding and !s.swapped) spare += 1;
                    }
                    self.first_swapped_ns = first_swapped orelse 0;
                    // A prompt that has waited longer goes first (it is eligible then).
                    if (oldest != null and first_prompt != null and first_prompt.? < self.slot[oldest.?].swapped_ns) oldest = null;
                    if (oldest == null) {
                        // Time slice: the longest waiter waited a slice; the sequence that has
                        // run longest since it came in (at least a slice) makes room.
                        const slice = self.options.swap_slice orelse break :swap;
                        const since = @min(first_swapped orelse std.math.maxInt(i96), first_prompt orelse std.math.maxInt(i96));
                        if (since == std.math.maxInt(i96)) break :swap;
                        const t = self.now();
                        if (t - since < slice.nanoseconds) {
                            self.slice_wait_ns = since + slice.nanoseconds;
                            break :swap;
                        }
                        var victim: ?usize = null;
                        for (self.slot[0..self.options.slots], 0..) |*v, j| {
                            // Only a sequence whose next step waits (a finished one is about
                            // to leave; swapping it would be a wasted copy).
                            if (!v.used or v.closing or v.swapped or v.running or !v.decoding or v.op != .step) continue;
                            if (t - v.resumed_ns < slice.nanoseconds) continue;
                            if (victim == null or v.resumed_ns < self.slot[victim.?].resumed_ns) victim = j;
                        }
                        if (victim == null) {
                            // No decoding sequence to swap: a prompt that waits for memory
                            // while holding some (a restored prefix or its earlier segments)
                            // makes room, the most recent such waiter, never the longest one
                            // (otherwise prompts holding pages could wait on each other
                            // forever).
                            for (self.slot[0..self.options.slots], 0..) |*v, j| {
                                if (!v.used or v.closing or v.swapped or v.running or v.admitted or v.op != .prefill or v.base == 0) continue;
                                if (v.wait_ns == since or t - v.resumed_ns < slice.nanoseconds) continue;
                                if (victim == null or v.wait_ns > self.slot[victim.?].wait_ns) victim = j;
                            }
                        }
                        const j = victim orelse {
                            self.slice_wait_ns = t + slice.nanoseconds; // try again later
                            break :swap;
                        };
                        const v = &self.slot[j];
                        v.running = true;
                        self.mutex.unlock(self.io);
                        const t0 = self.now();
                        const out = self.backend.swapOut(@intCast(j));
                        const t1 = self.now();
                        self.mutex.lockUncancelable(self.io);
                        v.running = false;
                        if (out) |ok| {
                            if (ok) {
                                v.swapped = true;
                                v.swapped_ns = t1;
                                self.swapped_slots += 1;
                                self.stats.swap_outs += 1;
                                self.stats.slice_swaps += 1;
                                self.account(&self.stats.swap_ns, t0, t1);
                                self.admit_epoch += 1; // its memory is free: swap-ins retry
                            } else v.resumed_ns = t1; // no room on the host: next slice
                        } else |_| v.resumed_ns = t1;
                        if (v.closing) {
                            self.release(v);
                            self.free(v);
                        }
                        self.mutex.unlock(self.io);
                        continue;
                    }
                    const i = oldest.?;
                    const s = &self.slot[i];
                    s.running = true;
                    self.mutex.unlock(self.io);
                    const t0 = self.now();
                    const in = self.backend.swapIn(@intCast(i), spare);
                    const t1 = self.now();
                    self.mutex.lockUncancelable(self.io);
                    s.running = false;
                    if (in) |ok| {
                        if (ok) {
                            s.swapped = false;
                            s.resumed_ns = t1;
                            self.swapped_slots -= 1;
                            self.stats.swap_ins += 1;
                            self.account(&self.stats.swap_ns, t0, t1);
                        } else s.swap_epoch = self.admit_epoch; // retried after a release
                    } else |_| {
                        // Not swapped back: its next step fails in `grow` (the backend
                        // still holds it as swapped out).
                        s.swapped = false;
                        self.swapped_slots -= 1;
                    }
                    if (s.closing) {
                        self.release(s);
                        self.free(s);
                    }
                    self.mutex.unlock(self.io);
                    continue;
                }
                if (self.pack) |*pk| {
                    var live: u32 = 0;
                    for (pk.slots[0..pk.n], 0..) |slot, i| {
                        const s = &self.slot[slot];
                        if (!pk.gone[i] and (s.canceled or s.closing)) {
                            pk.gone[i] = true;
                            self.dropMember(s);
                        }
                        if (!pk.gone[i]) live += 1;
                    }
                    if (live == 0) {
                        // Every generation of the chunk went away: drop it.
                        const done = pk.*;
                        self.pack = null;
                        self.stats.aborted_chunks += 1;
                        self.endPack(done, error.Canceled);
                        self.mutex.unlock(self.io);
                        self.backend.abortChunk();
                        continue;
                    }
                }
                var pre: ?u32 = null;
                var n_steps: u32 = 0;
                var n_decoding: u32 = 0;
                for (self.slot[0..self.options.slots], 0..) |*s, i| {
                    if (!s.used or s.closing) continue;
                    if (s.swapped) continue; // waits for `swapIn`
                    if (s.decoding) n_decoding += 1;
                    switch (s.op) {
                        .reset, .begin, .checkpoint, .prefill => if (!s.running and self.eligible(s) and (pre == null or self.before(s, &self.slot[pre.?]))) {
                            pre = @intCast(i);
                        },
                        .step => n_steps += 1,
                        .none => {},
                    }
                }
                const prefill_ready = self.pack != null or (if (pre) |i| self.slot[i].op != .prefill or self.prefill_holds == 0 else false);
                // Nanoseconds the batch still waits for missing slots (null: not ready yet).
                var wait_ns: ?i96 = null;
                if (n_steps > 0 and self.held_rows == 0) {
                    const left = self.options.gather.nanoseconds - (std.Io.Clock.awake.now(self.io).nanoseconds - self.released_ns);
                    wait_ns = if (n_steps >= n_decoding) 0 else @max(left, 0);
                }
                const batch_ready = if (wait_ns) |w| w == 0 else false;
                if (!prefill_ready and !batch_ready) {
                    // A pending time slice wakes the scheduler too.
                    if ((self.swapped_slots > 0 or self.blocked != null) and self.slice_wait_ns > 0) {
                        const left = @max(self.slice_wait_ns - self.now(), 0);
                        wait_ns = if (wait_ns) |w| @min(w, left) else left;
                    }
                    self.slice_wait_ns = 0;
                    self.mutex.unlock(self.io);
                    const timeout: std.Io.Timeout = if (wait_ns) |w| .{ .duration = .{ .raw = .{ .nanoseconds = w }, .clock = .awake } } else .none;
                    self.io.futexWaitTimeout(u32, &self.wake.raw, seen, timeout) catch {};
                    continue;
                }
                const preempt = switch (self.options.stall) {
                    .chunk => self.pack == null,
                    .ns => |ns| std.Io.Clock.awake.now(self.io).nanoseconds - self.last_batch_end >= ns,
                };
                if (batch_ready and (!prefill_ready or (!self.last_batch and preempt))) {
                    var n: usize = 0;
                    var ages: [max_slots]u64 = undefined;
                    for (self.slot[0..self.options.slots], 0..) |*s, i| if (s.used and !s.closing and !s.swapped and s.op == .step) {
                        rows[n] = .{ .slot = @intCast(i), .token = s.token };
                        ages[n] = s.order;
                        s.running = true;
                        n += 1;
                    };
                    self.stats.batches += 1;
                    self.stats.batch_rows += n;
                    if (n < n_decoding) self.stats.partial_batches += 1;
                    self.last_batch = true;
                    self.stats.sizes[n] += 1;
                    if (self.pack != null) self.stats.batches_in_chunk += 1;
                    self.mutex.unlock(self.io);
                    // A row that cannot run fails alone; the others still run.
                    var good: [max_slots]Row = undefined;
                    var index: [max_slots]?usize = @splat(null);
                    var bad: [max_slots]anyerror = undefined;
                    var ok: [max_slots]bool = @splat(true);
                    // Memory for each row's next position, oldest sequence first; when the
                    // pool is exhausted, the youngest row still in the batch is swapped out
                    // (itself, last). Rows are running: no other thread frees their slots.
                    var out_rows: [max_slots]bool = @splat(false);
                    var by_age: [max_slots]usize = undefined;
                    for (0..n) |r| {
                        var j = r;
                        while (j > 0 and ages[by_age[j - 1]] > ages[r]) : (j -= 1) by_age[j] = by_age[j - 1];
                        by_age[j] = r;
                    }
                    var swap_t: u64 = 0;
                    for (by_age[0..n]) |r| {
                        if (out_rows[r] or !ok[r]) continue;
                        while (true) {
                            const grown = self.backend.grow(rows[r].slot) catch |e| {
                                bad[r] = e;
                                ok[r] = false;
                                break;
                            };
                            if (grown) break;
                            var victim: usize = r;
                            for (by_age[0..n]) |v| if (!out_rows[v] and ok[v]) {
                                victim = v;
                            };
                            const t0 = self.now();
                            const swapped = self.backend.swapOut(rows[victim].slot) catch |e| {
                                bad[r] = e;
                                ok[r] = false;
                                break;
                            };
                            swap_t += @intCast(self.now() - t0);
                            if (!swapped) {
                                // No room in host memory: the victim fails (last resort) and
                                // its memory goes back now; the older rows keep running.
                                bad[victim] = error.PoolExhausted;
                                ok[victim] = false;
                                self.backend.release(rows[victim].slot);
                                if (victim == r) break;
                                continue;
                            }
                            out_rows[victim] = true;
                            if (victim == r) break;
                        }
                    }
                    var k: usize = 0;
                    for (rows[0..n], 0..) |row, r| {
                        if (out_rows[r] or !ok[r]) continue;
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
                    self.stats.swap_ns += swap_t;
                    self.last_batch_end = t1;
                    for (rows[0..n], 0..) |row, r| {
                        const s = &self.slot[row.slot];
                        if (out_rows[r]) {
                            // Swapped out: its step stays queued until `swapIn` (unless its
                            // generation went away meanwhile).
                            s.swapped = true;
                            s.swapped_ns = t1;
                            self.swapped_slots += 1;
                            self.stats.swap_outs += 1;
                            if (s.canceled or s.closing) {
                                s.op = .none;
                                s.err = error.Canceled;
                                self.complete(s);
                            } else s.running = false;
                            continue;
                        }
                        if (!ok[r] and bad[r] == error.PoolExhausted) self.stats.swap_failures += 1;
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
                } else if (self.pack == null and self.slot[pre.?].op != .prefill) {
                    // A reset, begin or checkpoint: host bookkeeping and short copies.
                    const s = &self.slot[pre.?];
                    const op = s.op;
                    s.running = true;
                    self.last_batch = false;
                    self.mutex.unlock(self.io);
                    const t0 = self.now();
                    var start: u32 = 0;
                    const out: anyerror!void = switch (op) {
                        .reset => self.backend.reset(pre.?),
                        .begin => if (self.backend.begin(pre.?, s.tokens)) |v| {
                            start = v;
                        } else |e| e,
                        .checkpoint => self.backend.checkpoint(pre.?, s.tokens),
                        else => unreachable,
                    };
                    const t1 = self.now();
                    self.mutex.lockUncancelable(self.io);
                    self.account(&self.stats.prefill_ns, t0, t1);
                    s.op = .none;
                    s.tokens = &.{};
                    if (op != .checkpoint) {
                        // A reset or begin returned the slot's memory to the pool (a begin
                        // maps the restored prefix again).
                        s.admitted = false;
                        s.base = start;
                        s.value = start;
                        self.admit_epoch += 1;
                    }
                    if (out) |_| {} else |e| s.err = e;
                    self.complete(s);
                } else {
                    // A prefill unit: the next one of the chunk in flight, or a new chunk of the
                    // first pending prompt plus those that pack with it (in `before` order).
                    var items: [max_pack]Item = undefined;
                    var n: usize = 0;
                    if (self.pack == null) {
                        var remaining: [max_pack]usize = undefined;
                        var taken: u64 = 0;
                        var next: ?u32 = pre;
                        while (next) |i| {
                            const s = &self.slot[i];
                            remaining[n] = s.tokens.len - s.done;
                            if (n > 0 and !self.backend.packFits(remaining[0 .. n + 1])) break;
                            // Memory for the whole sequence before its prompt starts (the
                            // backend call is host bookkeeping and a page-table copy).
                            if (!s.admitted) {
                                if (self.backend.admit(i, @max(s.reserve, s.base + s.tokens.len))) {
                                    s.admitted = true;
                                    s.resumed_ns = self.now();
                                    if (self.blocked != null and self.blocked.? == s.order) self.blocked = null;
                                } else {
                                    s.failed_epoch = self.admit_epoch;
                                    if (self.blocked == null or s.order < self.blocked.?) self.blocked = s.order;
                                    self.stats.admission_waits += 1;
                                    break;
                                }
                            }
                            items[n] = .{ .slot = i, .tokens = s.tokens[s.done..] };
                            taken |= @as(u64, 1) << @intCast(i);
                            n += 1;
                            if (n == self.options.pack) break;
                            next = null;
                            for (self.slot[0..self.options.slots], 0..) |*c, j| {
                                if (!c.used or c.closing or c.running or c.op != .prefill or !self.eligible(c) or taken & (@as(u64, 1) << @intCast(j)) != 0) continue;
                                if (next == null or self.before(c, &self.slot[next.?])) next = @intCast(j);
                            }
                        }
                        if (n == 0) {
                            // Its memory is not free yet: wait for a release (not eligible
                            // again until then).
                            self.mutex.unlock(self.io);
                            continue;
                        }
                        var pk: Pack = .{ .n = @intCast(n), .slots = undefined, .gone = @splat(false) };
                        for (items[0..n], 0..) |item, k| {
                            pk.slots[k] = item.slot;
                            self.slot[item.slot].running = true;
                        }
                        self.pack = pk;
                        if (n > 1) self.stats.packed_chunks += 1;
                    }
                    self.last_batch = false;
                    self.mutex.unlock(self.io);
                    const t0 = self.now();
                    const out = self.backend.prefillUnit(items[0..n]);
                    const t1 = self.now();
                    self.mutex.lockUncancelable(self.io);
                    self.account(&self.stats.prefill_ns, t0, t1);
                    self.stats.prefill_units += 1;
                    const pk = self.pack.?;
                    if (out) |unit| {
                        if (unit.done) {
                            self.stats.prefill_chunks += 1;
                            self.pack = null;
                            self.endPack(pk, unit);
                        }
                    } else |e| {
                        // A failed unit leaves no chunk in flight (the backend drops it).
                        self.pack = null;
                        self.endPack(pk, e);
                    }
                }
                self.mutex.unlock(self.io);
            }
        }
    };
}

// Host tests with a fake backend (tests/batcher.zig).
