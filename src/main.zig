//! zerv: native Qwen3.8-27B server, OpenAI Chat Completions v1 (`POST /v1/chat/completions`).
const std = @import("std");
const zerv = @import("zerv");

const usage =
    \\usage: zerv --model PATH [--host 127.0.0.1] [--port 8080] [--context 8192]
    \\            [--alias qwen3.8-27b] [--max-waiting 16] [--vram-budget-gib 23]
    \\            [--prefill-chunk 512]  (0 = token-by-token prefill)
    \\            [--drain-timeout 30]  (seconds; SIGINT/SIGTERM drain, a second one exits)
    \\            [--prefill-precision fp32]  (fp32 | f16: explicit f16 WMMA prompt projections)
    \\            [--prefix-cache-slots 8]  (recurrent-state snapshots, ~150 MiB VRAM each; 0 = no prefix cache)
    \\
;

/// Set by the first SIGINT/SIGTERM; `Server.run` polls it and drains.
var stop_requested: std.atomic.Value(bool) = .init(false);

fn onStopSignal(_: std.posix.SIG) callconv(.c) void {
    // Async-signal-safe: one atomic swap, or an immediate exit on the second signal.
    if (stop_requested.swap(true, .acq_rel)) std.os.linux.exit_group(130);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var model_path: ?[]const u8 = null;
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var context: u32 = 8192;
    var alias: []const u8 = "qwen3.8-27b";
    var max_waiting: u32 = 16;
    var budget_gib: u64 = 23;
    var prefill_chunk: u32 = 512;
    var drain_s: u32 = 30;
    var precision: zerv.model.gemm.Precision = .fp32;
    var snapshot_slots: u32 = 8;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (i + 1 >= args.len) {
            std.debug.print("{s}", .{usage});
            return error.InvalidArguments;
        }
        const value = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, arg, "--model")) model_path = value //
        else if (std.mem.eql(u8, arg, "--host")) host = value //
        else if (std.mem.eql(u8, arg, "--port")) port = try std.fmt.parseInt(u16, value, 10) //
        else if (std.mem.eql(u8, arg, "--context")) context = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--alias")) alias = value //
        else if (std.mem.eql(u8, arg, "--max-waiting")) max_waiting = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--vram-budget-gib")) budget_gib = try std.fmt.parseInt(u64, value, 10) //
        else if (std.mem.eql(u8, arg, "--prefill-chunk")) prefill_chunk = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--drain-timeout")) drain_s = try std.fmt.parseInt(u32, value, 10) //
        else if (std.mem.eql(u8, arg, "--prefill-precision")) precision = std.meta.stringToEnum(zerv.model.gemm.Precision, value) orelse return error.InvalidArguments //
        else if (std.mem.eql(u8, arg, "--prefix-cache-slots")) snapshot_slots = try std.fmt.parseInt(u32, value, 10) //
        else {
            std.debug.print("unknown option {s}\n{s}", .{ arg, usage });
            return error.InvalidArguments;
        }
    }
    const path = model_path orelse {
        std.debug.print("{s}", .{usage});
        return error.InvalidArguments;
    };

    // Own the port before the (long) model load, so a busy port fails immediately.
    // Connections that arrive while loading wait in the kernel backlog and are served
    // once the model is ready (docs/specs/serving.md, "Startup").
    const address = try std.Io.net.IpAddress.parse(host, port);
    var listener = zerv.serve.listen.listen(address, 128) catch |e| {
        if (e == error.AddressInUse) std.debug.print("zerv: {s}:{d} is already in use (another server is listening there)\n", .{ host, port });
        return e;
    };
    var listener_owned = true; // until `Server.run` takes it
    defer if (listener_owned) listener.deinit(io);

    // The mapping is needed only until the weights are resident and the tokenizer has
    // copied its tables; it is released before serving (otherwise ~15.5 GB stays in RSS).
    var file = try zerv.artifact.MappedFile.open(io, path, 64 * 1024 * 1024 * 1024);
    var file_live = true;
    defer if (file_live) file.deinit();
    var container = try zerv.artifact.gguf.Container.parse(gpa, file.bytes, .{});
    var container_live = true;
    defer if (container_live) container.deinit();
    var tokenizer = try zerv.tokenizer.fromGGUF(gpa, &container, .{});
    defer tokenizer.deinit();
    var device = try zerv.gpu.Device.open(.{ .max_allocated_bytes = budget_gib * 1024 * 1024 * 1024, .cooperative_matrix = zerv.model.gemm.deviceNeeds(precision).cooperative_matrix, .subgroup_size_control = zerv.model.gemm.deviceNeeds(precision).subgroup_size_control });
    defer device.deinit() catch @panic("device resources still live");
    std.debug.print("zerv: loading {s} on {s} (context {d}, prefill chunk {d}, prefill precision {s}, prefix cache slots {d} = {d} MiB)\n", .{ path, device.name(), context, prefill_chunk, @tagName(precision), snapshot_slots, snapshot_slots * zerv.model.snapshot_bytes / (1024 * 1024) });
    var model: zerv.model.Model = undefined;
    model.init(&device, &container, .{ .context = context, .prefill_rows = prefill_chunk, .prefill_precision = precision, .snapshots = snapshot_slots }) catch |e| {
        if (e == error.InsufficientVram) std.debug.print("zerv: not enough free VRAM: the model needs {d} MiB (including {d} MiB headroom), {d} MiB are free (another process is using the GPU?)\n", .{ model.vram.needed >> 20, zerv.model.vram_headroom >> 20, model.vram.free.? >> 20 });
        return e;
    };
    if (model.vram.free) |free| std.debug.print("zerv: VRAM: needed {d} MiB of {d} MiB free\n", .{ model.vram.needed >> 20, free >> 20 }) //
    else std.debug.print("zerv: VRAM: needed {d} MiB (the driver reports no budget; not checked)\n", .{model.vram.needed >> 20});
    defer model.deinit();
    var native = try zerv.serve.Native.init(io, gpa, &model, &tokenizer);
    defer native.deinit(gpa);
    const ids = [_][]const u8{alias};
    const defaults = try samplingDefaults(&container);
    container.deinit();
    container_live = false;
    file.deinit();
    file_live = false;
    var server = zerv.serve.http.Server.init(io, gpa, native.engine(&ids, defaults), .{ .max_waiting = max_waiting, .drain_timeout = .fromSeconds(drain_s) });
    const action: std.posix.Sigaction = .{ .handler = .{ .handler = onStopSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &action, null);
    std.posix.sigaction(.TERM, &action, null);
    std.debug.print("zerv: serving {s} at http://{s}:{d}/v1/chat/completions\n", .{ alias, host, port });
    listener_owned = false;
    server.run(&listener, &stop_requested) catch |e| {
        if (e != error.EngineFailed) return e;
        // The device is gone: its objects cannot be torn down normally (commands may
        // still be pending). Exiting releases them; a supervisor can restart the server.
        std.debug.print("zerv: stopped: the GPU device was lost\n", .{});
        std.process.exit(3);
    };
    std.debug.print("zerv: stopped\n", .{});
}

/// The artifact's recommended sampling (general.sampling.*), else the model card values.
fn samplingDefaults(container: *const zerv.artifact.gguf.Container) !zerv.serve.api.Defaults {
    var d: zerv.serve.api.Defaults = .{};
    if (container.findMetadata("general.sampling.temp")) |v| d.temperature = try v.scalar(f32);
    if (container.findMetadata("general.sampling.top_p")) |v| d.top_p = try v.scalar(f32);
    if (container.findMetadata("general.sampling.top_k")) |v| d.top_k = @intCast(try v.scalar(i32));
    return d;
}
