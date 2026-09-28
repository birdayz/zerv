//! TCP listener that owns its port exclusively (docs/specs/serving.md, "Startup").
//! std.Io.net `listen(.{ .reuse_address = true })` sets SO_REUSEPORT as well as
//! SO_REUSEADDR, which lets a second process listen on the same port and silently
//! receive a share of the connections. This one sets SO_REUSEADDR only: a restart can
//! still bind while old connections are in TIME_WAIT, but another listener on the same
//! address makes it fail with `AddressInUse`. Linux only (the server's target).
const std = @import("std");
const linux = std.os.linux;
const Threaded = std.Io.Threaded;

pub const Error = error{ AddressInUse, AddressNotAvailable, AccessDenied, SystemResources, Unexpected };

/// A listening socket, returned as the std server type (closed with `Server.deinit`).
pub fn listen(address: std.Io.net.IpAddress, backlog: u31) Error!std.Io.net.Server {
    const family = Threaded.posixAddressFamily(&address);
    const fd: i32 = @intCast(try check(linux.socket(family, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, linux.IPPROTO.TCP)));
    errdefer _ = linux.close(fd);
    const one: c_int = 1;
    _ = try check(linux.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int)));
    var storage: Threaded.PosixAddress = undefined;
    var len = Threaded.addressToPosix(&address, &storage);
    _ = try check(linux.bind(fd, &storage.any, len));
    _ = try check(linux.listen(fd, backlog));
    _ = try check(linux.getsockname(fd, &storage.any, &len));
    return .{ .socket = .{ .handle = fd, .address = Threaded.addressFromPosix(&storage) }, .options = {} };
}

fn check(rc: usize) Error!usize {
    return switch (linux.errno(rc)) {
        .SUCCESS => rc,
        .ADDRINUSE => error.AddressInUse,
        .ADDRNOTAVAIL => error.AddressNotAvailable,
        .ACCES, .PERM => error.AccessDenied,
        .MFILE, .NFILE, .NOBUFS, .NOMEM => error.SystemResources,
        else => error.Unexpected,
    };
}
