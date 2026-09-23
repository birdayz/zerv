//! OpenAI Chat Completions v1 serving: request/response encoding, HTTP server, engine.
pub const api = @import("api.zig");
pub const http = @import("http.zig");
pub const listen = @import("listen.zig");
pub const Native = @import("engine.zig").Native;
