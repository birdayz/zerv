//! The `zerv` umbrella module: every package, for executables that use several (the server,
//! benchmarks, tools, GPU tests). Packages are separate modules with declared dependencies
//! (src/NAME/BUILD.bazel `deps`); unit tests import only theirs.
pub const quant = @import("quant");
pub const gpu = @import("gpu");
pub const matvec = @import("matvec");
pub const artifact = @import("artifact");
pub const chat = @import("chat");
pub const text = @import("text");
pub const tokenizer = @import("tokenizer");
pub const model = @import("model");
pub const session = @import("session");
pub const serve = @import("serve");
