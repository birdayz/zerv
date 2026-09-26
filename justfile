# zerv tasks. Everything goes through Bazel (docs/development.md, "Bazel").

default:
    @just --list

# Build every target (server, benchmarks, tools, tests).
build:
    bazelisk build //...

# All required checks: zig fmt, the Zig unit tests in Debug and ReleaseFast, the Python tests.
test:
    bazelisk test //...

# Inner loop: Debug unit tests and the Python tests (no ReleaseFast variants).
quick:
    bazelisk test --config=quick //...

# Real-device Vulkan tests (needs the GPU to itself).
gpu:
    bazelisk test //tests:gpu

# ReleaseFast, native-CPU binaries to measure: the server, benchmarks and tools.
release:
    bazelisk build --config=release //src:zerv //bench/... //tools/...
