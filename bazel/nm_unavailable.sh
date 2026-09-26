#!/bin/sh
# libdrm's meson build requires an `nm` program, which only its symbol tests run (not built
# here, bazel/third_party/mesa.BUILD). No host nm may be used; this fails if anything runs it.
echo "nm: not provided in the hermetic build (bazel/nm_unavailable.sh)" >&2
exit 1
