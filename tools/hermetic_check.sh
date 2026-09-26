#!/usr/bin/env bash
# Hermeticity proof (docs/specs/hermetic-build.md, "Acceptance"): `bazel test` of the given
# targets (default //...) in a container that has no compiler, no Python, no Vulkan and no
# shader tools — a slim Debian (bash, coreutils, glibc) plus the Bazel binary of
# .bazelversion — with empty repository, disk and output caches. Everything the build needs
# must come from pinned downloads or the graph itself.
#
#   tools/hermetic_check.sh [--gpu] [--log FILE] [BAZEL ARGS...]
#
# The working tree (tracked and untracked files, not ignored ones) is copied into the
# container. --gpu passes the render node through (phase 4). Needs docker and network access.
set -euo pipefail

IMAGE=debian:trixie-slim@sha256:a99cfc517144bc59b1978475ec53b46ecabec7e43635402ee5b77cc54cd1b20a
BAZEL_VERSION=9.2.0
BAZEL_SHA256=7668a95db1250f12c40407251e4e203b4ec8bf39bc495d2f485b2d8c99048694

root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ $(cat "$root/.bazelversion") == "$BAZEL_VERSION" ]] || { echo "update BAZEL_VERSION/BAZEL_SHA256 for .bazelversion" >&2; exit 2; }

devices=()
log=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) devices=(--device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)"); shift ;;
    --log) log=$2; shift 2 ;;
    *) break ;;
  esac
done
args=("$@")
[[ ${#args[@]} -gt 0 ]] || args=(//...)

work=$(mktemp -d /tmp/zerv-hermetic.XXXXXX)
trap 'chmod -R u+w "$work" 2>/dev/null; rm -rf "$work"' EXIT

# The Bazel binary: bazelisk's cached download, else GitHub's release; checked either way.
bazel="$HOME/.cache/bazelisk/downloads/sha256/$BAZEL_SHA256/bin/bazel"
if [[ ! -x $bazel ]]; then
  bazel="$work/bazel"
  curl -fsSL -o "$bazel" "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64"
  chmod +x "$bazel"
fi
echo "$BAZEL_SHA256  $bazel" | sha256sum --check --quiet

mkdir -p "$work/src" "$work/home"
git -C "$root" ls-files -z --cached --others --exclude-standard | (cd "$root" && tar --null -T - -cf -) | tar -xf - -C "$work/src"
echo "workspace: $(git -C "$root" rev-parse HEAD) + working tree ($(find "$work/src" -type f | wc -l) files); image $IMAGE"

set +e
docker run --rm --network host "${devices[@]}" \
  --user "$(id -u):$(id -g)" -e HOME=/home/zerv -e USER=zerv \
  -v "$work/home:/home/zerv" -v "$work/src:/zerv" -v "$bazel:/usr/local/bin/bazel:ro" \
  -w /zerv "$IMAGE" \
  bash -c 'set -e
    echo "host tools in the container:"; for t in cc gcc clang python3 python glslc spirv-val zig; do command -v $t || true; done
    ls /usr/lib/x86_64-linux-gnu/libvulkan* 2>/dev/null || true
    exec bazel --nohome_rc --nosystem_rc test --jobs=12 "$@"' _ "${args[@]}" 2>&1 | tee "$work/log.txt"
status=${PIPESTATUS[0]}
set -e
[[ -z $log ]] || cp "$work/log.txt" "$log"
echo "exit $status${log:+ (log: $log)}"
exit "$status"
