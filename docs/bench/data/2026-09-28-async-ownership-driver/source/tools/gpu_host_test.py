#!/usr/bin/env python3
"""Runs a GPU test binary on the host's installed Vulkan stack (docs/specs/hermetic-build.md,
"Host-driver GPU tests"): //tests:gpu_host and //tests:gpu_host_release_fast.

  gpu_host_test.py BINARY

The host driver is an input of these tests: ZERV_HOST_VULKAN_ID (--test_env, set by
tools/zerv_build.py test_host_gpu()) must equal tools/host_info.py host_vulkan_id() of the
driver present now, so a cached result always belongs to the installed driver. The binary runs
with ZERV_TEST_GPU_RUNTIME=host: its native-kernel test requires the driver's pipeline key to be
the one of src/model/native/ (the binary production uses).
"""
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).absolute().parent))
import host_info  # noqa: E402

HINT = "run it as tools/py tools/zerv_build.py --test-host-gpu (it passes the id of the installed driver)"


def main():
    if len(sys.argv) != 2: sys.exit(__doc__)
    given, actual = os.environ.get("ZERV_HOST_VULKAN_ID"), host_info.host_vulkan_id()
    if given is None: sys.exit(f"ZERV_HOST_VULKAN_ID is not set: {HINT}")
    if given != actual: sys.exit(f"ZERV_HOST_VULKAN_ID {given} is not the installed Vulkan stack's ({actual}): {HINT}")
    print("host Vulkan stack:", host_info.json.dumps(host_info.host_vulkan(), indent=1), flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VK_", "RADV_", "ACO_", "MESA_", "AMD_")) and k != "LD_LIBRARY_PATH"}
    env["ZERV_TEST_GPU_RUNTIME"] = "host"
    sys.exit(subprocess.run([sys.argv[1]], env=env).returncode)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
