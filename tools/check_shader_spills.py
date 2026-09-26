#!/usr/bin/env python3
"""Gate: no pipeline created by COMMAND spills VGPRs (RADV/ACO statistics).

Mesa 26.2.3's ACO can miscompile VGPR spills that it places in LDS (single-wave
workgroups): the spill loads/stores carry no memory-sync info and the pre-RA scheduler
reorders them across each other (docs/bench/2026-09-24-aco-lds-spill.md). Our shaders must
therefore not spill VGPRs into LDS. This runs COMMAND with the shader cache disabled and
RADV_DEBUG=shaderstats (or parses an earlier log with --from-log), and classifies every
pipeline's statistics:
- FAIL: Spilled VGPRs > 0 with Scratch size 0, i.e. every spill slot is in LDS.
- WARN: Spilled VGPRs > 0 with scratch in use: scratch spills (ordered correctly, but slow;
  a performance item). Limitation: the statistics cannot show a mixed LDS + scratch spill,
  which ACO can emit for a single-wave workgroup; check such a pipeline's workgroup size.
SGPR spills are reported only (ACO keeps them in VGPR lanes, not in LDS).
Usage: tools/check_shader_spills.py --log FILE -- COMMAND [ARGS...]
       tools/check_shader_spills.py --from-log FILE
As a Bazel test (//tests:gpu_spills) the log defaults to the test's undeclared outputs."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def stats(text):
    for block in text.split("Driver pipeline hash")[1:]:
        def field(name):
            m = re.search(name+r": (\d+)", block)
            return int(m.group(1)) if m else 0
        yield dict(vgprs=field("VGPRs"), spilled_vgprs=field("Spilled VGPRs"), spilled_sgprs=field("Spilled SGPRs"),
                   lds=field("LDS size"), scratch=field("Scratch size"), code=field("Code size"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    p.add_argument("--log", type=Path, default=Path(outputs)/"shaderstats.txt" if outputs else None,
                   help="where the raw statistics output is written")
    p.add_argument("--from-log", type=Path, help="parse an existing log instead of running")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.from_log:
        output, returncode = a.from_log.read_text(), 0
    else:
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not command or not a.log: p.error("--log and COMMAND required")
        env = dict(os.environ, MESA_SHADER_CACHE_DISABLE="true", RADV_DEBUG="shaderstats")
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        a.log.parent.mkdir(parents=True, exist_ok=True)
        a.log.write_text(result.stdout)
        output, returncode = result.stdout, result.returncode
    pipelines = list(stats(output))
    lds = [s for s in pipelines if s["spilled_vgprs"] and not s["scratch"]]
    scratch = [s for s in pipelines if s["spilled_vgprs"] and s["scratch"]]
    sgpr = [s for s in pipelines if s["spilled_sgprs"]]
    print(f"command exit {returncode}; pipelines {len(pipelines)}; FAIL (VGPR spills in LDS) {len(lds)}; WARN (VGPR spills in scratch) {len(scratch)}; SGPR-spilling {len(sgpr)}")
    for s in lds: print("  FAIL:", s)
    for s in scratch: print("  WARN:", s)
    if returncode != 0: print("command failed; see", a.log); sys.exit(2)
    if not pipelines: print("no statistics found (not RADV?)"); sys.exit(2)
    sys.exit(1 if lds else 0)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
