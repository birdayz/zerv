#!/usr/bin/env python3
"""Configure and build a meson project inside one Bazel action (bazel/meson.bzl).

Hermetic: the compilers are the Bazel C/C++ toolchain's, meson runs under the pinned Python
(with the packages of this binary on its path), ninja and every named tool come from the
graph, PATH holds only those tools, HOME is a scratch directory, and subprojects must already
be present (no downloads).
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="the project's root (its meson.build)")
    p.add_argument("--meson", type=Path, required=True, help="meson.py")
    p.add_argument("--ninja", type=Path, required=True)
    p.add_argument("--cc", type=Path, required=True, help="C and C++ compiler driver")
    p.add_argument("--ar", type=Path, required=True)
    p.add_argument("--subproject", action="append", default=[], help="DIR=PATH: subprojects/DIR from PATH")
    p.add_argument("--tool", action="append", default=[], help="NAME=PATH on PATH")
    p.add_argument("--option", action="append", default=[], help="meson -D option")
    p.add_argument("--target", action="append", default=[], help="ninja target to build")
    p.add_argument("--output", action="append", default=[], help="BUILD_RELATIVE=DEST: copy after the build")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    a = p.parse_args()

    execroot = Path.cwd()
    work = Path(tempfile.mkdtemp(prefix="meson-"))
    # The source tree: symlinks to the (read-only) project, with its subprojects added.
    src = work / "src"
    src.mkdir()
    for child in (execroot / a.source).iterdir():
        if child.name != "subprojects":
            (src / child.name).symlink_to(child.resolve())
    subprojects = src / "subprojects"
    subprojects.mkdir()
    upstream = execroot / a.source / "subprojects"
    if upstream.is_dir():
        for child in upstream.iterdir():
            (subprojects / child.name).symlink_to(child.resolve())
    for item in a.subproject:
        name, path = item.split("=", 1)
        (subprojects / name).symlink_to((execroot / path).resolve())

    # PATH: the named tools and python3 (the interpreter running this, with our packages).
    bin_dir = work / "bin"
    bin_dir.mkdir()
    for item in a.tool:
        name, path = item.split("=", 1)
        (bin_dir / name).symlink_to((execroot / path).resolve())
    (bin_dir / "python3").symlink_to(sys.executable)
    # Coreutils are part of the execution platform (docs/specs/hermetic-build.md): the ones
    # meson's generated rules call (static archives: `rm -f`), and nothing else of the host.
    for name in ("rm", "tr"):
        (bin_dir / name).symlink_to(shutil.which(name, path="/usr/bin:/bin"))
    (bin_dir / "ninja").symlink_to((execroot / a.ninja).resolve())
    # The toolchain's driver selects its target from the path it is called by: keep the path.
    # Zig's linker rejects -Wl,--fatal-warnings, which meson adds to every link-argument check
    # (so -Wl,--build-id=sha1 and others would test as unsupported): drop that one argument.
    # The embedded clang answers --print-search-dirs with the host's GCC library directories
    # (/usr/lib, ...), where meson's find_library() would then find host libraries (it found
    # /usr/lib/libelf.so): answer with none. Zig's own links search no host path.
    cc_path, ar_path = work / "cc", work / "ar"
    # Meson makes thin archives (ar ...T) of internal static libraries, which Zig's linker does
    # not read: make regular ones.
    ar_path.write_text("#!/bin/sh\nop=$1; shift\n"
                       f"exec '{execroot / a.ar}' \"$(printf %s \"$op\" | tr -d T)\" \"$@\"\n")
    ar_path.chmod(0o755)
    # Mesa passes a version script as `-Wl,--version-script PATH`, the path a separate driver
    # argument that GNU ld and lld take as a linker script; Zig's driver rejects the file:
    # join them into -Wl,--version-script=PATH.
    cc_path.write_text("""#!/bin/sh
for a in "$@"; do case "$a" in --print-search-dirs|-print-search-dirs) printf 'programs: =\\nlibraries: =\\n'; exit 0;; esac; done
script=
for a in "$@"; do
  shift
  if [ -n "$script" ]; then set -- "$@" "-Wl,--version-script=$a"; script=; continue; fi
  case "$a" in
    -Wl,--fatal-warnings) ;;
    -Wl,--version-script) script=1 ;;
    *) set -- "$@" "$a" ;;
  esac
done
""" + f"exec '{execroot / a.cc}' \"$@\"\n")
    cc_path.chmod(0o755)
    native = work / "native.ini"
    native.write_text(f"""[binaries]
c = ['{cc_path}']
cpp = ['{cc_path}']
ar = ['{ar_path}']
python = ['{sys.executable}']
""")
    env = {
        "PATH": str(bin_dir),
        "HOME": str(work / "home"),
        "PYTHONPATH": os.pathsep.join(p for p in sys.path[1:] if p),
        "LC_ALL": "C.UTF-8",
        # Reproducible outputs: generators that iterate sets or embed dates.
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": "0",
        "TMPDIR": str(work / "tmp"),
    }
    (work / "tmp").mkdir()
    build = work / "build"
    meson = [sys.executable, str((execroot / a.meson).resolve())]
    subprocess.run(meson + ["setup", str(build), str(src), "--native-file", str(native), "--wrap-mode=nodownload",
                            "--buildtype=release", *("-D" + o for o in a.option)], env=env, check=True)
    subprocess.run([str(bin_dir / "ninja"), "-C", str(build), f"-j{a.jobs}", *a.target], env=env, check=True)
    for item in a.output:
        rel, dest = item.split("=", 1)
        shutil.copyfile(build / rel, execroot / dest)
        os.chmod(execroot / dest, 0o755)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
