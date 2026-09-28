#!/usr/bin/env python3
"""What a measurement ran on, recorded without host tools (docs/specs/hermetic-build.md,
phase 3). Benchmark manifests record the host they measured; harnesses must not execute host
programs (lscpu, ldd, readelf, vulkaninfo, ...) to do so. This module reads the kernel's and
the ELF files' own descriptions instead, and uses only the platform's dynamic linker (glibc,
part of the execution platform) and tools built in the graph.

- `cpu()`: the CPU as /proc/cpuinfo and sysfs describe it (lscpu's sources).
- `elf_dynamic(path)`: DT_NEEDED names and whether RPATH/RUNPATH is set; None if static.
- `loaded_libraries(path, env)`: the shared objects the dynamic linker resolves for an
  executable (what `ldd` prints, from `ld.so --list`, which never runs the program).
- `host_vulkan()`: the host's Vulkan loader and installed ICDs (manifest and driver library,
  with hashes). Production benchmarks run on the host driver (docs/specs/hermetic-build.md,
  decision 1a); this records which one. Tests and GPU oracles use the test-only runtime.
- `vulkaninfo(env, summary)`: the output of vulkaninfo built from source (@vulkan_tools).
- `processes(pattern)`: running processes whose command line matches (what `pgrep -a -f`
  prints, from /proc).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]

# Where the dynamic linker and the Vulkan loader look on common x86_64 distributions (for the
# record only; nothing is loaded from here by this module).
LIBRARY_DIRS = ("/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/lib64", "/lib")
ICD_DIRS = ("/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d", "/usr/share/vulkan/icd.d")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def cpu():
    """Model, logical CPUs, sockets/cores, flags and frequency limits of this machine."""
    blocks = [b for b in Path("/proc/cpuinfo").read_text().split("\n\n") if b.strip()]
    procs = [{k.strip(): v.strip() for k, v in (line.split(":", 1) for line in b.splitlines() if ":" in line)} for b in blocks]
    first = procs[0]
    cores = {(p.get("physical id"), p.get("core id")) for p in procs}
    sysfs = Path("/sys/devices/system/cpu")
    read = lambda p: p.read_text().strip() if p.exists() else None
    freq = sysfs / "cpu0/cpufreq"
    return dict(model_name=first.get("model name"), vendor=first.get("vendor_id"), family=first.get("cpu family"),
                model=first.get("model"), stepping=first.get("stepping"), microcode=first.get("microcode"),
                logical_cpus=len(procs), physical_cores=len(cores), online=read(sysfs / "online"),
                flags=sorted(first.get("flags", "").split()),
                min_khz=read(freq / "cpuinfo_min_freq"), max_khz=read(freq / "cpuinfo_max_freq"),
                boost=read(sysfs / "cpufreq/boost"))


def _elf(path):
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1: raise ValueError(f"{path}: not ELF64 LE")
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    return data, [struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize) for i in range(phnum)]


def interpreter(path):
    """The program interpreter (PT_INTERP) of an executable; None if static."""
    data, headers = _elf(path)
    for p_type, _, p_offset, _, _, p_filesz, _, _ in headers:
        if p_type == 3: return data[p_offset:p_offset + p_filesz].rstrip(b"\0").decode()
    return None


def elf_dynamic(path):
    """(DT_NEEDED names, has RPATH/RUNPATH) of an ELF64 little-endian file; None if static."""
    data, headers = _elf(path)
    loads = [(v, o, size) for t, _, o, v, _, size, _, _ in headers if t == 1]
    dyn = next(((o, size) for t, _, o, _, _, size, _, _ in headers if t == 2), None)
    if dyn is None: return None

    def offset(vaddr):
        for v, o, size in loads:
            if v <= vaddr < v + size: return o + vaddr - v
        raise ValueError(f"{path}: address {vaddr:#x} outside the loaded segments")
    entries = [struct.unpack_from("<qQ", data, dyn[0] + i) for i in range(0, dyn[1], 16)]
    strtab = offset(next(v for t, v in entries if t == 5))
    name = lambda at: data[strtab + at:data.index(b"\0", strtab + at)].decode()
    return [name(v) for t, v in entries if t == 1], any(t in (15, 29) for t, _ in entries)


def loaded_libraries(path, env=None):
    """{soname: resolved absolute path} as the platform's dynamic linker resolves them for
    `path` under `env` (its `--list` mode; the program is not run); {} for a static file."""
    interp = interpreter(path)
    if interp is None: return {}
    out = subprocess.run([interp, "--list", str(path)], env=env, capture_output=True, text=True, check=True).stdout
    libs = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "=>":
            if parts[2] == "not": raise ValueError(f"{path}: {parts[0]} not found")
            libs[parts[0]] = parts[2]
        elif parts and parts[0].startswith("/"):
            libs[Path(parts[0]).name] = parts[0]
    return libs


def library_record(path, env=None):
    """Resolved shared objects of `path` with their hashes."""
    return {name: dict(path=p, sha256=sha(p)) for name, p in loaded_libraries(path, env).items()}


def _library(name, base=None):
    if "/" in name:
        p = Path(name) if Path(name).is_absolute() else (base / name)
        return p if p.exists() else None
    return next((Path(d) / name for d in LIBRARY_DIRS if (Path(d) / name).exists()), None)


def host_vulkan():
    """The host's installed Vulkan loader and ICDs: paths and hashes (of the resolved files)."""
    loader = _library("libvulkan.so.1")
    icds = []
    for d in ICD_DIRS:
        for manifest in sorted(Path(d).glob("*.json")) if Path(d).is_dir() else []:
            try: library = json.loads(manifest.read_text())["ICD"]["library_path"]
            except (ValueError, KeyError): library = None
            lib = _library(library, manifest.parent) if library else None
            icds.append(dict(manifest=str(manifest), manifest_sha256=sha(manifest), library_path=library,
                             library=str(lib.resolve()) if lib else None, library_sha256=sha(lib) if lib else None))
    return dict(loader=str(loader.resolve()) if loader else None, loader_sha256=sha(loader) if loader else None, icds=icds)


def processes(pattern):
    """["PID COMMAND LINE", ...] of the processes whose command line (arguments joined by
    spaces) matches the regular expression `pattern` (re.search), as `pgrep -a -f PATTERN`;
    this process excluded."""
    found, regex = [], re.compile(pattern)
    for d in sorted(Path("/proc").iterdir(), key=lambda p: (len(p.name), p.name)):
        if not d.name.isdigit() or d.name == str(os.getpid()): continue
        try: raw = (d / "cmdline").read_bytes()
        except OSError: continue  # exited, or not ours to read
        line = " ".join(raw.rstrip(b"\0").replace(b"\0", b" ").decode(errors="replace").split())
        if line and regex.search(line): found.append(f"{d.name} {line}")
    return found


def host_vulkan_id():
    """sha256 of host_vulkan(): changes whenever the host's loader, an ICD manifest or a driver
    library changes (the cache key of the host-driver GPU tests)."""
    return hashlib.sha256(json.dumps(host_vulkan(), sort_keys=True).encode()).hexdigest()


def vulkaninfo(env=None, summary=True):
    """vulkaninfo (built from source, @vulkan_tools) under `env`: the Vulkan stack of that
    environment (the host's by default)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import zerv_build
    exe = zerv_build.binary("vulkaninfo", config=None)
    return subprocess.run([str(exe), *(["--summary"] if summary else [])], env=env, capture_output=True, text=True, check=True).stdout


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    print(json.dumps(dict(cpu=cpu(), host_vulkan=host_vulkan(), host_vulkan_id=host_vulkan_id()), indent=2))
