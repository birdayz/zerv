#!/usr/bin/env python3
"""Download one pinned Hugging Face model revision for use as a benchmark competitor's
weights, with integrity and safety checks (docs/bench/2026-09-25-vllm.md).

  tools/fetch_hf.py --repo RedHatAI/Qwen3.8-27B-INT4 --revision <40-hex sha> --dest models/

- The revision must be a full commit sha (no branch names).
- Refused before anything is downloaded: file types that can carry code (pickle
  checkpoints, Python, shared objects, scripts, archives) and a config.json with `auto_map`
  (custom modeling code). Only safetensors, JSON, text, jinja, YAML and Markdown are
  accepted.
- LFS files are checked against the Hub's sha256, other files against their git blob sha1.
- Every .safetensors file is parsed: an 8-byte header length, a JSON header of tensors with
  known dtypes, and data ranges that tile the data section exactly (no gaps, overlaps or
  trailing bytes that could hide a payload). The format holds no code; with pickle refused,
  nothing downloaded here is executable.
- `--verify` re-checks an existing download (hashes and structure) without the network
  beyond the Hub's file list.
- Files go to DEST/REPO/REVISION/; MANIFEST.json there records every file, size, sha256 and
  the source URL. Existing verified files are skipped.
"""
import argparse, hashlib, json, pathlib, re, subprocess, sys, urllib.request

ALLOWED = (".safetensors", ".json", ".txt", ".jinja", ".yaml", ".yml", ".md", ".gitattributes")
API = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
URL = "https://huggingface.co/{repo}/resolve/{rev}/{name}"


def file_hashes(path):
    sha256, sha1 = hashlib.sha256(), hashlib.sha1()
    size = path.stat().st_size
    sha1.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            sha256.update(block)
            sha1.update(block)
    return sha256.hexdigest(), sha1.hexdigest()


DTYPES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
          "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def check_safetensors(path):
    """Structure of a safetensors file; returns the tensor count or raises SystemExit."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if n <= 0 or 8 + n > size or n > 100 << 20: raise SystemExit(f"{path.name}: bad header length {n}")
        header = json.loads(f.read(n))
    data = size - 8 - n
    ranges = []
    for name, t in header.items():
        if name == "__metadata__":
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in t.items()): raise SystemExit(f"{path.name}: bad metadata")
            continue
        if set(t) != {"dtype", "shape", "data_offsets"} or t["dtype"] not in DTYPES: raise SystemExit(f"{path.name}: bad tensor {name}")
        count = 1
        for d in t["shape"]: count *= d
        a, b = t["data_offsets"]
        if b - a != count * DTYPES[t["dtype"]]: raise SystemExit(f"{path.name}: {name} size does not match its shape")
        ranges.append((a, b))
    ranges.sort()
    at = 0
    for a, b in ranges:
        if a != at: raise SystemExit(f"{path.name}: gap or overlap at byte {a}")
        at = b
    if at != data: raise SystemExit(f"{path.name}: data section {data} bytes, tensors cover {at}")
    return len(ranges)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--dest", type=pathlib.Path, required=True)
    p.add_argument("--skip", action="append", default=[], help="file names not to download (e.g. an unused draft head)")
    p.add_argument("--verify", action="store_true", help="only re-check an existing download")
    a = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision): p.error("--revision must be a full commit sha")
    with urllib.request.urlopen(API.format(repo=a.repo, rev=a.revision), timeout=60) as r:
        info = json.load(r)
    if info["sha"] != a.revision: raise SystemExit(f"revision mismatch: {info['sha']}")
    files = [s for s in info["siblings"] if s["rfilename"] not in a.skip]
    bad = [s["rfilename"] for s in files if not s["rfilename"].endswith(ALLOWED) or "/" in s["rfilename"] and s["rfilename"].split("/")[-1].startswith(".")]
    if bad: raise SystemExit(f"refused (file types that can carry code or unexpected paths): {bad}")
    out = a.dest / a.repo / a.revision
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(repo=a.repo, revision=a.revision, files=[])
    # Small files first: config.json is checked for custom code before the weights come.
    for s in sorted(files, key=lambda s: s.get("size") or 0):
        name = s["rfilename"]
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.format(repo=a.repo, rev=a.revision, name=name)
        want256 = (s.get("lfs") or {}).get("sha256")
        ok = False
        if path.exists() and path.stat().st_size == s.get("size"):
            h256, h1 = file_hashes(path)
            ok = h256 == want256 if want256 else h1 == s["blobId"]
        if not ok and a.verify: raise SystemExit(f"missing or mismatching: {name}")
        if not ok:
            print(f"downloading {name} ({s.get('size')} bytes)", flush=True)
            subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(path), url], check=True)
            h256, h1 = file_hashes(path)
            if (want256 and h256 != want256) or (not want256 and h1 != s["blobId"]):
                raise SystemExit(f"hash mismatch: {name}")
        if name == "config.json" and "auto_map" in json.loads(path.read_text()):
            raise SystemExit("config.json has auto_map (custom modeling code): refused")
        entry = dict(name=name, bytes=path.stat().st_size, sha256=h256, lfs=bool(want256), url=url)
        if name.endswith(".safetensors"): entry["tensors"] = check_safetensors(path)
        manifest["files"].append(entry)
        print(f"verified {name} sha256 {h256}", flush=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"ok: {len(manifest['files'])} files in {out}")


if __name__ == "__main__":
    main()
