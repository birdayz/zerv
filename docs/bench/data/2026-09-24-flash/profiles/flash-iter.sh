#!/bin/bash
# Rebuild the flash module, run its GPU gate, profile an 8k prompt (fp32), print attention time.
set -e
cd /home/USER/projects/zerv
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
tag=$1
D=third_party/model-build-flash-$tag; rm -rf $D
python3 tools/compile_model.py --output-dir $D > /dev/null
cp $D/attn_flash.spv $D/manifest.json src/model/shaders/
zig build gpu-test -Doptimize=ReleaseFast --summary all 2>&1 | grep -E "tests passed|failed|fused|TestUnexpected|p0 " | grep -v "failed command" | tail -3
zig build model-profile-build -Doptimize=ReleaseFast -Dcpu=native 2>&1 | head -3
MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats ./zig-out/bin/zerv-model-profile models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf 8704 512 8192 4 fp32 > third_party/flash-prof-$tag.jsonl 2> third_party/flash-prof-$tag.stats
python3 - $tag <<'PY'
import json,sys
tag=sys.argv[1]
rows=[json.loads(l) for l in open(f"third_party/flash-prof-{tag}.jsonl") if l.startswith("{")]
pre=[r for r in rows if r["kind"]=="prefill"]
att=[sum(v for k,v in r["phases_ms"].items() if k in ("scores","softmax","pv","attention")) for r in pre]
print(f"{tag}: prefill gpu {sum(r['gpu_ms'] for r in pre):.0f} ms, attention {sum(att):.0f} ms; chunk@0 {att[0]:.1f} @3584 {att[7]:.1f} @7680 {att[15]:.1f} ms")
t=open(f"third_party/flash-prof-{tag}.stats").read()
for b in t.split("*** SHADER STATS ***")[1:]:
    ls=dict(l.split(": ",1) for l in b.splitlines() if ": " in l)
    if "flash" in tag and ls.get("LDS size") in ("24576","25344","16384","24960","32768") and int(ls.get("VGPRs","0"))>0 and ls.get("Subgroups per SIMD")!="8":
        print({k:ls[k] for k in ("VGPRs","LDS size","Subgroups per SIMD","VALU","VMEM","SMEM","Latency","Inverse Throughput") if k in ls})
PY
