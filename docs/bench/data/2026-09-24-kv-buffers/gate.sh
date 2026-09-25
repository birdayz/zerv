#!/bin/bash
# KV buffers gate: new build (default KV buffers and a forced 6-buffer split) through the
# FP64/llama gates, then the pre-change baseline tool on the same inputs; bitwise compare.
set -u
cd /home/USER/projects/zerv
R=docs/bench/data/2026-09-24-kv-buffers
mkdir -p $R
W=third_party/model-native
python3 tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-23-default-regen --modes 0,512 --work-dir $W/2026-09-24-kv-default --report $R/verify-default.json; echo "default rc=$?"
python3 tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-23-default-regen --modes 0,512 --kv-capacity-mib 24 --work-dir $W/2026-09-24-kv-split-default --report $R/verify-split-default.json; echo "split-default rc=$?"
python3 tools/verify_model.py --fixture tests/fixtures/model/qwen38-oracle-long.json --oracle-dir third_party/model-oracle/2026-09-23-long --modes 0,512 --work-dir $W/2026-09-24-kv-long --report $R/verify-long.json; echo "long rc=$?"
python3 tools/verify_model.py --fixture tests/fixtures/model/qwen38-oracle-long.json --oracle-dir third_party/model-oracle/2026-09-23-long --modes 0,512 --kv-capacity-mib 24 --work-dir $W/2026-09-24-kv-split-long --report $R/verify-split-long.json; echo "split-long rc=$?"
# Baseline (pre-change tree) on exactly the new runs' inputs.
B=third_party/kv-split/baseline
mkdir -p $B
for run in 2026-09-24-kv-default 2026-09-24-kv-long; do
  for d in $W/$run/*/; do
    n=$(basename $d); mkdir -p $B/$run/$n
    cp $d/tokens.json $d/names.txt $B/$run/$n/
    mode=${n##*-}
    if [ "$mode" = "0" ]; then extra=""; else extra="$mode 3000"; fi
    third_party/kv-split/baseline-capture models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf $B/$run/$n/tokens.json $B/$run/$n/names.txt $B/$run/$n 1024 $extra 2> $B/$run/$n/stderr.txt || echo "baseline FAILED $run $n"
  done
done
echo gate-done
