#!/bin/bash
# 探针扫描：对每个 LSE_ROW_SRC 值 构建 -> 采样 -> 解析，结果落到 OUT/probe_<n>/
#
#   bash run_probe_sweep.sh <n> [<n> ...]
#
# 用法前提：卡空闲（ASCEND_RT_VISIBLE_DEVICES 指定），CWD 不在 /vllm-workspace。
set -uo pipefail

from=${1:?需要至少一个 LSE_ROW_SRC 值}
shift || true
VALUES="$from $*"
P=/vllm-workspace/ops/kernels/ascend-kernel/csrc/ops/lse_merge/prof
OUT=/tmp/lsemerge/probes
PY=/opt/miniconda3/envs/hust/bin/python
CARD="${ASCEND_RT_VISIBLE_DEVICES:-1}"
mkdir -p "$OUT"

cd /tmp
for n in $VALUES; do
  echo "################ LSE_ROW_SRC=$n ################"
  # 构建失败即硬停：绝不用旧二进制继续测量（2026-09-18 踩过）
  if ! bash "$P/build_probe.sh" "$n" /tmp/lsemergebuild 2>&1 | tail -6; then
    echo "FATAL: 构建失败，中止扫描"; exit 1
  fi
  rm -rf "$OUT/probe_$n"
  mkdir -p "$OUT/probe_$n"
  for t in tier0_stride1 tier1_fp32out tier0_stride8; do
    ASCEND_RT_VISIBLE_DEVICES="$CARD" HF_HUB_OFFLINE=1 timeout 600 \
      "$PY" -u "$P/run_prof.py" --mode prof --target "$t" \
      --prof-dir "$OUT/probe_$n" >/dev/null 2>&1 || { echo "FATAL: 采集 $t 失败"; exit 1; }
    echo "  captured $t"
  done
  "$PY" -u "$P/parse_prof.py" --prof-dir "$OUT/probe_$n" \
        --out "$OUT/probe_$n/parsed.json"
done
echo "=== SWEEP DONE: $VALUES ==="
