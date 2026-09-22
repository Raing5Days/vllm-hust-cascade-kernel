#!/bin/bash
# 仪器自检：在信任任何测量之前跑一次。
#
# 为什么需要它（2026-09-19 实测教训，三类错误都是"仪器没生效但没报警"）：
#   1) 环境 CXXFLAGS 被构建链吸收成 device 侧 -D 并长期缓存，静默压过源码默认值
#      ⇒ 两个不同"探针"其实编的是同一份代码；
#   2) build.sh 未在仓库根执行时 device 侧静默不重编，只有 host 侧与 wheel 更新
#      ⇒ 测的是旧二进制；
#   3) 预处理分支写成裸真值判断（#if X 而非 #if X == 1）⇒ 两个变体落进同一支。
# 三者的共同表象是"不同输入产出同一个二进制"。本脚本把该表象变成硬失败。
#
# 用法：bash selftest.sh [workdir]
# 退出码 0 = 仪器可信；非 0 = 先修仪器，不要开始测量。
set -uo pipefail

P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/vllm-workspace/ops/kernels/ascend-kernel
SP=/opt/miniconda3/envs/hust/lib/python3.12/site-packages/ascend_kernel
GOLDEN=/vllm-workspace/profiles/lse-merge-pipe-20260918/golden
PY=/opt/miniconda3/envs/hust/bin/python
OUT="${1:-/tmp/lsemerge/selftest}"
CARD="${ASCEND_RT_VISIBLE_DEVICES:-1}"
mkdir -p "$OUT"
cd /tmp || exit 1

md5so() { md5sum "$SP/lib/libascend_kernel.so" 2>/dev/null | awk '{print $1}'; }
FAIL=0

echo "=== 自检 1/3：探针宏是否真能改变 device 产物 ==="
echo "（断言：生产值 0 与探针值 3 必须产出**不同**的二进制；相同即宏未生效）"
bash "$P/build_probe.sh" 0 "$OUT" > "$OUT/st0.log" 2>&1 || { echo "FAIL: 生产构建失败"; tail -5 "$OUT/st0.log"; exit 1; }
M0=$(md5so)
bash "$P/build_probe.sh" 3 "$OUT" > "$OUT/st3.log" 2>&1 || { echo "FAIL: 探针构建失败"; tail -5 "$OUT/st3.log"; exit 1; }
M3=$(md5so)
if [ -z "$M0" ] || [ -z "$M3" ]; then
  echo "FAIL: 取不到 .so md5"; FAIL=1
elif [ "$M0" = "$M3" ]; then
  echo "FAIL: 探针 0 与 3 产出同一二进制（$M0）⇒ 宏没进 device 编译，所有探针测量都无效"
  FAIL=1
else
  echo "PASS: 探针能改变 device 产物（0→$M0，3→$M3）"
fi

echo
echo "=== 自检 2/3：回退到生产值后 bit 锚点是否全等 ==="
bash "$P/build_probe.sh" 0 "$OUT" > "$OUT/st0b.log" 2>&1 || { echo "FAIL: 回退构建失败"; tail -5 "$OUT/st0b.log"; exit 1; }
M0B=$(md5so)
if [ "$M0B" != "$M0" ]; then
  echo "WARN: 同一生产配置两次构建 md5 不同（$M0 vs $M0B）⇒ 构建非确定，跨轮比较需谨慎"
else
  echo "PASS: 生产配置可复现（md5 $M0B）"
fi
ASCEND_RT_VISIBLE_DEVICES="$CARD" timeout 900 "$PY" -u "$P/verify_golden.py" \
  --golden-dir "$GOLDEN" > "$OUT/bit.log" 2>&1
BIT=$(grep -o "BIT_COMPARE [0-9]*/[0-9]* identical, [0-9]* mismatch" "$OUT/bit.log" || echo "未取到结果")
echo "锚点: $BIT"
case "$BIT" in
  *"0 mismatch") echo "PASS: 生产路径与 v1 锚点逐元素 bit 相同" ;;
  *) echo "FAIL: 生产路径偏离 bit 锚点 ⇒ 当前装的不是已验证的生产代码"; FAIL=1 ;;
esac

echo
echo "=== 自检 3/3：device 产物新鲜度护栏是否在位 ==="
if grep -q "device 目标文件比源码旧\|OBJ_MTIME" "$P/build_probe.sh"; then
  echo "PASS: build_probe.sh 含 mtime 新鲜度断言"
else
  echo "FAIL: build_probe.sh 缺少新鲜度断言 ⇒ 无法发现'device 侧没重编'"; FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "=== 自检通过：仪器可信，可以开始测量 ==="
else
  echo "=== 自检失败：先修仪器，再谈测量 ==="
fi
exit "$FAIL"
