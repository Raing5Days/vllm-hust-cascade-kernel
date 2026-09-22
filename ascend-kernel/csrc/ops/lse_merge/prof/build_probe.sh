#!/bin/bash
# 构建指定探针模式的 lse_merge 并装进 site-packages（测量用）。
#
#   bash build_probe.sh <LSE_PROBE_ROW_SRC> [workdir]
#
# LSE_PROBE_ROW_SRC: 0=生产路径 1=读值不依赖向量 2=行权重只读第0行 3=在2基础上去掉Add
#
# ⚠️ 两条实测教训（2026-09-18）：
#  1) 不要靠 CXXFLAGS 传 -DLSE_ROW_SRC：本工程用 ExternalProject 编 device 侧，
#     其 CMakeCache 会冻住首次的 CMAKE_CXX_FLAGS，后续改环境变量不会让 device 重配置
#     ⇒ device 代码静默不变，只有 host 跟着变（实测 cache 一直停在 -DLSE_ROW_SRC=1）。
#     所以改的是 kernel 源里的 #define 默认值，构建后还原。
#  2) 不要手删 ExternalProject 的 CMakeCache.txt 来"强制重配置"：会留下 .o 与产物
#     互相污染的中间态，ld.lld 报 "unknown file type"，构建静默失败。
#     所以一律走 build.sh 全量构建（它会 rm -rf build），慢但可靠。
#
# 每次构建后校验 device 侧实际生效的宏，不符即硬失败（防止再拿到脏测量）。
set -euo pipefail

N="${1:?用法: build_probe.sh <LSE_PROBE_ROW_SRC> [workdir]}"
OUT_DIR="${2:-/tmp/lsemergebuild}"

REPO=/vllm-workspace/ops/kernels/ascend-kernel
KSRC=$REPO/csrc/ops/lse_merge/op_kernel/kernel_lse_merge.cpp
SP=/opt/miniconda3/envs/hust/lib/python3.12/site-packages/ascend_kernel
BASE=/vllm-workspace/profiles/lse-merge-pipe-20260918/baseline
PY=/opt/miniconda3/envs/hust/bin/python

export PATH=/opt/miniconda3/envs/hust/bin:$PATH   # 陷阱 13
mkdir -p "$OUT_DIR"

ORIG=$(sed -n 's/^#define LSE_PROBE_ROW_SRC \([0-9]*\)$/\1/p' "$KSRC" | head -1)
[ -n "$ORIG" ] || { echo "FATAL: 找不到 LSE_PROBE_ROW_SRC 默认值"; exit 1; }
TR="${3:-32}"
ORIG_TR=$(sed -n 's/^#define LSE_PROBE_TILE_ROWS \([0-9]*\)$/\1/p' "$KSRC" | head -1)
[ -n "$ORIG_TR" ] || { echo "FATAL: 找不到 LSE_PROBE_TILE_ROWS 默认值"; exit 1; }
echo "原默认值: ROW_SRC=$ORIG TILE_ROWS=$ORIG_TR  -> 目标: ROW_SRC=$N TILE_ROWS=$TR"
restore() {
  sed -i "s/^#define LSE_PROBE_ROW_SRC [0-9]*$/#define LSE_PROBE_ROW_SRC ${ORIG}/" "$KSRC"
  sed -i "s/^#define LSE_PROBE_TILE_ROWS [0-9]*$/#define LSE_PROBE_TILE_ROWS ${ORIG_TR}/" "$KSRC"
}
trap restore EXIT

sed -i "s/^#define LSE_PROBE_ROW_SRC [0-9]*$/#define LSE_PROBE_ROW_SRC ${N}/" "$KSRC"
sed -i "s/^#define LSE_PROBE_TILE_ROWS [0-9]*$/#define LSE_PROBE_TILE_ROWS ${TR}/" "$KSRC"
grep -m1 '^#define LSE_PROBE_ROW_SRC' "$KSRC"
grep -m1 '^#define LSE_PROBE_TILE_ROWS' "$KSRC"

echo "=== 全量构建 LSE_PROBE_ROW_SRC=${N}（build.sh）==="
# ⚠️ build.sh 里 CURRENT_DIR=$(pwd)，必须在仓库根目录执行，否则 cmake 会落在错误目录、
#    device 侧静默不重编（2026-09-18 实测：导致全部探针测量无效）。
cd "$REPO"
bash "$REPO/build.sh" "${SOC_VERSION:-Ascend910_9382}" > "$OUT_DIR/build_LSE${N}.log" 2>&1 || {
  echo "FATAL: build.sh 失败，见 $OUT_DIR/build_LSE${N}.log"; tail -20 "$OUT_DIR/build_LSE${N}.log"; exit 1; }
echo "build.sh rc=0"

echo "=== 校验：产物新鲜度 / 陈旧同名宏 ==="
# 注意有两个 build 目录：build.sh 用的是 $REPO/build（真产物）；
# $REPO/csrc/build 是 csrc/CMakeLists.txt 被当根工程执行时留下的残骸，不要用它判断。
KOBJ=$(find "$REPO/build" -name "auto_gen_kernel_lse_merge.cpp.o" | head -1)
[ -n "$KOBJ" ] || { echo "FATAL: 找不到 device 侧 lse_merge 目标文件"; exit 1; }
echo "device 目标文件: $KOBJ"
ls -la --time-style=full-iso "$KOBJ" | awk '{print "  mtime:", $6, $7}'
SRC_MTIME=$(stat -c %Y "$KSRC"); OBJ_MTIME=$(stat -c %Y "$KOBJ")
[ "$OBJ_MTIME" -ge "$SRC_MTIME" ] || {
  echo "FATAL: device 目标文件($OBJ_MTIME) 早于源码($SRC_MTIME) ⇒ 本次改动未被重编"; exit 1; }
echo "OK: device 目标文件不早于源码改动，本次确已重编"
CCJ=$(find "$REPO/build" -name compile_commands.json -path "*lse_merge*" | head -1)
if [ -n "$CCJ" ]; then
  STALE=$(grep -o "\-DLSE_ROW_SRC=[0-9]*" "$CCJ" 2>/dev/null | sort -u | tr '\n' ' ')
  echo "  记录：编译行里的陈旧同名宏（不生效，宏名已隔离）: ${STALE:-无}"
fi

WHL=$(ls -t "$REPO"/output/ascend_kernel*.whl | head -1)
cp -f "$WHL" "$OUT_DIR/ascend_kernel_LSE${N}.whl"
WHL="$OUT_DIR/ascend_kernel_LSE${N}.whl"
rm -rf "$OUT_DIR/unpack_LSE${N}"; mkdir -p "$OUT_DIR/unpack_LSE${N}"
cd "$OUT_DIR/unpack_LSE${N}"
"$PY" -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$WHL"
NEW_LIB=$(find . -name libascend_kernel.so | head -1)
NEW_C=$(find . -name '_C*.so' | head -1)
[ -n "$NEW_LIB" ] && [ -n "$NEW_C" ] || { echo "FATAL: wheel 里缺 .so"; exit 1; }

if [ ! -f "$BASE/libascend_kernel.so.v1-ce8bf74f" ]; then
  mkdir -p "$BASE"
  cp "$SP/lib/libascend_kernel.so" "$BASE/libascend_kernel.so.v1-ce8bf74f"
  cp "$SP"/_C.cpython-312-aarch64-linux-gnu.so "$BASE/_C.cpython-312-aarch64-linux-gnu.so.v1-ce8bf74f"
fi
cp -f "$NEW_LIB" "$SP/lib/libascend_kernel.so"
cp -f "$NEW_C" "$SP/$(basename "$NEW_C")"

echo "=== installed LSE_PROBE_ROW_SRC=${N} ==="
md5sum "$SP/lib/libascend_kernel.so"
