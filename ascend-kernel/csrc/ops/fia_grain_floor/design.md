# fia_grain_floor —— B1 粒度阶梯的测量专用核（KV 搬运粒度扫描）

> **性质**：**measurement-only**。服务于 `profiles/qwen14b-instruct-hotspot-20260910/probe-b1-floor/`
> 的 WP5 粒度阶梯实验；**不接任何 e2e 路径、不进插件 bundle、不参与 manifest、不得用于生产**。
> 判决与全部实测写在 `probe-b1-floor/REPORT-GRAIN.md`；本文件只写算子内部设计。

## 1. 它是什么 / 不是什么

它是生产算子 `csrc/ops/fa_fp32_stage1/` 的**逐字节副本**（owner = 吴天宇），唯一改动是：

- 入口名 `fa_fp32_stage1` → `fia_grain_floor`（host schema、launch、device `CATLASS_GLOBAL` 入口）；
- KV 栈深 `blockStackNum` 从硬编码 `4` 改为**源内 define** `B1_GRAIN_STACK_NUM`（默认 4）。

**它不是一个可用的新算子**：语义与 `fa_fp32_stage1` 完全相同（maskType=0 全网格、paged KV / GQA /
blockSize=128 / D=128 / bf16），存在的唯一意义是让「同一二进制、同一 shape、唯一变量 = KV 搬运粒度」
这件事可测。改动 `fa_fp32_stage1/` 任何文件是**明令禁止**的（生产算子）；因此必须做这份副本。

## 2. 被扫的单变量

`blockStackNum` = 一次 K/V 搬运里堆叠的分页块数（KV 栈行数 = `blockStackNum × blockSize(=128)`）。
源里两处使用它（AIC 与 AIV 各一次）：

- `op_kernel/kernel_fia_grain_floor.cpp:247`（AIC 主循环）
- `op_kernel/kernel_fia_grain_floor.cpp:557`（AIV 主循环）

两者必须同时改（阶梯脚本用同一个 define 供值，保证一致）。

| `B1_GRAIN_STACK_NUM` | KV 栈行数 | 说明 |
|---|---|---|
| 1 | 128 | 最碎：每页一搬 |
| 2 | 256 | |
| 4 | 512 | **= 生产默认**（`fa_fp32_stage1:223` 的 `= 4`） |
| 8 | 1024 | 工作区临界（见 §3） |
| 16 | 2048 | 预期越界（见 §3） |

## 3. 约束与预期失败面（写死，测前预注册）

工作区每核每 slot = `WORKSPACE_BLOCK_SIZE_DB = 131072` 元素（`kernel_common.hpp:24`；host 按
`aiCoreNum × 131072 × 4B × 3 slot` 分配，`op_host/fia_grain_floor.cpp` 的 `kWorkspaceBlockSizeDb/kPreLaunch`）。
本 shape（q=2048 / B=1 / H=40 / KVH=8 / D=128）`rowNum = qSBlockSize(128) × qNBlockSize(1) = 128`，
S/P 共用该 slot。

- S 槽需求 = `rowNum × (blockStackNum×128)` **fp32** = `128 × stackRows × 4B`。
- `blockStackNum=8` ⇒ `128×1024×4 = 512KB` = 恰好填满 slot（**临界**）。
- `blockStackNum=16` ⇒ `128×2048×4 = 1MB` > slot ⇒ **预期越界**（记录并跳过，不做补救）。

同时 `blockStackNum=16` 的 QK K-tile = `128(embed) × 2048` bf16 = 512KB ≥ L1 容量，也可能编译期/运行期失败。
**任何编译失败、越界、崩溃、非有限输出都原样记录，不掩盖。**

源内 define 纪律：`B1_GRAIN_STACK_NUM` **只能在源内 define**（同 `CATLASS_ARCH` 的先例），
**禁止 target-wide `-D`**——那会泄漏进同 target 的其他编译单元（仓内已有先例：`kernel_fa_fp32_stage1.cpp`
头注 + `docs/multi-operator-dev.md` §1.2）。阶梯每次构建前用 `sed` 改源内默认值，再全量重建。

## 4. 构建接入

照 `f3_floor` 的 measurement-only 先例：

- `csrc/CMakeLists.txt`：`OP_SRCS` 加 `op_host/fia_grain_floor.cpp`；
  新增 `ascendc_library(no_workspace_kernel_graingfloor STATIC … op_kernel/kernel_fia_grain_floor.cpp)`
  + catlass include 目录（`third_party/catlass/include` 与 CANN aarch64 include）；加入 `target_link_libraries`。
- `csrc/ops.h` / `csrc/register.cpp`：加 schema（照 `fa_fp32_stage1` 写法）。
- `aclrtlaunch_fia_grain_floor.h` 由 `ascendc_library` 的 INTERFACE include 自动生成并暴露（同 f3）。

## 5. 自证（最低核验，非逐位对拍）

本核是测量用，**不做**逐位对拍；每次运行只要求：

1. `torch.isfinite(out).all() == True`（防测到被优化掉/算错的核）；
2. 记录 `out` 的 mean / absmax，跨粒度**稳定**（同一输入随机种子）；
3. `lse` 形状 = `[T*H*8]`。

## 6. 使用

```
# 全阶梯（构建 + 采集 + 解析）：
bash profiles/qwen14b-instruct-hotspot-20260910/probe-b1-floor/grain_ladder.sh
# 单格：
ASCEND_RT_VISIBLE_DEVICES=6 python profiles/.../probe-b1-floor/run_grain.py 4 <outdir>
```

装载：`PYTHONPATH=<repo>/python/ascend_kernel`（`import ascend_kernel` 只做
`torch.ops.load_library(lib/libascend_kernel.so)`，无需 pip）。

**构建注意**：必须 `rm -rf build` 全量重建（`build.sh` 已内建）；增量重建会在
`merge_aic/aiv_obj_text` 处报 `ld.lld: … unknown file type`（本仓既有问题，见 f3_floor/design.md §7）。

## 7. 与生产算子的关系（2026-09-16 更新）

- 本核是 `fa_fp32_stage1` 的副本，**唯一差异** = 入口名 + `B1_GRAIN_STACK_NUM` 这个独立旋钮。
- **生产侧已加契约守卫**（栈深派生自 catlass 模板常量 + 两条 `static_assert`，见
  `fa_fp32_stage1/design.md` §12）⇒ 生产侧**不再**存在"硬编码 4"，也因此**无法偏离**契约。
  本核**刻意保留独立旋钮**，正是为了把契约被破坏时的失败模式继续留在可复现范围内。
- 本核同样带那两条断言，但它们只约束**模板侧**（恒 4×128==512），
  因此**不妨碍**扫 `STACKN` 到 1/2/8/16（那才是本核的用途）。
- 根因分析（为何只有 4 可跑）：`probe-b1-floor/REPORT-HANG.md`。
