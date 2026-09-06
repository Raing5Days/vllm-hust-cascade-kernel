# ascend-kernel：cascade 自研算子工程（fa_fp32_stage1 + lse_merge）

> 本工程 = 两个 CCE（Ascend C）算子的 torch extension：**共享前缀注意力核 `fa_fp32_stage1`**（fp32-out + LSE）与 **LSE 空间合并核 `lse_merge`**。二者合起来实现 vLLM cascade decode 的"两段式 + 数值稳定合并"，是精度分层（Tier1 fp32）的算子底座。
> wheel：`ascend_kernel-2026.3.9`（已安装、与源码树构建一致）；主 shape = Qwen2.5-14B decode（H=40/KVH=8/D=128，bf16）。
> **单一事实源分工**：使用方法/场景/优势 = 本 README；算子内部设计 = `csrc/ops/<op>/design.md`；验证判据与用例 = `csrc/ops/<op>/test/*-test-cases.md`。

## 1. 两个算子一览

| | `fa_fp32_stage1` | `lse_merge` |
|---|---|---|
| 一句话 | B=1 摊平读一段 paged KV 的 flash attention，**O 与 LSE 均以 fp32 输出** | 两分支注意力结果在 **LSE（log-sum-exp）空间数值稳定合并** |
| 签名 | `(q, key, value, block_table, actual_q_seqlens, actual_kv_seqlens, q_seqlen_value=0) -> (out, lse)` | `(o1, o2, lse1, lse2, out_code=0) -> out` |
| 在 cascade 中的角色 | **stage-1**：全 batch 摊平读一遍共享前缀（消 n 遍重读） | **合并**：stage-1(fp32) × stage-2(bf16 suffix) → 最终输出 |
| 精度意义 | 长前缀 softmax 大归约以 fp32 累加（长上下文误差收敛） | 消 stage-1 bf16 舍入项，残余 ≈ w2·ε2 + ε_order |

## 2. 使用方法

### 2.1 安装与注册

```bash
pip install output/ascend_kernel-2026.3.9-cp312-cp312-linux_aarch64.whl --force-reinstall --no-deps
```

```python
import ascend_kernel  # import 即注册 torch.ops.npu.fa_fp32_stage1 / lse_merge
# 验证：
assert hasattr(torch.ops.npu, "fa_fp32_stage1")
assert hasattr(torch.ops.npu, "lse_merge")
```

### 2.2 `fa_fp32_stage1`（stage-1 前缀注意力）

```python
out, lse = torch.ops.npu.fa_fp32_stage1(
    q,                        # (T, H, D) bf16 连续；T = Σactual_q_seqlens
    k_blocks, v_blocks,       # (numBlocks, 128, KVH, D) bf16 连续；blockSize 恒 128
    block_table,              # (B, cols) int32；cols = 行 stride ≥ ceil(maxKv/128)
    actual_q_seqlens,         # (B,) int64 device
    actual_kv_seqlens,        # (B,) int64 device
    q_seqlen_value,           # int，见下
)
# out: (T, H, D) **fp32**；lse: (T*H*8,) fp32 —— 每 (t,h) 行 8×fp32 复制的 32B padded 行
# lse 消费：lse.view(T*H, 8)[:, 0]  ←与 lse_merge stride=8 零拷贝直传
```

- **两种 q 形态**：
  - `q_seqlen_value=1`（**decode/cascade 生产形态**）：B 个请求各 q_len=1，T=B。op_host 零 D2H、tiling 纯 shape 计算（**图捕获安全**，实测比默认路径快 40%：225→133.9µs @C3 shape）；
  - `q_seqlen_value=0`（默认 legacy）：允许每请求不同 q_len（T=Σq_seqlens 摊平），op_host 两次 D2H 拉 seqlen——泛化/测试用，**不可图捕获**。
  - 变长 kv：两形态都支持（kv seqlen 留在 device，kernel 按 `ceil(kv/128)` 读块；`block_table` 的 cols 是容量上界契约）。
- **硬约束**（op_host TORCH_CHECK）：D==128、blockSize==128、H % KVH == 0、全部输入连续、q_seqlen_value<0 拒。

### 2.3 `lse_merge`（LSE 空间合并）

```python
merged = torch.ops.npu.lse_merge(
    o1,        # (..., D)：bf16（Tier0）或 fp32（Tier1，直接吃 stage-1 fp32 out）
    o2,        # (..., D) bf16（stage-2 输出）
    lse1,      # fp32；行 stride = numel/行数：1=compact（FIA v2 .out），8=padded（fa_fp32_stage1 直传）
    lse2,      # 同上
    out_code,  # 0=跟随 o1 dtype（默认）| 1=bf16（合并直进 bf16 output，省 cast）| 2=fp32（量化前残差探针）
)
```

- 数学：`m=max(lse1,lse2); out=(o1·exp(lse1−m)+o2·exp(lse2−m))/(exp(lse1−m)+exp(lse2−m))`，全 fp32 中间计算，任意有限 lse 安全（先减 max，无溢出路径）。
- 约束：`dim % 16 == 0`、o2 必须 bf16、lse 必须 fp32、stride∈[1,1024]；o1/o2 任意有限值。

### 2.4 两段式最小语义例（stage-1 + stage-2 + merge）

```python
import math, torch, torch_npu
from torch_npu import npu_fused_infer_attention_score_v2
import ascend_kernel  # noqa: F401

B, H, KVH, D, BS = 64, 40, 8, 128, 128
shared, suffix = 4356, 260
scale = 1.0 / math.sqrt(D)
q = (torch.randn(B, H, D, device="npu") * 0.5).to(torch.bfloat16)       # 每 token q_len=1
pool = (torch.randn(100, BS, KVH, D, device="npu") * 0.5).to(torch.bfloat16)  # paged KV 池

# stage-1：全 batch 摊平读共享前缀（前 sb 个块，q_seqlen_value=B fast path）
sb = shared // BS
bt1 = torch.arange(sb, dtype=torch.int32, device="npu").unsqueeze(0)
o1, l1 = torch.ops.npu.fa_fp32_stage1(
    q, pool[:sb], pool[:sb], bt1,
    torch.tensor([B], dtype=torch.int64, device="npu"),
    torch.tensor([shared], dtype=torch.int64, device="npu"), B)
# o1 (B,H,D) fp32；l1 (B*H*8,) fp32 padded（stride=8，零拷贝直传 lse_merge）

# stage-2：每请求 paged 读各自 suffix（CANN FIA v2，BNSD per-request 形态）
suf_b = (suffix + BS - 1) // BS
k_suf = pool[sb:sb + suf_b].transpose(1, 2).contiguous()
v_suf = pool[sb + 10:sb + 10 + suf_b].transpose(1, 2).contiguous()
bt2 = torch.arange(suf_b, dtype=torch.int32, device="npu").repeat(B, 1)
q3d = q.unsqueeze(2)
ws = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
    query=q3d, key=k_suf, value=v_suf, block_table=bt2,
    input_layout="BNSD", block_size=BS, actual_seq_qlen=[1]*B,
    actual_seq_kvlen=[suffix]*B, num_key_value_heads=KVH,
    softmax_scale=scale, num_query_heads=H)
o2 = torch.empty(B, H, 1, D, dtype=torch.bfloat16, device="npu")
l2 = torch.empty(B, H, 1, dtype=torch.float32, device="npu")
torch_npu.npu_fused_infer_attention_score_v2.out(
    query=q3d, key=k_suf, value=v_suf, block_table=bt2,
    input_layout="BNSD", block_size=BS, actual_seq_qlen=[1]*B,
    actual_seq_kvlen=[suffix]*B, num_key_value_heads=KVH,
    num_query_heads=H, softmax_scale=scale, workspace=ws, out=(o2, l2))

# merge：stride 自动推断（lse1: B*H*8/B*H → 8=padded；lse2: numel=行数 → 1=compact）
merged = torch.ops.npu.lse_merge(o1, o2.squeeze(2), l1, l2.reshape(B * H))
# merged (B,H,D) bf16（out_code=0 跟随 o1 的 fp32；集成形态用 out_code=1 直出 bf16）
```

## 3. 使用场景

### 3.1 设计场景：vLLM cascade decode（当前唯一生产消费者）

两段式的由来：CANN FIA 的 prefix-shared 输入与 paged KV **互斥（561002 硬拦截）**，vLLM decode 下 B 个请求对共享前缀 KV 的 n 遍重读无法用单 kernel 消除——必须 stage-1 摊平读一遍 + stage-2 suffix-only + LSE merge。

生产接线（**宿主零修改，全部插件侧**）：

| 环节 | 位置 |
|---|---|
| 插件入口 | `vllm-ascend-split-batch-hust`，`vllm.general_plugins` entry `cascade-attention`，分支 `feat/cascade-attention-plug` |
| 激活 | env `VLLM_ASCEND_ENABLE_CASCADE_DECODE=1`（默认关，default-off 零差异）；图模式另需 `VLLM_ASCEND_ENABLE_CASCADE_GRAPH` |
| eager 路径 | backend `build()` 透传 `cascade_shared_len` → `_forward_cascade_decode` 两段式（Tier0 bf16 / Tier1 fp32 由 `VLLM_ASCEND_CASCADE_PRECISION` 选） |
| graph 路径 | 图级 cascade plugin：stage-1/stage-2 = FIA task group，capture 期接入，replay 期 update-before-replay 重参数化；**stage-1 用 `q_seqlen_value=B` fast path（零 D2H，捕获安全）**；`lse_merge(o1, o2, l1, l2)` 图内直调，失败 fail-open torch 合并式 |
| 自适应 gate | `cascade_gate_self.py` 子进程 micro-bench（BNSD 形态），按 (B, shared) 桶实测决定走不走 cascade，亏区自动回落 FULL |

### 3.2 优势场景（何时值得开）

| 条件 | 说明 |
|---|---|
| **共享前缀长且占比极高**（8k+ 系统提示词/多轮对话，大量请求共享） | 主要收益来源：均匀等长 batch 铺满 tile + 前缀只读一遍；实测 8k 段 e2e −6.6%~−28%、16k 段 −21%~−38%（910B2） |
| **B 大**（64~256） | 摊薄两段式固定开销（第二次 launch/merge/重参数化） |
| **需要长上下文精度** | Tier1（fp32 stage-1 + fp32 LSE 合并）无条件值得：前缀大归约 fp32 化，成本仅比全 bf16 略高。全链 vs fp64 误差 7.574e-3，merge 抑制因子实测 0.069 ≈ 理论 w2 |
| **图模式** | 等长 shared 段桶化干净，捕获参数面小；fast path 免每步 D2H 同步 |

### 3.3 不适用 / 亏本场景

- **B 小 + suffix 占比不小**（如 4k shared × B=64 × suffix 260 的 C3 形态）：算术量 4096+260 ≈ full 4356 无节省，固定开销主导，**实测净亏 ~31%**——靠 gate 自动回落，不要手动常开；
- **prefix 短或请求间几乎无共享**：均匀收益不存在，纯付两段开销；
- **prefill（q_len>1）**：stage-1 的摊平形态支持 q_len>1（`q_seqlen_value=0`），但该场景未做产品化验证，图捕获路径仅覆盖 decode。

### 3.4 潜在复用场景（未验证，仅记录）

`lse_merge` 是通用"两分支注意力结果概率空间合并"原语：speculative decoding 的 draft/target 合并、PD 分离的跨实例 KV 续算、上下文并行的分段 softmax 归并——数学同构，均属潜在场景，**未做任何验证**，使用前按测试模板补对照。

## 4. 环境配对与迁移纪律（重要）

**wheel 兼容四元组**（`output/ascend_kernel-2026.3.9-cp312-cp312-linux_aarch64.whl`，ldd 实测 2026-09）：

| 维度 | 值 | 说明 |
|---|---|---|
| 平台 tag | `linux_aarch64` | `NpuExtension` 自动产出正确 tag，禁止假 `py3-none-any` 携带 .so |
| Python ABI | `cp312` | `_C.cpython-312-aarch64-linux-gnu.so` |
| torch / torch_npu ABI | torch 2.10.0 + torch_npu 2.10.0.post2 | `_C.so` 的 torch 系 so 为 import 期延迟解析；torch 小版本升级必须重验 |
| CANN runtime | 9.0.1（`libascendcl` / `libopapi`） | 仅运行时链接；目标机需 CANN runtime 在 `LD_LIBRARY_PATH` |

四元组任一变化 = 重编 + 重发布，并同步更新本表。

- **链接纪律**：CANN/torch 的 so 一律运行时链接，**严禁打进 wheel**；发布前审计包内容，只允许 `ascend_kernel/_C*.so`、`ascend_kernel/lib/libascend_kernel.so` 与纯 Python 文件。
- **依赖方向**：本包只暴露 `torch.ops.npu.*`，**禁止 import 任何 vllm / vllm-ascend / 插件模块**；插件壳（`vllm-ascend-split-batch-hust`）单向依赖本包，本包对插件机制（manifest / entry point）不可见。

- **版本配对**：wheel 按特定 CANN/torch_npu 工具链编译（当前配 CANN 9.0.1 + torch_npu 2.10.0.post2 + 910B2/`CATLASS_ARCH=2201`）。**换 CANN/soc 必须重编**（`./build.sh`，~2min），并按顺序回归：S1 bit 锚点 → 精度套件 → 图捕获冒烟。
- **正确性权威锚点**：S1 = 与 catlass example 二进制同输入 O/LSE **逐 bit 相等**（仅 q_len=1 域有效；摊平 q_len>1 域的权威 = fp64 对拍，见 design.md §11.4 历史口径订正）。
- **同进程危害**（W0 实测）：`fa_fp32_stage1` 执行后，同进程再跑 kv≥8k 的 FIA v2 TND 块表形态触发 fftsplus aicore 0x800000——micro-bench 必须分组进程（生产不混用，无风险）。
- **线程约束**：op 内 tiling 暂存区 registry 设计前提是 vllm worker 单线程调用；同 device 并发调用会竞争。

## 5. 已知边界与坑（红线速查）

| 红线 | 后果 |
|---|---|
| D≠128 或 blockSize≠128 | TORCH_CHECK 拒（L1TileShape 硬约束） |
| 摊平形态子块行数 >32（T≥17 @group=5） | 已修复的 LSE staging 竞态历史缺陷；判据与回归见 design.md §11，升级改动后必跑 `test_lse_flatten_regression.py` |
| 图捕获内 D2H/同步拷贝 | capture 拒绝（107027/107030）——本 op 的 v4 暂存区（pinned+non_blocking+内容去重）已内建，自写 op 引以为鉴：**进程级 registry 严禁持有需析构的 at::Tensor**（退出 GIL abort） |
| CANN FIA v2（stage-2 搭档）的 TND 均匀变长+无 mask 角落 | 静默 NaN/aicore 异常（bug report 草稿在 `cascade-c3-results/probes/`）；生产形态（ragged+mask）与 BNSD 形态均安全；**任何 FIA 计时前先同数据对拍正确性，存活≠正确** |
| lse_merge 输入不连续 / dim 非 16 倍数 | TORCH_CHECK 拒 |

## 6. 实测锚点（910B2 / CANN 9.0.1，2026-09）

| 项 | 数字 | 口径 |
|---|---|---|
| fa_fp32_stage1 @T=64,kv=4356（fast path） | 133.9–135.6µs | eager 上界；4k→8k 平坦（固定开销主导）；B3 锚点 64.2µs |
| fa_fp32_stage1 fast vs default | −40%（133.9 vs 225.0µs） | 同 shape 同法 host wall |
| lse_merge @B=64,H=40 | 25.4–33.6µs（图内 30.5µs，eager 26.2µs 首版基线） | M2/W0 |
| 全链精度（stage1+FIA+merge vs fp64） | 7.574e-03（与生产路径完全一致） | C3 形态 |
| merge 抑制因子 | 0.069 ≈ 理论 w2=0.059 | Tier1 数值主张直接证据 |
| S1 bit 锚点 | O/LSE 逐 bit 相等（example 二进制） | q_len=1 域 |
| e2e（插件两段式 + gate） | 8k 段 −6.6%~−28%；16k 段 −21%~−38%；4k×B64 亏 ~31%（gate 自动回落） | 9/9 格 × 两轮，Qwen2.5-Coder-14B 替身 |

## 7. 文档与代码索引

| 内容 | 路径 |
|---|---|
| fa_fp32_stage1 设计（含 §11 LSE 竞态缺陷史） | `csrc/ops/fa_fp32_stage1/design.md` |
| fa_fp32_stage1 测试用例与判据 | `csrc/ops/fa_fp32_stage1/test/fa_fp32_stage1-test-cases.md` |
| fa_fp32_stage1 回归脚本 | `test/test_fa_fp32_stage1_smoke.py`（S1 锚点）、`run_precision_suite.py`（30/30）、`test_q_seqlen_fastpath.py`、`test_lse_flatten_regression.py` |
| lse_merge 设计 / 用例 | `csrc/ops/lse_merge/design.md` / `test/lse_merge-test-cases.md` |
| 注册面 | `csrc/register.cpp`（torch.ops.npu schema） |
| 构建 | `./build.sh`（CATLASS_ARCH=2201 源内 define；catlass 整树在 `third_party/catlass/include/`） |
| 生产消费者（插件） | `vllm-ascend-split-batch-hust/src/vllm_ascend_split_batch/cascade_{plugin,graph_plugin,runner_patch,gate,gate_self}.py` |
| 测量报告 | `cascade-c3-results/probes/W0-B2-适配测量报告.md`（B2 适配）、`W1-stageB-报告.md`（融合线 NO-GO 依据） |
| 知识库 | `资料/flashinfer-移植知识库/02-本地对照面.md`（自研算子与 CANN FIA 事实条目） |
| 多人协作/新算子流程 | `../docs/multi-operator-dev.md` |
