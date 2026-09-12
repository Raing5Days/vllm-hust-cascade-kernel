# ascend-kernel：自研算子工程（fa_fp32_stage1 + lse_merge + add_rms_norm_stats）

> 本工程 = CCE（Ascend C）算子的 torch extension：**共享前缀注意力核 `fa_fp32_stage1`**（fp32-out + LSE）、
> **LSE 空间合并核 `lse_merge`**（二者合起来实现 vLLM cascade decode 的"两段式 + 数值稳定合并"，是精度分层 Tier1 的算子底座），
> 以及 **F2 融合的 norm 阶段核 `add_rms_norm_stats`**（AddRmsNormBias→GEMM 融合立项第一程的测量仪器 + 阶段算子，见其 design.md）。
> wheel：`ascend_kernel-2026.9.12`（CANN 9.1.0 / torch_npu 2.13.0rc1 环境重编）；主 shape = Qwen2.5-14B（H=40/KVH=8/D=128，hidden 5120，bf16）。
> **单一事实源分工**：使用方法/场景/优势 = 本 README；算子内部设计 = `csrc/ops/<op>/design.md`；验证判据与用例 = `csrc/ops/<op>/test/*-test-cases.md`。

## 1. 三个算子一览

| | `fa_fp32_stage1` | `lse_merge` | `add_rms_norm_stats` |
|---|---|---|---|
| 一句话 | B=1 摊平读一段 paged KV 的 flash attention，**O 与 LSE 均以 fp32 输出** | 两分支注意力结果在 **LSE（log-sum-exp）空间数值稳定合并** | AddRmsNormBias 的**残差加 / 行统计 / norm 施加**三段解耦成三 mode（F2 融合的 norm 阶段核） |
| 签名 | `(q, key, value, block_table, actual_q_seqlens, actual_kv_seqlens, q_seqlen_value=0) -> (out, lse)` | `(o1, o2, lse1, lse2, out_code=0) -> out` | `(x1, x2, gamma, beta, eps, mode) -> (x_out, rstd, y)`（mode 0/1/2，未用输出为空 tensor） |
| 在 cascade 中的角色 | **stage-1**：全 batch 摊平读一遍共享前缀（消 n 遍重读） | **合并**：stage-1(fp32) × stage-2(bf16 suffix) → 最终输出 | **F2 融合前哨**：给出三种融合拓扑"暴露在 GEMM 之外"的下界，供 gate ② 判定；亦可作独立 norm 阶段算子 |
| 精度意义 | 长前缀 softmax 大归约以 fp32 累加（长上下文误差收敛） | 消 stage-1 bf16 舍入项，残余 ≈ w2·ε2 + ε_order | 与 CANN `npu_add_rms_norm_bias` golden 同舍入口径（目标是与现役链数值不可区分） |

## 2. 使用方法

### 2.1 安装与注册

```bash
pip install output/ascend_kernel-2026.9.12-cp312-cp312-linux_aarch64.whl --force-reinstall --no-deps
```

```python
import ascend_kernel  # import 即注册 torch.ops.npu.fa_fp32_stage1 / lse_merge / add_rms_norm_stats
# 验证：
assert hasattr(torch.ops.npu, "fa_fp32_stage1")
assert hasattr(torch.ops.npu, "lse_merge")
assert hasattr(torch.ops.npu, "add_rms_norm_stats")
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

### 2.5 `add_rms_norm_stats`（F2 融合的 norm 阶段核）

```python
x_out, rstd, y = torch.ops.npu.add_rms_norm_stats(
    x1,        # (M, K) bf16/fp16 连续；残差加左项（生产语义 = 上游 GEMM 输出）
    x2,        # (M, K) 同 dtype；残差加右项（生产语义 = 跨层 residual）；mode 1 忽略
    gamma,     # (K,) 同 dtype 或 None（mode 1 必需）
    beta,      # (K,) 同 dtype 或 None（mode 1 的 norm bias；Qwen2.5-14B 生产为 None）
    eps,       # float，> 0
    mode,      # 0 = 残差加 + 行统计 | 1 = 行统计 + norm 施加 | 2 = 只统计
)
# mode 0 -> (x_out (M,K), rstd (ceil(M/8),1) fp32, y=空)
# mode 1 -> (x_out=空,      rstd, y (M,K))
# mode 2 -> (x_out=空,      rstd, y=空)
# 语义（= CANN npu_add_rms_norm_bias golden）：
#   x_out = round_dtype(x1+x2);  rstd = 1/sqrt(mean_k(x_out^2)+eps)
#   y     = round_dtype( round_dtype(x_out*rstd) * gamma + beta )
```

- 用途有二：① **独立 norm 阶段算子**（把"残差加 / 行统计 / 施加"拆开，供上层按融合形态取用）；
  ② **F2 门控的测量仪器**——三 mode 分别给出"后继 GEMM prologue 融 norm（mode 0）"、
  "上游 epilogue 融残差加后重算 norm（mode 1）"、"两侧都融只留统计（mode 2）"三种拓扑
  中暴露在 GEMM 之外的下界（判据式见 `csrc/ops/add_rms_norm_stats/design.md` §4/§8）。
- 硬约束：2D 连续、两输入同 dtype（bf16/fp16，**不支持 fp32**）、`K % 16 == 0`、**`K <= 5120`**（v1 整行 UB 驻留）、
  `mode ∈ {0,1,2}`、`eps > 0`；mode 1 要求 `gamma.numel() == K`。
- `rstd` 按 `ceil(M/8)` 行分配（每核 8 行一组写 32B），第 M 行之后为 padding 零，消费取 `rstd[:M]`。

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

**wheel 兼容四元组**（`output/ascend_kernel-2026.9.12-cp312-cp312-linux_aarch64.whl`，2026-09-12 实编）：

| 维度 | 值 | 说明 |
|---|---|---|
| 平台 tag | `linux_aarch64` | `NpuExtension` 自动产出正确 tag，禁止假 `py3-none-any` 携带 .so |
| Python ABI | `cp312` | `_C.cpython-312-aarch64-linux-gnu.so` |
| torch / torch_npu ABI | torch 2.13.0+cpu + torch_npu 2.13.0rc1 | `_C.so` 的 torch 系 so 为 import 期延迟解析；torch 小版本升级必须重验 |
| CANN runtime | 9.1.0（`libascendcl` / `libopapi`，`/usr/local/ascend91`） | 仅运行时链接；目标机需 CANN runtime 在 `LD_LIBRARY_PATH` |

四元组任一变化 = 重编 + 重发布，并同步更新本表。
**回归范围声明（2026-09-12，新算子合入）**：本程全量重编（三个 kernel target），
`add_rms_norm_stats` 走完精度 64/64 + oracle 48/48 + 负例 + S1 锚点；
`fa_fp32_stage1` / `lse_merge` 在本程**只重编、未重跑各自 S1 位锚点与精度套件**
（上一次全量回归 = commit `73bf96a`，CANN 9.1.0）。按四元组纪律，消费方接 wheel 前
如需要 fa/lse 的重新背书，按下面顺序补跑：S1 bit 锚点 → 精度套件 → 图捕获冒烟。

- **链接纪律**：CANN/torch 的 so 一律运行时链接，**严禁打进 wheel**；发布前审计包内容，只允许 `ascend_kernel/_C*.so`、`ascend_kernel/lib/libascend_kernel.so` 与纯 Python 文件。
- **依赖方向**：本包只暴露 `torch.ops.npu.*`，**禁止 import 任何 vllm / vllm-ascend / 插件模块**；插件壳（`vllm-ascend-split-batch-hust`）单向依赖本包，本包对插件机制（manifest / entry point）不可见。

- **版本配对**：wheel 按特定 CANN/torch_npu 工具链编译（当前配 CANN 9.1.0 + torch_npu 2.13.0rc1 + 910B2/`CATLASS_ARCH=2201`）。**换 CANN/soc 必须重编**（`./build.sh`，~2min），并按顺序回归：S1 bit 锚点 → 精度套件 → 图捕获冒烟。
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
| add_rms_norm_stats：`K > 5120` / `K % 16 != 0` / 两输入不同 dtype / mode∉{0,1,2} / fp32 输入 | TORCH_CHECK 拒（v1 整行 UB 驻留 + 32B 行拷贝约束） |
| add_rms_norm_stats：AICore 无标量 `sqrtf`、且拒绝 `uint32→float` 强转 | 编译期即失败：1/K 由 host 传入、开方走向量 `Rsqrt`（勿在 kernel 里写 `sqrtf((float)kDim)`） |
| add_rms_norm_stats：行数分组写 rstd | 每核 8 行（32B）一写；4B 单写挂（与 fa 的 LSE 行 32B padded 同源教训），`rstd` 因此按 `ceil(M/8)` 行分配 |
| add_rms_norm_stats：`Rsqrt` 在 910B 上是 ~2^-11 近似 | 单用 `Rsqrt` 得到的 rstd 与 CANN 差 2.29e-3（已超 2^-7 档位）；kernel 内必须跟一次 Newton 细化（`r *= 1.5 − 0.5·a·r²`）才回到 ~2^-22（上板复验：1.13e-5） |
| add_rms_norm_stats：`rstd` 输出是 **(ceil(M/8), 1)** 二维 | 与 `(M,)` 的一维参考直接相减会被 torch **静默广播**成 `(M,M)` 两两配对，产出假精度结论（曾报 `rel_l2 0.64` 而逐元素最大相对误差只 0.066——对齐时 `rel_l2 ≤ MARE` 恒成立，见 test-cases.md §3.1）；比对前先 flatten，`test/test_add_rms_norm_stats_ref.py` 已固化该守卫 |
| add_rms_norm_stats：行内平方和的**归约精度**决定 mode-1 `y` 的舍入边界 | 本核 `ReduceSum` 在 K=5120 上的误差 ≈ **30 eps**，现役 CANN ≈ 0.5 eps；该差使 `mid=round_dtype(x·rstd)` 在 3e-4 的元素上翻转，个别撞上 `y` 边界 ⇒ `y` 越 CANN 档位（4 ulp，2/1048 万元素）。**与现役算子对照时，归一化类算子的归约误差必须做到 ~1e-7 级**（补偿求和/分段树），否则单元素绝对档位必然被点状击穿（test-cases.md §3.2） |
| add_rms_norm_stats：rstd staging 的 **MTE3→S 未同步候选** | `FlushRstd` 只在 `DataCopy` **前**排 barrier（覆盖 S→MTE3 正方向），**下一组** 8 次标量 `SetValue` 与该 MTE3 读之间没有守护 ⇒ 规范上是未同步 WAR（实测三轮 >100 万次 flush 未见错乱：两组间有 8 整行流水）。修法两行：flush 后 `SetFlag/WaitFlag<HardEvent::MTE3_S>`（首组跳过），或把 barrier 复制到每组写 staging **之前**。见 design.md §5.3 |
| **判据口径**：容差一律取 `.agents/skills/ops-precision-standard/`（浮点计算类的混合容差 + `matched_ratio ≥ 0.99`），**不是** CANN 自带用例的档位 | 用错档位会得到相反的验收结论（本 op：标准档位 64/64 通过 vs CANN 严档位 52/64）。写用例时必须按**输出** dtype 取表（fp32 的 `rstd` 用 fp32 行），并把更严档位作为**诊断列**保留；改档位/规则必须留 diff 可核对（本仓 `STD` / `CANN_TEST_TIER` 两块常量） |

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
| add_rms_norm_stats @M=2048,K=5120（F2 norm 段三 mode，bf16，**F2 冻结构建 `2026.9.12`**） | mode0 157.8 / mode1 153.1 / mode2 151.1µs；oracle `npu_add_rms_norm` 156.5µs | 910B2/CANN 9.1.0，warmup 20 + 100 次中位 × 3 轮 host wall，卡 7 + `flock /tmp/w3-npu.lock` |
| add_rms_norm_stats @同 shape（**D3 构建 `2026.9.12.post1`**，硬化的 batch+pair+Newton2） | host wall：mode0 147.3 / mode1 153.7 / mode2 134.8µs；oracle 144.2µs（同法同轮次，卡 7 空闲 HBM 5%） | 同上口径；**设备时长**以 profiler 为准（§9.2/§9.4），host wall 含 ~65–116µs 固定开销，二者不可混比 |
| **F2 gate ② 投影**（norm→GEMM 融合，判据式 design.md §4 实测前冻结） | A **−0.045%** / B''' 0.127% / B'' **0.200%** prefill | <1% ⇒ 按预注册规则**诚实关闭** A 变体；B 系上界一并上报 |
| F2 结论：norm 段"读带宽受限 565GB/s" | ⚠ **已由 D3 更正**：565 是 `wall − c` 反推值，不是实测；profiler 实测设备时长 54.45µs（mode0）/56.39µs（mode2）⇒ mode2 真实读带宽 **743GB/s**，且**瓶颈不是带宽**而是 VEC+scalar 指令（见下两行） | 该行原判定"融合收益被带宽地板锁在 ~0.2%"的**结论仍成立**（F2 判据式等价于 wall 直比，c 相消），但**内因解释作废** |
| **D3 剖因**（`torch_npu.profiler` 硬件时长 + 逐 pipe 占用，主 shape M=2048/K=5120 bf16） | mode0 54.45µs：**vec 0.629 / scalar 0.367** / mte2 0.217 / mte3 0.167；mode1 50.66µs（vec 0.789）；mode2 56.39µs（vec 0.635）。**同 62.91MB 的 CANN `aclnnAdd` 只需 17.54µs** ⇒ **3.13x 结构性头寸**，且 mte2 仅占 0.22（搬得动） | 归因：每行 17 条向量指令里 **10 条只服务 1 个标量**（Rsqrt+Newton 链 9 条 count-1 + 1 个 V→S barrier），且行间**无预取**（深度 2 队列未跨行流水） |
| **D3 改动（已硬化为唯一行为）** | ① rstd 链批量化到每 `RSTD_GROUP=8` 行一次（元素级等价，仅调度）→ mode0 **−11.9%**（54.45→47.96µs）、mode2 −11.6%；② 行和**二分折叠** + **第二次 Newton**（精度，见下） | 变体原始数据 `profiles/qwen14b-instruct-hotspot-20260910/f2-kernel/raw/d3_*`；开关式 A/B 已删除，host 常量 `kBatchStats/kPairSum` |
| **D3 gate 判定** | ① **通过**（标准档位 64/64 + 48/48）；② **不过** ⇒ 按预注册规则**关闭归档** | ② 最好读带宽 874GB/s（mode0,batch）/<1.0TB/s；交付配置 806/788GB/s；折算 pass 0.30%~0.54%（同口径设备时长差）**全部 <1%** |
| **D3 精度修复（mode-1 `y`）** | CANN 严档位越界 **5/16 → 0/16**（控制组 CANN 0/16）；标准档位 64/64 + 48/48 不变；设备套件 46/46 | 根因拆两层：① 行和顺序合并 ≈30 eps（二分折叠修）；② 向量 `Rsqrt` ~2^-11 近似 × 一次 Newton 残留 ~1.5ε²（第二次 Newton 修）——**实测定位**：行和做准后越界元素的 rstd 相对差几乎不动（−3.389e-6→−3.450e-6） |
| add_rms_norm_stats 精度（**容差按 ops-precision-standard**） | 64/64（vs CPU fp64 参考）+ 48/48（vs CANN oracle）；`matched_ratio` 全 1.0、`max_abs_error` 仅用掉上限 3.9% | 判据 = 该 skill 的混合容差 + `matched_ratio ≥ 0.99`；档位表逐字复制并有 CPU 用例对拍其 checker |
| add_rms_norm_stats 残余差距（严档位诊断列） | CANN 自带用例更严档位（2^-7/2^-10，全元素）下 mode-1 `y` 12 行 × 1–20 元素越档 | 根因 = 行内平方和归约 ≈30 eps（CANN 0.5 eps）翻转 `mid` 舍入边界；修法（补偿求和/分段树）已定位未实施 |
| add_rms_norm_stats 精度（CANN 严档位，**诊断列**） | 同批数据改用 CANN 自带用例档位（2^-7/2^-10，全元素）时对照 fp64 参考 52/64、CANN oracle 42/48；失败全在 mode-1 `y`、每例 1–20 元素 | 不参与判定（判定用上一行）；保留它使"与现役算子的边际距离"可复判，见 test-cases.md §3.2/§3.3 |

## 7. 文档与代码索引

| 内容 | 路径 |
|---|---|
| fa_fp32_stage1 设计（含 §11 LSE 竞态缺陷史） | `csrc/ops/fa_fp32_stage1/design.md` |
| fa_fp32_stage1 测试用例与判据 | `csrc/ops/fa_fp32_stage1/test/fa_fp32_stage1-test-cases.md` |
| fa_fp32_stage1 回归脚本 | `test/test_fa_fp32_stage1_smoke.py`（S1 锚点）、`run_precision_suite.py`（30/30）、`test_q_seqlen_fastpath.py`、`test_lse_flatten_regression.py` |
| lse_merge 设计 / 用例 | `csrc/ops/lse_merge/design.md` / `test/lse_merge-test-cases.md` |
| add_rms_norm_stats 设计（F2 融合选型 + 预注册 gate ② 判据式） | `csrc/ops/add_rms_norm_stats/design.md` |
| add_rms_norm_stats D3（读带宽立项）：剖因 + 变体 + 判据 | `csrc/ops/add_rms_norm_stats/design.md` §9 |
| add_rms_norm_stats IO 归因工具（profiler pipe 表 + wall/pipelined 扫描） | `csrc/ops/add_rms_norm_stats/test/run_io_profile.py`、`parse_kernel_pipe_csv.py`（用法见文件头） |
| add_rms_norm_stats 行和累加深度 CPU 研究（校准到实测） | `csrc/ops/add_rms_norm_stats/test/run_sum_accuracy_study.py` |
| add_rms_norm_stats 用例与判据 / 精度报告 / S1 锚点脚本 | `csrc/ops/add_rms_norm_stats/test/add_rms_norm_stats-test-cases.md`（§3 判据（容差按 ops-precision-standard）+ §3.1 判据实现缺陷 + §3.2 CANN 严档位下的 `y` 归约精度差距 + §3.3 标准档位复测） / `f2_prec.{json,md}`（标准档位运行）+ `f2_prec_r2_cann_tier.{json,md}`（严档位运行） / `run_s1_anchor.py` / `run_golden_selfcheck.py`（两档位 × 三种配对，含"判据是否可达"的控制实验） / `test_add_rms_norm_stats_ref.py`（CPU-only 判据守卫，含与 skill checker 的对拍） |
| F2 立项与判定记录（画像侧） | `profiles/qwen14b-instruct-hotspot-20260910/f2-kernel/` |
| 注册面 | `csrc/register.cpp`（torch.ops.npu schema） |
| 构建 | `./build.sh`（CATLASS_ARCH=2201 源内 define；catlass 整树在 `third_party/catlass/include/`） |
| 生产消费者（插件） | `vllm-ascend-split-batch-hust/src/vllm_ascend_split_batch/cascade_{plugin,graph_plugin,runner_patch,gate,gate_self}.py` |
| 测量报告 | `cascade-c3-results/probes/W0-B2-适配测量报告.md`（B2 适配）、`W1-stageB-报告.md`（融合线 NO-GO 依据） |
| 知识库 | `资料/flashinfer-移植知识库/02-本地对照面.md`（自研算子与 CANN FIA 事实条目） |
| 多人协作/新算子流程 | `../docs/multi-operator-dev.md` |
