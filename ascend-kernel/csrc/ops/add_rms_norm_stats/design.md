# add_rms_norm_stats 设计文档（F2「AddRmsNormBias→GEMM 融合」第一程，2026-09-12）

> 立项依据：`profiles/qwen14b-instruct-hotspot-20260910/`（D3/P3 = `AddRmsNormBias` 1.83%(D) / 2.94%(P)）
> 与 `fusion-candidates.md` F2 行。**预注册 gate（用户 2026-09-12 立项时给定，不得事后放宽）**：
> ① 精度套件达标（对照参考实现，fp16/bf16 混合容差）；
> ② S1 实测折算的 **pass 级投影 ≥1% prefill**；
> ③ fused 不慢于对比链（`npu_add_rms_norm_bias` + `aclnnMatmul`）。
> 三条全过 → Loop B；任一不过 → 诚实关闭归档。
>
> **本文件的特殊之处**：§4 先把 ② 的**判据式**与**实测判据阈值**写死（本程实测前提交），
> §8 再填实测数字与判定。判据不允许在拿到数字后修改。

## 0. 本程（第一程）交付范围声明

| 项 | 状态 |
|---|---|
| 融合策略选型（按真实调用链论证） | 本文件 §2/§3 |
| ② 判据式 + 阈值（预注册） | 本文件 §4（**先于实测提交**） |
| norm 阶段核（三模式，实测 ② 分子下界的仪器） | `op_host/` + `op_kernel/` + `test/` |
| ② 的判定（实测 + 解析上界） | 本文件 §8 |
| ③ 的判定 | **未测**（需 GEMM 侧实现，见 §9 断点续作清单） |
| 融合 GEMM 本体 | 本程未实现，见 §9（判定结论决定是否续作） |

**为什么先做 norm 阶段核而不是先写融合 GEMM**：② 的分子 = 「融合省掉的 norm 时间」，
其**严格上界** = `norm 全量时间 − 融合中无法隐藏的 norm 部分`。融合里无法隐藏的部分
（下称 **exposed 段**）只由三个候选拓扑决定（§3），三者的 exposed 段都能用**一个** norm 阶段核
（三模式）真实测出来，而融合 GEMM 本体的实现（catlass prologue/epilogue 定制）是高风险长活。
即：**先花小钱把 ② 的判定做成实测，再用判定结果决定要不要花大钱写 GEMM。**

## 1. 算子签名与 dtype

```python
x_out, rstd, y = torch.ops.npu.add_rms_norm_stats(
    x1,        # (M, K) bf16/fp16 连续；残差加的左项（生产语义 = 上游 GEMM 输出）
    x2,        # (M, K) 同 dtype 连续；残差加右项（生产语义 = 跨层 residual）
    gamma,     # (K,) 同 dtype 连续（RMSNorm weight）
    beta,      # (K,) 同 dtype 或 None（= 无 norm bias，Qwen2.5-14B 生产形态）；None 时跳过加 bias
    eps,       # float
    mode,      # 0/1/2，见下
)
# 返回三个 tensor，未使用的位置返回 0 元素空 tensor（host 不分配，避免白付 20MB 分配开销）：
#   mode 0（残差+统计）  : x_out (M,K), rstd (M,1) fp32, y = 空
#   mode 1（统计+施加）  : x_out = 空,   rstd (M,1) fp32, y (M,K)
#   mode 2（只统计）     : x_out = 空,   rstd (M,1) fp32, y = 空
```

语义（与 CANN `npu_add_rms_norm_bias` golden 逐式对齐，见 `test/add_rms_norm_stats_ref.py`）：

```
s   = fp32(x1) + fp32(x2)                    # 残差加按 CANN golden：dtype 内加
xo  = round_dtype(s)                          # bf16/fp16 舍入（CAST_RINT，= torch .to(dtype)）
sq  = fp32(xo) ** 2                           # ★ 统计用「舍入后的 x_out」，不是未舍入的 s（golden 如此）
rstd= 1 / sqrt( mean_k(sq) + eps )            # fp32，M 行各一个
y   = round_dtype( fp32(xo) * rstd )          # 中间量按 golden 先舍入回 dtype
      * fp32(gamma) + fp32(beta)               # 再以更宽精度乘 gamma 加 beta
      → 输出 y 为输入 dtype
```

- x_out（残差输出）与 y（norm 输出）的舍入口径严格照 CANN golden（bf16：`xOut=(x1+x2) in bf16`，
  统计基于 `xOut` 的 fp32 值；`y = (x_out*rstd).to(bf16) * gamma + beta`）——目标是与现役链
  **数值不可区分**（§6 判据 1e-3 量级），而不是"更准"。
- 硬约束（op_host `TORCH_CHECK`）：两输入同 dtype（bf16/fp16）、连续、2D、同 shape、
  **K % 16 == 0**（32B 行拷贝）、**K ≤ 5120**（整行 UB 驻留，v1 限制；生产 hidden=5120 恒在域内）、
  mode ∈ {0,1,2}、eps > 0；mode 1 要求 `gamma` 连续且 numel == K（`beta` 可缺省）。
- **不支持 fp32**（本画像生产 dtype 是 bf16；fp16 为测试/对照 dtype）。mode 1 的 y 与 x1 同 dtype。
- **rstd 输出按 `ceil(M/8)` 行分配**（每核按 8 行一组写 32B，4B 写在 910B 上挂），
  第 M 行之后是 padding 零；消费方取 `rstd[:M]`。

### 1.1 为什么这个 op 是「F2 融合」的组成部分，而不是一个独立算子

F2 的融合目标（把 norm 塞进后继/上游 GEMM）在实现上必然切成「GEMM 侧定制」+「norm 侧计算」两块。
本 op 就是 norm 侧那一块的可测形态，三种 mode 分别对应三种候选拓扑里**唯一暴露在
GEMM 之外、因而决定 ② 上界**的那部分计算：

| mode | 读 | 写 | 对应拓扑（§3） |
|---|---|---|---|
| 0 | x1, x2（各 M·K） | x_out（M·K）+ rstd（M） | A：后继 GEMM prologue 融 norm |
| 1 | x_out（M·K） | y（M·K）+ rstd（M） | B'''：上游 GEMM epilogue 融残差加 |
| 2 | x_out（M·K） | rstd（M） | B''：上游 epilogue + 后继 prologue 两侧融 |

## 2. 真实调用链（只读证据）

调用链（`vllm-hust` 只读，`vllm/model_executor/models/qwen2.py:294-310`）：

```python
# Qwen2DecoderLayer.forward
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)          # 层 0：无残差项
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)   # L301
hidden_states = self.self_attn(positions, hidden_states)         # → qkv_proj(Addmm, N=7168, 带 bias) … o_proj
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # L308
hidden_states = self.mlp(hidden_states)                          # → gate_up(MatmulV3, N=27648) … down_proj
```

- 两层 norm 都走 `vllm-ascend-hust: vllm_ascend/ops/layernorm.py:73`（`AscendRMSNorm.forward_oot`）→
  `torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, self.bias, eps)`；
  本画像（Qwen2.5-14B bf16 非量化）`self.bias is None` → **bias 项恒为 0**（保留在签名里只为契约完整）。
- **后继**（consumer）GEMM：`input_layernorm → qkv_proj`（profile 里的 `aclnnAddmm`，每层 1 次，
  M=2048, K=5120, N=7168，带 qkv bias）；`post_attention_layernorm → gate_up`（`aclnnMatmulV3`，
  M=2048, K=5120, N=27648）。**上游**（producer）GEMM：`input_layernorm ← down_proj`（跨层，
  K=13824, N=5120）；`post_attention_layernorm ← o_proj`（同层，K=5120, N=5120）。
- 关键事实（决定选型的两个约束）：
  1. **y（norm 输出）没有任何其他消费方**：只喂后继 GEMM，是"写一次读一次"的纯中继张量；
  2. **x_out（残差输出）必须落 GM**：它是下一层 norm 的 x2（`residual`），不能只在寄存器里。
- profile 口径（`tables/step_ops_prefill.csv`，M=2048 chunk，48 层）：
  `AddRmsNormBias` 96 次 / 7651.3µs / **2.9236%**；`aclnnAddmm` 48 次 / 505.45µs avg；
  `aclnnMatmulV3` 144 次 / 1155.79µs avg；prefill pass 设备时长 **261.71ms**（`pass_prefill.json`）。

## 3. 融合策略选型（按真实调用链论证）

### 3.1 四变体的字节账（每对 norm→GEMM，M=2048 / K=5120 / N=7168 / bf16，M·K = 20MB 粒度）

| 变体 | 拓扑 | 参与 GM 的数据流（20MB 单位） | 相对基线省 | exposed 段 |
|---|---|---|---|---|
| 基线 | `norm(y,x_out) → GEMM(A=y)` | 读 x1+x2(2) 写 x_out(1) 写 y(1) ‖ GEMM 读 y(1) | — | norm 全量（80MB） |
| **A** | **norm 融进后继 GEMM 的 A-prologue** | 读 x1+x2(2) 写 x_out(1) ‖ **prologue 读 x_out(1)** | **1**（y 的写） | mode 0（60MB） |
| A'' | 只算 rstd，残差加也搬进 prologue | 读 x1+x2(2) ‖ prologue 读 x1+x2(2) 写 x_out(1) | 0（字节等价，收益全在"搬进影子"） | mode 2（20MB… 需读 x1+x2 实为 40MB） |
| **B'''** | **残差加融进上游 GEMM epilogue** + mode 1 | epilogue 读 x2(1) 写 x_out(1) ‖ mode1 读 x_out(1) 写 y(1) ‖ GEMM 读 y(1) | 1（x1 的写+读） | mode 1（40MB） |
| **B''** | 上游 epilogue 残差加 + mode 2 + **后继 prologue 融施加** | epilogue 读 x2(1) 写 x_out(1) ‖ mode2 读 x_out(1) ‖ prologue 读 x_out(1) | 2（x1 写+读 + y 写） | mode 2（20MB） |

（A'' 的彻底形态需要 prologue 里写 x_out，而 A 块会被多个 n-block 重复读 → 冗余写/跨核同步，
本程不取。它是"A 与 B 之间的中间态"，头寸与 B'' 同源但实现风险高，不作为选型。）

### 3.2 选型结论：A 为**本仓既有 scope** 的唯一落地形态，但其 ② 上界本身就偏薄；B 系是唯一有头寸的方向

- **A（后继 GEMM prologue）**：最小改造面——模型侧只把 `(norm, qkv/gate_up)` 这一对换成一次融合调用，
  residual 契约不动，且 catlass 树里**有现成的 prologue 通道**
  （`gemm/block/block_mmad_pingpong_with_prologue.hpp` + `gemm/tile/tile_traits.hpp::PrologueTraits`，
  参照实现 `gemm/kernel/padding_matmul.hpp::PaddingMatrixNZ`）。
  代价：它只省掉 y 的**一次写**（20MB/对 100MB），**movement 上限 = 20% of norm 时间**，
  即 ② 的**理论天花板 ≈ 0.2 × 2.92% = 0.58% prefill** —— **先天低于 1% 门槛**。
- **B''' / B''（上游 GEMM epilogue 融残差加）**：省的是 x1 的**一次写 + 一次读**（40MB），
  且残差加/施加这两段 20–40MB 的计算搬进了上游 GEMM（o_proj 436µs / down_proj ~1176µs，cube-bound，
  DRAM 利用率仅 ~210GB/s = 上限的 17%）的影子，exposed 只剩 mode 1（40MB）/ mode 2（20MB）。
  ② 的天花板 ≈ (79.7µs − t_exposed)/261710µs × 96 —— 按流式上限 1.2TB/s 估：
  B''' ≈ **1.7%**、B'' ≈ **2.3%**（§8 有实测替代值）。
  代价：要改**上游** GEMM 的 epilogue（catlass `epilogue/block/block_epilogue_elemwise_one_source.hpp`
  是"读一路外部源做逐元素"的现成通道，x_out = C + x2 正好是它的形态），且 op 契约变成
  "matmul + 残差加 + norm"，即 F2 立项书里被列为另一支的形态。

**按真实调用链的取舍**：两者都在同一张图里、都能与 `qkv/gate_up` 或 `o_proj/down_proj` 换接，
改造面同量级（换一次调用点）。差别是**收益方向相反**：A 省下游的写，B 省上游的写+读。
在 §3.1 的字节账下 A 的头寸是 B''' 的一半、B'' 的三分之一，**且 A 的天花板（0.58%）已低于 ② 的门槛（1%）**。
本程因此：**以实现 A 的仪器（norm 阶段核）先把 ② 判死/判活，并把 B 系作为"唯一能过 ② 的方向"上报**
（本仓 scope 归属由规划方裁决，worker 不擅自把 op 改成 B 系形态）。

## 4. ② 判据式与预注册阈值（**本节先于实测提交**）

**分子**（每个 norm→GEMM 对省掉的设备时间）：
```
saved_us_per_pair = t_norm_device − t_exposed_device
  t_norm_device   = 现役 CANN AddRmsNormBias 的设备时长（本画像实测锚点 79.70µs，
                    来源 tables/step_ops_prefill.csv：7651.3µs / 96 次，M=2048,K=5120,bf16,910B2）
  t_exposed_device= 该变体暴露在 GEMM 之外的 norm 段设备时长（本 op 三种 mode 实测，
                    口径 = 同 shape 同轮次 wall 中位 − dispatch 常数，见 §7；另附 msprof 复核）
```
**分母**：prefill 单个 2048-token chunk pass 的设备时长 **261710µs**（= 上表同一次 profile 的
`pass_prefill.json`，与 `REPORT.md §6` 分母完全一致，保证分子分母同源同口径）。

**折算公式**：
```
投影_prefill(%) = (t_norm_device − t_exposed_device) × 96 / 261710 × 100
                  其中 96 = 48 层 × 2 次 norm/层（profile 实测次数）
```
**上界性声明**（本式为何是上界，而非估计）：
1. 融合 GEMM 侧的 prologue/epilogue 定制开销 ≥ 0，一律按 0 计入（最乐观）；
2. rstd 的跨核/全局归约开销按 0 计入；
3. 图模式（生产口径）下没有 kernel launch/dispatch 头寸，故不把"少一次 launch"计入收益
   （eager 下那部分 ~63µs/对**不纳入 ②**，只在 §8 作旁证——F3 gelu 融合的教训：立项投影必须按图模式口径）；
4. 唯一被计入的收益 = 被真实消除的 GM 往返，读/写均按"现役实测算子在同 shape 上的有效带宽"折算。

**阈值判定规则（预注册，拿到数字后不得修改）**：
```
若 投影_prefill(A) < 1%  → gate ② 不过 → 本 scope（后继 GEMM prologue 形态）诚实关闭归档，
                            并把 B 系实测上界一并上报（供规划方决定是否另立 scope）
若 投影_prefill(A) ≥ 1%  → ② 通过（在本式的乐观假设下成立）→ 续作 §9 的 GEMM 侧实现，
                            再用真实 fused 实现复测 ②/③（届时以真实实现数字为准，允许更差）
```
注意本式不依赖任何"融合后 GEMM 变快"的假设；它只假设 **prologue 白送**。
因此一旦它给出 <1%，结论是**否证性的**（任何真实实现只会更差），可直接关闭，无需再写 GEMM。

## 5. Tiling / UB 规划 / buffer 分配（norm 阶段核）

- **多核切分**：按行切（本 op 的归约只在行内，行与行完全独立 → 天然零跨核通信）。
  每核连续 `rowsPerCore = ceil(M / aivNum)` 行，且**向上取整到 8 行**（保证每核起始行号
  8 对齐 → rstd 的 32B 对齐 DataCopy 合法；对齐纪律照 `lse_merge` 的 M2 修复口径）。
  尾核越界行直接不处理（`if (rowStart >= M) return;`）。
- **行内不切 K（v1 实现口径）**：整行 UB 驻留，所有向量算子是**单条行长**算子（K ≤ 5120
  由 op_host 硬拒），行尾 pad 问题不存在（K % 16 == 0 已拒非对齐值 → 每笔行列拷贝都是
  32B 整数倍）。**K > 5120 需 K 方向二级切分（v2）**；生产形态（norm 的行宽 = hidden = 5120）
  恒在域内。
- **UB 规划**（K=5120 bf16 实测口径，单核）：

| 缓冲 | 大小（K=5120） | 说明 |
|---|---|---|
| inQueX1 / inQueX2（depth 2 乒乓） | 2×10KB ×2 | bf16/fp16 整行 tile |
| outQue（depth 2） | 2×10KB | x_out / y 输出 |
| fp32 工作区 bufA/bufB/ReduceSum work | 3×20KB | 残差和 / 统计输入 / 归约临时 |
| bufSum（1 元素 + 对齐） | 32B | 归约目的 + rstd 向量算子载体 |
| bufGamma / bufBeta（mode 1） | 2×20KB | gamma/beta 单次读入后常驻（避免逐行重读 20MB） |
| bufRstd staging | 32B | 攒 8 行做一次 32B MTE3 |
| bufMid（mode 2） | 10KB | 私有舍入中间量 |

  合计 ≈ 140KB（mode 1）/ 130KB（mode 0/2），余量 ~50KB。
- **rstd 计算全程向量化**：`Muls(·, kInv)` → `Adds(·, eps)` → `Rsqrt` → `Muls(stage[staged], ·, 1.0f)`
  ——AICore 代码**没有标量 `sqrtf`，也拒绝 uint32→float 强转**（两者都实测编译失败），
  故 1/K 由 host 传入、开方走向量 `Rsqrt`；scalar 读只在 mode 1 施加步用一次。
- **归约**：行内 `ReduceSum`（Level-2，fp32，work buffer = 行长）得到该行 Σx²。
  归约顺序与 CANN kernel 不保证一致 → 属 §6 容差内的合法分叉。
- **buffer 分配**：`TQue<VECIN, 2>`×2、`TQue<VECOUT, 2>`×1、`TBuf<VECCALC>`（fp32 工作区/常驻 gamma/beta/staging）若干。

### 5.1 实现期实测发现（两次上板，2026-09-12，910B2/CANN 9.1.0）

1. **AICore 无标量 `sqrtf`、且拒绝 `uint32→float` 强转**（编译期两次报错）：1/K 由 host 传入（`kInv`），
   `1/sqrt` 走向量 `Rsqrt`。
2. **VEC 指令的 UB 地址必须 32B 对齐**（实测：首版把 rstd 逐行写进 staging 的 `stage[staged]`（元素偏移
   1..7 = +4B..+28B）→ 全核 `aivec error ... "The UB address accessed by the VEC instruction is not
   aligned"`，kernel task retCode=0x31）。修法：staging 写回退为**标量 `SetValue`**（标量无该约束，
   与 `lse_merge` 同款），flush 前用 `PipeBarrier<PIPE_ALL>` 兜 S→MTE3 序；相同纪律也适用于
   `Duplicate(stage[staged], ...)` 这类带元素偏移的向量写。
3. **`Rsqrt` 在 910B 上是 ~2^-11 近似**（实测：未加修正时本核 rstd 与 CANN `npu_add_rms_norm` 的 rstd
   偏差 `maxabs=2.29e-3`，且本核结果呈 10-bit 尾数特征 1.38671875 vs CANN 1.38575196）。
   修法：**一次 Newton-Raphson 细化** `r *= 1.5 − 0.5·a·r²`（每行多 4 个 1 元素向量算子，
   UB 成本 64B），精度回到 ~2^-22 —— 因为融合的目标是"与现役链不可区分"，不能把预算花在近似上。
4. **残差加（mode 0 的 x_out）与 CANN 逐位一致**（smoke：M=7/K=128/bf16，`maxabs = 0.0`）——
   与 §1 的舍入口径设计一致。

## 6. 精度参考实现与判据
- 参考实现：`test/add_rms_norm_stats_ref.py`（纯 torch CPU/GPU 无关，**同 CANN golden 语义**，
  见 §1 公式；fp64 累加版本另作"更准参考"用于误差来源分解）。
- 判据（写进 `test/add_rms_norm_stats-test-cases.md`）：
  - `rel_l2 = ||out − ref||₂ / ||ref||₂`：
    - x_out（残差加，纯 dtype 舍入）→ **≤ 1e-3**（bf16）/ ≤ 1e-3（fp16）；
    - y（norm 输出）→ **≤ 5e-3**（bf16）/ 2e-3（fp16）；
    - rstd → 逐元素 `rel ≤ 5e-3`；
  - 逐元素容差表：bf16 `atol=8e-3, rtol=8e-3`（对齐 CANN 自带用例的
    `0.0079345703125 = 2^-7`）、fp16 `atol=1.1e-3, rtol=1.1e-3`（`0.0010986328125`），
    即**直接采用 CANN 官方 `test_add_rms_norm_bias.py` 的 dtype 档位**，不自造更松口径。
  - 边界形状：K=16/128（最小对齐）、K 非 2 的幂（K=5184）、M 不整除核数（M=77/255）、
    M=1、K=16384（行内二级切分上界）、beta=None / beta 非零两态、fp16/bf16 两 dtype。
- 负例：非连续、dtype 不一致、K%16≠0、mode∉{0,1,2}、eps≤0 → 全部 `TORCH_CHECK` 拒。

## 7. S1 锚点方案（NPU）

- 设备/卡：910B2（c220），CANN 9.1.0，torch_npu 2.13.0rc1；`flock /tmp/w3-npu.lock` 串行化，
  显式 `ASCEND_RT_VISIBLE_DEVICES=<空闲卡>`；卡 7 水位（HBM/AICore）记录前后。
- 计时：`warmup 20 + ≥100 次中位 × 3 轮`（host wall，`torch.npu.synchronize` 每轮后），
  同轮次同法测三条对照：
  1. `torch_npu.npu_add_rms_norm`（= 现役 CANN AddRmsNormBias 的 C 面入口，bias=None 与生产等价）；
  2. `torch.matmul`（aclnnMatmul，M=2048 K=5120 N=7168 / N=5120）；
  3. 本 op 三 mode。
- **device 口径折算**：`t_device ≈ t_wall − c_dispatch`，`c_dispatch` 用同一进程、同一 shape 的
  CANN norm op 标定（本画像 2026-09-12 实测 wall 142.9µs vs profile 设备 79.70µs ⇒ c ≈ 63µs）；
  另跑一次 msprof 复核 2–3 个 shape（复核失败则只用 wall + 标注口径，不掩盖）。
- ⚠ 不使用 `torch_npu.profiler` 的 step 内自动停（§pitfalls 7：`stop()` 后 analyse 段 segfault）。

## 8. S1 实测与 ② 判定（实测后回填，原文不改）

> 本节由实测回填；§4 的公式与阈值在上述提交中已冻结。

（回填见本文件末尾「实测附录」与本仓 commit 的 REPORT。）

## 9. 断点续作清单（判定为"活"时才执行）

1. GEMM 侧（A 形态）：`catlass Gemm::Kernel::OptimizedMatmul<PrologueA, void, BlockMmad<MmadAtlasA2PingPongWithPrologue>, BlockEpilogue, BlockScheduler>`，
   自写 `PrologueA`（接口照 `PaddingMatrixNZ`：`paddingTag`/`GetWorkspaceSize`/`GetWorkspaceLayout`/`operator()`）：
   AIV 读 x_out → UB 施加 `(xo*gamma+beta)*rstd` → 写 L1/workspace → cube 消费；
   epilogue = identity（+ 可选 per-column bias 加）。tile 形状先取 128×256×128 起调。
2. S1/精度补全：fused vs 对比链（`npu_add_rms_norm_bias` + `aclnnMatmul`）同 shape 同法，
   ③ 判据：`t_fused_device ≤ t_norm_device + t_matmul_device`（同口径同轮次）。
3. 若 ②/③ 全过：Loop B（插件侧换接 `input_layernorm`/`post_attention_layernorm` 调用点，
   由插件库按自身 release.md 流程推进）。
4. 若 ② 不过而 B 系被立项：复用本 op 的 mode 1（B'''）/mode 2（B''）作 exposed 段，
   GEMM 侧改用 epilogue-one-source 形态（`block_epilogue_elemwise_one_source.hpp`）。
