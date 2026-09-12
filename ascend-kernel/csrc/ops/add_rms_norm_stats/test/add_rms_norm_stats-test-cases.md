# add_rms_norm_stats 测试用例与判据（F2 norm 阶段核，2026-09-12）

> 判据与容差在此声明；实现与设计见 `../design.md`；预注册 gate 见 design.md §4。

## 1. 测什么、为什么

`add_rms_norm_stats` 是 F2（AddRmsNormBias→GEMM 融合）里**暴露在 GEMM 之外的那部分 norm 计算**
的可测形态（三种 mode，见 design.md §1）。它同时是：

1. 一个可用算子（mode 0 = 残差加 + 行统计；mode 1 = 行统计 + 施加；mode 2 = 只统计）；
2. **gate ② 的测量仪器**：三种 mode 的实测耗时分别给出三种融合拓扑的"exposed 下界"，
   从而给出 ② 的严格上界（design.md §4/§8）。

## 2. 参考实现

| 参考 | 语义 | 用途 |
|---|---|---|
| `add_rms_norm_stats_ref.cpu_ref` | CANN `npu_add_rms_norm_bias` golden 舍入口径（残差按 dtype 舍入→平方→fp64 累加求均值；`y` 先舍 `(x·rstd)` 再乘 gamma 加 beta） | 主参考（唯一覆盖 beta≠None 与三 mode 的参考） |
| `torch_npu.npu_add_rms_norm` | 生产链上的现役 CANN 算子（= profile 里的 `AddRmsNormBias` kernel） | **生产 oracle 交叉核对**（beta=None 域），要求比 fp64 参考更严 |

主参考与 golden 的两处有意差异（写出以便复核，非"放宽"）：
1. 平方和按 fp64 累加（golden 用 fp32 `torch.sum`）→ 参考是"精确侧"，容差吸收 kernel 的 fp32 归约序差异；
2. fp16 档 gamma 乘法在 fp32 中完成（golden 的 fp16 路径在 fp16 中做乘加）→ fp16 MARE 容差吸收 ≤2^-11 的差。

## 3. 判据与容差（**容差按 ops-precision-standard**）

判据采用工作区只读参考 skill `ops-precision-standard`（浮点计算类；立项书原文即"容差按
fp16/bf16 标准混合容差"）。实现与表值直接取该 skill 的
`references/float_compute.md` §2–§4 与 `scripts/mixed_tolerance_check.py`，**逐字复制、不自行改编**：

- **逐元素**：`|out − golden| ≤ atol + rtol·|golden|`
- **整体（用例判定）**：`matched_ratio ≥ 0.99` **且** `max_abs_error ≤ max(fixed_limit, 32·ULP@1.0)`
- **档位按输出 dtype 取表**：fp16 `atol=rtol=2^-9`、bf16 `atol=rtol=2^-6`、
  `rstd` 是 **fp32 输出**故取 fp32 行 `atol=2^-16, rtol=2^-10`；`fixed_limit` 分别 1e-1 / 1e0 / 1e-2；
  `ULP@1.0` 分别 2^-10 / 2^-7 / 2^-23。
- 判据是两个条件**同时**成立；`matched_ratio` 允许 1% 元素越界，但 `max_abs_error` 上限拦住"个别元素离谱"
  的逃逸（本 op 实测两者都远未触线：matched_ratio 全为 1.0，max_abs_error 最大只到上限的 **3.9%**）。

| 输出 | 标准档位（按输出 dtype） | 判定 |
|---|---|---|
| `x_out`（mode 0） | bf16 2^-6 / fp16 2^-9 | matched_ratio ≥ 0.99 且 max_abs ≤ 1e0 / 1e-1 |
| `y`（mode 1） | 同上 | 同上 |
| `rstd`（全 mode） | fp32 2^-16/2^-10 | matched_ratio ≥ 0.99 且 max_abs ≤ 1e-2 |

报告同时全量留档三项**诊断量（不参与判定）**：`matched_ratio`（标准判据量本身）、`rel_l2`（批级相对 L2）、
以及 `cann_viol/n` = 改用 **CANN 官方用例更严档位**（`atol=rtol=2^-7 bf16 / 2^-10 fp16`，要求**全元素**）
时的越界元素数——用它把"与现役算子的边际距离"量化留档（见 §3.3）。

**结构判据**（额外）：`rstd` 的 padding 行（`M` 向上取整到 8 的行）必须为 0。

**生产 oracle 判据**（beta=None，与 `torch_npu.npu_add_rms_norm` 逐输出比，比标准**更严**）：
按标准档位但要求 `matched_ratio = 1.0`（逐元素全过）且 max_abs ≤ 上限。之所以更严：这条路径的目的是
证明"与现役算子不可区分"，不是满足某个 spec。
（mode 1 vs oracle 用 `x2=0` 喂 CANN，使其残差输出恒等输入，比较同一件事。）

**SKILL 一致性**：`test_add_rms_norm_stats_ref.py::test_matches_skill_checker_if_available`
在 skill 在场时直接加载其 `mixed_tolerance_check.py`，逐项比对档位表/`ULP@1.0`/`matched_ratio`/
`max_abs_error`/`is_pass`——防止我方实现与其参考实现漂移（skill 缺失时该用例 skip）。

### 3.1 判据实现缺陷与订正（2026-09-12，首轮实测后；档位数值未动）

首轮实测 64 例报 42/64、oracle 报 6/48。逐条定位后确认**两条都不是算子缺陷，而是判据实现缺陷**，
容差档位自始至终一个数都没改：

1. **把绝对档位当相对界 + 漏掉 rtol 项**：首版 `judge` 判 `MaxAbsErr ≤ atol` 且
   `MARE（逐元素最大相对误差）≤ atol`。对 `y`（|y| 最大 ~8，bf16 在 [4,8) 的 ulp = 2^-5）而言，
   `2^-7` 的**绝对**界比 1 个可表示步长还严 ~7 倍——任何正确核都过不了，也不是 CANN 官方用例的断言。
   典型读数：`y` 的 `MaxAbsErr = 3.125e-2 = 2^-5`（**恰好 1 ulp**）、`MARE = 1.37e-2`（出现在近零 y
   元素上，本该由 atol 项兜底），而同一 case 的 `rel_l2` 只有 1.1e-4。订正：按 §3 表恢复 rtol 项。
2. **度量函数静默广播**：`rstd` 由 host 分配为 `(ceil(M/8)*8, 1)`，而 oracle 侧的 CANN `rstd` 被
   `reshape(-1)` 成 `(M,)`；首版 `metrics` 直接相减，torch 把两者广播成 `(M, M)` 的两两配对。
   症状是**自相矛盾的读数**：oracle `rstd` 报 `rel_l2 = 0.64` 而 `MARE = 6.6e-2`；对逐元素对齐的一对数，
   `rel_l2 ≤ MARE` 恒成立，0.64 > 0.066 只能是配对错位。核对每个 case 的 `rel_l2 / MERE` 比值 =
   `sqrt(M) × 1.25`（2048→56.8、1000→39.6、32→7.2、7→3.3、M=1→1.0）与广播预测**逐项吻合**。
   订正：`metrics` 先各自 flatten、元素数不等即抛错（`refusing a broadcast compare`）；
   该守卫固化为 CPU 用例（`test_add_rms_norm_stats_ref.py`，NPU 无关）。
   影响面：**oracle 那 48 例全部作废重测**；对照 fp64 参考的 64 例中 `rstd` 两侧同为 `(M,1)`、
   未受影响（其读数一直正常：rel_l2 ~3e-6、MARE ~1.5e-5）。

订正只改实现、不改档位；报告保留更严读数列，避免"改判据换通过"的嫌疑。

### 3.2 复测结果与 mode-1 `y` 未闭环缺陷（2026-09-12，档位仍未动）

复测（`f2_prec_r2x.json|md`，本文件 §3 之前的旧档位口径）：对照 fp64 参考 **52/64**、
对照 CANN oracle **42/48**。**全部失败都在 mode 1 的 `y`**，每例只有 1–20 个元素越界（分母 26 万–1048 万），
同 case 的 `rel_l2` 仅 4e-5~1.3e-4（比旧 5e-3 档位好 40–125 倍）；**mode 0/2 与全部 rstd 比对
100% 通过（32/32 + 32/32）**。⇒ 该口径下 gate ① **边际不达标**，缺陷锁定在 `y` 的施加路径。
（档位改按 `ops-precision-standard` 后本项转为**通过**，见 §3.3；本节的机制分析仍然有效，
故保留原文。）

**是否"任何实现都过不了"？——实测回答：不是。** `run_golden_selfcheck.py` 把同一档位用在
现役 CANN 算子上（`y` 对 fp64 参考）：**cann_vs_ref 越界 0/16 例**，而本核 5/16 例；
且在越界元素上 CANN 的 `y` 与 fp64 参考 **逐位相同**（`|cann-ref| = 0.0`）。

**机制（`f2_probe_transcripts.log` 逐元素复现）**：
- 本核 `y ≡ round_dtype(round_dtype(x·rstd_ours)·gamma)`，CANN/参考是同式用各自 rstd
  （probe 对每个元素逐条复现，无例外）⇒ 公式与舍入次序无缺陷，差异全来自 **rstd**。
- 行内平方和的相对误差：本核 ≈ **30 eps**（−3.34e-6 / −3.71e-6 两例），
  现役 CANN ≈ **0.5 eps**（+6.1e-8 / 0.0）——`AscendC::ReduceSum` 在 K=5120 上的累加深度
  远大于 CANN 的归约。
- 放大链：`x·rstd` 本核 2.53905593 vs 精确 2.53906440（差 8.5e-6），而 bf16 在 [2,4) 的
  舍入边界恰为 **2.5390625** ⇒ `mid` 翻 1 ulp（1.5625e-2），两侧乘 gamma 后又分别落在 `y`
  的舍入边界两侧 ⇒ `y` 差 **4 ulp**（2.78125 vs 2.8125），allowed = 3.025e-2 ⇒ **越界 3%**。
- 规模：`d>0` 的元素 bf16 **3321/10485760**（3.2e-4）、fp16 **29599/10485760**（2.8e-3，
  与 ulp 细 2^-11 vs 2^-8 相符）；其中撞上第二重边界而越界的只有 2–20 个。

**已定位的修法（本程未实施，写出以便续作）**：把行内归约换成补偿求和（Neumaier/Kahan）或
分段树（误差降 1–2 个数量级）。**为什么本程不改**：(a) 本 op 的融合价值已被 gate ② 判死
（A −0.045%，全家族最乐观 +0.200%，判据在实测前冻结，见 design.md §4/§8）；
(b) 该段已是带宽受限（565GB/s，峰值 ~1/2.5），Neumaier 每 64 元块要多 ~8 条向量算子，
有把带宽受限段变成算力受限段的风险；(c) 改动需重跑 ①（另一把锁）。⇒ 如实记为**未闭环的边际缺陷**，
在**标准档位**下不构成 gate ① 失败（§3.3），但它是本核与现役算子之间唯一的数值质量差距。

### 3.3 标准档位复测（2026-09-12，判据改按 `ops-precision-standard`）

按 §3 的标准档位与整体规则重跑（`f2_prec.json|md`，同一 64+48 用例矩阵、同一设备、同一锁）：

| 口径 | 结果 |
|---|---|
| **标准档位（gate ① 判据）**：64 例 vs fp64 参考 | **64/64 通过**；`matched_ratio` **全为 1.0**；`max_abs_error` 最大仅到上限的 **3.9%**（fp16 `y` 3.906e-3 / 1e-1） |
| **标准档位**：48 例 vs CANN oracle（要求逐元素全过） | **48/48 通过** |
| 诊断列：CANN 官方更严档位（全元素） | 12 个输出行非零（**仅** mode-1 `y`，bf16 1–6 个、fp16 1–20 个元素）；其余 100 行为 0 |
| 诊断列：`rel_l2` | `y` 4e-5~1.3e-4、`rstd` 3e-6 量级、`x_out` **0.0**（逐位一致） |

两档位对同一批数据给出"标准过、更严档位不过"的差异是**档位差值本身**（bf16 2^-6 vs 2^-7、
fp16 2^-9 vs 2^-10），不是判定口径的双标：§3.2 的 5/16 例在最严档位下仍然存在，已作为
**剩余数值质量风险**记录（修法同上），并保留在每次报告的诊断列里。

**控制实验的意义**：`cann_vs_ref` 在最严档位下 0/16 例，证明"该档位不可达"的辩解不成立——
差距是本核的，不是判据的。这一点必须在关闭/续作时一并交代，不得只用"标准过了"盖过。

### 3.4 D3：mode-1 `y` 缺陷已闭环（2026-09-12，档位数值未动）

§3.2 记录的缺陷（CANN 严档位下 mode-1 `y` 12 行 × 1–20 元素越档）在 D3 立项内修复。**判据与档位
一个字没动**，修的是算子：

| 运行配置 | 标准档位（判定口径） | CANN 严档位（诊断列） |
|---|---|---|
| D3 前（`ReduceSum` + 一次 Newton） | 64/64 + 48/48 PASS | ours 5/16、CANN 对照 0/16 |
| D3 后（二分折叠 + 二次 Newton），**交付配置** | **64/64 + 48/48 PASS** | **ours 0/16**、CANN 对照 0/16 |

缺陷由两个叠加项构成，**用力学实验分开**而非推测：

1. **行内平方和按 ~80 个 partial 顺序合并**（≈30 eps）：CPU 侧建模并校准到实测
   （`run_sum_accuracy_study.py`：顺序模型 376 flips/百万元素 vs 实测 317）。
   修法 = 先**二分折叠**（skill `references/reduction/alg-dichotomy.md`）到 64 项再 `ReduceSum`
   ⇒ 累加深度从 K 降到 log2(K)，最坏误差降 ~18x。
2. **向量 `Rsqrt` 的 ~2^-11 近似 × 一次 Newton 的收敛底**（残留 ~1.5ε² ≈ 1e-5）：
   把第 1 项修掉后，越界元素的 rstd 相对差**几乎不动**（row 1512：−3.389e-6 → −3.450e-6；
   row 1590 逐位相同）⇒ 残余不在行和。加**第二次 Newton** 把迭代误差压到 1.5ε⁴，剩下的底是
   公式的 fp32 求值（~2 eps）⇒ 严档位 0/16。

回归面：设备侧 46/46（`test_add_rms_norm_stats_precision.py` + `test_add_rms_norm_stats_ref.py`）、
标准档位 64/64 + 48/48、负例面（§5）全过。**批量化改动（§3.5）另证为逐位等价**。

### 3.5 批量化 rstd 链的逐位等价性（2026-09-12，双构建 A/B 实测）

D3 把 mode 0/2 的 rstd 链从"每行 count=1"改为"每 `RSTD_GROUP=8` 行 count=64"（§5/§9.3）。
这是**调度改动**，用双构建实测而非论证：同一份源码分别以 `kBatchStats=1` / `=0` 各构建一次，
对 2 种 dtype × 3 个 shape × 3 个 mode 的**全部输出张量**取 sha256 指纹对比 ⇒
**18/18 全组逐位相同**（含 rstd）。恢复构建后指纹与交付构建再次相同。

⇒ 本程唯一的数值变化是**有意的精度修复**（§3.4），批量化不改变任何一位。
`ARMNS_STATS`/`ARMNS_SUM` 开关已在实测后删除，胜出组合硬化为 host 常量
（`kBatchStats=1`/`kPairSum=1`）。

## 4. 用例矩阵（共 64 例 + oracle 48 例）

- **mode** 0/1/2；**dtype** bf16/fp16；**shape**（M, K）：

| case | (M, K) | 覆盖的边界 |
|---|---|---|
| prefill-main | (2048, 5120) | F2 主 shape（profile M=2048 chunk，K=hidden） |
| decode | (32, 5120) | decode 形态（B=32，< 一核 8 行组整除） |
| single-row | (1, 5120) | M < RSTD_GROUP（部分 rstd 组 + 尾零填充） |
| odd-rows-77 | (77, 128) | M % 8 ≠ 0 且不整除核数 |
| k-not-pow2 | (255, 5008) | K 非 2 的幂（16 对齐） |
| k-min | (7, 16) | 最小 K（一个 32B 块）+ M < 8 |
| m-1000 | (1000, 5120) | 核间非均匀切分 |
| k-2048 | (128, 2048) | 中等 K |

- mode 1 额外跑 beta=None / beta≠0 两态（+16 例）→ 总 64 例。
- **负例**（`torch.ops.npu.add_rms_norm_stats` 必须 TORCH_CHECK 拒，见 §5）。

## 5. 负例清单（拒绝面）

| 输入 | 期望 |
|---|---|
| `x1` 非连续 / 3D | 拒（"must be contiguous" / "must be 2D"） |
| `x2` shape 或 dtype 与 `x1` 不一致 | 拒 |
| `K % 16 != 0` | 拒（32B 行拷贝约束） |
| `K > 5120` | 拒（单行 UB 驻留预算，v1 限制） |
| `mode` ∉ {0,1,2} | 拒 |
| `eps <= 0` | 拒 |
| mode 1 未给 `gamma` / `gamma.numel() != K` | 拒 |
| `x1` dtype ∉ {bf16, fp16}（如 fp32） | 拒 |

## 6. S1 锚点（性能，见 `run_s1_anchor.py`）

- 910B2（c220）/ CANN 9.1.0 / torch_npu 2.13.0rc1；`flock /tmp/w3-npu.lock`；卡 7；
  `warmup 20 + 100 次中位 × 3 轮`（host wall，每轮 sync）。
- 对照：`torch_npu.npu_add_rms_norm`（= 现役 `AddRmsNormBias` kernel 的 C 面入口）、
  `torch.matmul`（aclnnMatmul，K=5120 → N=7168 / N=5120）。
- 折算：`t_device ≈ t_wall − c`，`c = t_wall(oracle) − 79.70µs`（79.70µs = 生产 profile 的
  `AddRmsNormBias` 设备时长，`tables/step_ops_prefill.csv`）；② 的判据式与阈值见 design.md §4。
- **口径警告**：② 只用设备口径（图模式下无 launch/dispatch 头寸，见 design.md §4 第 3 条）；
  wall 口径的差值只作旁证。

## 7. 已知边界（写清而不是掩盖）

- `K > 5120` 需要 K 方向二级切分（v2），当前 host 拒绝；
- `rstd` 输出按 `ceil(M/8)` 行分配（32B MTE3 组），消费方取 `[:M]`；
- 同 shape 三次运行稳定性：由 §4 报告中的 3 轮中位一致性体现（无 aicore 异常/NaN 才算过；
  报告里 NaN 会直接体现为 FAIL）。
