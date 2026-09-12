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

## 3. 判据与容差（dtype 档位直接取 CANN 官方用例，不自造）

CANN 官方 `test_add_rms_norm_bias.py` 的档位：bf16 `atol=rtol=0.0079345703125 (2^-7)`、
fp16 `atol=rtol=0.0010986328125 (2^-10)`。该档位在 CANN 官方用例里就是
`torch.allclose(out, ref, rtol, atol)`，即**逐元素**判 `|out−ref| ≤ atol + rtol·|ref|`——
**rtol 项是档位的一部分**（对 |y| 最大 ~8 的 `y`，它就是"1 个可表示步长"的判据）。按输出分别判：

| 输出 | bf16 | fp16 | 判定式 |
|---|---|---|---|
| `x_out`（mode 0） | rel_l2 ≤ 1e-3 | 同构 | 违例元素数 = 0，违例 := `\|d\| > atol + rtol·\|ref\|` |
| `y`（mode 1） | rel_l2 ≤ 5e-3 | rel_l2 ≤ 3e-3 | 同上 |
| `rstd`（全 mode） | rel_l2 ≤ 5e-3 | 同构 | 违例元素数 = 0（无 CANN atol 档位 → atol=0, rtol=2^-6，等价于 max 相对误差 ≤ 2^-6） |

报告同时给出诊断量（**不参与判定**，全量留档）：`rel_l2`、`viol/n`、`MARE`（逐元素最大相对误差，
近零元素上由 atol 项兜底）、`MaxAbsErr`，以及 `strict_abs_only` 列 = 是否**同时**满足更严读数
"MaxAbsErr ≤ atol 且 max 相对误差 ≤ atol"（便于复核者按更严口径自行判读）。

**结构判据**（额外）：`rstd` 的 padding 行（`M` 向上取整到 8 的行）必须为 0。

**生产 oracle 判据**（beta=None，与 `torch_npu.npu_add_rms_norm` 逐输出比）：
`rstd` rel_l2 ≤ 2e-3 且逐元素相对违例 0；`x_out`/`y` rel_l2 ≤ 5e-3 且 CANN 逐元素档位违例 0。
（mode 1 vs oracle 用 `x2=0` 喂 CANN，使其残差输出恒等输入，比较同一件事。）

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

复测（`f2_prec.json|md`）：对照 fp64 参考 **52/64**、对照 CANN oracle **42/48**。
**全部失败都在 mode 1 的 `y`**，每例只有 1–20 个元素越界（分母 26 万–1048 万），
同 case 的 `rel_l2` 仅 4e-5~1.3e-4（比 5e-3 档位好 40–125 倍）；**mode 0/2 与全部 rstd 比对
100% 通过（32/32 + 32/32）**。⇒ 判定：**gate ① 不达标（边际）**，缺陷锁定在 `y` 的施加路径。

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
有把带宽受限段变成算力受限段的风险；(c) 改动需重跑 ①（另一把锁）。⇒ 如实记为**未闭环缺陷**。

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
