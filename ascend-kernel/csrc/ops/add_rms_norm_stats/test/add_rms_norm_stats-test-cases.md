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
fp16 `atol=rtol=0.0010986328125 (2^-10)`。本算子按输出分别判：

| 输出 | bf16 | fp16 | 说明 |
|---|---|---|---|
| `x_out`（mode 0） | rel_l2 ≤ 1e-3，MARE ≤ 2^-7，MaxAbsErr ≤ 2^-7 | 同构（2^-10） | 纯 dtype 舍入，应与参考几乎逐位一致 |
| `y`（mode 1） | rel_l2 ≤ 5e-3，MARE ≤ 2^-7，MaxAbsErr ≤ 2^-7 | rel_l2 ≤ 3e-3，2^-10 | 含 `(x·rstd)` 的 dtype 舍入点 |
| `rstd`（全 mode） | rel_l2 ≤ 5e-3，MARE ≤ 2^-6 | 同构 | fp32 统计量，无 CANN atol 档位，按相对判据 |

**结构判据**（额外）：`rstd` 的 padding 行（`M` 向上取整到 8 的行）必须为 0。

**生产 oracle 判据**（beta=None，与 `torch_npu.npu_add_rms_norm` 逐输出比）：
`rstd` rel_l2 ≤ 2e-3；`x_out`/`y` rel_l2 ≤ 5e-3 且 MaxAbsErr ≤ 对应 dtype 的 2^-7/2^-10。
（mode 1 vs oracle 用 `x2=0` 喂 CANN，使其残差输出恒等输入，比较同一件事。）

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
