# lse_merge 精度验证报告（混合精度扩展，2026-09-02）

- 用例总数 54（常规 shape 30 + 边界值 24），通过 54，失败 0，通过率 100.0%
- 判定标准：MERE/MARE（生态算子开源精度标准，分母 |ref|+1e-7）；bf16-out 阈值 7.81e-3 / 7.81e-2，fp32-out 阈值 1.22e-4 / 1.22e-3
- 参考实现：fp64 torch（FlashInfer 公式）；padded-LSE 用例使用 catlass FAInferBf16Fp32Out 的 32B padded 行布局（8×fp32 复制）
- 注：用例为自行设计（非 testcase-gen 产出，lse_merge 为 M2 既有算子补齐）

## 常规 Shape 用例

| 用例 | 输出 dtype | MERE | MARE | MaxAbsErr | 判定 |
|---|---|---|---|---|---|
| A_tier0/production/64x40x128 | bfloat16 | 1.52e-07 | 7.52e-03 | 7.81e-03 | PASS |
| A_tier0/0.5B/56x8x128 | bfloat16 | 1.31e-07 | 7.52e-03 | 3.91e-03 | PASS |
| A_tier0/tiny/7x4x16 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/odd-rows/13x8x128 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/single-row/1x40x128 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/small-tile/32x32x64 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| B_fp32out/production/64x40x128 | float32 | 1.03e-07 | 5.77e-06 | 4.77e-07 | PASS |
| B_fp32out/0.5B/56x8x128 | float32 | 1.63e-07 | 4.27e-06 | 4.77e-07 | PASS |
| B_fp32out/tiny/7x4x16 | float32 | 6.66e-08 | 1.78e-06 | 2.38e-07 | PASS |
| B_fp32out/odd-rows/13x8x128 | float32 | 8.68e-08 | 2.31e-06 | 4.77e-07 | PASS |
| B_fp32out/single-row/1x40x128 | float32 | 7.90e-08 | 2.51e-06 | 2.38e-07 | PASS |
| B_fp32out/small-tile/32x32x64 | float32 | 1.11e-07 | 5.09e-06 | 4.77e-07 | PASS |
| Bp_fp32out_pad/production/64x40x128 | float32 | 1.67e-07 | 5.41e-06 | 4.77e-07 | PASS |
| Bp_fp32out_pad/0.5B/56x8x128 | float32 | 1.08e-07 | 4.89e-06 | 4.77e-07 | PASS |
| Bp_fp32out_pad/tiny/7x4x16 | float32 | 9.35e-08 | 1.04e-06 | 2.38e-07 | PASS |
| Bp_fp32out_pad/odd-rows/13x8x128 | float32 | 1.48e-07 | 7.15e-06 | 2.38e-07 | PASS |
| Bp_fp32out_pad/single-row/1x40x128 | float32 | 7.98e-08 | 4.44e-06 | 2.38e-07 | PASS |
| Bp_fp32out_pad/small-tile/32x32x64 | float32 | 1.10e-07 | 4.09e-06 | 2.38e-07 | PASS |
| C_hybrid_bf16/production/64x40x128 | bfloat16 | 8.43e-08 | 6.80e-03 | 7.81e-03 | PASS |
| C_hybrid_bf16/0.5B/56x8x128 | bfloat16 | 8.11e-08 | 4.65e-03 | 3.91e-03 | PASS |
| C_hybrid_bf16/tiny/7x4x16 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C_hybrid_bf16/odd-rows/13x8x128 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C_hybrid_bf16/single-row/1x40x128 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C_hybrid_bf16/small-tile/32x32x64 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| D_tier0_pad/production/64x40x128 | bfloat16 | 1.04e-07 | 6.58e-03 | 7.81e-03 | PASS |
| D_tier0_pad/0.5B/56x8x128 | bfloat16 | 2.75e-07 | 6.71e-03 | 7.81e-03 | PASS |
| D_tier0_pad/tiny/7x4x16 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| D_tier0_pad/odd-rows/13x8x128 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| D_tier0_pad/single-row/1x40x128 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| D_tier0_pad/small-tile/32x32x64 | bfloat16 | 1.01e-07 | 6.61e-03 | 4.77e-07 | PASS |

## 边界值用例

| 用例 | 输出 dtype | MERE | MARE | MaxAbsErr | 判定 |
|---|---|---|---|---|---|
| A_tier0/equal_weights | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/dlse_p40 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/dlse_m40 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/large_mag_30_35 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| A_tier0/o2_zero | bfloat16 | 9.59e-08 | 6.94e-03 | 3.91e-03 | PASS |
| A_tier0/o1_zero | bfloat16 | 3.59e-08 | 5.88e-03 | 3.91e-03 | PASS |
| B_fp32out/equal_weights | float32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| B_fp32out/dlse_p40 | float32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| B_fp32out/dlse_m40 | float32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| B_fp32out/large_mag_30_35 | float32 | 2.95e-08 | 1.36e-07 | 4.77e-07 | PASS |
| B_fp32out/o2_zero | float32 | 3.94e-08 | 3.36e-07 | 4.77e-07 | PASS |
| B_fp32out/o1_zero | float32 | 3.96e-08 | 3.84e-07 | 2.38e-07 | PASS |
| C_hybrid_bf16/equal_weights | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C_hybrid_bf16/dlse_p40 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C_hybrid_bf16/dlse_m40 | bfloat16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C_hybrid_bf16/large_mag_30_35 | bfloat16 | 1.24e-08 | 4.05e-03 | 3.91e-03 | PASS |
| C_hybrid_bf16/o2_zero | bfloat16 | 3.22e-08 | 6.21e-03 | 4.88e-04 | PASS |
| C_hybrid_bf16/o1_zero | bfloat16 | 3.59e-08 | 5.88e-03 | 3.91e-03 | PASS |
| Bp_fp32out_pad/equal_weights | float32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| Bp_fp32out_pad/dlse_p40 | float32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| Bp_fp32out_pad/dlse_m40 | float32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| Bp_fp32out_pad/large_mag_30_35 | float32 | 2.95e-08 | 1.36e-07 | 4.77e-07 | PASS |
| Bp_fp32out_pad/o2_zero | float32 | 3.94e-08 | 3.36e-07 | 4.77e-07 | PASS |
| Bp_fp32out_pad/o1_zero | float32 | 3.96e-08 | 3.84e-07 | 2.38e-07 | PASS |

## 关键发现

1. fp32-out（Tier1 probe 形态）24 例全部通过，MERE 峰值 1.67e-07（阈值 1.22e-4）——混合合入误差在 fp32 求和噪声量级，Tier1 残差符合 w2·ε2 + ε_order 预期。
2. bf16-out（Tier0 回归 + Tier1 集成形态）30 例全部通过，MERE 峰值 2.75e-07（阈值 7.81e-3）——即最终 bf16 量化噪声主导。
3. padded-LSE（stride=8，catlass stage-1 kernel 布局）与 compact 结果逐 bit 相等（专项用例），Tier0 padded 回归通过——M-C 可零拷贝直传 stage-1 LSE。
4. 边界值（权重极端 dlse=±40、等权、大幅值 lse、零输入）全部通过——Exp 前减 max 的设计在 ±40 差值域无溢出路径。
5. rows=7/15/23（核内行数非 8 倍数）修复后全部通过——M2 潜伏 LSE DataCopy 非 32B 损坏 bug（DataCopyPad 修复）在边缘形状套件中暴露并已验证修复。
