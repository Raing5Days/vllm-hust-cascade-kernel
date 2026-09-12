# add_rms_norm_stats 精度验证报告（F2 norm 阶段核，2026-09-12）

- 用例总数 64，通过 64，失败 0，通过率 100.0%
- 参考实现：CPU fp64 累加 + CANN golden 舍入口径（`add_rms_norm_stats_ref.py`）
- **判据（容差按 ops-precision-standard）**：workspace skill `.agents/skills/ops-precision-standard/`（浮点计算类）的混合容差——逐元素 `|out−ref| <= atol + rtol·|ref|`，整体 `matched_ratio >= 0.99` 且 `max_abs_error <= max(fixed_limit, 32·ULP@1.0)`。档位按**输出** dtype 取表：fp16 atol=rtol=2^-9、bf16 2^-6、rstd（fp32）atol=2^-16/rtol=2^-10；abs 上限 fp16 0.1 / bf16 1.0 / fp32 1e-2
- 旁证列（**不参与判定**，留档以便复核）：`matched_ratio` 为标准判据量；`rel_l2` 为批级相对 L2；`cann_viol/n` 为改用 **CANN 自带用例更严档位**（atol=rtol=2^-7/2^-10 且要求全元素）时的违例数——用它把『与现役算子的边际距离』量化留档（见 `add_rms_norm_stats-test-cases.md` §3.1/§3.2）
- 生产 oracle 交叉核对（`torch_npu.npu_add_rms_norm`，beta=None）：48/48 通过；该路径按**更严**口径要求逐元素全过（matched_ratio = 1.0），因为它的目的是证明与现役算子不可区分
- shape/口径见 `add_rms_norm_stats-test-cases.md`

## 用例明细

| mode | case | shape | dtype | beta | 输出 | matched | rel_l2 | viol/n | MaxAbsErr | abs上限 | 判定 | CANN严档 viol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 residual_stats | prefill-main | 2048x5120 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/10485760 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | prefill-main | 2048x5120 | bfloat16 | N | rstd | 1.00000000 | 3.374e-06 | 0/2048 | 2.146e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | decode | 32x5120 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/163840 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | decode | 32x5120 | bfloat16 | N | rstd | 1.00000000 | 3.660e-06 | 0/32 | 1.705e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | single-row | 1x5120 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | single-row | 1x5120 | bfloat16 | N | rstd | 1.00000000 | 6.871e-06 | 0/1 | 9.775e-06 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | odd-rows-77 | 77x128 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/9856 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 1.00000000 | 3.080e-06 | 0/77 | 1.776e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | k-not-pow2 | 255x5008 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/1277040 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 1.00000000 | 3.906e-06 | 0/255 | 2.217e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | k-min | 7x16 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | k-min | 7x16 | bfloat16 | N | rstd | 1.00000000 | 2.999e-06 | 0/7 | 9.894e-06 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | m-1000 | 1000x5120 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/5120000 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | m-1000 | 1000x5120 | bfloat16 | N | rstd | 1.00000000 | 3.346e-06 | 0/1000 | 2.038e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | k-2048 | 128x2048 | bfloat16 | N | x_out | 1.00000000 | 0.000e+00 | 0/262144 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 0 residual_stats | k-2048 | 128x2048 | bfloat16 | N | rstd | 1.00000000 | 3.364e-06 | 0/128 | 1.776e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | prefill-main | 2048x5120 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/10485760 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | prefill-main | 2048x5120 | float16 | N | rstd | 1.00000000 | 3.326e-06 | 0/2048 | 2.074e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | decode | 32x5120 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/163840 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | decode | 32x5120 | float16 | N | rstd | 1.00000000 | 3.856e-06 | 0/32 | 1.669e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | single-row | 1x5120 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | single-row | 1x5120 | float16 | N | rstd | 1.00000000 | 8.265e-06 | 0/1 | 1.168e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | odd-rows-77 | 77x128 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/9856 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | odd-rows-77 | 77x128 | float16 | N | rstd | 1.00000000 | 2.617e-06 | 0/77 | 1.431e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | k-not-pow2 | 255x5008 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/1277040 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | k-not-pow2 | 255x5008 | float16 | N | rstd | 1.00000000 | 3.197e-06 | 0/255 | 2.038e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | k-min | 7x16 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | k-min | 7x16 | float16 | N | rstd | 1.00000000 | 2.429e-06 | 0/7 | 9.298e-06 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | m-1000 | 1000x5120 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/5120000 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | m-1000 | 1000x5120 | float16 | N | rstd | 1.00000000 | 3.475e-06 | 0/1000 | 2.217e-05 | 1.000e-02 | PASS | 0 |
| 0 residual_stats | k-2048 | 128x2048 | float16 | N | x_out | 1.00000000 | 0.000e+00 | 0/262144 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 0 residual_stats | k-2048 | 128x2048 | float16 | N | rstd | 1.00000000 | 3.205e-06 | 0/128 | 1.800e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | N | rstd | 1.00000000 | 2.734e-06 | 0/2048 | 2.277e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | N | y | 1.00000000 | 1.109e-04 | 0/10485760 | 3.125e-02 | 1.000e+00 | PASS | 6 |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | Y | rstd | 1.00000000 | 2.733e-06 | 0/2048 | 2.456e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | Y | y | 1.00000000 | 1.008e-04 | 0/10485760 | 3.125e-02 | 1.000e+00 | PASS | 5 |
| 1 stats_apply | decode | 32x5120 | bfloat16 | N | rstd | 1.00000000 | 3.713e-06 | 0/32 | 1.836e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | decode | 32x5120 | bfloat16 | N | y | 1.00000000 | 7.931e-05 | 0/163840 | 1.562e-02 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | decode | 32x5120 | bfloat16 | Y | rstd | 1.00000000 | 2.575e-06 | 0/32 | 1.717e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | decode | 32x5120 | bfloat16 | Y | y | 1.00000000 | 7.909e-05 | 0/163840 | 1.562e-02 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | N | rstd | 1.00000000 | 1.188e-07 | 0/1 | 2.384e-07 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | N | y | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | Y | rstd | 1.00000000 | 2.306e-06 | 0/1 | 4.768e-06 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | Y | y | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 1.00000000 | 3.695e-06 | 0/77 | 2.193e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | N | y | 1.00000000 | 7.930e-05 | 0/9856 | 7.812e-03 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | Y | rstd | 1.00000000 | 2.637e-06 | 0/77 | 1.717e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | Y | y | 1.00000000 | 2.183e-05 | 0/9856 | 1.953e-03 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 1.00000000 | 2.744e-06 | 0/255 | 1.609e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | N | y | 1.00000000 | 8.738e-05 | 0/1277040 | 1.562e-02 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | Y | rstd | 1.00000000 | 2.678e-06 | 0/255 | 1.454e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | Y | y | 1.00000000 | 1.108e-04 | 0/1277040 | 3.125e-02 | 1.000e+00 | PASS | 1 |
| 1 stats_apply | k-min | 7x16 | bfloat16 | N | rstd | 1.00000000 | 1.335e-06 | 0/7 | 6.437e-06 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-min | 7x16 | bfloat16 | N | y | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | k-min | 7x16 | bfloat16 | Y | rstd | 1.00000000 | 4.423e-06 | 0/7 | 1.574e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-min | 7x16 | bfloat16 | Y | y | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | N | rstd | 1.00000000 | 2.831e-06 | 0/1000 | 2.277e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | N | y | 1.00000000 | 1.064e-04 | 0/5120000 | 3.125e-02 | 1.000e+00 | PASS | 2 |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | Y | rstd | 1.00000000 | 2.846e-06 | 0/1000 | 2.384e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | Y | y | 1.00000000 | 1.275e-04 | 0/5120000 | 3.125e-02 | 1.000e+00 | PASS | 4 |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | N | rstd | 1.00000000 | 3.078e-06 | 0/128 | 2.241e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | N | y | 1.00000000 | 7.712e-05 | 0/262144 | 1.562e-02 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | Y | rstd | 1.00000000 | 2.782e-06 | 0/128 | 1.693e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | Y | y | 1.00000000 | 1.229e-04 | 0/262144 | 3.125e-02 | 1.000e+00 | PASS | 0 |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | N | rstd | 1.00000000 | 2.788e-06 | 0/2048 | 1.884e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | N | y | 1.00000000 | 4.075e-05 | 0/10485760 | 3.906e-03 | 1.000e-01 | PASS | 20 |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | Y | rstd | 1.00000000 | 2.699e-06 | 0/2048 | 1.955e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | Y | y | 1.00000000 | 3.882e-05 | 0/10485760 | 3.906e-03 | 1.000e-01 | PASS | 14 |
| 1 stats_apply | decode | 32x5120 | float16 | N | rstd | 1.00000000 | 2.904e-06 | 0/32 | 1.311e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | decode | 32x5120 | float16 | N | y | 1.00000000 | 4.074e-05 | 0/163840 | 1.953e-03 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | decode | 32x5120 | float16 | Y | rstd | 1.00000000 | 2.245e-06 | 0/32 | 9.537e-06 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | decode | 32x5120 | float16 | Y | y | 1.00000000 | 3.661e-05 | 0/163840 | 1.953e-03 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | float16 | N | rstd | 1.00000000 | 1.805e-07 | 0/1 | 3.576e-07 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | float16 | N | y | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | float16 | Y | rstd | 1.00000000 | 1.218e-07 | 0/1 | 2.384e-07 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | single-row | 1x5120 | float16 | Y | y | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | N | rstd | 1.00000000 | 2.572e-06 | 0/77 | 2.122e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | N | y | 1.00000000 | 2.874e-05 | 0/9856 | 1.953e-03 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | Y | rstd | 1.00000000 | 3.438e-06 | 0/77 | 1.764e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | Y | y | 1.00000000 | 6.017e-05 | 0/9856 | 3.906e-03 | 1.000e-01 | PASS | 1 |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | N | rstd | 1.00000000 | 2.846e-06 | 0/255 | 1.931e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | N | y | 1.00000000 | 3.860e-05 | 0/1277040 | 3.906e-03 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | Y | rstd | 1.00000000 | 2.507e-06 | 0/255 | 1.645e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | Y | y | 1.00000000 | 3.692e-05 | 0/1277040 | 3.906e-03 | 1.000e-01 | PASS | 1 |
| 1 stats_apply | k-min | 7x16 | float16 | N | rstd | 1.00000000 | 4.655e-06 | 0/7 | 2.384e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-min | 7x16 | float16 | N | y | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | k-min | 7x16 | float16 | Y | rstd | 1.00000000 | 3.103e-06 | 0/7 | 1.478e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-min | 7x16 | float16 | Y | y | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | N | rstd | 1.00000000 | 2.824e-06 | 0/1000 | 2.098e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | N | y | 1.00000000 | 4.059e-05 | 0/5120000 | 3.906e-03 | 1.000e-01 | PASS | 9 |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | Y | rstd | 1.00000000 | 2.707e-06 | 0/1000 | 2.193e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | Y | y | 1.00000000 | 3.831e-05 | 0/5120000 | 3.906e-03 | 1.000e-01 | PASS | 11 |
| 1 stats_apply | k-2048 | 128x2048 | float16 | N | rstd | 1.00000000 | 3.132e-06 | 0/128 | 1.884e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-2048 | 128x2048 | float16 | N | y | 1.00000000 | 4.041e-05 | 0/262144 | 1.953e-03 | 1.000e-01 | PASS | 0 |
| 1 stats_apply | k-2048 | 128x2048 | float16 | Y | rstd | 1.00000000 | 3.037e-06 | 0/128 | 2.003e-05 | 1.000e-02 | PASS | 0 |
| 1 stats_apply | k-2048 | 128x2048 | float16 | Y | y | 1.00000000 | 4.020e-05 | 0/262144 | 3.906e-03 | 1.000e-01 | PASS | 2 |
| 2 stats_only | prefill-main | 2048x5120 | bfloat16 | N | rstd | 1.00000000 | 3.435e-06 | 0/2048 | 2.110e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | decode | 32x5120 | bfloat16 | N | rstd | 1.00000000 | 2.992e-06 | 0/32 | 9.537e-06 | 1.000e-02 | PASS | 0 |
| 2 stats_only | single-row | 1x5120 | bfloat16 | N | rstd | 1.00000000 | 3.629e-06 | 0/1 | 5.126e-06 | 1.000e-02 | PASS | 0 |
| 2 stats_only | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 1.00000000 | 3.186e-06 | 0/77 | 1.693e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 1.00000000 | 3.335e-06 | 0/255 | 2.074e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | k-min | 7x16 | bfloat16 | N | rstd | 1.00000000 | 2.875e-06 | 0/7 | 1.037e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | m-1000 | 1000x5120 | bfloat16 | N | rstd | 1.00000000 | 3.180e-06 | 0/1000 | 2.015e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | k-2048 | 128x2048 | bfloat16 | N | rstd | 1.00000000 | 3.214e-06 | 0/128 | 2.027e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | prefill-main | 2048x5120 | float16 | N | rstd | 1.00000000 | 3.507e-06 | 0/2048 | 2.217e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | decode | 32x5120 | float16 | N | rstd | 1.00000000 | 3.485e-06 | 0/32 | 1.466e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | single-row | 1x5120 | float16 | N | rstd | 1.00000000 | 2.267e-06 | 0/1 | 3.219e-06 | 1.000e-02 | PASS | 0 |
| 2 stats_only | odd-rows-77 | 77x128 | float16 | N | rstd | 1.00000000 | 3.943e-06 | 0/77 | 1.669e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | k-not-pow2 | 255x5008 | float16 | N | rstd | 1.00000000 | 3.260e-06 | 0/255 | 2.027e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | k-min | 7x16 | float16 | N | rstd | 1.00000000 | 2.679e-06 | 0/7 | 9.775e-06 | 1.000e-02 | PASS | 0 |
| 2 stats_only | m-1000 | 1000x5120 | float16 | N | rstd | 1.00000000 | 3.579e-06 | 0/1000 | 2.062e-05 | 1.000e-02 | PASS | 0 |
| 2 stats_only | k-2048 | 128x2048 | float16 | N | rstd | 1.00000000 | 2.883e-06 | 0/128 | 1.681e-05 | 1.000e-02 | PASS | 0 |

## 生产 oracle 交叉核对（vs `torch_npu.npu_add_rms_norm`）

| mode | case | shape | dtype | 输出 | matched | rel_l2 | viol/n | MaxAbsErr | 判定 | CANN严档 viol |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | prefill-main | 2048x5120 | bfloat16 | rstd | 1.00000000 | 4.489e-05 | 0/2048 | 2.176e-04 | PASS | 0 |
| 0 | prefill-main | 2048x5120 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/10485760 | 0.000e+00 | PASS | 0 |
| 0 | decode | 32x5120 | bfloat16 | rstd | 1.00000000 | 3.519e-06 | 0/32 | 1.395e-05 | PASS | 0 |
| 0 | decode | 32x5120 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/163840 | 0.000e+00 | PASS | 0 |
| 0 | single-row | 1x5120 | bfloat16 | rstd | 1.00000000 | 1.175e-06 | 0/1 | 1.669e-06 | PASS | 0 |
| 0 | single-row | 1x5120 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | PASS | 0 |
| 0 | odd-rows-77 | 77x128 | bfloat16 | rstd | 1.00000000 | 3.594e-06 | 0/77 | 1.872e-05 | PASS | 0 |
| 0 | odd-rows-77 | 77x128 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/9856 | 0.000e+00 | PASS | 0 |
| 0 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 1.00000000 | 4.722e-05 | 0/255 | 1.987e-04 | PASS | 0 |
| 0 | k-not-pow2 | 255x5008 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/1277040 | 0.000e+00 | PASS | 0 |
| 0 | k-min | 7x16 | bfloat16 | rstd | 1.00000000 | 9.678e-07 | 0/7 | 2.027e-06 | PASS | 0 |
| 0 | k-min | 7x16 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | PASS | 0 |
| 0 | m-1000 | 1000x5120 | bfloat16 | rstd | 1.00000000 | 4.319e-05 | 0/1000 | 2.012e-04 | PASS | 0 |
| 0 | m-1000 | 1000x5120 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/5120000 | 0.000e+00 | PASS | 0 |
| 0 | k-2048 | 128x2048 | bfloat16 | rstd | 1.00000000 | 6.740e-05 | 0/128 | 3.037e-04 | PASS | 0 |
| 0 | k-2048 | 128x2048 | bfloat16 | x_out | 1.00000000 | 0.000e+00 | 0/262144 | 0.000e+00 | PASS | 0 |
| 0 | prefill-main | 2048x5120 | float16 | rstd | 1.00000000 | 3.506e-06 | 0/2048 | 2.086e-05 | PASS | 0 |
| 0 | prefill-main | 2048x5120 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/10485760 | 0.000e+00 | PASS | 0 |
| 0 | decode | 32x5120 | float16 | rstd | 1.00000000 | 2.626e-06 | 0/32 | 7.868e-06 | PASS | 0 |
| 0 | decode | 32x5120 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/163840 | 0.000e+00 | PASS | 0 |
| 0 | single-row | 1x5120 | float16 | rstd | 1.00000000 | 3.470e-06 | 0/1 | 4.888e-06 | PASS | 0 |
| 0 | single-row | 1x5120 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | PASS | 0 |
| 0 | odd-rows-77 | 77x128 | float16 | rstd | 1.00000000 | 4.011e-06 | 0/77 | 2.074e-05 | PASS | 0 |
| 0 | odd-rows-77 | 77x128 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/9856 | 0.000e+00 | PASS | 0 |
| 0 | k-not-pow2 | 255x5008 | float16 | rstd | 1.00000000 | 3.247e-06 | 0/255 | 2.098e-05 | PASS | 0 |
| 0 | k-not-pow2 | 255x5008 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/1277040 | 0.000e+00 | PASS | 0 |
| 0 | k-min | 7x16 | float16 | rstd | 1.00000000 | 2.652e-06 | 0/7 | 9.894e-06 | PASS | 0 |
| 0 | k-min | 7x16 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | PASS | 0 |
| 0 | m-1000 | 1000x5120 | float16 | rstd | 1.00000000 | 3.120e-06 | 0/1000 | 2.098e-05 | PASS | 0 |
| 0 | m-1000 | 1000x5120 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/5120000 | 0.000e+00 | PASS | 0 |
| 0 | k-2048 | 128x2048 | float16 | rstd | 1.00000000 | 2.927e-06 | 0/128 | 1.359e-05 | PASS | 0 |
| 0 | k-2048 | 128x2048 | float16 | x_out | 1.00000000 | 0.000e+00 | 0/262144 | 0.000e+00 | PASS | 0 |
| 1 | prefill-main | 2048x5120 | bfloat16 | rstd | 1.00000000 | 2.789e-06 | 0/2048 | 2.456e-05 | PASS | 0 |
| 1 | prefill-main | 2048x5120 | bfloat16 | y | 1.00000000 | 1.185e-04 | 0/10485760 | 3.125e-02 | PASS | 4 |
| 1 | decode | 32x5120 | bfloat16 | rstd | 1.00000000 | 2.438e-06 | 0/32 | 1.132e-05 | PASS | 0 |
| 1 | decode | 32x5120 | bfloat16 | y | 1.00000000 | 4.678e-05 | 0/163840 | 7.812e-03 | PASS | 0 |
| 1 | single-row | 1x5120 | bfloat16 | rstd | 1.00000000 | 1.179e-07 | 0/1 | 2.384e-07 | PASS | 0 |
| 1 | single-row | 1x5120 | bfloat16 | y | 1.00000000 | 0.000e+00 | 0/5120 | 0.000e+00 | PASS | 0 |
| 1 | odd-rows-77 | 77x128 | bfloat16 | rstd | 1.00000000 | 3.858e-06 | 0/77 | 2.480e-05 | PASS | 0 |
| 1 | odd-rows-77 | 77x128 | bfloat16 | y | 1.00000000 | 1.567e-04 | 0/9856 | 7.812e-03 | PASS | 0 |
| 1 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 1.00000000 | 2.575e-06 | 0/255 | 1.466e-05 | PASS | 0 |
| 1 | k-not-pow2 | 255x5008 | bfloat16 | y | 1.00000000 | 6.629e-05 | 0/1277040 | 1.562e-02 | PASS | 0 |
| 1 | k-min | 7x16 | bfloat16 | rstd | 1.00000000 | 2.296e-06 | 0/7 | 1.049e-05 | PASS | 0 |
| 1 | k-min | 7x16 | bfloat16 | y | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | PASS | 0 |
| 1 | m-1000 | 1000x5120 | bfloat16 | rstd | 1.00000000 | 2.715e-06 | 0/1000 | 2.170e-05 | PASS | 0 |
| 1 | m-1000 | 1000x5120 | bfloat16 | y | 1.00000000 | 1.040e-04 | 0/5120000 | 3.125e-02 | PASS | 1 |
| 1 | k-2048 | 128x2048 | bfloat16 | rstd | 1.00000000 | 3.014e-06 | 0/128 | 1.740e-05 | PASS | 0 |
| 1 | k-2048 | 128x2048 | bfloat16 | y | 1.00000000 | 1.252e-04 | 0/262144 | 1.562e-02 | PASS | 0 |
| 1 | prefill-main | 2048x5120 | float16 | rstd | 1.00000000 | 2.809e-06 | 0/2048 | 2.098e-05 | PASS | 0 |
| 1 | prefill-main | 2048x5120 | float16 | y | 1.00000000 | 3.938e-05 | 0/10485760 | 3.906e-03 | PASS | 20 |
| 1 | decode | 32x5120 | float16 | rstd | 1.00000000 | 2.531e-06 | 0/32 | 1.144e-05 | PASS | 0 |
| 1 | decode | 32x5120 | float16 | y | 1.00000000 | 3.743e-05 | 0/163840 | 3.906e-03 | PASS | 0 |
| 1 | single-row | 1x5120 | float16 | rstd | 1.00000000 | 1.815e-06 | 0/1 | 3.576e-06 | PASS | 0 |
| 1 | single-row | 1x5120 | float16 | y | 1.00000000 | 3.389e-06 | 0/5120 | 2.441e-04 | PASS | 0 |
| 1 | odd-rows-77 | 77x128 | float16 | rstd | 1.00000000 | 3.567e-06 | 0/77 | 2.480e-05 | PASS | 0 |
| 1 | odd-rows-77 | 77x128 | float16 | y | 1.00000000 | 3.053e-05 | 0/9856 | 9.766e-04 | PASS | 0 |
| 1 | k-not-pow2 | 255x5008 | float16 | rstd | 1.00000000 | 2.946e-06 | 0/255 | 2.146e-05 | PASS | 0 |
| 1 | k-not-pow2 | 255x5008 | float16 | y | 1.00000000 | 4.039e-05 | 0/1277040 | 3.906e-03 | PASS | 3 |
| 1 | k-min | 7x16 | float16 | rstd | 1.00000000 | 3.384e-06 | 0/7 | 1.407e-05 | PASS | 0 |
| 1 | k-min | 7x16 | float16 | y | 1.00000000 | 0.000e+00 | 0/112 | 0.000e+00 | PASS | 0 |
| 1 | m-1000 | 1000x5120 | float16 | rstd | 1.00000000 | 2.784e-06 | 0/1000 | 1.955e-05 | PASS | 0 |
| 1 | m-1000 | 1000x5120 | float16 | y | 1.00000000 | 4.010e-05 | 0/5120000 | 3.906e-03 | PASS | 8 |
| 1 | k-2048 | 128x2048 | float16 | rstd | 1.00000000 | 2.951e-06 | 0/128 | 2.110e-05 | PASS | 0 |
| 1 | k-2048 | 128x2048 | float16 | y | 1.00000000 | 3.965e-05 | 0/262144 | 3.906e-03 | PASS | 1 |
| 2 | prefill-main | 2048x5120 | bfloat16 | rstd | 1.00000000 | 4.467e-05 | 0/2048 | 2.073e-04 | PASS | 0 |
| 2 | decode | 32x5120 | bfloat16 | rstd | 1.00000000 | 4.111e-06 | 0/32 | 1.705e-05 | PASS | 0 |
| 2 | single-row | 1x5120 | bfloat16 | rstd | 1.00000000 | 2.183e-06 | 0/1 | 3.099e-06 | PASS | 0 |
| 2 | odd-rows-77 | 77x128 | bfloat16 | rstd | 1.00000000 | 3.613e-06 | 0/77 | 1.717e-05 | PASS | 0 |
| 2 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 1.00000000 | 4.547e-05 | 0/255 | 2.598e-04 | PASS | 0 |
| 2 | k-min | 7x16 | bfloat16 | rstd | 1.00000000 | 1.923e-06 | 0/7 | 5.722e-06 | PASS | 0 |
| 2 | m-1000 | 1000x5120 | bfloat16 | rstd | 1.00000000 | 4.332e-05 | 0/1000 | 1.832e-04 | PASS | 0 |
| 2 | k-2048 | 128x2048 | bfloat16 | rstd | 1.00000000 | 7.413e-05 | 0/128 | 3.612e-04 | PASS | 0 |
| 2 | prefill-main | 2048x5120 | float16 | rstd | 1.00000000 | 3.451e-06 | 0/2048 | 2.146e-05 | PASS | 0 |
| 2 | decode | 32x5120 | float16 | rstd | 1.00000000 | 3.946e-06 | 0/32 | 1.609e-05 | PASS | 0 |
| 2 | single-row | 1x5120 | float16 | rstd | 1.00000000 | 9.998e-07 | 0/1 | 1.431e-06 | PASS | 0 |
| 2 | odd-rows-77 | 77x128 | float16 | rstd | 1.00000000 | 4.513e-06 | 0/77 | 1.764e-05 | PASS | 0 |
| 2 | k-not-pow2 | 255x5008 | float16 | rstd | 1.00000000 | 3.586e-06 | 0/255 | 2.038e-05 | PASS | 0 |
| 2 | k-min | 7x16 | float16 | rstd | 1.00000000 | 3.743e-06 | 0/7 | 1.264e-05 | PASS | 0 |
| 2 | m-1000 | 1000x5120 | float16 | rstd | 1.00000000 | 3.214e-06 | 0/1000 | 2.027e-05 | PASS | 0 |
| 2 | k-2048 | 128x2048 | float16 | rstd | 1.00000000 | 3.365e-06 | 0/128 | 1.919e-05 | PASS | 0 |
