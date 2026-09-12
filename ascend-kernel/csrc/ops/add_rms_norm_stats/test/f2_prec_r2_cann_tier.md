# add_rms_norm_stats 精度验证报告（F2 norm 阶段核，2026-09-12）

- 用例总数 64，通过 52，失败 12，通过率 81.2%
- 参考实现：CPU fp64 累加 + CANN golden 舍入口径（`add_rms_norm_stats_ref.py`）
- 判据：rel_l2 + CANN 官方逐元素档位 `|out−ref| <= atol + rtol·|ref|`（bf16 atol=rtol=2^-7，fp16 atol=rtol=2^-10，viol=违例元素数，必须 0），rstd 无 CANN atol 档位，按相对判据（atol=0, rtol=2^-6）
- 旁证列 `MaxAbsErr/MARE` 为原始诊断量：`MARE` 是**逐元素最大相对误差**（近零元素上由 atol 项兜底），首版实现误用它当绝对档位、并漏掉 rtol 项；`strict_abs_only` 标记该更严读数是否也满足（不参与判定）
- 生产 oracle 交叉核对（`torch_npu.npu_add_rms_norm`，beta=None）：42/48 通过
- shape/口径见 `add_rms_norm_stats-test-cases.md`

## 用例明细

| mode | case | shape | dtype | beta | 输出 | rel_l2 | viol/n | MARE | MaxAbsErr | 判定 | 严读数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 residual_stats | prefill-main | 2048x5120 | bfloat16 | N | x_out | 0.000e+00 | 0/10485760 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | prefill-main | 2048x5120 | bfloat16 | N | rstd | 3.374e-06 | 0/2048 | 1.552e-05 | 2.146e-05 | PASS | over |
| 0 residual_stats | decode | 32x5120 | bfloat16 | N | x_out | 0.000e+00 | 0/163840 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | decode | 32x5120 | bfloat16 | N | rstd | 3.660e-06 | 0/32 | 1.205e-05 | 1.705e-05 | PASS | over |
| 0 residual_stats | single-row | 1x5120 | bfloat16 | N | x_out | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | single-row | 1x5120 | bfloat16 | N | rstd | 6.871e-06 | 0/1 | 6.871e-06 | 9.775e-06 | PASS | over |
| 0 residual_stats | odd-rows-77 | 77x128 | bfloat16 | N | x_out | 0.000e+00 | 0/9856 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 3.080e-06 | 0/77 | 1.285e-05 | 1.776e-05 | PASS | over |
| 0 residual_stats | k-not-pow2 | 255x5008 | bfloat16 | N | x_out | 0.000e+00 | 0/1277040 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 3.906e-06 | 0/255 | 1.604e-05 | 2.217e-05 | PASS | over |
| 0 residual_stats | k-min | 7x16 | bfloat16 | N | x_out | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | k-min | 7x16 | bfloat16 | N | rstd | 2.999e-06 | 0/7 | 7.242e-06 | 9.894e-06 | PASS | over |
| 0 residual_stats | m-1000 | 1000x5120 | bfloat16 | N | x_out | 0.000e+00 | 0/5120000 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | m-1000 | 1000x5120 | bfloat16 | N | rstd | 3.346e-06 | 0/1000 | 1.475e-05 | 2.038e-05 | PASS | over |
| 0 residual_stats | k-2048 | 128x2048 | bfloat16 | N | x_out | 0.000e+00 | 0/262144 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | k-2048 | 128x2048 | bfloat16 | N | rstd | 3.364e-06 | 0/128 | 1.285e-05 | 1.776e-05 | PASS | over |
| 0 residual_stats | prefill-main | 2048x5120 | float16 | N | x_out | 0.000e+00 | 0/10485760 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | prefill-main | 2048x5120 | float16 | N | rstd | 3.326e-06 | 0/2048 | 1.484e-05 | 2.074e-05 | PASS | over |
| 0 residual_stats | decode | 32x5120 | float16 | N | x_out | 0.000e+00 | 0/163840 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | decode | 32x5120 | float16 | N | rstd | 3.856e-06 | 0/32 | 1.185e-05 | 1.669e-05 | PASS | over |
| 0 residual_stats | single-row | 1x5120 | float16 | N | x_out | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | single-row | 1x5120 | float16 | N | rstd | 8.265e-06 | 0/1 | 8.265e-06 | 1.168e-05 | PASS | over |
| 0 residual_stats | odd-rows-77 | 77x128 | float16 | N | x_out | 0.000e+00 | 0/9856 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | odd-rows-77 | 77x128 | float16 | N | rstd | 2.617e-06 | 0/77 | 1.113e-05 | 1.431e-05 | PASS | over |
| 0 residual_stats | k-not-pow2 | 255x5008 | float16 | N | x_out | 0.000e+00 | 0/1277040 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | k-not-pow2 | 255x5008 | float16 | N | rstd | 3.197e-06 | 0/255 | 1.458e-05 | 2.038e-05 | PASS | over |
| 0 residual_stats | k-min | 7x16 | float16 | N | x_out | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | k-min | 7x16 | float16 | N | rstd | 2.429e-06 | 0/7 | 7.422e-06 | 9.298e-06 | PASS | over |
| 0 residual_stats | m-1000 | 1000x5120 | float16 | N | x_out | 0.000e+00 | 0/5120000 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | m-1000 | 1000x5120 | float16 | N | rstd | 3.475e-06 | 0/1000 | 1.604e-05 | 2.217e-05 | PASS | over |
| 0 residual_stats | k-2048 | 128x2048 | float16 | N | x_out | 0.000e+00 | 0/262144 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 residual_stats | k-2048 | 128x2048 | float16 | N | rstd | 3.205e-06 | 0/128 | 1.287e-05 | 1.800e-05 | PASS | over |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | N | rstd | 2.734e-06 | 0/2048 | 1.165e-05 | 2.277e-05 | PASS | over |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | N | y | 1.109e-04 | 6/10485760 | 1.370e-02 | 3.125e-02 | FAIL | over |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | Y | rstd | 2.733e-06 | 0/2048 | 1.196e-05 | 2.456e-05 | PASS | over |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | Y | y | 1.008e-04 | 5/10485760 | 1.185e+00 | 3.125e-02 | FAIL | over |
| 1 stats_apply | decode | 32x5120 | bfloat16 | N | rstd | 3.713e-06 | 0/32 | 9.357e-06 | 1.836e-05 | PASS | over |
| 1 stats_apply | decode | 32x5120 | bfloat16 | N | y | 7.931e-05 | 0/163840 | 1.015e-02 | 1.562e-02 | PASS | over |
| 1 stats_apply | decode | 32x5120 | bfloat16 | Y | rstd | 2.575e-06 | 0/32 | 8.399e-06 | 1.717e-05 | PASS | over |
| 1 stats_apply | decode | 32x5120 | bfloat16 | Y | y | 7.909e-05 | 0/163840 | 7.752e-03 | 1.562e-02 | PASS | over |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | N | rstd | 1.188e-07 | 0/1 | 1.188e-07 | 2.384e-07 | PASS | over |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | N | y | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | Y | rstd | 2.306e-06 | 0/1 | 2.306e-06 | 4.768e-06 | PASS | over |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | Y | y | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 3.695e-06 | 0/77 | 1.069e-05 | 2.193e-05 | PASS | over |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | N | y | 7.930e-05 | 0/9856 | 7.353e-03 | 7.812e-03 | PASS | ok |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | Y | rstd | 2.637e-06 | 0/77 | 8.431e-06 | 1.717e-05 | PASS | over |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | Y | y | 2.183e-05 | 0/9856 | 5.587e-03 | 1.953e-03 | PASS | ok |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 2.744e-06 | 0/255 | 8.204e-06 | 1.609e-05 | PASS | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | N | y | 8.738e-05 | 0/1277040 | 1.227e-02 | 1.562e-02 | PASS | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | Y | rstd | 2.678e-06 | 0/255 | 7.144e-06 | 1.454e-05 | PASS | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | Y | y | 1.108e-04 | 1/1277040 | 5.907e-02 | 3.125e-02 | FAIL | over |
| 1 stats_apply | k-min | 7x16 | bfloat16 | N | rstd | 1.335e-06 | 0/7 | 4.251e-06 | 6.437e-06 | PASS | over |
| 1 stats_apply | k-min | 7x16 | bfloat16 | N | y | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | k-min | 7x16 | bfloat16 | Y | rstd | 4.423e-06 | 0/7 | 6.204e-06 | 1.574e-05 | PASS | over |
| 1 stats_apply | k-min | 7x16 | bfloat16 | Y | y | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | N | rstd | 2.831e-06 | 0/1000 | 1.165e-05 | 2.277e-05 | PASS | over |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | N | y | 1.064e-04 | 2/5120000 | 1.370e-02 | 3.125e-02 | FAIL | over |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | Y | rstd | 2.846e-06 | 0/1000 | 1.162e-05 | 2.384e-05 | PASS | over |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | Y | y | 1.275e-04 | 4/5120000 | 2.800e-01 | 3.125e-02 | FAIL | over |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | N | rstd | 3.078e-06 | 0/128 | 1.147e-05 | 2.241e-05 | PASS | over |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | N | y | 7.712e-05 | 0/262144 | 1.105e-02 | 1.562e-02 | PASS | over |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | Y | rstd | 2.782e-06 | 0/128 | 8.314e-06 | 1.693e-05 | PASS | over |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | Y | y | 1.229e-04 | 0/262144 | 1.047e-02 | 3.125e-02 | PASS | over |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | N | rstd | 2.788e-06 | 0/2048 | 9.600e-06 | 1.884e-05 | PASS | over |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | N | y | 4.075e-05 | 20/10485760 | 2.038e-03 | 3.906e-03 | FAIL | over |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | Y | rstd | 2.699e-06 | 0/2048 | 9.721e-06 | 1.955e-05 | PASS | over |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | Y | y | 3.882e-05 | 14/10485760 | 3.967e-01 | 3.906e-03 | FAIL | over |
| 1 stats_apply | decode | 32x5120 | float16 | N | rstd | 2.904e-06 | 0/32 | 6.634e-06 | 1.311e-05 | PASS | over |
| 1 stats_apply | decode | 32x5120 | float16 | N | y | 4.074e-05 | 0/163840 | 1.753e-03 | 1.953e-03 | PASS | over |
| 1 stats_apply | decode | 32x5120 | float16 | Y | rstd | 2.245e-06 | 0/32 | 4.705e-06 | 9.537e-06 | PASS | over |
| 1 stats_apply | decode | 32x5120 | float16 | Y | y | 3.661e-05 | 0/163840 | 1.970e-02 | 1.953e-03 | PASS | over |
| 1 stats_apply | single-row | 1x5120 | float16 | N | rstd | 1.805e-07 | 0/1 | 1.805e-07 | 3.576e-07 | PASS | over |
| 1 stats_apply | single-row | 1x5120 | float16 | N | y | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | single-row | 1x5120 | float16 | Y | rstd | 1.218e-07 | 0/1 | 1.218e-07 | 2.384e-07 | PASS | over |
| 1 stats_apply | single-row | 1x5120 | float16 | Y | y | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | N | rstd | 2.572e-06 | 0/77 | 1.032e-05 | 2.122e-05 | PASS | over |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | N | y | 2.874e-05 | 0/9856 | 1.224e-03 | 1.953e-03 | PASS | over |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | Y | rstd | 3.438e-06 | 0/77 | 8.781e-06 | 1.764e-05 | PASS | over |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | Y | y | 6.017e-05 | 1/9856 | 1.153e-02 | 3.906e-03 | FAIL | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | N | rstd | 2.846e-06 | 0/255 | 9.447e-06 | 1.931e-05 | PASS | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | N | y | 3.860e-05 | 0/1277040 | 1.727e-03 | 3.906e-03 | PASS | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | Y | rstd | 2.507e-06 | 0/255 | 8.080e-06 | 1.645e-05 | PASS | over |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | Y | y | 3.692e-05 | 1/1277040 | 7.014e-02 | 3.906e-03 | FAIL | over |
| 1 stats_apply | k-min | 7x16 | float16 | N | rstd | 4.655e-06 | 0/7 | 1.140e-05 | 2.384e-05 | PASS | over |
| 1 stats_apply | k-min | 7x16 | float16 | N | y | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | k-min | 7x16 | float16 | Y | rstd | 3.103e-06 | 0/7 | 6.019e-06 | 1.478e-05 | PASS | over |
| 1 stats_apply | k-min | 7x16 | float16 | Y | y | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | N | rstd | 2.824e-06 | 0/1000 | 1.043e-05 | 2.098e-05 | PASS | over |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | N | y | 4.059e-05 | 9/5120000 | 1.864e-03 | 3.906e-03 | FAIL | over |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | Y | rstd | 2.707e-06 | 0/1000 | 1.122e-05 | 2.193e-05 | PASS | over |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | Y | y | 3.831e-05 | 11/5120000 | 2.872e-01 | 3.906e-03 | FAIL | over |
| 1 stats_apply | k-2048 | 128x2048 | float16 | N | rstd | 3.132e-06 | 0/128 | 9.273e-06 | 1.884e-05 | PASS | over |
| 1 stats_apply | k-2048 | 128x2048 | float16 | N | y | 4.041e-05 | 0/262144 | 1.808e-03 | 1.953e-03 | PASS | over |
| 1 stats_apply | k-2048 | 128x2048 | float16 | Y | rstd | 3.037e-06 | 0/128 | 9.759e-06 | 2.003e-05 | PASS | over |
| 1 stats_apply | k-2048 | 128x2048 | float16 | Y | y | 4.020e-05 | 2/262144 | 1.187e-02 | 3.906e-03 | FAIL | over |
| 2 stats_only | prefill-main | 2048x5120 | bfloat16 | N | rstd | 3.435e-06 | 0/2048 | 1.526e-05 | 2.110e-05 | PASS | over |
| 2 stats_only | decode | 32x5120 | bfloat16 | N | rstd | 2.992e-06 | 0/32 | 6.815e-06 | 9.537e-06 | PASS | over |
| 2 stats_only | single-row | 1x5120 | bfloat16 | N | rstd | 3.629e-06 | 0/1 | 3.629e-06 | 5.126e-06 | PASS | over |
| 2 stats_only | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 3.186e-06 | 0/77 | 1.238e-05 | 1.693e-05 | PASS | over |
| 2 stats_only | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 3.335e-06 | 0/255 | 1.484e-05 | 2.074e-05 | PASS | over |
| 2 stats_only | k-min | 7x16 | bfloat16 | N | rstd | 2.875e-06 | 0/7 | 7.691e-06 | 1.037e-05 | PASS | over |
| 2 stats_only | m-1000 | 1000x5120 | bfloat16 | N | rstd | 3.180e-06 | 0/1000 | 1.441e-05 | 2.015e-05 | PASS | over |
| 2 stats_only | k-2048 | 128x2048 | bfloat16 | N | rstd | 3.214e-06 | 0/128 | 1.466e-05 | 2.027e-05 | PASS | over |
| 2 stats_only | prefill-main | 2048x5120 | float16 | N | rstd | 3.507e-06 | 0/2048 | 1.604e-05 | 2.217e-05 | PASS | over |
| 2 stats_only | decode | 32x5120 | float16 | N | rstd | 3.485e-06 | 0/32 | 1.037e-05 | 1.466e-05 | PASS | over |
| 2 stats_only | single-row | 1x5120 | float16 | N | rstd | 2.267e-06 | 0/1 | 2.267e-06 | 3.219e-06 | PASS | over |
| 2 stats_only | odd-rows-77 | 77x128 | float16 | N | rstd | 3.943e-06 | 0/77 | 1.260e-05 | 1.669e-05 | PASS | over |
| 2 stats_only | k-not-pow2 | 255x5008 | float16 | N | rstd | 3.260e-06 | 0/255 | 1.450e-05 | 2.027e-05 | PASS | over |
| 2 stats_only | k-min | 7x16 | float16 | N | rstd | 2.679e-06 | 0/7 | 6.480e-06 | 9.775e-06 | PASS | over |
| 2 stats_only | m-1000 | 1000x5120 | float16 | N | rstd | 3.579e-06 | 0/1000 | 1.475e-05 | 2.062e-05 | PASS | over |
| 2 stats_only | k-2048 | 128x2048 | float16 | N | rstd | 2.883e-06 | 0/128 | 1.190e-05 | 1.681e-05 | PASS | over |

## 生产 oracle 交叉核对（vs `torch_npu.npu_add_rms_norm`）

| mode | case | shape | dtype | 输出 | rel_l2 | viol/n | MARE | MaxAbsErr | 判定 | 严读数 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | prefill-main | 2048x5120 | bfloat16 | rstd | 4.489e-05 | 0/2048 | 1.526e-04 | 2.176e-04 | PASS | over |
| 0 | prefill-main | 2048x5120 | bfloat16 | x_out | 0.000e+00 | 0/10485760 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | decode | 32x5120 | bfloat16 | rstd | 3.519e-06 | 0/32 | 9.866e-06 | 1.395e-05 | PASS | over |
| 0 | decode | 32x5120 | bfloat16 | x_out | 0.000e+00 | 0/163840 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | single-row | 1x5120 | bfloat16 | rstd | 1.175e-06 | 0/1 | 1.175e-06 | 1.669e-06 | PASS | over |
| 0 | single-row | 1x5120 | bfloat16 | x_out | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | odd-rows-77 | 77x128 | bfloat16 | rstd | 3.594e-06 | 0/77 | 1.339e-05 | 1.872e-05 | PASS | over |
| 0 | odd-rows-77 | 77x128 | bfloat16 | x_out | 0.000e+00 | 0/9856 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 4.722e-05 | 0/255 | 1.411e-04 | 1.987e-04 | PASS | over |
| 0 | k-not-pow2 | 255x5008 | bfloat16 | x_out | 0.000e+00 | 0/1277040 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | k-min | 7x16 | bfloat16 | rstd | 9.678e-07 | 0/7 | 1.580e-06 | 2.027e-06 | PASS | over |
| 0 | k-min | 7x16 | bfloat16 | x_out | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | m-1000 | 1000x5120 | bfloat16 | rstd | 4.319e-05 | 0/1000 | 1.428e-04 | 2.012e-04 | PASS | over |
| 0 | m-1000 | 1000x5120 | bfloat16 | x_out | 0.000e+00 | 0/5120000 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | k-2048 | 128x2048 | bfloat16 | rstd | 6.740e-05 | 0/128 | 2.107e-04 | 3.037e-04 | PASS | over |
| 0 | k-2048 | 128x2048 | bfloat16 | x_out | 0.000e+00 | 0/262144 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | prefill-main | 2048x5120 | float16 | rstd | 3.506e-06 | 0/2048 | 1.492e-05 | 2.086e-05 | PASS | over |
| 0 | prefill-main | 2048x5120 | float16 | x_out | 0.000e+00 | 0/10485760 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | decode | 32x5120 | float16 | rstd | 2.626e-06 | 0/32 | 5.621e-06 | 7.868e-06 | PASS | over |
| 0 | decode | 32x5120 | float16 | x_out | 0.000e+00 | 0/163840 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | single-row | 1x5120 | float16 | rstd | 3.470e-06 | 0/1 | 3.470e-06 | 4.888e-06 | PASS | over |
| 0 | single-row | 1x5120 | float16 | x_out | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | odd-rows-77 | 77x128 | float16 | rstd | 4.011e-06 | 0/77 | 1.484e-05 | 2.074e-05 | PASS | over |
| 0 | odd-rows-77 | 77x128 | float16 | x_out | 0.000e+00 | 0/9856 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | k-not-pow2 | 255x5008 | float16 | rstd | 3.247e-06 | 0/255 | 1.501e-05 | 2.098e-05 | PASS | over |
| 0 | k-not-pow2 | 255x5008 | float16 | x_out | 0.000e+00 | 0/1277040 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | k-min | 7x16 | float16 | rstd | 2.652e-06 | 0/7 | 5.930e-06 | 9.894e-06 | PASS | over |
| 0 | k-min | 7x16 | float16 | x_out | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | m-1000 | 1000x5120 | float16 | rstd | 3.120e-06 | 0/1000 | 1.501e-05 | 2.098e-05 | PASS | over |
| 0 | m-1000 | 1000x5120 | float16 | x_out | 0.000e+00 | 0/5120000 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 0 | k-2048 | 128x2048 | float16 | rstd | 2.927e-06 | 0/128 | 9.825e-06 | 1.359e-05 | PASS | over |
| 0 | k-2048 | 128x2048 | float16 | x_out | 0.000e+00 | 0/262144 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 | prefill-main | 2048x5120 | bfloat16 | rstd | 2.789e-06 | 0/2048 | 1.196e-05 | 2.456e-05 | PASS | over |
| 1 | prefill-main | 2048x5120 | bfloat16 | y | 1.185e-04 | 4/10485760 | 1.370e-02 | 3.125e-02 | FAIL | over |
| 1 | decode | 32x5120 | bfloat16 | rstd | 2.438e-06 | 0/32 | 5.752e-06 | 1.132e-05 | PASS | over |
| 1 | decode | 32x5120 | bfloat16 | y | 4.678e-05 | 0/163840 | 7.299e-03 | 7.812e-03 | PASS | ok |
| 1 | single-row | 1x5120 | bfloat16 | rstd | 1.179e-07 | 0/1 | 1.179e-07 | 2.384e-07 | PASS | over |
| 1 | single-row | 1x5120 | bfloat16 | y | 0.000e+00 | 0/5120 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 | odd-rows-77 | 77x128 | bfloat16 | rstd | 3.858e-06 | 0/77 | 1.088e-05 | 2.480e-05 | PASS | over |
| 1 | odd-rows-77 | 77x128 | bfloat16 | y | 1.567e-04 | 0/9856 | 6.536e-03 | 7.812e-03 | PASS | ok |
| 1 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 2.575e-06 | 0/255 | 7.446e-06 | 1.466e-05 | PASS | over |
| 1 | k-not-pow2 | 255x5008 | bfloat16 | y | 6.629e-05 | 0/1277040 | 1.047e-02 | 1.562e-02 | PASS | over |
| 1 | k-min | 7x16 | bfloat16 | rstd | 2.296e-06 | 0/7 | 4.424e-06 | 1.049e-05 | PASS | over |
| 1 | k-min | 7x16 | bfloat16 | y | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 | m-1000 | 1000x5120 | bfloat16 | rstd | 2.715e-06 | 0/1000 | 1.057e-05 | 2.170e-05 | PASS | over |
| 1 | m-1000 | 1000x5120 | bfloat16 | y | 1.040e-04 | 1/5120000 | 1.361e-02 | 3.125e-02 | FAIL | over |
| 1 | k-2048 | 128x2048 | bfloat16 | rstd | 3.014e-06 | 0/128 | 8.515e-06 | 1.740e-05 | PASS | over |
| 1 | k-2048 | 128x2048 | bfloat16 | y | 1.252e-04 | 0/262144 | 1.235e-02 | 1.562e-02 | PASS | over |
| 1 | prefill-main | 2048x5120 | float16 | rstd | 2.809e-06 | 0/2048 | 1.074e-05 | 2.098e-05 | PASS | over |
| 1 | prefill-main | 2048x5120 | float16 | y | 3.938e-05 | 20/10485760 | 1.867e-03 | 3.906e-03 | FAIL | over |
| 1 | decode | 32x5120 | float16 | rstd | 2.531e-06 | 0/32 | 5.711e-06 | 1.144e-05 | PASS | over |
| 1 | decode | 32x5120 | float16 | y | 3.743e-05 | 0/163840 | 1.657e-03 | 3.906e-03 | PASS | over |
| 1 | single-row | 1x5120 | float16 | rstd | 1.815e-06 | 0/1 | 1.815e-06 | 3.576e-06 | PASS | over |
| 1 | single-row | 1x5120 | float16 | y | 3.389e-06 | 0/5120 | 4.890e-04 | 2.441e-04 | PASS | ok |
| 1 | odd-rows-77 | 77x128 | float16 | rstd | 3.567e-06 | 0/77 | 1.147e-05 | 2.480e-05 | PASS | over |
| 1 | odd-rows-77 | 77x128 | float16 | y | 3.053e-05 | 0/9856 | 9.506e-04 | 9.766e-04 | PASS | ok |
| 1 | k-not-pow2 | 255x5008 | float16 | rstd | 2.946e-06 | 0/255 | 1.046e-05 | 2.146e-05 | PASS | over |
| 1 | k-not-pow2 | 255x5008 | float16 | y | 4.039e-05 | 3/1277040 | 1.850e-03 | 3.906e-03 | FAIL | over |
| 1 | k-min | 7x16 | float16 | rstd | 3.384e-06 | 0/7 | 6.910e-06 | 1.407e-05 | PASS | over |
| 1 | k-min | 7x16 | float16 | y | 0.000e+00 | 0/112 | 0.000e+00 | 0.000e+00 | PASS | ok |
| 1 | m-1000 | 1000x5120 | float16 | rstd | 2.784e-06 | 0/1000 | 9.564e-06 | 1.955e-05 | PASS | over |
| 1 | m-1000 | 1000x5120 | float16 | y | 4.010e-05 | 8/5120000 | 1.895e-03 | 3.906e-03 | FAIL | over |
| 1 | k-2048 | 128x2048 | float16 | rstd | 2.951e-06 | 0/128 | 1.080e-05 | 2.110e-05 | PASS | over |
| 1 | k-2048 | 128x2048 | float16 | y | 3.965e-05 | 1/262144 | 1.847e-03 | 3.906e-03 | FAIL | over |
| 2 | prefill-main | 2048x5120 | bfloat16 | rstd | 4.467e-05 | 0/2048 | 1.486e-04 | 2.073e-04 | PASS | over |
| 2 | decode | 32x5120 | bfloat16 | rstd | 4.111e-06 | 0/32 | 1.219e-05 | 1.705e-05 | PASS | over |
| 2 | single-row | 1x5120 | bfloat16 | rstd | 2.183e-06 | 0/1 | 2.183e-06 | 3.099e-06 | PASS | over |
| 2 | odd-rows-77 | 77x128 | bfloat16 | rstd | 3.613e-06 | 0/77 | 1.336e-05 | 1.717e-05 | PASS | over |
| 2 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 4.547e-05 | 0/255 | 1.838e-04 | 2.598e-04 | PASS | over |
| 2 | k-min | 7x16 | bfloat16 | rstd | 1.923e-06 | 0/7 | 3.590e-06 | 5.722e-06 | PASS | over |
| 2 | m-1000 | 1000x5120 | bfloat16 | rstd | 4.332e-05 | 0/1000 | 1.301e-04 | 1.832e-04 | PASS | over |
| 2 | k-2048 | 128x2048 | bfloat16 | rstd | 7.413e-05 | 0/128 | 2.550e-04 | 3.612e-04 | PASS | over |
| 2 | prefill-main | 2048x5120 | float16 | rstd | 3.451e-06 | 0/2048 | 1.552e-05 | 2.146e-05 | PASS | over |
| 2 | decode | 32x5120 | float16 | rstd | 3.946e-06 | 0/32 | 1.138e-05 | 1.609e-05 | PASS | over |
| 2 | single-row | 1x5120 | float16 | rstd | 9.998e-07 | 0/1 | 9.998e-07 | 1.431e-06 | PASS | over |
| 2 | odd-rows-77 | 77x128 | float16 | rstd | 4.513e-06 | 0/77 | 1.291e-05 | 1.764e-05 | PASS | over |
| 2 | k-not-pow2 | 255x5008 | float16 | rstd | 3.586e-06 | 0/255 | 1.458e-05 | 2.038e-05 | PASS | over |
| 2 | k-min | 7x16 | float16 | rstd | 3.743e-06 | 0/7 | 9.768e-06 | 1.264e-05 | PASS | over |
| 2 | m-1000 | 1000x5120 | float16 | rstd | 3.214e-06 | 0/1000 | 1.450e-05 | 2.027e-05 | PASS | over |
| 2 | k-2048 | 128x2048 | float16 | rstd | 3.365e-06 | 0/128 | 1.373e-05 | 1.919e-05 | PASS | over |
