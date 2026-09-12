# add_rms_norm_stats 精度验证报告（F2 norm 阶段核，2026-09-12）

- 用例总数 64，通过 42，失败 22，通过率 65.6%
- 参考实现：CPU fp64 累加 + CANN golden 舍入口径（`add_rms_norm_stats_ref.py`）
- 判据：rel_l2 + MARE + MaxAbsErr，容差取 CANN 官方用例档位（bf16 atol=rtol=2^-7，fp16 atol=rtol=2^-10），rstd 按相对判据（fp32 归约序差异）
- 生产 oracle 交叉核对（`torch_npu.npu_add_rms_norm`，beta=None）：6/48 通过
- shape/口径见 `add_rms_norm_stats-test-cases.md`

## 用例明细

| mode | case | shape | dtype | beta | 输出 | rel_l2 | MARE | MaxAbsErr | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 0 residual_stats | prefill-main | 2048x5120 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | prefill-main | 2048x5120 | bfloat16 | N | rstd | 3.374e-06 | 1.552e-05 | 2.146e-05 | PASS |
| 0 residual_stats | decode | 32x5120 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | decode | 32x5120 | bfloat16 | N | rstd | 3.660e-06 | 1.205e-05 | 1.705e-05 | PASS |
| 0 residual_stats | single-row | 1x5120 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | single-row | 1x5120 | bfloat16 | N | rstd | 6.871e-06 | 6.871e-06 | 9.775e-06 | PASS |
| 0 residual_stats | odd-rows-77 | 77x128 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 3.080e-06 | 1.285e-05 | 1.776e-05 | PASS |
| 0 residual_stats | k-not-pow2 | 255x5008 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 3.906e-06 | 1.604e-05 | 2.217e-05 | PASS |
| 0 residual_stats | k-min | 7x16 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | k-min | 7x16 | bfloat16 | N | rstd | 2.999e-06 | 7.242e-06 | 9.894e-06 | PASS |
| 0 residual_stats | m-1000 | 1000x5120 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | m-1000 | 1000x5120 | bfloat16 | N | rstd | 3.346e-06 | 1.475e-05 | 2.038e-05 | PASS |
| 0 residual_stats | k-2048 | 128x2048 | bfloat16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | k-2048 | 128x2048 | bfloat16 | N | rstd | 3.364e-06 | 1.285e-05 | 1.776e-05 | PASS |
| 0 residual_stats | prefill-main | 2048x5120 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | prefill-main | 2048x5120 | float16 | N | rstd | 3.326e-06 | 1.484e-05 | 2.074e-05 | PASS |
| 0 residual_stats | decode | 32x5120 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | decode | 32x5120 | float16 | N | rstd | 3.856e-06 | 1.185e-05 | 1.669e-05 | PASS |
| 0 residual_stats | single-row | 1x5120 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | single-row | 1x5120 | float16 | N | rstd | 8.265e-06 | 8.265e-06 | 1.168e-05 | PASS |
| 0 residual_stats | odd-rows-77 | 77x128 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | odd-rows-77 | 77x128 | float16 | N | rstd | 2.617e-06 | 1.113e-05 | 1.431e-05 | PASS |
| 0 residual_stats | k-not-pow2 | 255x5008 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | k-not-pow2 | 255x5008 | float16 | N | rstd | 3.197e-06 | 1.458e-05 | 2.038e-05 | PASS |
| 0 residual_stats | k-min | 7x16 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | k-min | 7x16 | float16 | N | rstd | 2.429e-06 | 7.422e-06 | 9.298e-06 | PASS |
| 0 residual_stats | m-1000 | 1000x5120 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | m-1000 | 1000x5120 | float16 | N | rstd | 3.475e-06 | 1.604e-05 | 2.217e-05 | PASS |
| 0 residual_stats | k-2048 | 128x2048 | float16 | N | x_out | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 0 residual_stats | k-2048 | 128x2048 | float16 | N | rstd | 3.205e-06 | 1.287e-05 | 1.800e-05 | PASS |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | N | rstd | 2.734e-06 | 1.165e-05 | 2.277e-05 | PASS |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | N | y | 1.109e-04 | 1.370e-02 | 3.125e-02 | FAIL |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | Y | rstd | 2.733e-06 | 1.196e-05 | 2.456e-05 | PASS |
| 1 stats_apply | prefill-main | 2048x5120 | bfloat16 | Y | y | 1.008e-04 | 1.185e+00 | 3.125e-02 | FAIL |
| 1 stats_apply | decode | 32x5120 | bfloat16 | N | rstd | 3.713e-06 | 9.357e-06 | 1.836e-05 | PASS |
| 1 stats_apply | decode | 32x5120 | bfloat16 | N | y | 7.931e-05 | 1.015e-02 | 1.562e-02 | FAIL |
| 1 stats_apply | decode | 32x5120 | bfloat16 | Y | rstd | 2.575e-06 | 8.399e-06 | 1.717e-05 | PASS |
| 1 stats_apply | decode | 32x5120 | bfloat16 | Y | y | 7.909e-05 | 7.752e-03 | 1.562e-02 | FAIL |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | N | rstd | 1.188e-07 | 1.188e-07 | 2.384e-07 | PASS |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | N | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | Y | rstd | 2.306e-06 | 2.306e-06 | 4.768e-06 | PASS |
| 1 stats_apply | single-row | 1x5120 | bfloat16 | Y | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 3.695e-06 | 1.069e-05 | 2.193e-05 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | N | y | 7.930e-05 | 7.353e-03 | 7.812e-03 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | Y | rstd | 2.637e-06 | 8.431e-06 | 1.717e-05 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | bfloat16 | Y | y | 2.183e-05 | 5.587e-03 | 1.953e-03 | PASS |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 2.744e-06 | 8.204e-06 | 1.609e-05 | PASS |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | N | y | 8.738e-05 | 1.227e-02 | 1.562e-02 | FAIL |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | Y | rstd | 2.678e-06 | 7.144e-06 | 1.454e-05 | PASS |
| 1 stats_apply | k-not-pow2 | 255x5008 | bfloat16 | Y | y | 1.108e-04 | 5.907e-02 | 3.125e-02 | FAIL |
| 1 stats_apply | k-min | 7x16 | bfloat16 | N | rstd | 1.335e-06 | 4.251e-06 | 6.437e-06 | PASS |
| 1 stats_apply | k-min | 7x16 | bfloat16 | N | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | k-min | 7x16 | bfloat16 | Y | rstd | 4.423e-06 | 6.204e-06 | 1.574e-05 | PASS |
| 1 stats_apply | k-min | 7x16 | bfloat16 | Y | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | N | rstd | 2.831e-06 | 1.165e-05 | 2.277e-05 | PASS |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | N | y | 1.064e-04 | 1.370e-02 | 3.125e-02 | FAIL |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | Y | rstd | 2.846e-06 | 1.162e-05 | 2.384e-05 | PASS |
| 1 stats_apply | m-1000 | 1000x5120 | bfloat16 | Y | y | 1.275e-04 | 2.800e-01 | 3.125e-02 | FAIL |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | N | rstd | 3.078e-06 | 1.147e-05 | 2.241e-05 | PASS |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | N | y | 7.712e-05 | 1.105e-02 | 1.562e-02 | FAIL |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | Y | rstd | 2.782e-06 | 8.314e-06 | 1.693e-05 | PASS |
| 1 stats_apply | k-2048 | 128x2048 | bfloat16 | Y | y | 1.229e-04 | 1.047e-02 | 3.125e-02 | FAIL |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | N | rstd | 2.788e-06 | 9.600e-06 | 1.884e-05 | PASS |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | N | y | 4.075e-05 | 2.038e-03 | 3.906e-03 | FAIL |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | Y | rstd | 2.699e-06 | 9.721e-06 | 1.955e-05 | PASS |
| 1 stats_apply | prefill-main | 2048x5120 | float16 | Y | y | 3.882e-05 | 3.967e-01 | 3.906e-03 | FAIL |
| 1 stats_apply | decode | 32x5120 | float16 | N | rstd | 2.904e-06 | 6.634e-06 | 1.311e-05 | PASS |
| 1 stats_apply | decode | 32x5120 | float16 | N | y | 4.074e-05 | 1.753e-03 | 1.953e-03 | FAIL |
| 1 stats_apply | decode | 32x5120 | float16 | Y | rstd | 2.245e-06 | 4.705e-06 | 9.537e-06 | PASS |
| 1 stats_apply | decode | 32x5120 | float16 | Y | y | 3.661e-05 | 1.970e-02 | 1.953e-03 | FAIL |
| 1 stats_apply | single-row | 1x5120 | float16 | N | rstd | 1.805e-07 | 1.805e-07 | 3.576e-07 | PASS |
| 1 stats_apply | single-row | 1x5120 | float16 | N | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | single-row | 1x5120 | float16 | Y | rstd | 1.218e-07 | 1.218e-07 | 2.384e-07 | PASS |
| 1 stats_apply | single-row | 1x5120 | float16 | Y | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | N | rstd | 2.572e-06 | 1.032e-05 | 2.122e-05 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | N | y | 2.874e-05 | 1.224e-03 | 1.953e-03 | FAIL |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | Y | rstd | 3.438e-06 | 8.781e-06 | 1.764e-05 | PASS |
| 1 stats_apply | odd-rows-77 | 77x128 | float16 | Y | y | 6.017e-05 | 1.153e-02 | 3.906e-03 | FAIL |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | N | rstd | 2.846e-06 | 9.447e-06 | 1.931e-05 | PASS |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | N | y | 3.860e-05 | 1.727e-03 | 3.906e-03 | FAIL |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | Y | rstd | 2.507e-06 | 8.080e-06 | 1.645e-05 | PASS |
| 1 stats_apply | k-not-pow2 | 255x5008 | float16 | Y | y | 3.692e-05 | 7.014e-02 | 3.906e-03 | FAIL |
| 1 stats_apply | k-min | 7x16 | float16 | N | rstd | 4.655e-06 | 1.140e-05 | 2.384e-05 | PASS |
| 1 stats_apply | k-min | 7x16 | float16 | N | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | k-min | 7x16 | float16 | Y | rstd | 3.103e-06 | 6.019e-06 | 1.478e-05 | PASS |
| 1 stats_apply | k-min | 7x16 | float16 | Y | y | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | N | rstd | 2.824e-06 | 1.043e-05 | 2.098e-05 | PASS |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | N | y | 4.059e-05 | 1.864e-03 | 3.906e-03 | FAIL |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | Y | rstd | 2.707e-06 | 1.122e-05 | 2.193e-05 | PASS |
| 1 stats_apply | m-1000 | 1000x5120 | float16 | Y | y | 3.831e-05 | 2.872e-01 | 3.906e-03 | FAIL |
| 1 stats_apply | k-2048 | 128x2048 | float16 | N | rstd | 3.132e-06 | 9.273e-06 | 1.884e-05 | PASS |
| 1 stats_apply | k-2048 | 128x2048 | float16 | N | y | 4.041e-05 | 1.808e-03 | 1.953e-03 | FAIL |
| 1 stats_apply | k-2048 | 128x2048 | float16 | Y | rstd | 3.037e-06 | 9.759e-06 | 2.003e-05 | PASS |
| 1 stats_apply | k-2048 | 128x2048 | float16 | Y | y | 4.020e-05 | 1.187e-02 | 3.906e-03 | FAIL |
| 2 stats_only | prefill-main | 2048x5120 | bfloat16 | N | rstd | 3.435e-06 | 1.526e-05 | 2.110e-05 | PASS |
| 2 stats_only | decode | 32x5120 | bfloat16 | N | rstd | 2.992e-06 | 6.815e-06 | 9.537e-06 | PASS |
| 2 stats_only | single-row | 1x5120 | bfloat16 | N | rstd | 3.629e-06 | 3.629e-06 | 5.126e-06 | PASS |
| 2 stats_only | odd-rows-77 | 77x128 | bfloat16 | N | rstd | 3.186e-06 | 1.238e-05 | 1.693e-05 | PASS |
| 2 stats_only | k-not-pow2 | 255x5008 | bfloat16 | N | rstd | 3.335e-06 | 1.484e-05 | 2.074e-05 | PASS |
| 2 stats_only | k-min | 7x16 | bfloat16 | N | rstd | 2.875e-06 | 7.691e-06 | 1.037e-05 | PASS |
| 2 stats_only | m-1000 | 1000x5120 | bfloat16 | N | rstd | 3.180e-06 | 1.441e-05 | 2.015e-05 | PASS |
| 2 stats_only | k-2048 | 128x2048 | bfloat16 | N | rstd | 3.214e-06 | 1.466e-05 | 2.027e-05 | PASS |
| 2 stats_only | prefill-main | 2048x5120 | float16 | N | rstd | 3.507e-06 | 1.604e-05 | 2.217e-05 | PASS |
| 2 stats_only | decode | 32x5120 | float16 | N | rstd | 3.485e-06 | 1.037e-05 | 1.466e-05 | PASS |
| 2 stats_only | single-row | 1x5120 | float16 | N | rstd | 2.267e-06 | 2.267e-06 | 3.219e-06 | PASS |
| 2 stats_only | odd-rows-77 | 77x128 | float16 | N | rstd | 3.943e-06 | 1.260e-05 | 1.669e-05 | PASS |
| 2 stats_only | k-not-pow2 | 255x5008 | float16 | N | rstd | 3.260e-06 | 1.450e-05 | 2.027e-05 | PASS |
| 2 stats_only | k-min | 7x16 | float16 | N | rstd | 2.679e-06 | 6.480e-06 | 9.775e-06 | PASS |
| 2 stats_only | m-1000 | 1000x5120 | float16 | N | rstd | 3.579e-06 | 1.475e-05 | 2.062e-05 | PASS |
| 2 stats_only | k-2048 | 128x2048 | float16 | N | rstd | 2.883e-06 | 1.190e-05 | 1.681e-05 | PASS |

## 生产 oracle 交叉核对（vs `torch_npu.npu_add_rms_norm`）

| mode | case | shape | dtype | 输出 | rel_l2 | MaxAbsErr | 判定 |
|---|---|---|---|---|---|---|---|
| 0 | prefill-main | 2048x5120 | bfloat16 | rstd | 6.405e-01 | 9.085e-02 | FAIL |
| 0 | prefill-main | 2048x5120 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | decode | 32x5120 | bfloat16 | rstd | 9.523e-02 | 7.850e-02 | FAIL |
| 0 | decode | 32x5120 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | single-row | 1x5120 | bfloat16 | rstd | 1.175e-06 | 1.669e-06 | PASS |
| 0 | single-row | 1x5120 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | odd-rows-77 | 77x128 | bfloat16 | rstd | 7.803e-01 | 4.781e-01 | FAIL |
| 0 | odd-rows-77 | 77x128 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 2.114e-01 | 7.557e-02 | FAIL |
| 0 | k-not-pow2 | 255x5008 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | k-min | 7x16 | bfloat16 | rstd | 5.812e-01 | 6.402e-01 | FAIL |
| 0 | k-min | 7x16 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | m-1000 | 1000x5120 | bfloat16 | rstd | 4.501e-01 | 9.487e-02 | FAIL |
| 0 | m-1000 | 1000x5120 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | k-2048 | 128x2048 | bfloat16 | rstd | 2.688e-01 | 1.356e-01 | FAIL |
| 0 | k-2048 | 128x2048 | bfloat16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | prefill-main | 2048x5120 | float16 | rstd | 6.321e-01 | 9.042e-02 | FAIL |
| 0 | prefill-main | 2048x5120 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | decode | 32x5120 | float16 | rstd | 7.191e-02 | 4.445e-02 | FAIL |
| 0 | decode | 32x5120 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | single-row | 1x5120 | float16 | rstd | 3.470e-06 | 4.888e-06 | PASS |
| 0 | single-row | 1x5120 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | odd-rows-77 | 77x128 | float16 | rstd | 7.428e-01 | 3.963e-01 | FAIL |
| 0 | odd-rows-77 | 77x128 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | k-not-pow2 | 255x5008 | float16 | rstd | 2.232e-01 | 8.287e-02 | FAIL |
| 0 | k-not-pow2 | 255x5008 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | k-min | 7x16 | float16 | rstd | 7.312e-01 | 1.082e+00 | FAIL |
| 0 | k-min | 7x16 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | m-1000 | 1000x5120 | float16 | rstd | 4.505e-01 | 8.258e-02 | FAIL |
| 0 | m-1000 | 1000x5120 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 0 | k-2048 | 128x2048 | float16 | rstd | 2.591e-01 | 1.274e-01 | FAIL |
| 0 | k-2048 | 128x2048 | float16 | x_out | 0.000e+00 | 0.000e+00 | PASS |
| 1 | prefill-main | 2048x5120 | bfloat16 | rstd | 6.296e-01 | 1.250e-01 | FAIL |
| 1 | prefill-main | 2048x5120 | bfloat16 | y | 1.185e-04 | 3.125e-02 | FAIL |
| 1 | decode | 32x5120 | bfloat16 | rstd | 7.754e-02 | 7.410e-02 | FAIL |
| 1 | decode | 32x5120 | bfloat16 | y | 4.678e-05 | 7.812e-03 | PASS |
| 1 | single-row | 1x5120 | bfloat16 | rstd | 1.179e-07 | 2.384e-07 | PASS |
| 1 | single-row | 1x5120 | bfloat16 | y | 0.000e+00 | 0.000e+00 | PASS |
| 1 | odd-rows-77 | 77x128 | bfloat16 | rstd | 7.227e-01 | 5.021e-01 | FAIL |
| 1 | odd-rows-77 | 77x128 | bfloat16 | y | 1.567e-04 | 7.812e-03 | PASS |
| 1 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 2.407e-01 | 1.150e-01 | FAIL |
| 1 | k-not-pow2 | 255x5008 | bfloat16 | y | 6.629e-05 | 1.562e-02 | FAIL |
| 1 | k-min | 7x16 | bfloat16 | rstd | 5.462e-01 | 9.872e-01 | FAIL |
| 1 | k-min | 7x16 | bfloat16 | y | 0.000e+00 | 0.000e+00 | PASS |
| 1 | m-1000 | 1000x5120 | bfloat16 | rstd | 4.436e-01 | 1.288e-01 | FAIL |
| 1 | m-1000 | 1000x5120 | bfloat16 | y | 1.040e-04 | 3.125e-02 | FAIL |
| 1 | k-2048 | 128x2048 | bfloat16 | rstd | 2.534e-01 | 2.109e-01 | FAIL |
| 1 | k-2048 | 128x2048 | bfloat16 | y | 1.252e-04 | 1.562e-02 | FAIL |
| 1 | prefill-main | 2048x5120 | float16 | rstd | 6.335e-01 | 1.346e-01 | FAIL |
| 1 | prefill-main | 2048x5120 | float16 | y | 3.938e-05 | 3.906e-03 | FAIL |
| 1 | decode | 32x5120 | float16 | rstd | 8.518e-02 | 8.199e-02 | FAIL |
| 1 | decode | 32x5120 | float16 | y | 3.743e-05 | 3.906e-03 | FAIL |
| 1 | single-row | 1x5120 | float16 | rstd | 1.815e-06 | 3.576e-06 | PASS |
| 1 | single-row | 1x5120 | float16 | y | 3.389e-06 | 2.441e-04 | PASS |
| 1 | odd-rows-77 | 77x128 | float16 | rstd | 8.080e-01 | 5.844e-01 | FAIL |
| 1 | odd-rows-77 | 77x128 | float16 | y | 3.053e-05 | 9.766e-04 | PASS |
| 1 | k-not-pow2 | 255x5008 | float16 | rstd | 2.373e-01 | 1.233e-01 | FAIL |
| 1 | k-not-pow2 | 255x5008 | float16 | y | 4.039e-05 | 3.906e-03 | FAIL |
| 1 | k-min | 7x16 | float16 | rstd | 4.773e-01 | 8.980e-01 | FAIL |
| 1 | k-min | 7x16 | float16 | y | 0.000e+00 | 0.000e+00 | PASS |
| 1 | m-1000 | 1000x5120 | float16 | rstd | 4.453e-01 | 1.368e-01 | FAIL |
| 1 | m-1000 | 1000x5120 | float16 | y | 4.010e-05 | 3.906e-03 | FAIL |
| 1 | k-2048 | 128x2048 | float16 | rstd | 2.633e-01 | 1.955e-01 | FAIL |
| 1 | k-2048 | 128x2048 | float16 | y | 3.965e-05 | 3.906e-03 | FAIL |
| 2 | prefill-main | 2048x5120 | bfloat16 | rstd | 6.321e-01 | 9.766e-02 | FAIL |
| 2 | decode | 32x5120 | bfloat16 | rstd | 8.127e-02 | 6.324e-02 | FAIL |
| 2 | single-row | 1x5120 | bfloat16 | rstd | 2.183e-06 | 3.099e-06 | PASS |
| 2 | odd-rows-77 | 77x128 | bfloat16 | rstd | 7.553e-01 | 4.477e-01 | FAIL |
| 2 | k-not-pow2 | 255x5008 | bfloat16 | rstd | 2.191e-01 | 7.872e-02 | FAIL |
| 2 | k-min | 7x16 | bfloat16 | rstd | 5.794e-01 | 7.416e-01 | FAIL |
| 2 | m-1000 | 1000x5120 | bfloat16 | rstd | 4.267e-01 | 7.918e-02 | FAIL |
| 2 | k-2048 | 128x2048 | bfloat16 | rstd | 2.577e-01 | 1.091e-01 | FAIL |
| 2 | prefill-main | 2048x5120 | float16 | rstd | 6.437e-01 | 1.051e-01 | FAIL |
| 2 | decode | 32x5120 | float16 | rstd | 7.279e-02 | 7.222e-02 | FAIL |
| 2 | single-row | 1x5120 | float16 | rstd | 9.998e-07 | 1.431e-06 | PASS |
| 2 | odd-rows-77 | 77x128 | float16 | rstd | 7.587e-01 | 4.493e-01 | FAIL |
| 2 | k-not-pow2 | 255x5008 | float16 | rstd | 2.080e-01 | 7.324e-02 | FAIL |
| 2 | k-min | 7x16 | float16 | rstd | 5.702e-01 | 7.038e-01 | FAIL |
| 2 | m-1000 | 1000x5120 | float16 | rstd | 4.488e-01 | 8.015e-02 | FAIL |
| 2 | k-2048 | 128x2048 | float16 | rstd | 2.433e-01 | 1.173e-01 | FAIL |
