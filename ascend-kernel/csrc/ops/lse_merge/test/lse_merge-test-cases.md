# lse_merge 测试用例文档（2026-09-02，混合精度扩展配套）

> 注：用例为自行设计（非 testcase-gen 产出）——lse_merge 为 M2 既有算子，本次按 operator-dev skill 回退口径补齐。

## 测试配置

SUPPORTED_DTYPES（模式矩阵，输入 dtype 组合）：

| 模式 | o1 | o2 | lse | out | 语义 | 判定阈值 (MERE/MARE) |
|---|---|---|---|---|---|---|
| A tier0 | bf16 | bf16 | fp32 stride1 | bf16 | 既有行为回归（须 bit 级等于旧 whl 输出） | 7.81e-3 / 7.81e-2 |
| B fp32-out | fp32 | bf16 | fp32 stride1 | fp32 | Tier1 probe 形态 | 1.22e-4 / 1.22e-3 |
| Bp fp32-out padded | fp32 | bf16 | fp32 stride8 | fp32 | Tier1 + catlass padded LSE 零拷贝 | 1.22e-4 / 1.22e-3 |
| C bf16-out 混合 | fp32 | bf16 | fp32 stride1 | bf16 | Tier1 集成形态（M-C 直连 output） | 7.81e-3 / 7.81e-2 |
| D tier0 padded | bf16 | bf16 | fp32 stride8 | bf16 | stride 路径 Tier0 回归 | 7.81e-3 / 7.81e-2 |

TEST_SHAPES（rows=T*H）：
- ("production", "14B decode T=64 H=40 D=128", (64, 40, 128))
- ("0.5B", "0.5B decode T=56 H=8 D=128", (56, 8, 128))
- ("tiny", "rows=7 非整除核边界", (7, 4, 16))
- ("odd-rows", "rows=104 非整除 tile(32) 尾块", (13, 8, 128))
- ("single-row", "rows=1", (1, 40, 128))
- ("small-tile", "rows=1024 tile 边界", (32, 32, 64))

BOUNDARY_VALUES（lse 差值域与输入极值，shape 固定 (64,40,128)，模式 B+C 各跑）：
- ("equal weights lse1==lse2", 0.0)
- ("o2 vanishes dlse=+40", +40.0)   # lse1-lse2=+40 → w2≈0 → out≈o1
- ("o1 vanishes dlse=-40", -40.0)   # out≈o2
- ("large lse magnitude 30/35", 5.0)  # lse1=30, lse2=35
- ("o2 all zeros", None)            # o2=0 边界
- ("o1 all zeros", None)            # o1=0 边界

## 算子标杆

- NPU_CALL：`torch.ops.npu.lse_merge(o1, o2, lse1, lse2, out_code)`
- CPU_REF：fp64 torch（见 design.md §6），对 padded 模式 lse 先按 stride 展开（[..., 0] 语义 = 每 stride 组取首元素，与 kernel GetValue(r*stride) 一致）
- 附加回归：Tier0 模式输出与改版前 whl 捕获的 golden（/tmp/lse_merge_tier0_golden/*.pt）逐元素 bit 相等（torch.equal）
