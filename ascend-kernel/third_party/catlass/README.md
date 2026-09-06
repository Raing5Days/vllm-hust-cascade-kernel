# vendored catlass include tree

来源：vllm-ascend-hust 仓库内置 submodule `csrc/third_party/catlass`（gitlink `b50cad68`）的
`include/catlass` 整树拷贝（2026-09-03，fa_fp32_stage1 算子化，plan-20260903 §2.2-1 设计决策：
拷贝而非引 submodule include 路径——聚合头交叉引用深、整树是唯一可靠封闭方案，同时解除本独立
工程对 asc 仓的路径依赖）。

相对上游的本地改动（= M-A probe 改动 + 本算子化新增修正，随树带来）：
- `include/catlass/epilogue/block/block_epilogue_rescale_o_no_split_row.hpp`：
  fp32-out（ElementOutput=float 跳过 cast、CopyOToGm 按 ElementOutput 宽度）+
  fp32 LSE 写出（Ln(gl)+hm，32B padded 行，staging 双缓冲 + EVENT_ID0 复用）。
  详见 csrc/ops/fa_fp32_stage1/design.md §3/§6。
- 同文件（2026-09-03 本算子化修正）：LSE scatter 行映射 token-major → **head-major**
  （tile 行序实为 head 外层/qSBlockSize token 内层，q_len>1 时原映射 LSE 错行 0.36 →
  修复后 6.98e-7；q_len=1 位点等价，decode stage-1 主语义与 bit 锚点不受影响）。
  ComputeLseAndWrite/Invoke/SubCoreCompute 新增 qsBlockSize 传参。
  ⚠ asc 仓 submodule 的 M-A probe 版仍带此 token-major 限制（decode q_len=1 不触发），
  M-C/社区出口前需回移本修正。

许可证：CANN Open Software License Agreement Version 2.0（各头文件头部保留原声明）。
社区出口（plan-20260902 §M-E）时再评估：本 vendor 树随 PR 携带 vs 引用 asc 仓 submodule。
