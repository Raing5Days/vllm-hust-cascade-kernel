# bw_probe — D3 访问模式天花板锚点（benchmark-only，非生产算子）

> 立项：`profiles/qwen14b-instruct-hotspot-20260910/f2-kernel/REPORT.md` §8（F2 判负后的
> "norm 段是否真的带宽受限"复核）。本算子**只用于 benchmark**，不参与任何 e2e 路径，
> 也不进插件仓 bundle。

## 0. 为什么需要它（gate 修正的第一性理由）

F2 的 ② 判定写的是"norm 段读带宽 ~565GB/s ⇒ 融合天花板 ~0.2%"，该结论的推导链是：

```
bytes / (t_wall − c)   ，  c = t_wall(oracle) − 79.70us（生产 profile 的 oracle 器件时间）
```

这条链有两个隐含假设，实测都不成立（证据见 §3）：

1. **假设 c 与算子无关**。实测 mode0 的每调用 wall 比 oracle 多 ~9us 的**宿主开销**
   （配置器 + 分配 21MB 输出张量），于是 `t_wall(mode) − c` **不是** mode 的器件时间。
   用真机 profiler 直接读器件时间：mode0 = 54.76us，而该公式给出 80.92us（高估 48%）。
2. **假设"读字节/时间"就是带宽瓶颈**。真机 pipe 数据（§3）显示 MTE2 利用率仅 **0.22**，
   VEC 0.63、S 0.37 —— 器件时间是**计算+标量同步**约束，不是读带宽约束。

所以"565GB/s"既不是正确的时间口径，也不是正确的瓶颈口径。要判定"这个 pattern 最多能跑多快"，
必须**直接把计算全部拿掉、只留同字节的搬运**，实测其速度。这就是本算子。

## 1. 变体阶梯（单变量）

访问模式与 `add_rms_norm_stats` **逐字节相同**：同一 per-core 行切分（`ceil(M/cores)` 向上取 8
并对齐 32B）、同一条 10KB 行 `DataCopy`、同一组 VECIN/VECOUT 双缓冲队列、**同一套 UB buffer
（含 mid/零计算变体不需要的 buffer）**。唯一变化是"流水内容"：

| variant | 内容 | 读/写 | 用途 |
|---|---|---|---|
| 0 `copy` | 读 x1、读 x2、写 out=in1，**零 vector 指令** | 2R + 1W | **模式天花板 anchor** |
| 1 `apply` | + 两次 Cast(fp32)、Add、round Cast、写 out | 2R + 1W | mode0 输出路径的净成本 |
| 2 `mode0` | + Cast 回 fp32、Mul(平方)、ReduceSum、rstd 向量数学、标量 GetValue/SetValue staging、写 rstd | 2R + 1W | **= add_rms_norm_stats mode0（自证有效）** |
| 3 `no_reduce` | 同 v2，**去掉行归约**（用平方后 lane 0 顶替行和） | 2R + 1W | 剥离 `ReduceSum` 成本 |
| 4 `no_rstd` | 同 v2，**去掉 rsqrt + Newton 链**（保留归约与标量回读） | 2R + 1W | 剥离 count-1 rstd 数学成本 |

UD 占用对所有变体**相同**（寄存器/UB 压力是常量），故变体之间的差值是干净的单变量差。

**自证条款**：若 variant 2 的器件时间与 `add_rms_norm_stats` mode 0 在同条件实测不一致
（>5%），则本探针**没有**在复现生产模式，其余所有 rung 的数据都不得用于归因。

⚠ **已知未闭环项（如实登记）**：v2 的 `out` 与生产 mode0 **逐位相等**，但 `rstd` 不是——
探针相对 fp64 参考 **1.4e-05**（~60 ulp），生产算子 **1.7e-07**（1 ulp）。两者各自确定性
（3 次重复逐位一致）。探针只用于**时间**归因，其 `rstd` 数值不作为任何证据；差异原因
**未定位**（v2 与生产 mode0 的指令序列逐条相同，仅 in1 的 FreeTensor 位置与 variant 分支不同）。

## 2. 使用

```python
out, rstd = torch.ops.npu.bw_probe(x1, x2, 0, 1e-6)   # variant, eps
```
限制与 mode0 相同（`K <= 5120`、`K % 16 == 0`、bf16/fp16、2D contiguous）。返回的 `rstd`
只有 variant 2 有定义；variant 0/1 的 `rstd` 未写入。

## 3. Phase 0 实测结果

见 `profiles/qwen14b-instruct-hotspot-20260910/f2-kernel/`（D3 报告 §Phase 0）。摘要写在
`add_rms_norm_stats/design.md` §9。

## 4. 纪律

- 本算子**不是**候选融合算子，不进 manifest、不进 wheel 的生产面语义承诺；保留在算子库仅因
  它是 D3 结论的可复现工具（reproducible evidence > 一次性脚本）。
- 若 D3 方向关闭，本算子仍保留（它是"关闭"结论的证据本身），但不再维护。
