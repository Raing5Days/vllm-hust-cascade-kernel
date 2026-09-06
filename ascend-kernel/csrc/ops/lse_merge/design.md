# lse_merge 设计文档（含混合精度扩展，2026-09-02）

> 依据 ascendc-operator-design / operator-dev skill 流程补齐（原 M2 落地时未归档设计文档）。
> 数学定义（FlashInfer 公式，逐行 elementwise）：
> `m = max(lse1, lse2); out = (o1*exp(lse1-m) + o2*exp(lse2-m)) / (exp(lse1-m)+exp(lse2-m))`
> 行 r 对应一个 (token, head)。全部中间计算 fp32（CANN 8.5.1 Muls 无 bf16 支持）。

## 1. 函数签名与 dtype 支持（扩展后）

```
torch.ops.npu.lse_merge(Tensor o1, Tensor o2, Tensor lse1, Tensor lse2, int out_code=0) -> Tensor
```

| 参数 | 约束 |
|---|---|
| o1 | (…, D) 连续，**bf16（Tier0）或 fp32（Tier1 混合：stage-1 fp32-out kernel 输出）** |
| o2 | (…, D) 连续，bf16（stage-2 FIA 输出） |
| lse1/lse2 | fp32 连续，行数 = o1.numel()/D；**行 stride = numel/行数**（1=compact（FIA v2 .out()）；8=32B padded（catlass FAInferBf16Fp32Out 的 LSE 布局，每 (t,h) 行 8×fp32 复制，消费取 elem0），零拷贝直传） |
| out_code | 0=跟随 o1 dtype（默认，向后兼容）；1=bf16（集成形态：合并结果直进 bf16 output，免额外 cast kernel）；2=fp32（probe 形态：量化最终 cast 前的 Tier1 残差） |
| 返回 | (…, D)，dtype 按 out_code |

精度分层语义：Tier0（o1 bf16）= 既有行为（bit 级不变）；Tier1（o1 fp32）= 消除 stage-1 bf16 舍入项 w1·ε1，残余 ≈ w2·ε2 + ε_order（plan-20260902 §1）。

## 2. Tiling 策略（Block 级 + UB 级）

- **Block 级**：8 AIV 均分行；`rowsPerCore = ceil(totalRows/cores)` 再向上对齐 8 → 每核 rowStart 8 对齐，保证 compact LSE（stride=1）DataCopy 起始 32B 对齐（M2 教训：0.5B 56 行场景 28B 非对齐数据损坏）。padded stride 任意行距 ×8 对齐行号 → 偏移 32B*k*stride 恒对齐。
- **LSE 拷贝（2026-09-02 修正）**：统一 DataCopyPad（Ext 版）。M2 潜伏 bug：核内行数非 8 倍数（rows=7）时 compact LSE DataCopy blockLen=28B 非 32B 倍数 → **末行数据损坏**（非确定性；起始对齐不能替代长度对齐）。生产 shape T×H 恒为 8 倍数从未暴露。O tile 因 dim%16 字节数恒为 32B 倍数，保持 DataCopy。
- **同步**：行循环标量 GetValue 前加 `PipeBarrier<PIPE_V>()`（V→S 同步，sync-audit SYNC-02 红线要求；M2 原代码依赖同核序隐式成立，已显式化）。
- **UB 级**：行 tile = 32 行（TILE_ROWS）。lse 读入 rows*stride 连续元素；向量段（Max/Sub/Exp/Add，全 elementwise）作用 rows*stride 全长——stride>1 时行间垃圾 lane 参与运算但不被读取（无跨 lane 归约，NaN/Inf 不传播到有效 lane）；行循环 `GetValue(r*stride)`。
- 尾块：rows<32 按实际行数；myRows=0 的核空转退出。

## 3. UB 分配表（dim=128，TILE_ROWS=32；扩展后按 dtype 定容）

| Buffer | 位置 | Tier0 (bf16 o1/out) | Tier1 (fp32 o1/out) | 说明 |
|---|---|---|---|---|
| inQueO1 | VECIN ×1 | 8 KB | 16 KB | fp32 直读跳过 Cast |
| inQueO2 | VECIN ×1 | 8 KB | 8 KB | bf16 |
| inQueLse1/2 | VECIN ×1 | 1 KB×2 | 1 KB×2 | 32 行×8 stride fp32 |
| outQue | VECOUT ×1 | 8 KB | 16 KB | fp32 out 跳过 Cast |
| o1F32Buf | VECCALC | 16 KB | —（o1 已 fp32，复用 inQueO1 张量） | |
| o2F32Buf | VECCALC | 16 KB | 16 KB | 行循环融合目标 |
| w1/w2/s | VECCALC | 1 KB×3 | 1 KB×3 | 按 TILE_ROWS*stride 定容 |
| **合计** | | **~75 KB** | **~59 KB** | ≪ AIV UB 上限，压力无关紧要 |

bufferCoefficient：bf16=2B、fp32=4B（Cast 进出各一次，Tier1 fp32 路径省两次 Cast）。

## 4. 计算伪代码（AscendC 序）

```
Init: 分核(8对齐) → SetGlobalBuffer(o1 按 dtype / o2 bf16 / lse / out 按 dtype)
      InitBuffer(按 dtype 定容)
循环 tile (r += 32):
  AllocTensor; DataCopy GM→UB (o1 按 dtype; o2 bf16; lse rows*stride 连续)
  EnQue/DeQue (MTE2→V 同步)
  [o1 bf16] Cast o1F32 ← o1Local          # fp32 o1 跳过
  Cast o2F32 ← o2Local
  Max s ← lse1,lse2 (rows*stride); Sub lse−s; Exp w1,w2; Add s ← w1+w2
  行循环 r: Muls o1F32[off]*w1 → Muls o2F32[off]*w2 → Add → Muls *s⁻¹   # fp32
  [out bf16] Cast outLocal ← o2F32 (CAST_RINT)  # fp32 out 跳过
  EnQue/DeQue (V→MTE3 同步); DataCopy UB→GM
```

## 5. 约束与边界（op_host TORCH_CHECK）

- 全输入连续；dim % 16 == 0（bf16 32B 对齐；覆盖 fp32 的 8 元素对齐）
- o2 必须 bf16；lse 必须 fp32；lse.numel() % totalRows == 0，stride ∈ [1,1024]
- out_code ∈ {0,1,2}
- totalRows == 0 → 直接返回空 out
- 输入域：lse 任意 fp32（Exp 前已减 max，无溢出路径）；o1/o2 任意有限值

## 6. 算子标杆

- NPU_CALL：`torch.ops.npu.lse_merge(o1, o2, lse1, lse2, out_code)`
- 参考：fp64 torch（`m=lse1.maximum(lse2); w1=(lse1-m).exp(); w2=(lse2-m).exp(); out=(o1.double()*w1[...,None]+o2.double()*w2[...,None])/(w1+w2)[...,None]`）
- 精度标准：MERE/MARE（生态开源标准），bf16-out 阈值 7.81e-3/7.81e-2，fp32-out 阈值 1.22e-4/1.22e-3

## 7. 性能基线（M2 实测，Tier0 路径）

图内 2.28×（69.7→30.5µs）、eager 10.89×（287.7→26.2µs）@ T=64/N=40/D=128。Tier0 热路径（bf16 o1 + stride 1 + bf16 out）本扩展零改动（同指令序列），重编译后须对 25.4µs/call host 基线与 M2 图内数据复测不回归。混合路径（fp32 o1）省一次 Cast、out fp32 省一次 Cast，GM 流量 O1/out ×2 字节——性能正式评测随 M-C 集成做 msprof 对比。
