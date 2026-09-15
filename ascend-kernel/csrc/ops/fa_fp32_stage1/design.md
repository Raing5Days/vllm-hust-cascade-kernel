# fa_fp32_stage1 设计文档（stage-1 fp32-out flash attention 算子化，2026-09-03）

> 依据 ascendc-operator-design / operator-dev skill 流程；实现路径 = **CATLASS 模板库**（GEMM/FlashAttention 类，skill 路由表）。
> 源 = catlass example 23 `FAInferBf16Fp32Out`（M-A probe 已上板验证：O vs fp64 1.87e-6、LSE 9.5e-7、10/10 稳定、fp32-out 零计时开销）。
> 本算子 = 该 kernel 移植进 ascend-kernel 框架成为 torch op（plan-20260903 §2）。**移植不改数据流、不改指令序列**——判据含与 example 同输入 bit 相等。

## 1. 函数签名与 dtype 支持

```
torch.ops.npu.fa_fp32_stage1(Tensor query, Tensor key, Tensor value, Tensor block_table,
                             Tensor actual_q_seqlens, Tensor actual_kv_seqlens,
                             int q_seqlen_value=0) -> (Tensor out, Tensor lse)
```

| 参数 | 约束 |
|---|---|
| query | (T, H, D) bf16 连续；T = Σactual_q_seqlens（decode stage-1 时 q_len=1，T=B）；**q_seqlen_value>0 时 T == B×q_seqlen_value** |
| key/value | (numBlocks, blockSize, KVH, D) bf16 连续；**blockSize == 128**（fai_tiling 硬约束） |
| block_table | (B, cols) int32 连续；cols ≥ ceil(maxKv/128)；**cols 即 kernel 的 blockTable 行 stride**。q_seqlen_value>0 时该约束为 **caller 契约**（kv seqlen 留在 device、kernel 内读取，host 无 D2H 可查——vLLM cascade 集成侧保证） |
| actual_q_seqlens / actual_kv_seqlens | (B,) int64（device，kernel 内 GM 读取）。**q_seqlen_value=0（默认，legacy）**：op_host 两次 D2H 拉 seqlen 算 tiling + 全量 host 校验；**q_seqlen_value>0（fast path）**：所有请求 q seqlen == 该值（decode=1），op_host 零 D2H、tiling 纯 shape 计算（uniform q 下 GetQSBlockTile 恒 128） |
| out | (T, H, D) **fp32**（≈1.3MB/层 @ 64×40×128，GM 带宽可接受，plan §2.1） |
| lse | (T*H*8,) **fp32**，行布局 = 每 (t,h) 行 8×fp32 复制的 32B padded 行（消费取 elem0）——与 lse_merge stride=8 **零拷贝直传** |

**q_seqlen_value 语义（plan-20260903 M-C C0-1）**：默认 0 = 现行为（泛化/测试路径保留 D2H）；>0 = graph-capture 安全路径——tiling 内容对给定 (B,H,KVH,cols) 桶 shape 静态，捕获期 H2D 重放字节不变；sumQ 校验退化为 `T == B×q_seqlen_value`；`q_seqlen_value` 为负直接 TORCH_CHECK 拒。

语义：单请求摊平（B=1/TND 类）、无 mask（maskType=0 固定）、paged KV、GQA（H=KVH*group）、共享前缀 stage（cascade stage-1）。H=40/KVH=8/D=128 为主 shape；fp16 入口不移植（fp16 本仓不用），保留 `FAInferBf16` 入口作 bf16 回归锚点。

## 2. 移植映射表（example → 本工程）

| example 23 | 本工程 | 改动 |
|---|---|---|
| `fai_kernel.cpp`（FAInferKernel 模板类 + 3 entry） | `csrc/ops/fa_fp32_stage1/op_kernel/kernel_fa_fp32_stage1.cpp` | 保留 FAInferKernel 类逐行不变（含 fp32 分支的 epilogueRescaleO lse overload 调用）；entry 重整：`FAInferBf16`（回归锚点）+ `fa_fp32_stage1`（= probe `FAInferBf16Fp32Out`，ElementO=float + params.lse）；**删除 FAInferFp16**（本仓不用，减模板实例化） |
| `kernel_common.hpp`（FATilingData/FAIKernelParams/常量） | `op_kernel/kernel_common.hpp` + `op_kernel/fai_tiling_data.hpp` | FATilingData 单独拆出（host/device 共享同一结构体头，杜绝双份漂移）；kernel 侧其余逐行不变 |
| `fai_tiling.cpp`（host tiling 计算） | `op_host/fa_fp32_stage1.cpp` 内联 | GetQNBlockTile/GetQSBlockTile/FillBasic/FillSplitCore/FillWorkSpace 逐行迁入；**改动一处：maxNumBlocksPerBatch = block_table.size(1)（真实行 stride），example 版 = ceil(maxKv/128)（其数据恰巧相等）** |
| `fai.cpp`（host 分配/launch/对拍） | `op_host/fa_fp32_stage1.cpp`（torch op 形态） | 分配改 at::empty；launch 改 EXEC_KERNEL_CMD（ACLRT_LAUNCH_KERNEL）；对拍不进 op（走 test/） |
| `include/catlass/**`（上游 + probe 改 2 处） | `third_party/catlass/include/catlass/`（整树拷贝 2.9MB/207 hpp） | ① `block_epilogue_rescale_o_no_split_row.hpp` = probe 版（fp32-out cast 跳过 + CopyOToGm 按 ElementOutput 宽度 + LSE 写出 + staging 双缓冲）；② 其余上游原样。**整树拷贝理由**：聚合头（block_mmad.hpp/block_epilogue.hpp）交叉引用深，子集抽取易漏；整树解除 submodule 依赖（任务书 §2.2-1 决策：倾向拷贝）。社区出口（M-E）再评估 vendor vs 引 asc 仓 submodule |

## 3. Kernel 数据流（移植保持，逐行=example）

```
AIC (blockDim = aicCoreNum, 每 core 独立任务循环 taskIdx += coreNum):
  for task (batch → qSBlock × qNBlock):
    QK^T mmad → S workspace (fp32 GM, ping-pong preLaunch=2, 每 core 3 slot)
    CrossCoreSetFlag(qkReady)
    [preLaunch 深水后] PV mmad → oTemp (fp32 GM, 同 ping-pong)
    CrossCoreSetFlag(pvReady)
AIV (每 AIC 配对 2 AIV, coreIdx = blockIdx/2):
  for task 同序:
    WaitFlag(qkReady) → OnlineSoftmax epilogue: S→P(bf16 GM)+lm/hm/gm/gl 维护
    SetFlag(softmaxReady) → WaitFlag(pvReady) → RescaleO epilogue:
      isLast tile: oUpdate 加权合并 → gl 归一化 → [fp32-out: 跳过 cast] CopyOToGm(out fp32)
                                          → LSE = Ln(gl)+hm → 32B padded 行写出 GM
```

关键既有机制（M-A probe 踩坑固化，勿重踩）：
1. **EVENT_ID2 对 V_MTE3/MTE3_V 不可用**（910B3 稳定死等）→ LSE 写出用 staging 双缓冲（奇偶 isLast 交替）+ 复用 EVENT_ID0（V_MTE3）Set/Wait 对；
2. **4B UbToGm DataCopyPad 挂** → LSE 行 32B padded（8×fp32 复制）保每笔 DataCopy 32B 对齐；
3. **上游 dm 区域写超 512B**（3 slot×128rows=1536B > 预留 1KB）→ [10*UB+14KB, 11*UB) 不安全，LSE scratch 放 mask 区（**仅 maskType=0 安全，本算子 maskType 恒 0**）；
4. **挂起判定纪律**：A/B 版本各 ≥5 次对比，单次 timeout 不作数。

## 4. Tiling 策略

- **kernel 内嵌 tiling**（example 形态，非两级 tiling 框架）：task 级切分 = batch × qSBlock(128 kv 行) × qNBlock(qN tile)；qN tile = min((128/qlen)/2*2, groupSize)；workspace ping-pong = 每 core 3 slot。
- **FATilingData**（88B：10×u32 + 5×u64 + f32，device 侧 `__gm__` 直接 reinterpret）：

| 字段 | 来源（op_host） |
|---|---|
| numHeads/kvHeads/embeddingSize | query/key shape |
| numBlocks / blockSize | key.size(0)/key.size(1)（==128 校验） |
| maxKvSeqlen | actual_kv_seqlens.cpu().max() |
| **maxNumBlocksPerBatch** | **block_table.size(1)**（行 stride，修正点见 §2） |
| batch / firstBatchTaskNum / totalTaskNum | B；FillSplitCoreTilingData 逐 batch 累计（需 host seqlens） |
| maskType=0 / scaleValue=1/sqrt(D) | 固定/计算 |
| mm1OutSize 等 4×u64 | FillWorkSpaceTilingData(blockDim)（kernel 读而不用，保真填充） |

- op_host 校验：全输入连续；D==128（L1TileShape::K 约束）；blockSize==128；H%KVH==0；T==Σq_seqlens；ceil(maxKv/128)≤cols；q_seqlens≥1、kv_seqlens≥1。

## 5. Workspace 与 GM 布局（host 分配，全部 at::empty）

| Buffer | 大小（字节） | 说明 |
|---|---|---|
| s | aicNum×131072×4×3 | QK^T 输出 S（fp32），每 core 3 slot ping-pong |
| p | aicNum×131072×2×3 | softmax 后 P（bf16） |
| oTemp | aicNum×131072×4×3 | O 累加（fp32） |
| oUpdate | aicNum×131072×4×3 | O 更新（fp32） |
| tiling | sizeof(FATilingData)=88 | host 填 → uint8 device tensor |
| out / lse | T×H×D×4 / T×H×8×4 | op 输出 |

⚠ **M-C 图捕获集成注记**：workspace 4 buffer + tiling 必须图池化稳定地址；cascade 两段 workspace 先例 = task group 并发分开 buffer；tiling 内容按桶静态化（capture 时 D2H 一次）。本步（M-B）eager 形态不阻塞。

⚠ **C0 tiling 暂存区生命周期（2026-09-03，plan-20260903 M-C C0-2 落地，v4 最终形态）**：原实现 `from_blob(&tiling).clone().to(device)` 的 H2D 源是 clone 出的临时 CPU tensor，op 返回即析构——eager 下驱动同步拷贝侥幸安全；**图捕获会把该 H2D 记为节点、重放时读已释放主机内存**。最终形态 = **按内容去重的进程级暂存区**（每 device 每 FATilingData 字节模式一对 host/device 缓冲）：
  - **host 侧必须走 torch pinned 分配器**（`at::empty(...).pin_memory()`）：裸 `aclrtMallocHost` 指针不被 CachingHostAllocator 认可，non_blocking 拷贝走同步 fallback → capture 拒绝（probe 实测 107027 "Not allow to synchronize captured-stream"）。pinned tensor **故意泄漏**（`new at::Tensor` 永不析构）——析构若运行必触发 teardown abort（见 A/B 记录）；每内容 88B 可忽略。
  - **拷贝必须 non_blocking**：pageable 源 H2D = 同步 rtMemcpy，capture 直接拒绝（probe 实测 107030）。
  - **内容去重（memcmp 88B 精确判重）**：uniform-q fast path 下 tiling 是纯 shape 函数 → 每 (B,H,KVH,cols) 桶内容唯一 → 每桶独立 staging，**多桶捕获互不覆写**；命中时不重写 host 字节，消除在图重放 DMA 读与后续调用 memcpy 的竞态。device 侧裸 `aclrtMalloc(88)`。
  - **纪律：进程级 registry 严禁持有需析构的 at::Tensor**（静态 C++ tensor 析构晚于解释器 finalization → PyEval_SaveThread GIL abort；A/B 实证）。泄漏的 pinned tensor 与裸指针均无析构路径。
  - 线程约束：vllm worker 单线程调用前提；同 device 并发调用会竞争 registry 判重与拷贝。

## 6. UB 分配表（AIV 侧，固化 M-A 遗留项——任务书 §2.2-4）

以 UB_UINT8_BLOCK_SIZE=16KB 为块单位（上限 11 块 + 14KB 内为上游自用区）：

| 区间 | 内容 | 来源 |
|---|---|---|
| [0, 4*UB) | ls（S fp32 tile） | OnlineSoftmax |
| [4*UB, 10*UB) | lp（P bf16）∥ mask32（fp32） | OnlineSoftmax |
| [10*UB, +8KB) | tv | softmax+rescale 共享 |
| [10*UB+8K, +1K) | lm | OnlineSoftmax |
| [10*UB+9K, +1K) | hm | OnlineSoftmax（LSE 输入） |
| [10*UB+10K, +1K) | gm | OnlineSoftmax |
| [10*UB+11K, +1K) | ll | OnlineSoftmax |
| [10*UB+12K, +1K) | gl | OnlineSoftmax（LSE 输入） |
| [10*UB+13K, +1K) | dm | rescale 权重（**实际写 1536B 越界至 +14K+512B**，故 [10*UB+14KB, 11*UB) 禁用） |
| **[11*UB, +512B)** | **LSE val（Ln(gl)+hm 结果）** | **本算子新增（probe 版）**，仅 maskType=0 安全 |
| **[11*UB+512, +2×1KB)** | **LSE staging 双缓冲（32 行×32B slot ×2 bank）** | **本算子新增（probe 版）** |

## 7. 同步红线核查清单（移植保持形态，sync_audit 范围）

- AIC↔AIV：qkReady/softmaxReady/pvReady 三组 CrossCoreFlag（ID 1/2/3），`SetSyncBaseAddr(fftsAddr)` 由 **ascendc 框架自动接管**（实测：框架对 mix kernel 的 auto-gen wrapper 注入 `GM_ADDR ffts_addr` 首参并在核内 `set_ffts_base_addr`，host stub 侧 `GetAscendCoreSyncAddr` 取址）——因此 **kernel entry 不声明 ffts 参数、entry 内不再调 SetSyncBaseAddr、op_host 不调 rtGetC2cCtrlAddr**（example 的手工路径被框架机制取代；这是与 example 的既知 entry 签名差异）。
- launch 面：`EXEC_KERNEL_CMD(fa_fp32_stage1, aicNum, q, k, v, mask(nullptr, maskType=0 恒不引用), blockTable, out, qSeq, kvSeq, s, p, oTemp, oUpdate, tiling, lse)`。
- framework 侧同步（SetFlag 初始化/收尾 WaitFlag 对）逐行 = example；probe 已 10/10 稳定。
- **已知风险（编译期揭晓，已闭环）**：ascendc_library 对 mix kernel 的支持实测成立——框架自动分类 MIX_SOURCES（AIC+AIV 双编译 + merge 二进制 + stub 双注册 type 0/1）并自动注入 ffts。**构建结构（2026-09-03 定稿）**：lse_merge 与本算子**各占独立 ascendc_library 目标**（`no_workspace_kernel` / `no_workspace_kernel_fa`，见 csrc/CMakeLists.txt 注释）——共享目标曾改变 lse_merge 的 stub 流并伴随其 fp32 路径潜伏缺陷显性化（test-cases.md §8）；`CATLASS_ARCH=2201` 为源内 define（目标级 -D 会泄漏进同目标其他源编译）。
- **sync_audit 红线裁决记录（2026-09-03 用户裁决：不改码）**——核心：本 kernel 为**任务环 + 跨核标志**设计，AIC/AIV 循环同一任务序列，逐 stack tile 生产者 Set 一次、消费者 Wait 一次；静态计数数"位点数"、动态配对数"每执行次数"，位点数本就不要求相等：
  - **SYNC-04 qkReady（Set=2/Wait=1）动态配对表**：

    | 位点 | 侧 | 执行时机 |
    |---|---|---|
    | Set@258（主循环） | AIC | 每 stack tile 一次（QK mmad 后） |
    | Set@329（次循环） | AIC | **仅 maskType≠0**；本算子恒 maskType=0 → 次循环 `kvSIdx < kvSLoopNumTotal` 恒假，动态死代码 |
    | Wait@559（主循环） | AIV | 每 stack tile 一次（online softmax 前） |

    maskType=0 动态配对 = 1↔1 逐 tile 平衡；Set=2 因次循环位点为 maskType≠0 保留（其配对 Wait 在 masked softmax epilogue 重载内，亦跨文件）。
  - **SYNC-04 softmaxReady（Set=2/Wait=0）**：Wait 在 `block_mmad_fai_pv_normal.hpp:227`（PV mmad 首个 k 分片前，每 tile 一次，`pvCVItr` 门控）；flag 对象经两层形参传递（kernel `softmaxReady` → `operator()` → `computePV`），工具不做跨函数 flag 身份追踪 → 报 Wait=0；扩展扫描（纳入 mmad 文件）后同一盲区从另一侧报「Wait 无对应 Set」——两侧互证为**跨文件/跨参数追踪盲区**，非缺配对。
  - **判定依据三层**：① 机制层——动态配对表如上，CrossCoreFlag 按 ID（1/2/3）配对、无 buffer 索引维度；② 工具层——真缺配对的表现是死等挂死或确定性错乱，均未出现；③ 实测层——同一数据流 M-A probe 10/10 + 本日 30/30 + 10/10 bit 级稳定 + 与 example 二进制逐 bit 相等（约 500+ 次 launch 零挂死）。**改码有害**：任何 sync 结构改动破坏「移植不改数据流」与 bit 等价锚点。
  - **重审触发条件**：未来启用 maskType≠0 路径时，次循环 Set@329 与 masked epilogue 内的配对 Wait 须重走 full-audit。其余候选：SYNC-01 高（epilogue:164 跨循环双缓冲合法序，M-A 既有；mmad 模板内 5 项 ping-pong 初始化序，工具自标"可能为合法序"）、SYNC-09 性能（PipeBarrier 计数，informational）。

## 8. 算子标杆与测试要点（详见 test/fa_fp32_stage1-test-cases.md）

- NPU_CALL：`out, lse = torch.ops.npu.fa_fp32_stage1(q, k, v, bt, qseq, kvseq)`；lse 消费：`lse.view(T*H, 8)[:, 0]`。
- 参考实现：numpy fp64 语义重建（bf16 输入升精度 + P 保留 bf16 量化点 + online 512 分块，M-A probe 口径）。
- **移植等价性**：与 example 二进制（同 shape 同输入 `--outfp32 --dump`）O/LSE **逐 bit 相等**。
- 精度判据（2026-09-03 修订，tie-band 口径详见 test-cases.md §7）：vs fp64 `|O−O_ref| ≤ A + 3e-6`（A = bf16 近平局模糊带，fp32 mmad 噪声使中点附近舍入方向合法分叉）、LSE ≤1e-3；稳定性 ≥10/10 bit 级。
- 端到端 merge probe：stage-1 本算子（真实）+ FIA v2 suffix bf16（真实 stage-2）+ lse_merge(out_code=2) → 全链 vs fp64，判据 = 抑制因子 merged_err/stage2_err ≈ w2（=0.059 @ S_P=8192/S_S=512）——Tier1 数值主张直接证据（实测 0.069 PASS）。
- 性能：msprof Task Duration 与 example 同量级；同卡同法 host wall 对照。

## 9. C0 执行记录（2026-09-03，plan-20260903 M-C C0，NPU0，全验证绿）

- **改动面**：`op_host/fa_fp32_stage1.cpp`（q_seqlen_value fast path + 暂存区生命周期修复）+ `csrc/ops.h`/`csrc/register.cpp`（schema 加 `int q_seqlen_value=0`，旧调用点零改动）。**kernel（op_kernel/*）零改动**（重装后 `_C.…so` md5 `68f08206…` 不变）；构建 107-120s 无告警；whl 重装 lib md5 `a9f9c52f…`（C0 版）。
- **回归门（C0-3 全绿）**：
  - S1 bit 锚点（默认路径）：vs example 二进制 O/LSE **逐 bit 相等**；S3 fp64 1.866e-6 / 7.58e-7；S4 稳定 10/10；
  - 30/30 精度套件**两种口径**：默认路径 30/30 + 负例 3/3 拒；`--fast`（q_seqlen_value=q_len）30/30（tie-band 判据 real-fail=0；fast 路径 "cols too small" 负例按设计不适用=caller 契约，套件自动跳过）；
  - **fast-path 专项套件**（`test/test_q_seqlen_fastpath.py`，新增）：8 shape fast-vs-default **逐 bit 相等**（含 14B 主 shape/B=1 最小/varlen kv/padded cols/kv=128 tie-band 显性形状）+ fp64 对拍全过 + 负例 `q_seqlen_value=2, T=4≠8` 拒 + fast 10/10 bit 级稳定；
  - lse_merge pytest **56/56**（零改动确认）。
- **性能锚点**：同 shape 同法 host wall：默认 225.0µs → **fast 133.9µs（-40%）**——两次 D2H 同步消除兑现（预期收益，14B 集成形态每次 decode 步每层省 2 次同步）。
- **A/B 事故记录（暂存区生命周期）**：初版 registry 持 pinned+device at::Tensor → 退出 abort（PyEval_SaveThread GIL，exit=134）；**A/B 裁决**：byte-精确还原改前源码重建（lib md5 `39ee53be…` 与改前安装版一致）→ 干净 exit=0，确认非 torch_npu 全局伪影、责任在 static tensor teardown。v2 = 裸内存 + 非持有 from_blob 视图 → exit=0 + 全部回归绿。教训入 §5 注记：**进程级 registry 严禁持 at::Tensor**。
- **v3→v4（C2 机制 probe 驱动，同日）**：v2/v3 的 pageable-host + blocking copy 在 capture 内被拒（107030）；v3 改裸 aclrtMallocHost + non_blocking 仍被拒（107027——CachingHostAllocator 不认外来 pinned 指针，non_blocking 路径仍含同步 fallback）；**v4 = torch pin_memory() 分配 + 故意泄漏 + non_blocking + 内容去重（memcmp）** → C2 机制 probe 全 PASS（见 §10）。

## 10. C2 机制 probe（2026-09-03，NPU0，PASS——plan §2-C2.1 纪律门）

脚本：`/tmp/c2_probe_fa_fp32_stage1_graph.py`（stage-1 单请求 B=1 摊平形态，q_seqlen_value=T fast path）。

| 项 | 结果 |
|---|---|
| capture | **OK**——op 内 7×at::empty（out/lse/4 workspace/—）由 graph pool 正常承接，tiling H2D（v4 pinned+non_blocking）成功记为节点，无挂死 |
| replay（同参） | O/LSE **逐 bit 相等** eager fast-path 输出（静态 out/lse buffer 先清零再 replay，证明 replay 真实写出） |
| replay（变参 kv=2048/6144） | 仅改 device kv seqlen（block_table 桶静态——共享前缀块 ID 不变，kernel 按 device seqlen 读 ceil(kv/128) 块；C0 设计 maxKvSeqlen=cols×128 配套）→ O/LSE **逐 bit 相等** fresh eager |

结论：§1.3-1（tiling 纯 shape 函数）、§1.3-3/4（op 内分配图池承接）两条预判全部实测成立；**§1.3-2 生命周期缺陷的 v4 修复经图捕获实测验证**。C2 实现（build_for_cascade_graph_capture 接入 + 变参重发面扩展）机制风险已清除。

## 11. LSE 摊平形态缺陷：定位与修复（2026-09-04，plan-20260903-lse-flatten-fix，全验证绿）

> 现象（C3 验收暴露）：B=1 摊平（kernel 视角单请求 q_len=T）形态下 LSE 相对 fp64 出现时序依赖错乱（14B T=64 错误 ~1e-2-2e0 量级，逐次运行漂移），O 路径始终正常（6.2e-5 优于 FIA）。立项假设「head-major 行映射公式未覆盖摊平形态」被 P0 证伪。

### 11.1 真实机制（P0 定位，probe 驱动，预测全部命中）

**staging bank 同任务复用竞态**：`ComputeLseAndWrite`（vendored `block_epilogue_rescale_o_no_split_row.hpp`）以 32 行（LSE_CHUNK_ROWS）为 chunk 将 lseVal Brcb 进 staging bank；原实现每任务只按 `lseParity` 选一次 bank → **子块行数 >32（≥2 chunk）时 chunk1 的 Brcb 覆写同一 bank，而 chunk0 的 MTE3 DataCopy 异步在途**。`Set/WaitFlag(V_MTE3, EVENT_ID0)` 只定序 V pipe、不跟踪 MTE3 完成（EVENT_ID2 对 V_MTE3/MTE3_V 不可用是 910B3 既有实测，正是本 staging 设计的成因）→ chunk0 的 GM 行收到 chunk1 的值，程度随时间漂移。

**判据公式（与全部 26 个扫描点 + 证伪集吻合）**：摊平形态 LSE 正确 ⟺ 子块行数 = qSBlockSize × (qNBlockSize/2 整除) ≤ 32（单 chunk）。qnB=1 尾块（rows=qsB/2≤32）恒正确；14B（group=5）T≤16、3B（group=8）T≤8 正确；14B T≥17、3B T≥16 混淆。证伪集 T=17/18/20（34/36/40 行）预测混淆实测混淆。错值来源逐格验证：out(head_local,t) ← 同任务 chunk1 同偏移行（T=40 out(h=1,t=7)←ref(head1,t=39)；3B T=16 out(head0)←ref(head2)，chunk 边界预测精确命中）；T=40 重复 5 次错误集漂移 = 竞态而非索引。**O 路径不受影响的机理**：goUbTensor 在下一 tile 前不会被复用（oTemp 3-slot ping-pong + 任务管线距离），LSE staging 是唯一在同一 isLast tile 内快速复用的缓冲。

### 11.2 修复（P1，一处公式级改动）

staging bank 改为**按 chunk 连续交替**：成员计数 `lseStageSeq`，每 chunk `stageBaseElems = (lseStageSeq++ & 1) ? bank1 : bank0`——同 bank 复用距离 ≥2 chunk（≥1KB staging 完整排空 + 一次 QK/PV 管线间隔，裕度数量级充分）；跨任务同样交替（epilogue 对象每 AIV 核一个，存活于任务循环外）。**单 chunk 任务（含全部 q_len=1 形态）bank 序列与旧 `lseParity` 逐步一致 → S1 锚点域结构性 bit 不变**（实测 vs example 二进制逐 bit 相等维持）。O 路径、lse_merge、asc 集成、C2 回退形态零改动。构建后 lib md5 `be5f10a9`→`f654c2ce`，`_C` `68f08206` 不变。

### 11.3 回归与 S1 锚点有效域声明

- **新增免疫回归** `test/test_lse_flatten_regression.py`：8 摊平 shape（kv 128/4096/4356、T=16/17 chunk 边界、T=48/63、3B T=8）× default+fast 双模式 + 3 对照形态（q_len=1 B=64 / uniform q_len=2），19/19 PASS，lse ≤8.4e-7（阈值 1e-5），stride8 不变量维持。
- 全量门：30/30 default + 30/30 fast + 负例 3/3 + fastpath 8-shape bit 等 + S4 10/10 + lse_merge pytest 56/56 全绿。
- **S1 锚点有效域声明**：S1（op vs example 二进制逐 bit）在 q_len=1 per-request 域维持有效；**摊平 q_len>1 域不再以 example 为参照**（example 二进制自带本竞态缺陷，修复后两者在该域预期分叉），正确性权威 = fp64 对拍（§8 判据 + §7 tie-band 口径不变）。asc 仓 `csrc/third_party/catlass` submodule 内 example 树保持 probe 旧态（含本竞态），仅为移植参照、无构建产物消费。

### 11.4 历史口径订正（重要）

1. **M-B/S1 锚点/30-30/fast 套件对该缺陷「互盲」的真因** = 全部用例子块行 ≤32（单 chunk），非「同 scramble 互盲」。
2. **C1 3B「3 发散臂（0,1,4）」与 LSE 缺陷无关**：3B eager T=8 摊平子块行=32（单 chunk），LSE 修复前后均正确；修复后复跑 fp32-vs-Tier0 仍为同 3 臂 [0,1,4] → 定性为 stage-2/merge bf16 常规效应（原「fp32 生效证据」改判）。
3. **M-B merge probe 0.069-vs-w2 0.059 的 +17% 不是置换偏置**：该 probe stage-1 为 per-request 形态（子块行 2，单 chunk，从未被污染）；修复后复跑仍 0.069，且新增摊平形态 merge probe（`cascade-c3-results/probes/p2_merge_probe_flat.py`）与 per-request 臂 **merged abs 逐 bit 相等（8.629e-05）**、抑制因子同为 0.069。+17% 为理论 w2（均匀扩散假设）的 realized-w2 方差，非偏置。

## 12. blockStackNum 契约守卫（2026-09-16）

> **缘起**：B1 地板核实验扫 KV 栈深时发现 1/2/8 **挂死**、16 抛 **aicore exception**，只有生产值 4 可跑；
> 根因定位（零卡静态分析）见 `profiles/qwen14b-instruct-hotspot-20260910/probe-b1-floor/REPORT-HANG.md`。

### 12.1 被守卫的契约

本核的 KV 栈深同时充当三件事，且**必须**与 catlass FAI 模板的硬编码几何相等：

```
blockStackNum (kernel：两处，原各写 4)
  ≡ BlockMmadQK::UNIT_BLOCK_STACK_NUM   (模板，public static constexpr = 4，8 个模板文件一致)
  ≡ BlockMmadPV::UNIT_BLOCK_STACK_NUM   (模板，同上)
  ≡ KV_BASE_BLOCK / pagedBlockSize      (512 / 128 = 4)
  ≡ 模板内 S stride 字面量 512 / 128     (block_mmad_fai_*_normal.hpp，非参数)
```

kernel 侧另把它用作 S/P layout stride（`stackSeqTilePad = blockStackNum × pagedBlockSize`）。

### 12.2 为什么必须守卫（原状：静默失败）

- 模板的 `operator()` 一次**固定吃 `UNIT_BLOCK_STACK_NUM` 块**（`block_mmad_fai_qk_normal.hpp:190`），
  而 kernel 以为只推进 `blockStackNum` 块 ⇒ 二者不等时必然产生**两处不一致**（代码事实）：
  **KV 覆盖不一致**（N<4 重复计页、N>4 跳过页）与 **AIV 窗口/AIC 产出尺寸不一致**
  （AIV 的 S 窗口 = `N×128` 列、stride `N×128`，而模板恒按 `KV_BASE_BLOCK = 512` 列、stride 512 写 S）；
- **S 容量**（纯算术）：读窗 `128 × N×128` 元素 vs op_host 每 slot `131072` ⇒ N=8 恰好顶格、N=16 越过 slot 1.0×；
- ⚠ **失效触发点未建立**：实测 1/2/8 挂死、16 抛 aicore exception，但**"为什么恰好是这种表象"未定位**。
  报告初版曾归因于"跨核 flag/L1 配对失衡"，**该归因已撤回**（逐次核算显示 N<4 时跨核 set/wait 仍配平）。
  详见 `REPORT-HANG.md` §2.3（含仍未被验证的候选机制）。**本守卫拦截的是契约违反本身，不依赖该机制成立。**
- **原本无任何守卫**：op_host 的 `TORCH_CHECK` 不校验栈深、模板无 `static_assert`，
  且 `kernel_common.hpp` 里那份 `UNIT_BLOCK_STACK_NUM` **定义了但从未被引用**。

### 12.3 处置（编译期，零运行时开销）

| 改动 | 位置 |
|---|---|
| 栈深改为**派生**模板常量（消除两处独立硬编码） | `kernel_fa_fp32_stage1.cpp` AIC/AIV 两处 `blockStackNum = BlockMmad{QK,PV}::UNIT_BLOCK_STACK_NUM` |
| 页大小取模板的 `KV_SPLIT_SIZE`（**不新造字面量**）+ 断言它 == 128（与 host `TORCH_CHECK` 对齐） | 同文件 `FAInferKernel` 类头 |
| 断言 QK/PV 栈深一致；断言 `栈深 × KV_SPLIT_SIZE == KV_BASE_BLOCK`（模板内 S stride 字面量） | 同上 |
| **工作区容量断言**：`kRowNumMax(128) × 栈深 × KV_SPLIT_SIZE ≤ WORKSPACE_BLOCK_SIZE_DB` | 同上；`kRowNumMax=128` 由暴力枚举核实（qSeqlen 1..4096 × group 1..32） |
| 删除 `kernel_common.hpp` 的死常量，改为指向本节的注释 | `kernel_common.hpp` |
| 副本同步全部断言（只约束模板侧，**不妨碍**副本扫 STACKN） | `fia_grain_floor/op_kernel/kernel_fia_grain_floor.cpp` |

### 12.4 验证（四层，2026-09-16；原始日志见 `probe-b1-floor/raw/guard/`）

1. **零语义变更**：重建后 `libascend_kernel.so` 与改前 **md5 逐字节相同**（`7c9f22df5512a05d8fc175857bf935a6`，多轮重建恒同）。
2. **断言真的会拦**（两次独立反证）：令 `kPagedBlockSize = KV_SPLIT_SIZE*2` ⇒ **RC=2**；
   令 `kRowNumMax = 4096` ⇒ **RC=2**。报错文本即对应断言消息。验证后已还原。
3. **本算子回归套件全绿**（README §4 要求的顺序）：
   `test_fa_fp32_stage1_smoke.py` → **S1 与 example 二进制逐 bit 相等（O=True/LSE=True）**、
   S3 vs fp64 `1.866e-06 / 7.582e-07`（与 §9 历史值一致）、S4 10/10 ⇒ PASS；
   `run_precision_suite.py` 完整版 → **30/30 PASS** + 3/3 负例；
   `test_lse_flatten_regression.py` → ALL PASS；`test_q_seqlen_fastpath.py` → PASS。
4. **实机冒烟**：q=2048/kv=4096/B=1/H40/KVH8/D128 host wall **1239.9 µs**（改前 1239.0/1239.9），`out` 全有限。

### 12.5 仍存在的边界（未覆盖，如实登记）

- 模板内 **S stride 字面量 512** 与 `KV_BASE_BLOCK` 在模板里同为 512，但二者由**人工**保持一致；
  若将来只改 S stride 而不改 `KV_BASE_BLOCK`，本守卫**察觉不到**（模板只读，无法断言其内部字面量）。
- 四条断言中只有两条**本地可证伪**（页大小 == 128、工作区容量）；另两条（QK/PV 一致、
  `栈深×页大小 == KV_BASE_BLOCK`）只在模板侧变化时触发 ⇒ 需改只读依赖才能证伪，**未验**。
- 编译器每个实例化只报**第一条**失败的断言：同一处打坏多个常量时，只看到最早那条消息。
- `WORKSPACE_BLOCK_SIZE_DB` 在 **host（`kWorkspaceBlockSizeDb`）与 kernel 头（`WORKSPACE_BLOCK_SIZE_DB`）
  各写一份 131072**（同类"双份常量"），本守卫只约束 kernel 侧那份；两份的一致性**无机械保障**
  （与本节被守卫的契约是同一类脆弱点，见 `REPORT-HANG.md` §6）。
- 工作区容量断言用的是 `≤`：栈深若变为 8 会**恰好等于** slot（零余量）而仍通过断言——
  实测中"恰好顶格"的档位也失败了，但**失败是否因零余量所致并未确定**，故未凭此设阈值。
