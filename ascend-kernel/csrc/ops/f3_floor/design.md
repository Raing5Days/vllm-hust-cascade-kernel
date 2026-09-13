# f3_floor —— F3（SwiGLU 融入 gate_up GEMM epilogue）Phase B 地板阶梯核

> **性质**：**measurement-only**。服务于 `profiles/qwen14b-instruct-hotspot-20260910/probe-f3/` 的 F3 立项判定，
> 不接任何 e2e 路径、不进插件 bundle、不参与 manifest。判决与结论见
> `probe-f3/PHASE-B.md`（**原设计 NO-GO；就地写形态硬门通过但复合账净亏 ⇒ 不立项**）；本文件只写算子内部设计。
> 上游设计输入：`probe-f3/DESIGN-phaseB.md`（Catlass 选型 / UB 预算 / 同步清单）。

## 1. 它回答的问题

F3 的目标是消掉独立的 `SwiGlu`（prefill in-pass **111.80µs/次**、占 pass 2.05%）。A2 无 L0C→UB 通路，
epilogue 必然 **C 过 GM**（AIC 写 bf16 workspace → AIV 读回），因此融合**不省任何 GM 字节**；
收益只能来自"**epilogue 的 169.9MB 额外流量被 GEMM 的 cube 时间吸收**"。

本算子就是把这句假设做成**可测的单变量阶梯**：一个二进制、**8 个 mode（0..5、7、8；6 留空）**，唯一的差是"AIV 搬什么 / 算不算"。

| mode | AIC | AIV | 物理含义 |
|---|---|---|---|
| 0 `floor` | 配对 gate_up GEMM → bf16 C workspace | 读 gate 窗口 + up 窗口，**D = gate 窗口原样** | **融合地板**：与真 epilogue **逐字节同流量**（113.2R + 56.6W），**零 SwiGLU 语义** |
| 1 `swiglu` | 同 0 | `D = SiLU(gate) * up` | **Step 2 挂点**。本程**未实现**（`if constexpr` 分支已预留，当前与 mode 0 同行为）——按止损规则停工 |
| 2 `gemm_only` | 同 0 | 只走 flag 握手，不搬数据 | 隔离"本核 GEMM + 同步"的时长 |
| 3 `epilogue_only` | 只走 flag 握手 | floor 拷贝 | 地板单独跑的速度（= Phase B §9.1 的 Stage B0 探针） |
| 4 `floor_up` | 同 0 | 读 gate + up，**D = up 窗口** | 证明 `+I` 列偏移的窗口寻址（值级） |
| 5 `gate_only` | 同 0 | 只读 gate 窗口，D = 它 | 字节扫描（56.6R + 56.6W） |
| 7 `read_only` | 同 0 | 读 gate + up，**不写** | **读写分解**：纯读 113.2MB |
| 8 `write_only` | 同 0 | **不读**，只写 D | **读写分解**：纯写 56.6MB |

（mode 6 留空；编号沿用驱动脚本 `b1c_attrib.py` 的约定。全阶梯共用同一 block/flag 数，故差值可与字节数、block 数分别对照。）

**判据**（预注册于驱动脚本 `probe-f3/scripts/b1_floor.py`，早于测量；测量后不放宽）：
`T[0] − T[2] < 5% × T[2]` 才继续 Step 2。该阈值不是任取的——`5% × T_gemm(2183.4µs) = 109.17µs`
与**被替换算子** `SwiGlu` 的 111.80µs 之比为 **0.976**，即它编码的正是经济学判据「地板增量必须小于被替换算子」。
- 原设计（D 写进**新张量**）：实测 `T[0] − T[2] = +112.0 / +121.1 / +135.8 / +148.0 µs`（4 个独立 session，`T[2]=2146.1~2184.7µs`）⇒ **判负（NO-GO）**。
- 变体（D **就地写**进 workspace 的 gate 半边，`dStride=2I`）：`T[0] − T[2] = +32.6µs = +1.49%` ⇒ **硬门通过**，但天花板只有 +0.9~1.45% of pass，且需先追平本核 GEMM 与生产的 ~11% 差距 ⇒ 综合判**不立项**（详见 `probe-f3/PHASE-B.md` §3/§5/§6）。

## 2. 结构

```
A [M, K] bf16 RowMajor            （x，生产 gate_up 的激活输入）
B [2I, K] bf16 RowMajor           （nn.Linear 权重原样；逻辑上是 K×2I 的 ColumnMajor）
        ↓ Catlass BlockMmad<MmadAtlasA2Pingpong<false>, <256,128,256>, <256,128,64>>
C [M, 2I] bf16 workspace          （fixpipe 直接落 bf16：quantPre = F322BF16）
        ↓ CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(id=0)   ← 两次 blockMmad 之后一次 flag
        ↓ CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(id=0, rv=1)
D [M, I] bf16                     （AIV：跨步窗口读 → 写）
```

- **GEMM 与非融合基线同构**：同 shape `[2048,5120]×[27648,5120]ᵀ`、同 dtype、**同权重布局（ND，不改权重、不重排）**、
  同样 fixpipe 落 bf16。实测 C 与生产 `aclnnMatMulV3` 的输出**逐 bit 相等**（见 §5）。
- **配对 N（设计 P2）**：block 的 N 只枚举 `[0, I/BN)`；一个 block 内对同一 A tile 做**两次** `blockMmad`，
  B 的列偏移分别是 `n0` 与 `n0 + I`，C 落到 workspace 的 `[m0:m0+bm, n0:n0+BN]` 与 `[…, I+n0:I+n0+BN]`。
  ⇒ gate/up 两半由**同一个 AIC block** 产出，AIV 一次 wait 即可配对（这是 F3 的结构性前提，本程已验证成立）。
- **epilogue 的"切列"**用**拷贝窗口的步长**表达，不是张量算子：源布局 `RowMajor(subM, BN, stride = 2I)`
  直接喂给 `Epilogue::Tile::CopyGm2Ub`（`DataCopyPad` 带 srcStride）。**融合路径上零 `Slice` / 零 `InplaceCopy`**。
- 骨架照 catlass `gemm/kernel/matmul_activation.hpp` 的 AIC→workspace→flag→AIV 形态自写 kernel 类；
  **catlass 只读**（`third_party/catlass` 不动），依赖的 7 个符号：`BlockMmad` / `MmadAtlasA2Pingpong` /
  `Arch::Resource` / `Arch::CrossCoreFlagWithReverse` / `CrossCoreSetFlagWithReverse` /
  `Epilogue::Tile::CopyGm2Ub` / `Epilogue::Tile::CopyUb2Gm`（catlass 版本钉在 `third_party/catlass` 现状）。

## 3. tiling 与 UB 预算（910B2：UB 192KB / L1 512KB / L0A=L0B=64KB / L0C=128KB）

约束（`block_mmad_pingpong.hpp` 的 static_assert，STAGES=2）：

| 约束 | 取值 | 判定 |
|---|---|---|
| `L0C`: `bm*bn*4B ≤ 128KB` | 256×128×4 = 128KB | ✅ 占满（**L0C 单缓冲**，故无 ping-pong：本核 GEMM 比生产慢 ~11% 的主因之一） |
| `L0A`: `bm*K0*2B*2 ≤ 64KB` | 256×64×2×2 = 64KB | ✅ |
| `L0B`: `K0*bn*2B*2 ≤ 64KB` | 64×128×2×2 = 32KB | ✅ |
| `L1`: `(bm*K + bn*K)*2B*2 ≤ 512KB` | (256+128)×256×2×2 = 384KB | ✅ |
| 32B 对齐（bf16 ⇒ 16 元素） | M=256/N=128/K=256/K0=64 | ✅ |

- **为什么 `bm=256 / bn=128`**：L0C 是硬约束（`bm*bn ≤ 32768`）；在此约束下 L2 流量
  `≈ sizeof(B)·M/bm + sizeof(A)·(I/bn)` 在 `bm/bn ≈ sqrt(sizeof(B)·M / (sizeof(A)·I))` 处最小 ⇒ 本问题解 ≈ (256,128)。
- **block 顺序 n-major**（`nIdx = loopIdx / numMBlocks`）：并发核共享同一批 B tile（B 只从 HBM 取一遍），
  A（21MB）常驻 L2 被反复读。
- **UB（每 AIV 子核，单段 96KB）**：gate 窗口 `[128,128]bf16`=32KB + up 窗口 32KB + 出 32KB。
  2 段需 192KB（贴顶，本程未用）；mode 1 的全 fp32 链在 `bm=256` 下需列分块（见 §6）。

## 4. 同步清单（照 `ascendc-sync-audit` 的写法）

| # | producer → consumer | 机制 | 判定 |
|---|---|---|---|
| (a) | MTE2（gate/up 窗口 GM→UB）→ V | `HardEvent::MTE2_V`（EVENT_ID0，两次拷贝后一次 set） | ✅ |
| (b) | V → MTE3（UB→D） | `HardEvent::V_MTE3` | ✅ |
| (c) | MTE3 → MTE2（UB 复用 WAR） | `HardEvent::MTE3_MTE2`，循环首尾各一次（首迭代前置 token） | ✅ |
| (d) | **AIC fixpipe 落 C（两次）→ AIV MTE2 读窗口** | `CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(id=0)` / `CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(id=0)` | ✅ 实测 864 block × 5 shape/形态 × 3 session 零挂死 |
| (e) | 深度流控（同 flagId 连续 set > 15 死机） | `CrossCoreFlagWithReverse<>(id=0, rv=1)`，`MAX_REVERSE_DEPTH=15` 不动；864 block/case ⇒ 反向握手必需 | ✅ 已定值 |
| (f) | flag-id 预算 | 只用 0/1（8/9/10 是跨核 barrier 保留值） | ✅ |
| (g) | 与 `AscendC::Matmul` 高阶 API 混用 | 未使用（走 catlass 显式 Mmad+fixpipe） | ✅ |
| (h) | SYNC-08 提前 return 跳过 SetFlag | AIV 的 `subRowOff >= mActual` 分支**仍在任何 flag 操作之后**（wait/set 都已在分支外完成） | ✅ |
| (i) | 核间对称性 | 每 iteration：AIC 1×set / AIV 1×wait；每 15 次各 1×反向 ⇒ 逐次对称 | ✅ |
| (j) | `SyncAll` / `AtomicAdd` / 双缓冲下标下溢 | 均不使用 | ✅ N/A（显式声明） |
| (k) | 信号与 buffer 索引一致性（SYNC-14） | 每个 block 的 flag 对与其唯一 (m0,n0) 绑定；AIV 的读基址由同一 (m0,n0) 推出（gate `n0`、up `I+n0`、D `n0`） | ⚠ 模板化过深，`ascendc_flow_analyzer.py` 追不进 ⇒ 以**逐 bit 自证**替代（§5 mode 0 的 `D == C[:, :I]`） |

## 5. 自证条款（先正确性后计时）

| 检查 | 结果 |
|---|---|
| C（mode 2）vs `F.linear`（生产 `aclnnMatMulV3`）@M=2048 / M=8192 | `max_abs_diff = 0.0`、**bitwise 1.000000** ⇒ ① GEMM 同构成立；② **catlass 的 `F322BF16` fixpipe 与生产舍入口径逐 bit 一致** |
| 同上 @M=32（生产走 `MatMulV2`） | bitwise 0.99967，`max_abs = 0.5~1.0`（1 ulp，0.03% 元素）⇒ 差异来源是生产侧换了核，不是 fixpipe |
| mode 0 / 4 / 5 的 D vs C 的 gate / up / gate 半边 | **bitwise 1.000000** ⇒ 跨步窗口寻址正确（gate 与 up 两个窗口都能取对） |
| mode 0 的 D vs C 的 up 半边 | 99.91% 元素不同 ⇒ D 不是误取 up 窗口 |
| 就地写（`out_d = ws[:, :I]`，返回张量 `stride(0)=2I`）| D 与 gate 半边 **bitwise 1.000000** ⇒ `dStride` 可配、就地写正确 |
| 失败模式 | 零 `0x31`、零 MTE 越界、零 device exception（5 个 shape/形态 × 3 session × 4 个驱动脚本，18 个导出目录全扫） |

## 6. 已知边界与未做项

- **mode 1 未实现**：`if constexpr (MODE == MODE_SWIGLU)` 分支当前与 mode 0 同行为（挂点已留）。
  要接续需：`Cast(gate→fp32, CAST_NONE)` → `Muls(-1) / Exp / Adds(1) / Div` → `Cast(up→fp32)` → `Mul` → `Cast(CAST_RINT)`
  （CANN `swi_glu_bf16.hpp` 的链，注意 `beta=-1.0f` 与收尾 `CAST_RINT`）；UB 在 `bm=256`（每子核 128 行）下
  全精度链需 288KB > 192KB ⇒ 必须**列分块**（32 列一块 × 双缓冲 ≈ 144KB）。
- **up 窗口的正确性已由 mode 4 间接证**（mode 4 的 D = up 窗口，与 C 的 up 半边逐 bit 相等）；真 SwiGLU（mode 1）仍未实现。
- **失败前提（必读）**：本核**不是**一个可用的融合算子——它的存在意义是给出"融合地板"的实测值；
  判决（`probe-f3/PHASE-B.md`）按**原设计判 NO-GO**：新 D 形态的地板增量 +112.0~+148.0µs > 被替换的 `SwiGlu` 111.80µs；
  就地写形态虽把增量压到 +32.6µs（硬门通过），但复合账仍净亏（本核 GEMM 比生产慢 ~11% = 4.5% of pass ≫ 融合头寸 1.45%），
  故**不立项**。复用的价值在三处：① 8-mode 阶梯的"边际成本"测量法（同二进制自比）；② §5 的 fixpipe 舍入事实；
  ③ **归因机制**：代价 = "往新内存写输出"的 L2 写分配污染（按 ~436GB/s 记账，就地写后 0.27×），不是带宽/端口争用。
- 形态硬约束（`op_host` TORCH_CHECK 硬拒，不静默）：bf16/连续、`K%16==0`、`M%16==0`、`I%128==0`、`K≥64`；
  `I` 非 128 倍数（尾块）**不在本核支持面内**。

## 7. 构建注意

- `csrc/CMakeLists.txt` 的 `no_workspace_kernel_f3` target：`ascendc_include_directories` 加
  `${PROJECT_SOURCE_DIR}/third_party/catlass/include`；`CATLASS_ARCH=2201` **源内 define**（勿 target-wide：
  会污染同 target 其它源，仓内已有先例记录）。
- **必须全量重建**：`rm -rf build` 之后再 cmake。增量重建会在 `merge_aic/aiv_obj_text` 处报
  `ld.lld: error: … .o: unknown file type`（既有 target 同样报错，与本次改动无关——该流程会把输入 .o 原地覆盖）。
- 装载：`PYTHONPATH=<repo>/python/ascend_kernel`（`import ascend_kernel` 只做 `torch.ops.load_library(lib/libascend_kernel.so)`），
  **无需 pip**。
