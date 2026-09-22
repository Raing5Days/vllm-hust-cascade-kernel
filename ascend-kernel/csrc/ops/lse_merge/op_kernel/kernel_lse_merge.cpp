// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// LSE-merge kernel for two-stage cascade decode attention (FlashInfer
// formula), see vllm-ascend plan-20260831-cascade-attn.md. Replaces ~12
// eager torch elementwise kernels with a single pass:
//   per row r: m = max(l1, l2); w1 = exp(l1-m); w2 = exp(l2-m);
//              out[r, :] = (o1[r, :]*w1 + o2[r, :]*w2) / (w1 + w2)
// Inputs: o2 (totalRows, dim) bf16 contiguous; lse1/lse2 (totalRows*stride,)
// fp32 contiguous; o1 (totalRows, dim) bf16 (legacy Tier-0) or fp32 (Tier-1
// mixed precision, stage-1 custom fp32-out kernel output).
// Output: out (totalRows, dim) bf16 or fp32 (out dtype flag).
//
// Mixed-precision extension (plan-20260902 M-B, Tier 1 fp32-partial):
//  - o1 fp32 skips the input Cast (GM fp32 -> UB fp32 direct); o1 bf16 keeps
//    the legacy instruction sequence bit-identical (Tier-0 regression guard).
//  - lse row stride = numel/rows: 1 = compact (FIA v2 .out() layout), 8 = the
//    32B padded-row layout the catlass FAInferBf16Fp32Out kernel writes (every
//    (t,h) row is 8 replicated fp32 so UbToGm writes stay 32B DataCopy - 4B
//    writes hang on 910B3). Strided rows are compacted in UB with the scalar
//    regroup pattern documented in ascendc-api-best-practices api-datacopy.md
//    (<= TILE_ROWS values per tile; cost negligible). stride==1 keeps the
//    original all-vector path untouched.
//  - out fp32 skips the final Cast: the row loop's last Muls writes straight
//    into the VECOUT tensor; out bf16 casts once (CAST_RINT) as before.
// All math in fp32 (CANN 8.5.1 Muls has no bf16 support).

#include "kernel_operator.h"

// ---------------------------------------------------------------------------
// 测量用探针（默认值 = 生产路径，指令序列与原文逐字节相同）
//   0 生产：行循环每行从 w1/w2/s 标量读取（96 次/tile）
//   1 读值不依赖向量结果（保留 GetValue 指令与条数，隔离数据依赖代价）
//   2 行权重只读第 0 行；GetValue 指令从 3*rows 降到 3（保留行循环与 Muls）
//   3 在 2 的基础上去掉每行的 Add（向量指令数 4*rows -> 3*rows）
//   4 行循环整段停用（测"行循环之外的结构性因素"是否支配总时长）
//
// ⚠️ 刻意不写 #ifndef 守卫（2026-09-18 实测教训）：本工程的构建链会把环境
//    CXXFLAGS 吸收成 device 侧的 -D 并**长期缓存**，那样会压过本文件的默认值、
//    令探针静默失效。不设守卫 ⇒ 一律以本文件为准；若外部再传同名宏则会直接
//    触发 -Wmacro-redefined 报错而不是静默走偏。
// 结果只能用于计时：1/2/3/4 的输出数值是错的，禁止进精度门。
// 构建与校验见 prof/build_probe.sh；测量结论见
// profiles/lse-merge-pipe-20260918/FINDINGS.md。
// ---------------------------------------------------------------------------
#define LSE_PROBE_ROW_SRC 0

// tile 行数（独立旋钮，用于测"时长是否随 tile 数变化"；默认 32 = 生产值）
#define LSE_PROBE_TILE_ROWS 32

constexpr uint32_t TILE_ROWS = LSE_PROBE_TILE_ROWS;

class LseMerge {
public:
    __aicore__ inline LseMerge() {}

    __aicore__ inline void Init(GM_ADDR o1, GM_ADDR o2, GM_ADDR lse1, GM_ADDR lse2, GM_ADDR out,
                                uint32_t totalRows, uint32_t dim, uint32_t o1IsF32,
                                uint32_t outIsF32, uint32_t lseStride1, uint32_t lseStride2)
    {
        this->dim = dim;
        this->o1IsF32 = o1IsF32;
        this->outIsF32 = outIsF32;
        this->lseStride1 = lseStride1;
        this->lseStride2 = lseStride2;
        uint32_t cores = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // Rows per core padded to a multiple of 8: every core's rowStart stays
        // 8-aligned, so the fp32 LSE DataCopy offsets stay 32B-aligned (MTE
        // requirement) for any totalRows (M2 fix: 0.5B 56-row shape corrupted
        // data with unaligned starts). Cores starting beyond totalRows idle.
        uint32_t rowsPerCore = (totalRows + cores - 1) / cores;
        rowsPerCore = (rowsPerCore + 7) / 8 * 8;
        uint32_t rowStart = blockIdx * rowsPerCore;
        uint32_t rowStop = rowStart + rowsPerCore;
        if (rowStop > totalRows) {
            rowStop = totalRows;
        }
        this->myRows = (rowStop > rowStart) ? (rowStop - rowStart) : 0;

        if (this->myRows > 0) {
            uint64_t elems = (uint64_t)this->myRows * dim;
            if (o1IsF32) {
                this->o1GmF32.SetGlobalBuffer((__gm__ float *)o1 + (uint64_t)rowStart * dim, elems);
            } else {
                this->o1GmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)o1 + (uint64_t)rowStart * dim, elems);
            }
            this->o2Gm.SetGlobalBuffer((__gm__ bfloat16_t *)o2 + (uint64_t)rowStart * dim, elems);
            if (outIsF32) {
                this->outGmF32.SetGlobalBuffer((__gm__ float *)out + (uint64_t)rowStart * dim, elems);
            } else {
                this->outGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)out + (uint64_t)rowStart * dim, elems);
            }
            this->lse1Gm.SetGlobalBuffer((__gm__ float *)lse1 + (uint64_t)rowStart * lseStride1,
                                         (uint64_t)this->myRows * lseStride1);
            this->lse2Gm.SetGlobalBuffer((__gm__ float *)lse2 + (uint64_t)rowStart * lseStride2,
                                         (uint64_t)this->myRows * lseStride2);

            uint32_t tileElems = TILE_ROWS * dim;
            uint32_t lseTile = TILE_ROWS * (lseStride1 > lseStride2 ? lseStride1 : lseStride2);
            this->pipe.InitBuffer(this->inQueO1, 1, tileElems * (o1IsF32 ? sizeof(float)
                                                                          : sizeof(bfloat16_t)));
            this->pipe.InitBuffer(this->inQueO2, 1, tileElems * sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->inQueLse1, 1, lseTile * sizeof(float));
            this->pipe.InitBuffer(this->inQueLse2, 1, lseTile * sizeof(float));
            this->pipe.InitBuffer(this->outQue, 1, tileElems * (outIsF32 ? sizeof(float)
                                                                          : sizeof(bfloat16_t)));
            if (!o1IsF32) {
                this->pipe.InitBuffer(this->o1F32Buf, tileElems * sizeof(float));
            }
            this->pipe.InitBuffer(this->o2F32Buf, tileElems * sizeof(float));
            this->pipe.InitBuffer(this->w1Buf, TILE_ROWS * sizeof(float));
            this->pipe.InitBuffer(this->w2Buf, TILE_ROWS * sizeof(float));
            this->pipe.InitBuffer(this->sBuf, TILE_ROWS * sizeof(float));
            this->pipe.InitBuffer(this->lseC1Buf, TILE_ROWS * sizeof(float));
            this->pipe.InitBuffer(this->lseC2Buf, TILE_ROWS * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < this->myRows; r += TILE_ROWS) {
            uint32_t rows = (this->myRows - r) < TILE_ROWS ? (this->myRows - r) : TILE_ROWS;
            ProcessTile(r, rows);
        }
    }

private:
    // Compact a strided lse tile (rows*stride lanes, row r's value at lane
    // r*stride; padded rows carry 8 replicated copies) into rows contiguous
    // lanes. UB scalar regroup per api-datacopy.md's documented pattern.
    __aicore__ inline void CompactLse(const AscendC::LocalTensor<float> &src, uint32_t stride,
                                      uint32_t rows, const AscendC::LocalTensor<float> &dst)
    {
        for (uint32_t r = 0; r < rows; r++) {
            dst.SetValue(r, src.GetValue(r * stride));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessTile(uint32_t rowOffset, uint32_t rows)
    {
        uint64_t elems = (uint64_t)rows * this->dim;

        AscendC::LocalTensor<float> lse1Local = this->inQueLse1.AllocTensor<float>();
        AscendC::LocalTensor<float> lse2Local = this->inQueLse2.AllocTensor<float>();
        // LSE copies go through DataCopyPad (Ext form, per api-datacopy.md):
        // compact layout with per-core rows not divisible by 8 gives a non-32B
        // blockLen (e.g. 7 rows = 28B) where plain DataCopy corrupts the last
        // row on 910B3 - latent M2 bug, exposed by the M-B edge-shape suite.
        // Plain DataCopy stays safe for the O tiles: dim % 16 makes their byte
        // counts 32B multiples for any row count.
        uint32_t lseCount1 = rows * this->lseStride1;
        uint32_t lseCount2 = rows * this->lseStride2;
        AscendC::DataCopyExtParams cpy1{1u, static_cast<uint32_t>(lseCount1 * sizeof(float)), 0u, 0u, 0u};
        AscendC::DataCopyExtParams cpy2{1u, static_cast<uint32_t>(lseCount2 * sizeof(float)), 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<float> pad1{true, 0u,
                                                  static_cast<uint8_t>((8u - lseCount1 % 8u) % 8u), 0.0f};
        AscendC::DataCopyPadExtParams<float> pad2{true, 0u,
                                                  static_cast<uint8_t>((8u - lseCount2 % 8u) % 8u), 0.0f};
        AscendC::DataCopyPad(lse1Local, this->lse1Gm[(uint64_t)rowOffset * this->lseStride1],
                             cpy1, pad1);
        AscendC::DataCopyPad(lse2Local, this->lse2Gm[(uint64_t)rowOffset * this->lseStride2],
                             cpy2, pad2);
        if (this->o1IsF32) {
            this->o1F32 = this->inQueO1.AllocTensor<float>();
            AscendC::DataCopy(this->o1F32, this->o1GmF32[(uint64_t)rowOffset * this->dim], elems);
            // EnQue is REQUIRED here (plan-20260903 fix): without it the TQue
            // depth-1 counter goes negative on DeQue and the queue bookkeeping
            // corrupts across tiles - non-deterministic MTE address faults on
            // the fp32 path (Tier0 bf16 branch was already balanced).
            this->inQueO1.EnQue(this->o1F32);
        } else {
            AscendC::LocalTensor<bfloat16_t> o1Local = this->inQueO1.AllocTensor<bfloat16_t>();
            AscendC::DataCopy(o1Local, this->o1GmBf16[(uint64_t)rowOffset * this->dim], elems);
            this->inQueO1.EnQue(o1Local);
        }
        AscendC::LocalTensor<bfloat16_t> o2Local = this->inQueO2.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(o2Local, this->o2Gm[(uint64_t)rowOffset * this->dim], elems);
        this->inQueO2.EnQue(o2Local);
        this->inQueLse1.EnQue(lse1Local);
        this->inQueLse2.EnQue(lse2Local);

        if (this->o1IsF32) {
            this->o1F32 = this->inQueO1.DeQue<float>();  // fp32 in, no Cast
        } else {
            AscendC::LocalTensor<bfloat16_t> o1Local = this->inQueO1.DeQue<bfloat16_t>();
            this->o1F32 = this->o1F32Buf.Get<float>();
            AscendC::Cast(this->o1F32, o1Local, AscendC::RoundMode::CAST_NONE, elems);
            this->inQueO1.FreeTensor(o1Local);
        }
        o2Local = this->inQueO2.DeQue<bfloat16_t>();
        lse1Local = this->inQueLse1.DeQue<float>();
        lse2Local = this->inQueLse2.DeQue<float>();

        AscendC::LocalTensor<float> o2F32 = this->o2F32Buf.Get<float>();
        // Cast the whole tile to fp32 once (o1 already fp32 in mixed mode).
        AscendC::Cast(o2F32, o2Local, AscendC::RoundMode::CAST_NONE, elems);

        // Compact strided lse rows, then run the row weights fully vectorized
        // (elementwise; safe on compact lanes). stride==1 sides use the queued
        // tensor directly - the legacy all-vector path, unchanged.
        AscendC::LocalTensor<float> lse1C;
        AscendC::LocalTensor<float> lse2C;
        if (this->lseStride1 == 1) {
            lse1C = lse1Local;
        } else {
            lse1C = this->lseC1Buf.Get<float>();
            CompactLse(lse1Local, this->lseStride1, rows, lse1C);
            this->inQueLse1.FreeTensor(lse1Local);
        }
        if (this->lseStride2 == 1) {
            lse2C = lse2Local;
        } else {
            lse2C = this->lseC2Buf.Get<float>();
            CompactLse(lse2Local, this->lseStride2, rows, lse2C);
            this->inQueLse2.FreeTensor(lse2Local);
        }

        // Vector part on the rows dimension: w1 = exp(l1 - m), w2 = exp(l2 - m),
        // s = w1 + w2 with m = max(l1, l2). All in fp32.
        AscendC::LocalTensor<float> w1Local = this->w1Buf.Get<float>();
        AscendC::LocalTensor<float> w2Local = this->w2Buf.Get<float>();
        AscendC::LocalTensor<float> sLocal = this->sBuf.Get<float>();
        AscendC::Max(sLocal, lse1C, lse2C, rows);
        AscendC::Sub(lse1C, lse1C, sLocal, rows);
        AscendC::Sub(lse2C, lse2C, sLocal, rows);
        AscendC::Exp(w1Local, lse1C, rows);
        AscendC::Exp(w2Local, lse2C, rows);
        AscendC::Add(sLocal, w1Local, w2Local, rows);
        // V->S sync before the scalar GetValue loop below (SYNC-02 red line:
        // vector-written w1/w2/s must be complete before scalar reads; the M2
        // original relied on same-core ordering - now explicit).
        AscendC::PipeBarrier<PIPE_V>();

        // Row loop: out = (o1*w1 + o2*w2) / s, fp32. The final scale writes
        // into the VECOUT tensor for fp32 out (no Cast), into the fp32 working
        // buffer for bf16 out (Cast after the loop, legacy path).
        uint32_t dim = this->dim;
        AscendC::LocalTensor<float> mergeDst;
        if (this->outIsF32) {
            mergeDst = this->outQue.AllocTensor<float>();
        } else {
            mergeDst = o2F32;
        }
        for (uint32_t r = 0; r < rows; r++) {
#if LSE_PROBE_ROW_SRC == 4
            // 行循环整段停用：只保留循环骨架，测"彻底删掉行循环的向量与标量工作"后
            // 总时长是否变化——若也不变，则 kernel 时长由行循环之外的结构性因素支配。
            break;
#elif LSE_PROBE_ROW_SRC == 1
            float w1 = 0.5f;
            float w2 = 0.5f;
            float sInv = 1.0f / sLocal.GetValue(r);
#elif LSE_PROBE_ROW_SRC == 2 || LSE_PROBE_ROW_SRC == 3
            float w1 = w1Local.GetValue(0);
            float w2 = w2Local.GetValue(0);
            float sInv = 1.0f / sLocal.GetValue(0);
#else
            float w1 = w1Local.GetValue(r);
            float w2 = w2Local.GetValue(r);
            float sInv = 1.0f / sLocal.GetValue(r);
#endif
            uint64_t off = (uint64_t)r * dim;
            AscendC::Muls(this->o1F32[off], this->o1F32[off], w1, dim);
            AscendC::Muls(o2F32[off], o2F32[off], w2, dim);
#if LSE_PROBE_ROW_SRC != 3
            AscendC::Add(o2F32[off], this->o1F32[off], o2F32[off], dim);
#endif
            AscendC::Muls(mergeDst[off], o2F32[off], sInv, dim);
        }
        if (this->outIsF32) {
            this->outQue.EnQue(mergeDst);
        } else {
            AscendC::LocalTensor<bfloat16_t> outLocal = this->outQue.AllocTensor<bfloat16_t>();
            AscendC::Cast(outLocal, o2F32, AscendC::RoundMode::CAST_RINT, elems);
            this->outQue.EnQue(outLocal);
        }
        if (this->o1IsF32) {
            this->inQueO1.FreeTensor(this->o1F32);  // fp32 path frees after the row loop
        }
        this->inQueO2.FreeTensor(o2Local);
        if (this->lseStride1 == 1) {
            this->inQueLse1.FreeTensor(lse1Local);
        }
        if (this->lseStride2 == 1) {
            this->inQueLse2.FreeTensor(lse2Local);
        }

        if (this->outIsF32) {
            AscendC::LocalTensor<float> outLocal = this->outQue.DeQue<float>();
            AscendC::DataCopy(this->outGmF32[(uint64_t)rowOffset * this->dim], outLocal, elems);
            this->outQue.FreeTensor(outLocal);
        } else {
            AscendC::LocalTensor<bfloat16_t> outLocal = this->outQue.DeQue<bfloat16_t>();
            AscendC::DataCopy(this->outGmBf16[(uint64_t)rowOffset * this->dim], outLocal, elems);
            this->outQue.FreeTensor(outLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueO1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueO2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueLse1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueLse2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> w1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> w2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> o1F32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> o2F32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lseC1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lseC2Buf;
    AscendC::GlobalTensor<bfloat16_t> o1GmBf16;
    AscendC::GlobalTensor<float> o1GmF32;
    AscendC::GlobalTensor<bfloat16_t> o2Gm;
    AscendC::GlobalTensor<bfloat16_t> outGmBf16;
    AscendC::GlobalTensor<float> outGmF32;
    AscendC::GlobalTensor<float> lse1Gm;
    AscendC::GlobalTensor<float> lse2Gm;
    AscendC::LocalTensor<float> o1F32;  // working fp32 o1 tile (queued or TBuf)
    uint32_t dim;
    uint32_t myRows;
    uint32_t o1IsF32;
    uint32_t outIsF32;
    uint32_t lseStride1;
    uint32_t lseStride2;
};

extern "C" __global__ __aicore__ void lse_merge(GM_ADDR o1, GM_ADDR o2, GM_ADDR lse1, GM_ADDR lse2,
                                                GM_ADDR out, uint32_t totalRows, uint32_t dim,
                                                uint32_t o1IsF32, uint32_t outIsF32,
                                                uint32_t lseStride1, uint32_t lseStride2)
{
    LseMerge op;
    op.Init(o1, o2, lse1, lse2, out, totalRows, dim, o1IsF32, outIsF32, lseStride1, lseStride2);
    op.Process();
}
