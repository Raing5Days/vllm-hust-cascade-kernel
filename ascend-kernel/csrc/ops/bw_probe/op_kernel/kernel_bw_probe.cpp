// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// bw_probe - D3 Phase-0 access-pattern anchor for add_rms_norm_stats (design.md
// of the probe op; ops/add_rms_norm_stats/design.md 9 is the consumer).
//
// Purpose: separate "how fast can this memory pattern possibly go" from "how
// fast does our kernel go". The pattern is copied from add_rms_norm_stats mode 0
// BYTE FOR BYTE - same per-core row split, same 10KB 32B-aligned row DataCopy,
// same double-buffered VECIN/VECOUT queues, same UB budget - and only the compute
// is removed, one stage at a time:
//
//   v0 copy     : read x1, read x2, write out = in1. Zero vector work.
//                 => the pure memory ceiling of this pattern (the anchor).
//   v1 apply    : + 2 casts to fp32, Add, round-cast to dtype, write out.
//                 => cost of the mode-0 output path alone.
//   v2 mode0    : + cast-back fp32, Mul (squares), ReduceSum, rstd vector math,
//                 scalar GetValue/SetValue staging, rstd write.
//                 => must reproduce add_rms_norm_stats mode 0 wall/device time.
//                 This rung is the probe's own validity check: if v2 does not land
//                 on mode 0's time, the probe is not measuring the production
//                 pattern and no other rung may be used.
//
// Every variant allocates the identical buffer set, so UB pressure is a constant
// of the experiment and only the pipeline content varies (single variable).
//
// Outputs of v0/v1 are deliberately not semantically meaningful for rstd (only
// `out` is); the probe is a benchmark, never a production op.

#include "kernel_operator.h"

namespace {
constexpr uint32_t RSTD_GROUP = 8;  // 8 fp32 rows = 32B: minimum legal MTE3 write unit
constexpr uint32_t MAX_K = 5120;    // per-row UB residency budget (op_host enforced)

enum Variant {
    V_COPY = 0,
    V_APPLY = 1,
    V_MODE0 = 2,
    // Ablations of V_MODE0: same ladder, one component removed, to split the
    // statistics path (33us of mode0) into the row reduction and the per-row
    // scalar/rstd tail.
    V_NO_REDUCE = 3,   // drop ReduceSum (lane 0 of the squares stands in for the row sum)
    V_NO_RSTD_MATH = 4,  // keep ReduceSum, drop the count-1 rsqrt/Newton chain
    // Round 2 candidate design: 8 rows x 640 columns per tile instead of 1 row x
    // 5120 (identical UB bytes), so the row sums come out of ONE AR-pattern
    // ReduceSum as 8 contiguous values and rstd is written as one legal 32B MTE3
    // with no scalar readback at all. V_TILE8_COPY is its own pattern floor: the
    // 8-row tile moves the same bytes as a strided copy (8 x 1280B segments),
    // which is not the same memory pattern as the contiguous 10KB row copy of
    // V_COPY, so the tiling gain and the strided-copy cost have to be separated.
    V_TILE8_COPY = 5,
    V_TILE8_FULL = 6,
};
constexpr uint32_t TILE_ROWS = 8;  // rows per tile for the V_TILE8_* design
}  // namespace

template <typename T>
class BwProbe {
public:
    __aicore__ inline BwProbe() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR xOut, GM_ADDR rstd, uint32_t mRows,
                                uint32_t kDim, uint32_t variantIn, float epsIn, float kInvIn)
    {
        this->kDim = kDim;
        this->variant = variantIn;
        this->eps = epsIn;
        this->kInv = kInvIn;

        uint32_t cores = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowsPerCore = (mRows + cores - 1u) / cores;
        rowsPerCore = (rowsPerCore + RSTD_GROUP - 1u) / RSTD_GROUP * RSTD_GROUP;
        uint32_t rowStart = blockIdx * rowsPerCore;
        uint32_t rowStop = rowStart + rowsPerCore;
        if (rowStop > mRows) {
            rowStop = mRows;
        }
        this->rowStart = rowStart;
        this->myRows = (rowStop > rowStart) ? (rowStop - rowStart) : 0u;
        if (this->myRows == 0u) {
            return;
        }

        this->x1Gm.SetGlobalBuffer((__gm__ T *)x1 + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        this->x2Gm.SetGlobalBuffer((__gm__ T *)x2 + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        this->xOutGm.SetGlobalBuffer((__gm__ T *)xOut + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        this->rstdGm.SetGlobalBuffer((__gm__ float *)rstd + rowStart, this->myRows);

        const uint32_t rowBytes = kDim * (uint32_t)sizeof(T);
        const uint32_t f32Bytes = kDim * (uint32_t)sizeof(float);
        // Identical buffer set for every variant: UB occupancy must not be a
        // difference between rungs of the ladder.
        this->pipe.InitBuffer(this->inQueX1, 2, rowBytes);
        this->pipe.InitBuffer(this->inQueX2, 2, rowBytes);
        this->pipe.InitBuffer(this->outQue, 2, rowBytes);
        this->pipe.InitBuffer(this->bufA, f32Bytes);
        this->pipe.InitBuffer(this->bufB, f32Bytes);
        this->pipe.InitBuffer(this->bufWork, f32Bytes);
        this->pipe.InitBuffer(this->bufSum, 8u * (uint32_t)sizeof(float));
        this->pipe.InitBuffer(this->bufTmp, 16u * (uint32_t)sizeof(float));
        this->pipe.InitBuffer(this->bufRstd, RSTD_GROUP * (uint32_t)sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->myRows == 0u) {
            return;
        }
        AscendC::LocalTensor<float> rstdStage = this->bufRstd.template Get<float>();
        uint32_t staged = 0u;
        for (uint32_t r = 0u; r < this->myRows; ++r) {
            ProcessRow(r, rstdStage, staged);
            if (++staged == RSTD_GROUP) {
                if (this->variant >= V_MODE0) {
                    FlushRstd(rstdStage, r + 1u - RSTD_GROUP);
                }
                staged = 0u;
            }
        }
        if (staged != 0u && this->variant >= V_MODE0) {
            for (uint32_t i = staged; i < RSTD_GROUP; ++i) {
                rstdStage.SetValue(i, 0.0f);
            }
            FlushRstd(rstdStage, this->myRows - staged);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t r, const AscendC::LocalTensor<float> &rstdStage,
                                      uint32_t staged)
    {
        const int32_t kI = static_cast<int32_t>(this->kDim);
        AscendC::LocalTensor<float> bufA = this->bufA.template Get<float>();
        AscendC::LocalTensor<float> bufB = this->bufB.template Get<float>();

        AscendC::LocalTensor<T> in1 = this->inQueX1.template AllocTensor<T>();
        AscendC::DataCopy(in1, this->x1Gm[(uint64_t)r * this->kDim], this->kDim);
        this->inQueX1.EnQue(in1);
        in1 = this->inQueX1.template DeQue<T>();

        AscendC::LocalTensor<T> in2 = this->inQueX2.template AllocTensor<T>();
        AscendC::DataCopy(in2, this->x2Gm[(uint64_t)r * this->kDim], this->kDim);
        this->inQueX2.EnQue(in2);
        in2 = this->inQueX2.template DeQue<T>();

        AscendC::LocalTensor<T> outT = this->outQue.template AllocTensor<T>();

        if (this->variant == V_COPY) {
            // Zero compute: the write still carries a full 10KB row, and both
            // reads are real (the x2 DataCopy above cannot be elided: it is an
            // intrinsic with a memory side effect).
            AscendC::DataCopy(outT, in1, this->kDim);
            this->inQueX1.FreeTensor(in1);
            this->inQueX2.FreeTensor(in2);
            this->outQue.EnQue(outT);
            AscendC::LocalTensor<T> outD0 = this->outQue.template DeQue<T>();
            AscendC::DataCopy(this->xOutGm[(uint64_t)r * this->kDim], outD0, this->kDim);
            this->outQue.FreeTensor(outD0);
            return;
        }

        AscendC::Cast(bufA, in1, AscendC::RoundMode::CAST_NONE, kI);
        AscendC::Cast(bufB, in2, AscendC::RoundMode::CAST_NONE, kI);
        this->inQueX1.FreeTensor(in1);
        this->inQueX2.FreeTensor(in2);
        AscendC::Add(bufA, bufA, bufB, kI);                              // fp32 residual sum
        AscendC::Cast(outT, bufA, AscendC::RoundMode::CAST_RINT, kI);    // = .to(dtype)
        if (this->variant >= V_MODE0) {
            // Cast back from the *queued* tensor, exactly as add_rms_norm_stats
            // mode 0 does. Reading it after EnQue/DeQue/FreeTensor instead makes
            // the rstd disagree with the production op by ~5e-6 relative (measured,
            // first probe build) - the probe must replicate the production data
            // flow byte for byte, including the order in which the rounded values
            // are consumed.
            AscendC::Cast(bufB, outT, AscendC::RoundMode::CAST_NONE, kI);
        }
        this->outQue.EnQue(outT);
        AscendC::LocalTensor<T> outD = this->outQue.template DeQue<T>();
        AscendC::DataCopy(this->xOutGm[(uint64_t)r * this->kDim], outD, this->kDim);
        this->outQue.FreeTensor(outD);

        if (this->variant == V_APPLY) {
            return;  // v1 stops here: the apply path is complete and written out
        }

        AscendC::Mul(bufA, bufB, bufB, kI);
        AscendC::LocalTensor<float> sumD = this->bufSum.template Get<float>();
        if (this->variant == V_NO_REDUCE) {
            // Ablation: keep everything after the reduction, drop only the row
            // reduction itself (lane 0 stands in for the row sum). Isolates the
            // cost of ReduceSum over 5120 elements per row.
            AscendC::Muls(sumD, bufA, 1.0f, 1);
        } else {
            AscendC::ReduceSum<float>(sumD, bufA, this->bufWork.template Get<float>(), kI);
        }
        AscendC::Muls(sumD, sumD, this->kInv, 1);
        AscendC::Adds(sumD, sumD, this->eps, 1);
        AscendC::LocalTensor<float> tmp0 = this->bufTmp.template Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp0[8];
        if (this->variant == V_NO_RSTD_MATH) {
            // Ablation: keep the reduction and the scalar readback, drop the
            // count-1 rsqrt + Newton-Raphson chain.
            AscendC::PipeBarrier<PIPE_V>();
            float rawVal = sumD.GetValue(0);
            rstdStage.SetValue(staged, rawVal);
            return;
        }
        AscendC::Muls(tmp0, sumD, 1.0f, 1);
        AscendC::Rsqrt(sumD, sumD, 1);
        AscendC::Mul(tmp1, tmp0, sumD, 1);
        AscendC::Mul(tmp1, tmp1, sumD, 1);
        AscendC::Muls(tmp1, tmp1, -0.5f, 1);
        AscendC::Adds(tmp1, tmp1, 1.5f, 1);
        AscendC::Mul(sumD, sumD, tmp1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        float rstdVal = sumD.GetValue(0);
        rstdStage.SetValue(staged, rstdVal);
    }

    __aicore__ inline void FlushRstd(const AscendC::LocalTensor<float> &stage, uint32_t firstRow)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(this->rstdGm[firstRow], stage, RSTD_GROUP);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueX1;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufWork;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSum;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRstd;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<T> xOutGm;
    AscendC::GlobalTensor<float> rstdGm;
    uint32_t kDim = 0u;
    uint32_t variant = 0u;
    uint32_t myRows = 0u;
    uint32_t rowStart = 0u;
    float eps = 1e-6f;
    float kInv = 1.0f;
};

// ---------------------------------------------------------------------------
// Round-2 candidate: 8 rows x KC columns per tile (KC = K / TILE_ROWS), i.e. the
// same UB bytes per tile as one full row, arranged so that the statistics path
// needs no scalar readback at all.
//
// Why this shape: rstd is 4 bytes per row while the smallest legal MTE3 write is
// 32 bytes, so with one row per tile the only way out is the scalar
// GetValue/SetValue staging that round 1 named as the bottleneck (~20.5us of the
// 33.7us statistics path). Holding 8 rows buys, for free, 8 contiguous row sums
// (AR-pattern ReduceSum), 8 contiguous rstd values (so the count-1 rsqrt/Newton
// chain becomes count-8) and one legal 32B rstd DataCopy.
//
// Cost: the copies become strided (8 segments of KC*2 bytes, K*2 apart) instead
// of one contiguous 10KB row. V_TILE8_COPY measures that cost with the compute
// removed, so the tiling gain and the layout cost stay separable.
//
// ⚠ STATUS: V_TILE8_FULL's memory path is verified correct (`out` is bit-equal to
// add_rms_norm_stats mode 0) and its timing is the round-2 result, but its
// STATISTICS ARE NOT CORRECT: rstd comes out as garbage (0 .. ~700 against a
// true range of 1.3719 .. 1.4664 on the 2048x5120 case). The row sums from the
// AR-pattern ReduceSum are not what this layout should produce, and neither the
// 4KB temp nor isReuseSource=false fixed it (the latter changed the numbers, so
// the reduce is running, just not on the data I think it is). Do NOT use
// V_TILE8_FULL's rstd as a reference or as evidence of anything but timing; the
// open item and the next step are recorded in design.md 3.
// ---------------------------------------------------------------------------
template <typename T>
class BwProbeTile8 {
public:
    __aicore__ inline BwProbeTile8() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR xOut, GM_ADDR rstd, uint32_t mRows,
                                uint32_t kDim, uint32_t variantIn, float epsIn, float kInvIn)
    {
        this->kDim = kDim;
        this->variant = variantIn;
        this->eps = epsIn;
        this->kInv = kInvIn;
        this->kTile = kDim / TILE_ROWS;
        this->nKTiles = TILE_ROWS;

        uint32_t cores = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowsPerCore = (mRows + cores - 1u) / cores;
        rowsPerCore = (rowsPerCore + RSTD_GROUP - 1u) / RSTD_GROUP * RSTD_GROUP;
        uint32_t rowStart = blockIdx * rowsPerCore;
        uint32_t rowStop = rowStart + rowsPerCore;
        if (rowStop > mRows) {
            rowStop = mRows;
        }
        this->rowStart = rowStart;
        this->myRows = (rowStop > rowStart) ? (rowStop - rowStart) : 0u;
        if (this->myRows == 0u) {
            return;
        }
        // Whole-tensor views for x1/x2/out (this design indexes by absolute row),
        // but rstd is a per-core window, so its index is relative to rowStart.
        this->x1Gm.SetGlobalBuffer((__gm__ T *)x1, (uint64_t)mRows * kDim);
        this->x2Gm.SetGlobalBuffer((__gm__ T *)x2, (uint64_t)mRows * kDim);
        this->xOutGm.SetGlobalBuffer((__gm__ T *)xOut, (uint64_t)mRows * kDim);
        this->rstdGm.SetGlobalBuffer((__gm__ float *)rstd + rowStart, this->myRows);

        const uint32_t tileBytes = kDim * (uint32_t)sizeof(T);        // TILE_ROWS * kTile elements
        const uint32_t tileF32Bytes = kDim * (uint32_t)sizeof(float);
        this->pipe.InitBuffer(this->inQueX1, 2, tileBytes);
        this->pipe.InitBuffer(this->inQueX2, 2, tileBytes);
        this->pipe.InitBuffer(this->outQue, 2, tileBytes);
        this->pipe.InitBuffer(this->bufA, tileF32Bytes);
        this->pipe.InitBuffer(this->bufB, tileF32Bytes);
        this->pipe.InitBuffer(this->bufWork, 4096u);   // AR-reduce temp (skill formula: >= 4KB)
        this->pipe.InitBuffer(this->bufAcc, 32u);
        this->pipe.InitBuffer(this->bufPart, 32u);
        this->pipe.InitBuffer(this->bufTmp, 64u);
    }

    __aicore__ inline void Process()
    {
        if (this->myRows == 0u) {
            return;
        }
        for (uint32_t g = 0u; g < this->myRows; g += TILE_ROWS) {
            ProcessGroup(g);
        }
    }

private:
    // Copy parameters: TILE_ROWS segments of kTile elements. Strides are the GAP
    // between adjacent blocks, and the unit depends on WHICH SIDE the parameter
    // describes - GM in bytes, UB in 32B units
    // (ascendc-operator-code-gen/references/data-copy-api.md). So the two
    // directions are not interchangeable: for GM->UB the gap sits on the src side
    // (bytes) and the UB side is contiguous; for UB->GM it is the mirror image.
    // Passing the GM gap as the UB stride interprets it as 32B units and reads far
    // past the end of UB (measured: "address for the MTE instruction to read on
    // chip buffer is out of bounds", retCode 0x31).
    __aicore__ inline AscendC::DataCopyExtParams CopyInParams() const
    {
        return AscendC::DataCopyExtParams(static_cast<uint16_t>(TILE_ROWS), SegBytes(), GapBytes(), 0u, 0u);
    }

    __aicore__ inline AscendC::DataCopyExtParams CopyOutParams() const
    {
        return AscendC::DataCopyExtParams(static_cast<uint16_t>(TILE_ROWS), SegBytes(), 0u, GapBytes(), 0u);
    }

    __aicore__ inline uint32_t SegBytes() const
    {
        return this->kTile * (uint32_t)sizeof(T);
    }

    __aicore__ inline uint32_t GapBytes() const
    {
        return (this->kDim - this->kTile) * (uint32_t)sizeof(T);
    }

    // relFirstRow is relative to this core's rowStart (rstd window); GM indexing
    // adds rowStart back.
    __aicore__ inline void ProcessGroup(uint32_t relFirstRow)
    {
        const int32_t tileElems = static_cast<int32_t>(this->kDim);  // TILE_ROWS * kTile
        const uint64_t rowBase = (uint64_t)(this->rowStart + relFirstRow) * this->kDim;
        AscendC::LocalTensor<float> acc = this->bufAcc.template Get<float>();
        AscendC::LocalTensor<float> part = this->bufPart.template Get<float>();
        AscendC::LocalTensor<float> tid = this->bufA.template Get<float>();
        AscendC::LocalTensor<float> tidB = this->bufB.template Get<float>();
        if (this->variant == V_TILE8_FULL) {
            AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(RSTD_GROUP));
        }

        for (uint32_t kt = 0u; kt < this->nKTiles; ++kt) {
            const uint64_t base = rowBase + (uint64_t)kt * this->kTile;
            AscendC::DataCopyExtParams cpIn = CopyInParams();
            AscendC::DataCopyExtParams cpOut = CopyOutParams();

            AscendC::LocalTensor<T> in1 = this->inQueX1.template AllocTensor<T>();
            AscendC::DataCopyPad(in1, this->x1Gm[base], cpIn,
                                 AscendC::DataCopyPadExtParams<T>(false, 0u, 0u, 0u));
            this->inQueX1.EnQue(in1);
            in1 = this->inQueX1.template DeQue<T>();

            AscendC::LocalTensor<T> in2 = this->inQueX2.template AllocTensor<T>();
            AscendC::DataCopyPad(in2, this->x2Gm[base], cpIn,
                                 AscendC::DataCopyPadExtParams<T>(false, 0u, 0u, 0u));
            this->inQueX2.EnQue(in2);
            in2 = this->inQueX2.template DeQue<T>();

            AscendC::LocalTensor<T> outT = this->outQue.template AllocTensor<T>();
            if (this->variant == V_TILE8_COPY) {
                AscendC::DataCopy(outT, in1, tileElems);  // pattern floor: 2R + 1W, no compute
                this->inQueX1.FreeTensor(in1);
                this->inQueX2.FreeTensor(in2);
                this->outQue.EnQue(outT);
                AscendC::LocalTensor<T> o = this->outQue.template DeQue<T>();
                AscendC::DataCopyPad(this->xOutGm[base], o, cpOut);
                this->outQue.FreeTensor(o);
                continue;
            }

            AscendC::Cast(tid, in1, AscendC::RoundMode::CAST_NONE, tileElems);
            AscendC::Cast(tidB, in2, AscendC::RoundMode::CAST_NONE, tileElems);
            this->inQueX1.FreeTensor(in1);
            this->inQueX2.FreeTensor(in2);
            AscendC::Add(tid, tid, tidB, tileElems);
            AscendC::Cast(outT, tid, AscendC::RoundMode::CAST_RINT, tileElems);
            // Statistics are taken on the rounded value (golden protocol).
            AscendC::Cast(tidB, outT, AscendC::RoundMode::CAST_NONE, tileElems);
            this->outQue.EnQue(outT);
            AscendC::LocalTensor<T> o = this->outQue.template DeQue<T>();
            AscendC::DataCopyPad(this->xOutGm[base], o, cpOut);
            this->outQue.FreeTensor(o);

            AscendC::Mul(tid, tidB, tidB, tileElems);
            uint32_t srcShape[2] = {TILE_ROWS, static_cast<uint32_t>(this->kTile)};
            // isReuseSource=false on purpose: with true the AR implementation uses
            // the source tensor itself as scratch (ReduceSumArReusedSrc ->
            // ReduceSumArCompute(dst, src, src, ...)), which silently corrupts the
            // partial sums here. The explicit-temp form is the documented one.
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
                part, tid, this->bufWork.template Get<uint8_t>(), srcShape, true);
            AscendC::Add(acc, acc, part, static_cast<int32_t>(RSTD_GROUP));
        }

        // rstd = 1/sqrt(mean + eps): once per 8 rows, on 8 contiguous lanes.
        AscendC::Muls(acc, acc, this->kInv, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Adds(acc, acc, this->eps, static_cast<int32_t>(RSTD_GROUP));
        AscendC::LocalTensor<float> t0 = this->bufTmp.template Get<float>();
        AscendC::LocalTensor<float> t1 = t0[8];
        AscendC::Muls(t0, acc, 1.0f, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Rsqrt(acc, acc, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Mul(t1, t0, acc, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Mul(t1, t1, acc, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Muls(t1, t1, -0.5f, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Adds(t1, t1, 1.5f, static_cast<int32_t>(RSTD_GROUP));
        AscendC::Mul(acc, acc, t1, static_cast<int32_t>(RSTD_GROUP));
        // V -> MTE3: the freshly written rstd must be visible to the copy out.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(this->rstdGm[relFirstRow], acc, static_cast<int32_t>(RSTD_GROUP));
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueX1;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufWork;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufAcc;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufPart;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<T> xOutGm;
    AscendC::GlobalTensor<float> rstdGm;
    uint32_t kDim = 0u;
    uint32_t kTile = 0u;
    uint32_t nKTiles = 0u;
    uint32_t variant = 0u;
    uint32_t myRows = 0u;
    uint32_t rowStart = 0u;
    float eps = 1e-6f;
    float kInv = 1.0f;
};
#define BW_PROBE_ENTRY(entry, elemType)                                                     \
    extern "C" __global__ __aicore__ void entry(GM_ADDR x1, GM_ADDR x2, GM_ADDR xOut,        \
                                                GM_ADDR rstd, uint32_t mRows, uint32_t kDim, \
                                                uint32_t variant, float eps, float kInv)     \
    {                                                                                        \
        if (variant >= 5u) {                                                                 \
            BwProbeTile8<elemType> op8;                                                      \
            op8.Init(x1, x2, xOut, rstd, mRows, kDim, variant, eps, kInv);                   \
            op8.Process();                                                                   \
            return;                                                                          \
        }                                                                                    \
        BwProbe<elemType> op;                                                                 \
        op.Init(x1, x2, xOut, rstd, mRows, kDim, variant, eps, kInv);                          \
        op.Process();                                                                         \
    }

BW_PROBE_ENTRY(bw_probe_bf16, bfloat16_t)
BW_PROBE_ENTRY(bw_probe_fp16, half)
