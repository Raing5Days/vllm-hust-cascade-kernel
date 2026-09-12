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
};
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

#define BW_PROBE_ENTRY(entry, elemType)                                                     \
    extern "C" __global__ __aicore__ void entry(GM_ADDR x1, GM_ADDR x2, GM_ADDR xOut,        \
                                                GM_ADDR rstd, uint32_t mRows, uint32_t kDim, \
                                                uint32_t variant, float eps, float kInv)     \
    {                                                                                        \
        BwProbe<elemType> op;                                                                 \
        op.Init(x1, x2, xOut, rstd, mRows, kDim, variant, eps, kInv);                          \
        op.Process();                                                                         \
    }

BW_PROBE_ENTRY(bw_probe_bf16, bfloat16_t)
BW_PROBE_ENTRY(bw_probe_fp16, half)
