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

// add_rms_norm_stats - F2 fusion "norm stage" (see csrc/ops/add_rms_norm_stats/design.md).
//
// Three modes, all purely row-local (a row's sum of squares never leaves the
// row: no cross-core communication, no workspace):
//   mode 0 ("residual + stats"): x_out = round_dtype(x1 + x2);
//                                rstd  = 1/sqrt(mean(x_out^2) + eps)
//   mode 1 ("stats + apply")  : rstd = 1/sqrt(mean(x1^2) + eps);
//                                y     = round_dtype(round_dtype(x1*rstd)*gamma + beta)
//   mode 2 ("stats only")     : rstd = 1/sqrt(mean(round_dtype(x1+x2)^2) + eps)
//
// The rounding/reduction protocol deliberately mirrors the CANN
// npu_add_rms_norm_bias golden (test/add_rms_norm_stats_ref.py):
//  - the sum of squares is taken on the *rounded* residual x_out, not on the
//    unrounded fp32 sum;
//  - in the apply step (x*rstd) is rounded back to the data dtype before the
//    gamma multiply (golden: result_mid.to(dtype) then * gamma in fp32).
// Target is indistinguishability from the production chain, not extra accuracy.
//
// Layout: one full row (K <= MAX_K) is resident in UB, so every vector op is a
// single full-row op - no K tiling, no per-tile tail handling. op_host enforces
// K % 16 == 0 (32B alignment) which makes every row copy a legal 32B-multiple
// DataCopy, so no DataCopyPad / tail padding is involved at all.
//
// K > MAX_K needs a second-level K tiling (design.md 5, v2 item); op_host
// rejects it explicitly.

#include "kernel_operator.h"

namespace {
constexpr uint32_t RSTD_GROUP = 8;  // 8 fp32 rows = 32B: minimum legal MTE3 write unit
constexpr uint32_t MAX_K = 5120;    // per-row UB residency budget (op_host enforced)
}  // namespace

template <typename T>
class AddRmsNormStats {
public:
    __aicore__ inline AddRmsNormStats() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                GM_ADDR xOut, GM_ADDR rstd, GM_ADDR y, uint32_t mRows,
                                uint32_t kDim, uint32_t modeIn, float epsIn, float kInvIn,
                                uint32_t hasBetaIn)
    {
        this->kDim = kDim;
        this->mode = modeIn;
        this->eps = epsIn;
        this->kInv = kInvIn;
        this->hasBeta = hasBetaIn;

        uint32_t cores = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // Rows per core padded to a multiple of RSTD_GROUP: every core's rowStart
        // stays 8-aligned, so the fp32 rstd DataCopy offsets stay 32B aligned
        // (MTE3 requirement) for any partial last group (lse_merge M2 lesson).
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
            return;  // core beyond the row range: idle, no buffers allocated
        }

        this->x1Gm.SetGlobalBuffer((__gm__ T *)x1 + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        if (this->mode != 1u) {
            this->x2Gm.SetGlobalBuffer((__gm__ T *)x2 + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        }
        if (this->mode == 0u) {
            this->xOutGm.SetGlobalBuffer((__gm__ T *)xOut + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        }
        if (this->mode == 1u) {
            this->gammaGm.SetGlobalBuffer((__gm__ T *)gamma, kDim);
            if (this->hasBeta != 0u) {
                this->betaGm.SetGlobalBuffer((__gm__ T *)beta, kDim);
            }
            this->yGm.SetGlobalBuffer((__gm__ T *)y + (uint64_t)rowStart * kDim, (uint64_t)this->myRows * kDim);
        }
        this->rstdGm.SetGlobalBuffer((__gm__ float *)rstd + rowStart, this->myRows);

        const uint32_t rowBytes = kDim * (uint32_t)sizeof(T);
        const uint32_t f32Bytes = kDim * (uint32_t)sizeof(float);
        this->pipe.InitBuffer(this->inQueX1, 2, rowBytes);
        if (this->mode != 1u) {
            this->pipe.InitBuffer(this->inQueX2, 2, rowBytes);
        }
        this->pipe.InitBuffer(this->outQue, 2, rowBytes);
        this->pipe.InitBuffer(this->bufA, f32Bytes);
        this->pipe.InitBuffer(this->bufB, f32Bytes);
        this->pipe.InitBuffer(this->bufWork, f32Bytes);
        this->pipe.InitBuffer(this->bufSum, 8u * (uint32_t)sizeof(float));
        if (this->mode == 2u) {
            // mode 2 only needs the rounded residual privately (no GM output).
            this->pipe.InitBuffer(this->bufMid, rowBytes);
        }
        if (this->mode == 1u) {
            this->pipe.InitBuffer(this->bufGamma, f32Bytes);
            if (this->hasBeta != 0u) {
                this->pipe.InitBuffer(this->bufBeta, f32Bytes);
            }
        }
        this->pipe.InitBuffer(this->bufRstd, RSTD_GROUP * (uint32_t)sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->myRows == 0u) {
            return;
        }
        if (this->mode == 1u) {
            PreloadGammaBeta();
        }
        AscendC::LocalTensor<float> rstdStage = this->bufRstd.template Get<float>();
        uint32_t staged = 0u;
        for (uint32_t r = 0u; r < this->myRows; ++r) {
            ProcessRow(r, rstdStage, staged);
            if (++staged == RSTD_GROUP) {
                // Partial groups never reach here (myRows is a multiple of
                // RSTD_GROUP for every core but the last one).
                FlushRstd(rstdStage, r + 1u - RSTD_GROUP);
                staged = 0u;
            }
        }
        if (staged != 0u) {
            // Partial group: the tail lanes are zeroed and the rstd buffer is
            // padded to a multiple of RSTD_GROUP rows by op_host, so this stays
            // inside the allocation. Vector write, so the flush barrier below
            // covers it as well.
            AscendC::Duplicate(rstdStage[staged], 0.0f, static_cast<int32_t>(RSTD_GROUP - staged));
            FlushRstd(rstdStage, this->myRows - staged);
        }
    }

private:
    __aicore__ inline void PreloadGammaBeta()
    {
        AscendC::LocalTensor<T> gT = this->inQueX1.template AllocTensor<T>();
        AscendC::DataCopy(gT, this->gammaGm, this->kDim);
        this->inQueX1.EnQue(gT);
        gT = this->inQueX1.template DeQue<T>();
        AscendC::Cast(this->bufGamma.template Get<float>(), gT, AscendC::RoundMode::CAST_NONE,
                      static_cast<int32_t>(this->kDim));
        this->inQueX1.FreeTensor(gT);
        if (this->hasBeta != 0u) {
            AscendC::LocalTensor<T> bT = this->inQueX1.template AllocTensor<T>();
            AscendC::DataCopy(bT, this->betaGm, this->kDim);
            this->inQueX1.EnQue(bT);
            bT = this->inQueX1.template DeQue<T>();
            AscendC::Cast(this->bufBeta.template Get<float>(), bT, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(this->kDim));
            this->inQueX1.FreeTensor(bT);
        }
    }

    // Runs one row and stages its rstd into `rstdStage[staged]` (vector write).
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

        AscendC::LocalTensor<T> outT;
        if (this->mode == 1u) {
            // The input already is the rounded residual: its fp32 value is the
            // norm input, no residual add and no rounding step here.
            AscendC::Cast(bufB, in1, AscendC::RoundMode::CAST_NONE, kI);
        } else {
            AscendC::LocalTensor<T> in2 = this->inQueX2.template AllocTensor<T>();
            AscendC::DataCopy(in2, this->x2Gm[(uint64_t)r * this->kDim], this->kDim);
            this->inQueX2.EnQue(in2);
            in2 = this->inQueX2.template DeQue<T>();
            AscendC::Cast(bufA, in1, AscendC::RoundMode::CAST_NONE, kI);
            AscendC::Cast(bufB, in2, AscendC::RoundMode::CAST_NONE, kI);
            this->inQueX2.FreeTensor(in2);
            AscendC::Add(bufA, bufA, bufB, kI);                                // fp32 residual sum
            if (this->mode == 0u) {
                outT = this->outQue.template AllocTensor<T>();
                AscendC::Cast(outT, bufA, AscendC::RoundMode::CAST_RINT, kI);  // = .to(dtype)
                AscendC::Cast(bufB, outT, AscendC::RoundMode::CAST_NONE, kI);  // stats on the rounded value
                this->outQue.EnQue(outT);
                AscendC::LocalTensor<T> outD = this->outQue.template DeQue<T>();
                AscendC::DataCopy(this->xOutGm[(uint64_t)r * this->kDim], outD, this->kDim);
                this->outQue.FreeTensor(outD);
            } else {
                // mode 2: round privately (no x_out output), stats on the rounded value.
                AscendC::LocalTensor<T> mid = this->bufMid.template Get<T>();
                AscendC::Cast(mid, bufA, AscendC::RoundMode::CAST_RINT, kI);
                AscendC::Cast(bufB, mid, AscendC::RoundMode::CAST_NONE, kI);
            }
        }
        this->inQueX1.FreeTensor(in1);

        // Sum of squares of the (rounded) norm input, reduced over the row.
        AscendC::Mul(bufA, bufB, bufB, kI);
        AscendC::LocalTensor<float> sumD = this->bufSum.template Get<float>();
        AscendC::ReduceSum<float>(sumD, bufA, this->bufWork.template Get<float>(), kI);
        // rstd = 1/sqrt(mean + eps) with vector ops only: AICore code has no
        // scalar sqrtf and rejects uint32 -> float casts, so 1/K comes from the
        // host (kInv) and the reciprocal square root is the vector Rsqrt.
        AscendC::Muls(sumD, sumD, this->kInv, 1);
        AscendC::Adds(sumD, sumD, this->eps, 1);
        AscendC::Rsqrt(sumD, sumD, 1);
        AscendC::Muls(rstdStage[staged], sumD, 1.0f, 1);

        if (this->mode == 1u) {
            AscendC::PipeBarrier<PIPE_V>();  // V -> S: sumD is read by the scalar unit below
            float rstdVal = sumD.GetValue(0);
            outT = this->outQue.template AllocTensor<T>();
            AscendC::Muls(bufA, bufB, rstdVal, kI);                          // x * rstd (fp32)
            AscendC::Cast(outT, bufA, AscendC::RoundMode::CAST_RINT, kI);    // round to dtype (golden)
            AscendC::Cast(bufA, outT, AscendC::RoundMode::CAST_NONE, kI);    // back to fp32
            AscendC::Mul(bufA, bufA, this->bufGamma.template Get<float>(), kI);
            if (this->hasBeta != 0u) {
                AscendC::Add(bufA, bufA, this->bufBeta.template Get<float>(), kI);
            }
            AscendC::Cast(outT, bufA, AscendC::RoundMode::CAST_RINT, kI);
            this->outQue.EnQue(outT);
            AscendC::LocalTensor<T> outD = this->outQue.template DeQue<T>();
            AscendC::DataCopy(this->yGm[(uint64_t)r * this->kDim], outD, this->kDim);
            this->outQue.FreeTensor(outD);
        }
    }

    // Writes RSTD_GROUP rows of rstd (32B) starting at row `firstRow` of this core.
    __aicore__ inline void FlushRstd(const AscendC::LocalTensor<float> &stage, uint32_t firstRow)
    {
        AscendC::PipeBarrier<PIPE_ALL>();  // stage the vector-written rstd before the MTE3 read
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMid;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufGamma;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufBeta;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRstd;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<T> gammaGm;
    AscendC::GlobalTensor<T> betaGm;
    AscendC::GlobalTensor<T> xOutGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<float> rstdGm;
    uint32_t kDim = 0u;
    uint32_t mode = 0u;
    uint32_t hasBeta = 0u;
    uint32_t myRows = 0u;
    uint32_t rowStart = 0u;
    float eps = 1e-6f;
    float kInv = 1.0f;
};

#define ADD_RMS_NORM_STATS_ENTRY(entry, elemType)                                                 \
    extern "C" __global__ __aicore__ void entry(                                                  \
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR xOut, GM_ADDR rstd,          \
        GM_ADDR y, uint32_t mRows, uint32_t kDim, uint32_t mode, float eps, float kInv,           \
        uint32_t hasBeta)                                                                          \
    {                                                                                             \
        AddRmsNormStats<elemType> op;                                                             \
        op.Init(x1, x2, gamma, beta, xOut, rstd, y, mRows, kDim, mode, eps, kInv, hasBeta);       \
        op.Process();                                                                             \
    }

ADD_RMS_NORM_STATS_ENTRY(add_rms_norm_stats_bf16, bfloat16_t)
ADD_RMS_NORM_STATS_ENTRY(add_rms_norm_stats_fp16, half)
