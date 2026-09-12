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

// D3 (design.md 9): the rstd chain (Rsqrt + one Newton step) is 9 count-1 vector
// ops plus a PIPE_V barrier and a scalar round trip *per row*, i.e. 10 of the 17
// vector instructions a row costs, all serving a single scalar. Batched mode
// (batchStats != 0, modes 0/2 only) lets each row's sum of squares land in lane 0
// of its own 32B slot, then runs the same chain once over the 8 slots
// (count=64) per RSTD_GROUP rows: 9 instructions + 1 barrier per 8 rows instead
// of per row. Every op in the chain is element-wise, so the per-lane results are
// bit-identical to the per-row form - this is a scheduling change, not a
// numeric one (the precision suite re-checks that claim).
//
// Mode 1 is excluded: its rstd is consumed as a scalar by Muls in the apply step
// of the *same* row, so the value cannot be deferred to the group boundary.

template <typename T>
class AddRmsNormStats {
public:
    __aicore__ inline AddRmsNormStats() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                GM_ADDR xOut, GM_ADDR rstd, GM_ADDR y, uint32_t mRows,
                                uint32_t kDim, uint32_t modeIn, float epsIn, float kInvIn,
                                uint32_t hasBetaIn, uint32_t batchStatsIn, uint32_t pairSumIn)
    {
        this->kDim = kDim;
        this->mode = modeIn;
        this->eps = epsIn;
        this->kInv = kInvIn;
        this->hasBeta = hasBetaIn;
        // Only modes 0/2 can batch (see the note above the class).
        this->batchStats = (modeIn == 1u) ? 0u : batchStatsIn;
        this->pairSum = pairSumIn;

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
        this->pipe.InitBuffer(this->bufTmp, 16u * (uint32_t)sizeof(float));  // 2 x 32B lanes
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
        if (this->batchStats != 0u) {
            // 8 slots x 8 lanes: lane 0 of slot r is row r's sum of squares, so
            // every ReduceSum destination is 32B aligned (a VEC write to element
            // offsets 1..7 of a row faults, which is why the per-row path uses
            // scalar SetValue). 256B each.
            this->pipe.InitBuffer(this->bufSums, RSTD_GROUP * 8u * (uint32_t)sizeof(float));
            this->pipe.InitBuffer(this->bufT64a, RSTD_GROUP * 8u * (uint32_t)sizeof(float));
            this->pipe.InitBuffer(this->bufT64b, RSTD_GROUP * 8u * (uint32_t)sizeof(float));
        }
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
        if (this->batchStats != 0u) {
            // Slot lanes other than lane 0 hold no row sum. Priming them with 0
            // keeps every lane finite through the chain (Rsqrt(0+eps) = 1e3, no
            // NaN), which matters because the chain covers all 64 lanes; the
            // lanes of an unused slot are overwritten by the tail zeroing below.
            PrimeSlots();
        }
        for (uint32_t r = 0u; r < this->myRows; ++r) {
            ProcessRow(r, rstdStage, staged);
            if (++staged == RSTD_GROUP) {
                // Partial groups never reach here (myRows is a multiple of
                // RSTD_GROUP for every core but the last one).
                if (this->batchStats != 0u) {
                    RstdChainBatch(rstdStage);
                }
                FlushRstd(rstdStage, r + 1u - RSTD_GROUP);
                staged = 0u;
                if (this->batchStats != 0u && r + 1u < this->myRows) {
                    PrimeSlots();
                }
            }
        }
        if (staged != 0u) {
            if (this->batchStats != 0u) {
                RstdChainBatch(rstdStage);
            }
            // Partial group: the tail lanes are zeroed and the rstd buffer is
            // padded to a multiple of RSTD_GROUP rows by op_host, so this stays
            // inside the allocation. Scalar writes only: a VEC instruction whose
            // UB address is not 32B aligned faults on 910B (measured, see
            // design.md 5), and element offsets 1..7 of a float row are not.
            // Runs after the batch chain, which fills all 8 lanes.
            for (uint32_t i = staged; i < RSTD_GROUP; ++i) {
                rstdStage.SetValue(i, 0.0f);
            }
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
        if (this->batchStats != 0u) {
            // Batched modes 0/2: ReduceSum writes lane 0 of slot `staged` itself
            // (32B aligned), and the chain is deferred to the group boundary -
            // so the rest of this function is never reached here. in1 was
            // already freed above.
            RowSumSq(this->bufSums.template Get<float>()[staged * RSTD_GROUP], bufA, kI);
            return;
        }
        AscendC::LocalTensor<float> sumD = this->bufSum.template Get<float>();
        RowSumSq(sumD, bufA, kI);
        // rstd = 1/sqrt(mean + eps) with vector ops only: AICore code has no
        // scalar sqrtf and rejects uint32 -> float casts, so 1/K comes from the
        // host (kInv) and the reciprocal square root is the vector Rsqrt.
        // Rsqrt itself is a ~2^-11 approximation on 910B (measured: 1.4e-3
        // relative deviation from the CANN npu_add_rms_norm rstd, see design.md
        // 5/8), so one Newton-Raphson step r *= 1.5 - 0.5*a*r*r is applied
        // (4 extra 1-element vector ops). All count-1 ops run on the 32B
        // aligned base of bufSum / bufTmp lanes.
        AscendC::Muls(sumD, sumD, this->kInv, 1);
        AscendC::Adds(sumD, sumD, this->eps, 1);
        AscendC::LocalTensor<float> tmp0 = this->bufTmp.template Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp0[8];  // +32B: keeps VEC addresses aligned
        AscendC::Muls(tmp0, sumD, 1.0f, 1);          // a = mean + eps
        AscendC::Rsqrt(sumD, sumD, 1);               // r0
        NewtonStep(sumD, tmp0, tmp1, 1);             // r1
        if (this->pairSum != 0u) {
            // D3: the vector Rsqrt is only a ~2^-11 approximation on 910B, and one
            // Newton step leaves its square, ~1.5*eps^2 ~ 1e-5 worst case - which
            // is exactly the residual rstd error measured on the elements that
            // miss the CANN strict tier (they are unchanged when the row sum is
            // made pairwise-accurate). A second step drives the iteration error to
            // 1.5*eps^4 (far below fp32), leaving the fp32 evaluation of the
            // formula (~2 eps) as the floor. Same formula, same rounding modes.
            NewtonStep(sumD, tmp0, tmp1, 1);         // r2
        }
        AscendC::PipeBarrier<PIPE_V>();  // V -> S: sumD is read by the scalar unit below
        float rstdVal = sumD.GetValue(0);
        // Scalar write into the 8-row staging buffer: `rstdStage[staged]` for
        // staged >= 1 is NOT 32B aligned, and a VEC instruction writing there
        // faults with "UB address accessed by the VEC instruction is not
        // aligned" (measured on 910B, first attempt of this kernel; scalar
        // SetValue has no such constraint - lse_merge uses the same pattern).
        rstdStage.SetValue(staged, rstdVal);

        if (this->mode == 1u) {
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

    // Row sum of squares. `src` is [kI] fp32 (the squared norm input) and is
    // destroyed - it is folded in place; the result lands in dst[0].
    //
    // D3 (design.md 9): the golden's rstd is good to ~0.5 eps, while a plain
    // AscendC::ReduceSum over K=5120 accumulates ~30 eps (measured, F2 raw
    // f2_mode1_y_defect_evidence.md): the sum is combined from ~80 partials
    // sequentially, so the error grows with the *number of partials*, not with
    // log(K). That error moves round_dtype(x*rstd) across a dtype rounding
    // boundary for ~3e-4 of the elements, which is what puts mode-1 `y` 1-2 ulp
    // outside the CANN tier on 5/16 cases. Folding pairwise first (the skill's
    // 二分累加, references/reduction/alg-dichotomy.md) shortens the accumulation
    // to a binary tree of depth log2(K), so the final ReduceSum only sees 64
    // elements and contributes ~8 eps of a partial instead of ~80 eps of the
    // whole row. Same protocol, same rounding, only a shorter sum.
    __aicore__ inline void RowSumSq(const AscendC::LocalTensor<float> &dst,
                                    const AscendC::LocalTensor<float> &src, int32_t kI)
    {
        if (this->pairSum == 0u) {
            AscendC::ReduceSum<float>(dst, src, this->bufWork.template Get<float>(), kI);
            return;
        }
        uint32_t p2 = 1u;
        while ((p2 << 1u) <= static_cast<uint32_t>(kI)) {
            p2 <<= 1u;  // largest power of two <= kI
        }
        if (static_cast<uint32_t>(kI) > p2) {
            // fold the tail onto the power-of-two window
            AscendC::Add(src, src, src[p2], static_cast<int32_t>(static_cast<uint32_t>(kI) - p2));
        }
        while (p2 > 64u) {  // binary fold; every offset here is >= 64 fp32 = 256B, so 32B aligned
            p2 >>= 1u;
            AscendC::Add(src, src, src[p2], static_cast<int32_t>(p2));
        }
        AscendC::ReduceSum<float>(dst, src, this->bufWork.template Get<float>(),
                                  static_cast<int32_t>(p2));
    }

    // One Newton-Raphson refinement of r = 1/sqrt(a): r *= 1.5 - 0.5*a*r^2.
    // `a` is preserved, `tmp` is scratch, `r` is refined in place. Element-wise,
    // so it is bit-identical regardless of how many lanes are active.
    __aicore__ inline void NewtonStep(const AscendC::LocalTensor<float> &r,
                                      const AscendC::LocalTensor<float> &a,
                                      const AscendC::LocalTensor<float> &tmp, int32_t n)
    {
        AscendC::Mul(tmp, a, r, n);        // a * r
        AscendC::Mul(tmp, tmp, r, n);      // a * r^2
        AscendC::Muls(tmp, tmp, -0.5f, n); // -0.5 * a * r^2
        AscendC::Adds(tmp, tmp, 1.5f, n);  // 1.5 - 0.5 * a * r^2
        AscendC::Mul(r, r, tmp, n);        // r * (...)
    }

    // Batched mode only: prime the 64 lanes of the slot buffer. Zero keeps every
    // lane finite through Rsqrt, so no NaN reaches the vector unit from the
    // lanes that carry no row sum.
    __aicore__ inline void PrimeSlots()
    {
        AscendC::Duplicate(this->bufSums.template Get<float>(), 0.0f, RSTD_GROUP * 8u);
    }

    // Batched mode only (modes 0/2): rstd = 1/sqrt(mean+eps), Newton-refined,
    // for RSTD_GROUP rows at once. Lane 0 of slot i holds row i's sum of squares;
    // only those 8 lanes are consumed. Every operation here is element-wise, so
    // each consumed lane sees exactly the sequence of values the per-row chain
    // computes (same RoundMode, same order, same constants).
    __aicore__ inline void RstdChainBatch(const AscendC::LocalTensor<float> &rstdStage)
    {
        AscendC::LocalTensor<float> s = this->bufSums.template Get<float>();
        AscendC::LocalTensor<float> a = this->bufT64a.template Get<float>();
        AscendC::LocalTensor<float> b = this->bufT64b.template Get<float>();
        AscendC::Muls(s, s, this->kInv, RSTD_GROUP * 8u);
        AscendC::Adds(s, s, this->eps, RSTD_GROUP * 8u);
        AscendC::Muls(a, s, 1.0f, RSTD_GROUP * 8u);            // a = mean + eps
        AscendC::Rsqrt(s, s, RSTD_GROUP * 8u);                 // r0
        NewtonStep(s, a, b, RSTD_GROUP * 8u);                  // r1
        if (this->pairSum != 0u) {
            NewtonStep(s, a, b, RSTD_GROUP * 8u);              // r2 (see the note in ProcessRow)
        }
        AscendC::PipeBarrier<PIPE_V>();  // V -> S: the lanes are read by the scalar unit
        for (uint32_t i = 0u; i < RSTD_GROUP; ++i) {
            rstdStage.SetValue(i, s.GetValue(i * RSTD_GROUP));
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMid;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufGamma;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufBeta;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRstd;
    // batched-stat scratch (modes 0/2 only): 8 slots x 8 lanes of fp32
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSums;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufT64a;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufT64b;
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
    uint32_t batchStats = 0u;  // D3: batch the rstd chain per RSTD_GROUP rows (modes 0/2)
    uint32_t pairSum = 0u;     // D3: pairwise (dichotomy) fold before the row ReduceSum
    uint32_t myRows = 0u;
    uint32_t rowStart = 0u;
    float eps = 1e-6f;
    float kInv = 1.0f;
};

#define ADD_RMS_NORM_STATS_ENTRY(entry, elemType)                                                 \
    extern "C" __global__ __aicore__ void entry(                                                  \
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR xOut, GM_ADDR rstd,          \
        GM_ADDR y, uint32_t mRows, uint32_t kDim, uint32_t mode, float eps, float kInv,           \
        uint32_t hasBeta, uint32_t batchStats, uint32_t pairSum)                                  \
    {                                                                                             \
        AddRmsNormStats<elemType> op;                                                             \
        op.Init(x1, x2, gamma, beta, xOut, rstd, y, mRows, kDim, mode, eps, kInv, hasBeta,        \
                batchStats, pairSum);                                                             \
        op.Process();                                                                             \
    }

ADD_RMS_NORM_STATS_ENTRY(add_rms_norm_stats_bf16, bfloat16_t)
ADD_RMS_NORM_STATS_ENTRY(add_rms_norm_stats_fp16, half)
