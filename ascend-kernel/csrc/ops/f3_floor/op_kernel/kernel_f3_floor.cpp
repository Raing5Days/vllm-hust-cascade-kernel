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

// f3_floor - F3 (SwiGLU into the gate_up GEMM epilogue) Phase B Step 1 ladder.
// See csrc/ops/f3_floor/design.md; the campaign plan lives in
// profiles/qwen14b-instruct-hotspot-20260910/probe-f3/DESIGN-phaseB.md.
//
// The point of this kernel is a single physical question: after the gate_up
// GEMM is split into "AIC writes a bf16 C workspace -> AIV reads it back and
// writes D", does the extra 169.9 MB of epilogue traffic hide inside the cube
// time (>= 2 ms for the prefill shape), or does it add its own duration on top?
// The ladder answers it with single-variable rungs that share ONE binary:
//
//   mode 0 "floor"   AIC: paired gate_up GEMM -> bf16 C workspace
//                    AIV: read the gate AND up windows of its own C tile,
//                         write D = gate window *unchanged* (zero SwiGLU
//                         semantics = the n2 "template floor")
//                    => exactly the production epilogue's byte pattern
//                       (113.2 MB read + 56.6 MB write) with zero compute.
//   mode 1 "swiglu"  same, but D = SiLU(gate) * up (Step 2; see design.md)
//   mode 2 "gemm"    AIC: same GEMM; AIV: flag handshake only, no data moved
//                    => this kernel's own GEMM time (isolates AIC + sync cost
//                       from the epilogue, so the marginal epilogue cost is
//                       T[0] - T[2] inside one binary).
//   mode 3 "epi"     AIC: flag handshake only; AIV: the mode-0 copy
//                    => the floor epilogue's standalone time (the "template
//                       floor" duration, comparable with the 111.80 us SwiGlu).
//
// N is enumerated in PAIRS: a block covers gate columns [n0, n0+BN) and up
// columns [I+n0, I+n0+BN) of the SAME A tile, and both are stored by the same
// AIC block before the single cross-core flag is raised (design P2). That is
// the structural prerequisite of F3 (R2): the epilogue never waits for two
// far-apart AIC writes. The weight tensor keeps its production layout
// ([2I, K] row major = a logical K x 2I column-major B), so nothing about the
// model weights changes.
//
// A2 has no L0C->UB path, so C must go through GM: this kernel does NOT save
// GM bytes, it only moves them into the cube's shadow (design.md 0/2.1).

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_operator.h"

using namespace Catlass;

namespace {
constexpr uint32_t FLAG_AIC_FINISH_STORE = 0;     // AIC -> AIV "C tile stored"
constexpr uint32_t RV_FLAG_AIC_FINISH_STORE = 1;  // AIV -> AIC reverse credit

// EPILOGUE_MODE semantics (see the header comment).
constexpr uint32_t MODE_FLOOR = 0;
constexpr uint32_t MODE_SWIGLU = 1;
constexpr uint32_t MODE_GEMM_ONLY = 2;
constexpr uint32_t MODE_EPILOGUE_ONLY = 3;
// Attribution rungs (added after the Step-1 verdict, purely additive: modes
// 0/2/3 compile to the identical code path as before):
//   4 floor_up        same as mode 0 but D = the UP window  (proves the +I
//                     window addressing value-by-value, not just "in bounds")
//   5 floor_gate_only read the gate window only, write D = it  (2/3 of mode 0's
//                     bytes: 113.2 MB instead of 169.9 MB) -> if the marginal
//                     cost of the epilogue scales with its bytes, the delta
//                     must drop to ~2/3; if it stays flat, the cost is the
//                     cross-core protocol / scheduling, not bandwidth.
constexpr uint32_t MODE_FLOOR_UP = 4;
constexpr uint32_t MODE_FLOOR_GATE_ONLY = 5;

using ElementA = bfloat16_t;
using LayoutA = layout::RowMajor;    // activation x: [M, K] row major
using ElementB = bfloat16_t;
using LayoutB = layout::ColumnMajor; // weight [2I, K] row major == logical K x 2I col major
using ElementC = bfloat16_t;         // C workspace element (fixpipe casts fp32 -> bf16)
using LayoutC = layout::RowMajor;
using ElementD = bfloat16_t;
using LayoutD = layout::RowMajor;

// Tiling (see design.md 3): L0C is the binding constraint (bm * bn * 4B <= 128KB)
// and the L2 traffic optimum sits at bm = 256 / bn = 128 for this problem
// (b_traffic = sizeof(B) * M/bm, a_traffic = sizeof(A) * (I/bn)).
using L1TileShape = GemmShape<256, 128, 256>;
using L0TileShape = GemmShape<256, 128, 64>;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<false>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

using CopyGm2Ub = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<ElementC, layout::RowMajor>>;
using CopyUb2Gm = Epilogue::Tile::CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<ElementD, layout::RowMajor>>;

constexpr uint32_t SUB_ROWS = L1TileShape::M / 2;  // rows per AIV subcore (2 subcores)
constexpr uint32_t TILE_N = L1TileShape::N;

struct F3Params {
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrC;  // bf16 workspace [M, 2I]
    GM_ADDR ptrD;  // bf16 output   [M, I]
    uint32_t m;
    uint32_t nHalf;  // I
    uint32_t k;
};

template <uint32_t MODE>
class F3GateUpKernel {
public:
    using ArchTag = Arch::AtlasA2;

    CATLASS_DEVICE
    F3GateUpKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(F3Params const &params);

    // ---------------------------------------------------------------- AIC ---
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(F3Params const &params)
    {
        uint32_t numMBlocks = CeilDiv(params.m, L1TileShape::M);
        uint32_t numNBlocks = CeilDiv(params.nHalf, L1TileShape::N);
        uint32_t coreLoops = numMBlocks * numNBlocks;

        if constexpr (MODE != MODE_EPILOGUE_ONLY) {
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            AscendC::GlobalTensor<ElementC> gmC;
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

            layout::RowMajor layoutA(params.m, params.k);
            // Logical B is K x 2I of bf16 stored as [2I, K] row major.
            layout::ColumnMajor layoutB(params.k, 2 * params.nHalf);
            layout::RowMajor layoutC(params.m, 2 * params.nHalf);

            BlockMmad blockMmad(resource);

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
                 loopIdx += AscendC::GetBlockNum()) {
                // n-major block order: the cores running at the same time share
                // the same B tiles (which are L2/HBM expensive) and only re-read
                // the 21 MB A matrix; see design.md 3.2.
                uint32_t nIdx = loopIdx / numMBlocks;
                uint32_t mIdx = loopIdx - nIdx * numMBlocks;
                uint32_t m0 = mIdx * L1TileShape::M;
                uint32_t n0 = nIdx * L1TileShape::N;
                uint32_t mActual = (params.m - m0 < L1TileShape::M) ? (params.m - m0) : L1TileShape::M;
                uint32_t nActual = (params.nHalf - n0 < L1TileShape::N) ? (params.nHalf - n0)
                                                                        : L1TileShape::N;
                GemmCoord actualShape{mActual, nActual, params.k};

                // gate half: columns [n0, n0 + nActual)
                MatrixCoord offA{m0, 0};
                MatrixCoord offBGate{0, n0};
                MatrixCoord offCGate{m0, n0};
                blockMmad(gmA[layoutA.GetOffset(offA)], layoutA, gmB[layoutB.GetOffset(offBGate)],
                          layoutB, gmC[layoutC.GetOffset(offCGate)], layoutC, actualShape);

                // up half: columns [I + n0, I + n0 + nActual)
                MatrixCoord offBUp{0, params.nHalf + n0};
                MatrixCoord offCUp{m0, params.nHalf + n0};
                blockMmad(gmA[layoutA.GetOffset(offA)], layoutA, gmB[layoutB.GetOffset(offBUp)],
                          layoutB, gmC[layoutC.GetOffset(offCUp)], layoutC, actualShape);

                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            }
        } else {
            // epilogue-only rung: the AIC contributes nothing but the flag
            // protocol (same flag cadence as the real kernel).
            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
                 loopIdx += AscendC::GetBlockNum()) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ---------------------------------------------------------------- AIV ---
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(F3Params const &params)
    {
        uint32_t subNum = AscendC::GetSubBlockNum();
        uint32_t subIdx = AscendC::GetSubBlockIdx();
        uint32_t subRowOff = subIdx * SUB_ROWS;
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / subNum;
        uint32_t aicoreNum = AscendC::GetBlockNum();

        uint32_t numMBlocks = CeilDiv(params.m, L1TileShape::M);
        uint32_t numNBlocks = CeilDiv(params.nHalf, L1TileShape::N);
        uint32_t coreLoops = numMBlocks * numNBlocks;

        if constexpr (MODE != MODE_GEMM_ONLY) {
            AscendC::GlobalTensor<ElementC> gmC;
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
            AscendC::GlobalTensor<ElementD> gmD;
            gmD.SetGlobalBuffer((__gm__ ElementD *)params.ptrD);
            layout::RowMajor layoutC(params.m, 2 * params.nHalf);
            layout::RowMajor layoutD(params.m, params.nHalf);

            // UB per subcore (single stage): gate tile + up tile, bf16.
            AscendC::LocalTensor<ElementC> ubGate = ub.GetBufferByByte<ElementC>(0);
            AscendC::LocalTensor<ElementC> ubUp = ub.GetBufferByByte<ElementC>(SUB_ROWS * TILE_N *
                                                                              sizeof(ElementC));

            CopyGm2Ub copyIn;
            CopyUb2Gm copyOut;

            // First iteration's "UB is free for MTE2" token.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
                uint32_t nIdx = loopIdx / numMBlocks;
                uint32_t mIdx = loopIdx - nIdx * numMBlocks;
                uint32_t m0 = mIdx * L1TileShape::M;
                uint32_t n0 = nIdx * L1TileShape::N;
                uint32_t mActual = (params.m - m0 < L1TileShape::M) ? (params.m - m0)
                                                                    : L1TileShape::M;
                uint32_t nActual = (params.nHalf - n0 < L1TileShape::N) ? (params.nHalf - n0)
                                                                        : L1TileShape::N;
                // The AIC's two fixpipe stores for this (m0, n0) are visible now.
                // The wait is issued on the MTE2 consumer side; the reverse
                // credit is tagged PIPE_MTE3 (catlass precedent).
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

                if (subRowOff < mActual) {
                    uint32_t rAct = (mActual - subRowOff < SUB_ROWS) ? (mActual - subRowOff)
                                                                     : SUB_ROWS;
                    uint32_t r0 = m0 + subRowOff;
                    layout::RowMajor lUb(rAct, nActual, TILE_N);  // packed UB tile
                    layout::RowMajor lWin(rAct, nActual, 2 * params.nHalf);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    copyIn(ubGate, gmC[layoutC.GetOffset(MatrixCoord{r0, n0})], lUb, lWin);
                    if constexpr (MODE != MODE_FLOOR_GATE_ONLY) {
                        copyIn(ubUp, gmC[layoutC.GetOffset(MatrixCoord{r0, params.nHalf + n0})],
                               lUb, lWin);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    if constexpr (MODE == MODE_SWIGLU) {
                        // Step 2 hook: D = SiLU(gate) * up (design.md 1.1 chain).
                        // Not implemented in the Step-1 binary; see design.md 9.
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    layout::RowMajor lD(rAct, nActual, params.nHalf);
                    if constexpr (MODE == MODE_FLOOR_UP) {
                        copyOut(gmD[layoutD.GetOffset(MatrixCoord{r0, n0})], ubUp, lD, lUb);
                    } else {
                        copyOut(gmD[layoutD.GetOffset(MatrixCoord{r0, n0})], ubGate, lD, lUb);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        } else {
            // gemm-only rung: consume the flags, move nothing.
            for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE,
                                                       RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
    Arch::LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> &ub = resource.ubBuf;
};

}  // namespace

CATLASS_GLOBAL void f3_gateup_epilogue(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GM_ADDR d,
                                       uint32_t m, uint32_t nHalf, uint32_t k, uint32_t mode)
{
    F3Params params{a, b, workspace, d, m, nHalf, k};
    if (mode == MODE_FLOOR) {
        F3GateUpKernel<MODE_FLOOR> kernel;
        kernel(params);
    } else if (mode == MODE_GEMM_ONLY) {
        F3GateUpKernel<MODE_GEMM_ONLY> kernel;
        kernel(params);
    } else if (mode == MODE_EPILOGUE_ONLY) {
        F3GateUpKernel<MODE_EPILOGUE_ONLY> kernel;
        kernel(params);
    } else if (mode == MODE_SWIGLU) {
        F3GateUpKernel<MODE_SWIGLU> kernel;
        kernel(params);
    } else if (mode == MODE_FLOOR_UP) {
        F3GateUpKernel<MODE_FLOOR_UP> kernel;
        kernel(params);
    } else if (mode == MODE_FLOOR_GATE_ONLY) {
        F3GateUpKernel<MODE_FLOOR_GATE_ONLY> kernel;
        kernel(params);
    }
}
