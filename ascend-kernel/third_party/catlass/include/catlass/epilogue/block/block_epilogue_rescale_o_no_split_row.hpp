/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP
 #define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/epilogue/dispatch_policy.hpp"
 #include "catlass/epilogue/tile/tile_copy.hpp"
 #include "catlass/gemm_coord.hpp"
 #include "catlass/matrix_coord.hpp"
 
 namespace Catlass::Epilogue::Block {
 
 template <
     class OutputType_,
     class InputType_,
     class UpdateType_>
 class BlockEpilogue<
     EpilogueAtlasA2RescaleO,
     OutputType_,
     InputType_,
     UpdateType_>
 {
 public:
     // Type aliases
     using DispatchPolicy = EpilogueAtlasA2RescaleO;
     using ArchTag = typename DispatchPolicy::ArchTag;
 
     using ElementOutput = typename OutputType_::Element;
     using ElementInput = typename InputType_::Element;
     using ElementUpdate = typename UpdateType_::Element;
 
     using LayoutOutput = typename OutputType_::Layout;
     using LayoutInput = typename InputType_::Layout;
     using LayoutUpdate = typename UpdateType_::Layout;
 
     static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
     static constexpr uint32_t BLOCK_SIZE = 16;
     static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
     static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
     static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
     static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
     static constexpr uint32_t MULTIPLIER = 2;
     static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
     static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
     static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
     static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
     static constexpr uint32_t HALF_DM_UB_SIZE = 64;
     static constexpr uint32_t HALF_LL_UB_SIZE = 256;
     static constexpr uint32_t VECTOR_SIZE = 128;
     static constexpr uint32_t NUM4 = 4;
     static constexpr uint32_t MAX_UB_O_ELEM_NUM = 4096;
     static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 128;
     static constexpr uint32_t ELEMENT_BYTES = sizeof(ElementOutput);
     // LSE probe scratch (BISECT-E): placed in the mask UB region, which is untouched when
     // maskType == 0. NOTE: the dm region [10*UB + 13KB, ...) is over-written by softmax/rescale
     // ping-pong (3 slots x 128 rows = 1536B > the 1024B reserved), so it must not be reused.
     // values: fp32 row results; staging: 2 ping-pong banks x 32 rows x 8-float 32B-aligned slots.
     static constexpr uint32_t LSE_CHUNK_ROWS = 32;
     static constexpr uint32_t LSE_VAL_UB_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
     static constexpr uint32_t LSE_STAGE_UB_OFFSET = LSE_VAL_UB_OFFSET + 512;
     static constexpr uint32_t LSE_STAGE_BANK_BYTES = LSE_CHUNK_ROWS * FLOAT_BLOCK_SIZE * ELEMENT_BYTES; // 1KB
 
     CATLASS_DEVICE
     BlockEpilogue(Arch::Resource<ArchTag> &resource)
     {
         // Allocate UB space
         constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
         constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;
         constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
         
         constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
         constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;
 
         loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
         dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
         glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
         tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
         goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
         goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
         hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
         lseValUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_VAL_UB_OFFSET);
         lseStageUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_STAGE_UB_OFFSET);
     }
 
     CATLASS_DEVICE
     ~BlockEpilogue()
     {
     }
 
     CATLASS_DEVICE
     void SetMask(int32_t len)
     {
         uint64_t mask = 0;
         uint64_t one = 1;
         uint64_t temp = len % FLOAT_VECTOR_SIZE;
         for (int64_t i = 0; i < temp; i++) {
             mask |= one << i;
         }
 
         if (len == VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         } else if (len >= FLOAT_VECTOR_SIZE) {
             AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
         } else {
             AscendC::SetVectorMask<int8_t>(0x0, mask);
         }
     }
 
     CATLASS_DEVICE
     void CopyOToGm(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         uint32_t curRowNum, uint32_t qSBlockSize, uint32_t embed,
         uint32_t embedRound, uint32_t qNThisSubBlock, uint32_t oHiddenSize)
     {
         if (qNThisSubBlock == 0) {
             AscendC::DataCopyPad(
                 gOutput,
                 goUbTensor16,
                 AscendC::DataCopyExtParams(curRowNum, embed * ELEMENT_BYTES, 0,
                                            (oHiddenSize - embed) * ELEMENT_BYTES, 0));
         } else {
             for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                 AscendC::DataCopyPad(
                     gOutput[qNIdx * embed],
                     goUbTensor16[qNIdx * embedRound * qSBlockSize],
                     AscendC::DataCopyExtParams(qSBlockSize, embed * ELEMENT_BYTES, 0,
                                                (oHiddenSize - embed) * ELEMENT_BYTES, 0));
             }
         }
     }
 
     CATLASS_DEVICE
     void SubCoreCompute(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         AscendC::GlobalTensor<ElementInput> gInput,
         const LayoutOutput &layoutOutput,
         const LayoutInput &layoutInput,
         uint32_t qNThisSubBlock,
         uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
         bool lseEnable = false, AscendC::GlobalTensor<float> gLse = AscendC::GlobalTensor<float>(),
         uint32_t lseNumHeads = 0, uint32_t lseTokenBase = 0, uint32_t lseHeadBase = 0,
         uint32_t lseQnBlockSize = 1, uint32_t lseQsBlockSize = 1, uint32_t lseRowOffset = 0)
     {
         uint32_t curRowNum = layoutInput.shape(0);
         uint32_t embed = layoutInput.shape(1);
         uint32_t embedRound = layoutInput.stride(0);
         uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
         uint32_t qSBlockSize = layoutOutput.shape(0);
         uint32_t oHiddenSize = layoutOutput.shape(1);
         uint32_t dmUbOffsetCurStackTile =
             curStackTileMod * MAX_ROW_NUM_SUB_CORE;
         
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
         if (!isFirstStackTile) {
             AscendC::DataCopy(loUbTensor, gInput,
                               AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                           dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(), curRowNumRound / FLOAT_BLOCK_SIZE,
                           AscendC::BrcbRepeatParams(1, 8));
             AscendC::PipeBarrier<PIPE_V>();
             // *** go = go * dm_block
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                 AscendC::Mul<float, false>(goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                                            tvUbTensor, (uint64_t)0,
                                            curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
             }
             if (embed % FLOAT_VECTOR_SIZE > 0) {
                 SetMask(embed % FLOAT_VECTOR_SIZE);
                 AscendC::Mul<float, false>(goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            tvUbTensor,
                                            (uint64_t)0, curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
                 AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             }
             AscendC::PipeBarrier<PIPE_V>();
             // *** go = lo + go
             AscendC::Add<float, false>(goUbTensor32, goUbTensor32,
                                        loUbTensor, (uint64_t)0,
                                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
             AscendC::PipeBarrier<PIPE_V>();
         } else {
             // *** go = lo
             AscendC::DataCopy(goUbTensor32, gInput,
                               AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
         }
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
 
         if (isLastStackTile) {
             // *** gl_block = expand_to_block(gl), 存放于 tv
             AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                           glUbTensor.ReinterpretCast<uint32_t>(),
                           curRowNumRound / FLOAT_BLOCK_SIZE,
                           AscendC::BrcbRepeatParams(1, 8));
             AscendC::PipeBarrier<PIPE_V>();
             // *** go = go / gl_block
             AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                 AscendC::Div<float, false>(goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                                            tvUbTensor, (uint64_t)0,
                                            curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
             }
             if (embed % FLOAT_VECTOR_SIZE > 0) {
                 SetMask(embed % FLOAT_VECTOR_SIZE);
                 AscendC::Div<float, false>(goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            tvUbTensor,
                                            (uint64_t)0, curRowNum,
                                            AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                        embedRound / FLOAT_BLOCK_SIZE, 1));
                 AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
             }
             AscendC::PipeBarrier<PIPE_V>();
 
             // *** go = castfp32to16(go); skipped for fp32-out (ElementOutput == float)
             if constexpr (!std::is_same_v<ElementOutput, float>) {
                 if (std::is_same<ElementOutput, bfloat16_t>::value) {
                     AscendC::Cast<ElementOutput, float, false>(
                         goUbTensor16, goUbTensor32,
                         AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                         (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                         AscendC::UnaryRepeatParams(1, 1, 4, 8));
                 } else {
                     AscendC::Cast<ElementOutput, float, false>(
                         goUbTensor16, goUbTensor32,
                         AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                         (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                         AscendC::UnaryRepeatParams(1, 1, 4, 8));
                 }
             }
             AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 
             // ***move O to GM
             CopyOToGm(
                 gOutput, curRowNum, qSBlockSize, embed,
                 embedRound, qNThisSubBlock, oHiddenSize);

             // *** lse = ln(gl) + hm, scatter to GM (probe layout [numTokens, numHeads] fp32)
             if (lseEnable) {
                 ComputeLseAndWrite(gLse, curRowNum, curRowNumRound, lseNumHeads, lseTokenBase,
                                    lseHeadBase, lseQnBlockSize, lseQsBlockSize, lseRowOffset);
             }
         }
     }
 
     CATLASS_DEVICE
     void operator()(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         AscendC::GlobalTensor<ElementInput> gInput,
         const LayoutOutput &layoutOutput,
         const LayoutInput &layoutInput,
         GemmCoord actualBlockShape,
         uint32_t qSBlockSize, uint32_t qNBlockSize,
         uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
     {
         Invoke(gOutput, gInput, layoutOutput, layoutInput, actualBlockShape, qSBlockSize, qNBlockSize,
                isFirstStackTile, isLastStackTile, curStackTileMod);
     }

     // LSE-out overload (probe): lse row layout [numTokens, numHeads] fp32,
     // row (token, head) -> gm offset token * numHeads + head.
     CATLASS_DEVICE
     void operator()(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         AscendC::GlobalTensor<ElementInput> gInput,
         const LayoutOutput &layoutOutput,
         const LayoutInput &layoutInput,
         GemmCoord actualBlockShape,
         uint32_t qSBlockSize, uint32_t qNBlockSize,
         uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
         AscendC::GlobalTensor<float> gLse, uint32_t numHeads, uint32_t tokenBase, uint32_t headBase)
     {
         Invoke(gOutput, gInput, layoutOutput, layoutInput, actualBlockShape, qSBlockSize, qNBlockSize,
                isFirstStackTile, isLastStackTile, curStackTileMod,
                true, gLse, numHeads, tokenBase, headBase);
     }

 private:
     CATLASS_DEVICE
     void Invoke(
         AscendC::GlobalTensor<ElementOutput> gOutput,
         AscendC::GlobalTensor<ElementInput> gInput,
         const LayoutOutput &layoutOutput,
         const LayoutInput &layoutInput,
         GemmCoord actualBlockShape,
         uint32_t qSBlockSize, uint32_t qNBlockSize,
         uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
         bool lseEnable = false, AscendC::GlobalTensor<float> gLse = AscendC::GlobalTensor<float>(),
         uint32_t lseNumHeads = 0, uint32_t lseTokenBase = 0, uint32_t lseHeadBase = 0)
     {
         uint32_t rowNum = actualBlockShape.m();
         uint32_t embed = actualBlockShape.n();
 
         uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
         uint32_t subBlockNum = AscendC::GetSubBlockNum();
 
         uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
         uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
             0: (subBlockIdx == 1) ?
             (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
         uint32_t inRowSplitSubBlock = (qNBlockSize == 1) ?
             (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
         uint32_t inRowActualThisSubBlock = (subBlockIdx == 1) ?
             (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
         uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
         uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1) ?
             inRowOffsetThisSubBlock : 0;
         uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1) ?
             0 : subBlockIdx * qNSplitSubBlock * embed;
 
         if (inRowActualThisSubBlock > 0) {
             int64_t offsetOutput =
                 layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));
             auto gOutputThisSubBlock = gOutput[offsetOutput];
             auto layoutOutputThisSubBlock = layoutOutput;
 
             int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(inRowOffsetThisSubBlock, 0));
             auto gInputThisSubBlock = gInput[offsetInput];
             auto layoutInputThisSubBlock = layoutInput.GetTileLayout(MatrixCoord(inRowActualThisSubBlock, embed));
             SubCoreCompute(
                 gOutputThisSubBlock,
                 gInputThisSubBlock,
                 layoutOutputThisSubBlock,
                 layoutInputThisSubBlock,
                 qNThisSubBlock, isFirstStackTile, isLastStackTile, curStackTileMod,
                 lseEnable, gLse, lseNumHeads, lseTokenBase, lseHeadBase,
                 qNBlockSize, qSBlockSize, inRowOffsetThisSubBlock);
         }
     }

     // lse = ln(gl) + hm (both fp32, maintained by the online softmax epilogue in shared UB).
     // Scatter per row. The O tile rows are HEAD-major (qSBlockSize tokens outer per head,
     // verified against the O write path and fp64 reference for q_len>1): taskRow r ->
     // headLocal = r/qsBlockSize, tokenLocal = r%qsBlockSize. For q_len=1 (decode stage-1)
     // qsBlockSize==1 and this is identical to the original token-major probe mapping.
     // GM layout is row-padded: each (token, head) row occupies 32B (8 copies of the fp32 value,
     // consumer reads element 0). This keeps every MTE3 write 32B-aligned: 4B DataCopyPad hangs
     // the kernel (measured on 910B3, CANN 8.5.1).
     CATLASS_DEVICE
     void ComputeLseAndWrite(
         AscendC::GlobalTensor<float> gLse, uint32_t curRowNum, uint32_t curRowNumRound,
         uint32_t numHeads, uint32_t tokenBase, uint32_t headBase, uint32_t qnBlockSize,
         uint32_t qsBlockSize, uint32_t rowOffset)
     {
         AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
         // lseVal = ln(gl)
         AscendC::Ln<float, false>(lseValUbTensor, glUbTensor, (uint64_t)0,
             (curRowNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
             AscendC::UnaryRepeatParams(1, 1, 8, 8));
         AscendC::PipeBarrier<PIPE_V>();
         // lseVal += hm
         AscendC::Add<float, false>(lseValUbTensor, lseValUbTensor, hmUbTensor, (uint64_t)0,
             (curRowNumRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
             AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
         AscendC::PipeBarrier<PIPE_V>();
         // staging bank ping-pong, ONE BANK PER 32-ROW CHUNK, alternating continuously
         // (within a task and across tasks). The V_MTE3 Set/Wait pair below only orders the
         // V pipe - it cannot track MTE3 completion (EVENT_ID2 proved unusable for
         // V_MTE3/MTE3_V on 910B3: stable hang; only EVENT_ID0/1 exist for these directions),
         // so chunk c's Brcb must not touch the bank chunk c-1's async MTE3 copies read.
         // History (plan-20260903-lse-flatten-fix P0): the original probe version picked the
         // bank once per isLast tile (`lseParity`), so any sub-block with rows > 32 (2+ chunks
         // in one task) hit chunk 1's Brcb overwriting the bank while chunk 0's MTE3 copies
         // were in flight - a timing-dependent scramble of the flattened q_len>1 form
         // (correct iff sub-block rows <= 32). Same-bank reuse is now >= 2 chunks apart
         // (>= one full 1 KB staging drain), orders of magnitude beyond the drain time.
         // NOTE: lseStageUbTensor is already based at LSE_STAGE_UB_OFFSET; bank index is relative.
         for (uint32_t chunkBase = 0; chunkBase < curRowNum; chunkBase += LSE_CHUNK_ROWS) {
             uint32_t chunkRows = (curRowNum - chunkBase < LSE_CHUNK_ROWS) ? (curRowNum - chunkBase)
                                                                           : LSE_CHUNK_ROWS;
             uint32_t chunkRowsRound = (chunkRows + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
             uint32_t stageBaseElems =
                 (lseStageSeq++ & 1u) ? (LSE_STAGE_BANK_BYTES / ELEMENT_BYTES) : 0;
             // stage each row value into an 8-float (32B) slot for aligned GM writes
             AscendC::Brcb(lseStageUbTensor[stageBaseElems].ReinterpretCast<uint32_t>(),
                           lseValUbTensor[chunkBase].ReinterpretCast<uint32_t>(),
                           chunkRowsRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
             AscendC::PipeBarrier<PIPE_V>();
             // reuse the proven EVENT_ID0 (V_MTE3) pair: one Set per produced bank, one Wait per copy
             AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
             AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
             for (uint32_t r = 0; r < chunkRows; r++) {
                 uint32_t taskRow = rowOffset + chunkBase + r;
                 uint32_t headLocal = taskRow / qsBlockSize;
                 uint32_t tokenLocal = taskRow - headLocal * qsBlockSize;
                 uint64_t rowIdx = (uint64_t)(tokenBase + tokenLocal) * numHeads + (headBase + headLocal);
                 AscendC::DataCopy(gLse[rowIdx * FLOAT_BLOCK_SIZE],
                                   lseStageUbTensor[stageBaseElems + r * FLOAT_BLOCK_SIZE],
                                   FLOAT_BLOCK_SIZE);
             }
         }
     }

 private:
     AscendC::LocalTensor<float> loUbTensor;
     AscendC::LocalTensor<float> dmUbTensor;
     AscendC::LocalTensor<float> hmUbTensor;
     AscendC::LocalTensor<float> glUbTensor;
     AscendC::LocalTensor<float> tvUbTensor;
     AscendC::LocalTensor<ElementOutput> goUbTensor16;
     AscendC::LocalTensor<float> goUbTensor32;
     AscendC::LocalTensor<float> lseValUbTensor;
     AscendC::LocalTensor<float> lseStageUbTensor;
     // continuous staging-bank sequence: one lse chunk per increment (see ComputeLseAndWrite)
     uint32_t lseStageSeq = 0;
 };
 }
 
 #endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP