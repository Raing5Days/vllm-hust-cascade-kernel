# Copyright (c) 2026 Huawei Technologies Co., Ltd
# BSD 3-Clause License.
#
# S7 perf driver: loop torch.ops.npu.fa_fp32_stage1 on the probe shape
# (B=4, q=1, kv=4096, H=40, KVH=8, D=128). Run under msprof for Task Duration,
# or standalone for host-wall timing. Usage: python run_perf_driver.py [reps]

import sys
import time
import numpy as np
import torch
import torch_npu
import ml_dtypes

DATA = "/vllm-workspace/cascade-merge-op/catlass-example-data"


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    torch.npu.set_device(0)
    import ascend_kernel

    q_seqlens = np.fromfile(f"{DATA}/q_seqlen.bin", dtype=np.int64)
    kv_seqlens = np.fromfile(f"{DATA}/kv_seqlen.bin", dtype=np.int64)
    B = len(q_seqlens)
    T = int(q_seqlens.sum())
    H, KVH, D = 40, 8, 128
    max_kv = int(kv_seqlens.max())
    cols = (max_kv + 127) // 128
    bt = np.fromfile(f"{DATA}/block_table.bin", dtype=np.int32).reshape(B, cols)
    q = np.fromfile(f"{DATA}/q.bin", dtype=ml_dtypes.bfloat16)[: T * H * D].reshape(T, H, D)
    num_blocks = int(bt.max()) + 1
    shape = (num_blocks, 128, KVH, D)
    k = np.fromfile(f"{DATA}/k.bin", dtype=ml_dtypes.bfloat16)[: np.prod(shape)].reshape(shape)
    v = np.fromfile(f"{DATA}/v.bin", dtype=ml_dtypes.bfloat16)[: np.prod(shape)].reshape(shape)

    def np2t(x):
        u16 = np.ascontiguousarray(x).view(np.uint16)
        return torch.from_numpy(u16).view(torch.bfloat16).to("npu")

    args = (np2t(q), np2t(k), np2t(v), torch.from_numpy(bt).to("npu"),
            torch.from_numpy(q_seqlens).to("npu"), torch.from_numpy(kv_seqlens).to("npu"))

    # warmup
    for _ in range(10):
        torch.ops.npu.fa_fp32_stage1(*args)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        torch.ops.npu.fa_fp32_stage1(*args)
    torch.npu.synchronize()
    wall = (time.perf_counter() - t0) / reps
    print(f"fa_fp32_stage1 host wall (incl. launch+tiling D2H): {wall * 1e6:.1f} us/call over {reps} reps")


if __name__ == "__main__":
    main()
