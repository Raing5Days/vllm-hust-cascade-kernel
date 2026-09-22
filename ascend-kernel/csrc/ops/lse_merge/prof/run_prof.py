# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""lse_merge 上板性能采集（pipe 级）。

补上该算子此前完全缺失的性能身份：kernel_details.csv 里的
aiv_vec_ratio / aiv_scalar_ratio / aiv_mte2_ratio / aiv_mte3_ratio。

范式照抄 profiles/.../f2-kernel/raw/d3/probe.py 的 phase_p：
  - 每个形态独立进程捕获（三个形态共用 kernel 名 lse_merge，同一进程会混淆）
  - aic_metrics=AiCMetrics.PipeUtilization 才产出 pipe 分列
  - p.stop() 后 os._exit(0)：规避本机 torch_npu.profiler 进程内 analyse 段 segfault
    （AGENTS.md 陷阱 7），kernel_details.csv 由 stop() 内的导出自动落盘。

用法（必须在 /tmp 等非仓库根目录下跑，见 AGENTS.md 陷阱 1）：
  python run_prof.py --mode wall
  python run_prof.py --mode prof --target tier0 --prof-dir /tmp/lsemerge/prof
"""

import argparse
import json
import os
import statistics
import sys
import time

import ascend_kernel  # noqa: F401  import 即注册 torch.ops.npu.lse_merge
import torch
import torch_npu  # noqa: F401  设备后端注册

WARMUP = 10
PROF_ITERS = 50
WALL_ITERS = 200

T, H, D = 64, 40, 128   # 生产 shape（14B decode）

# name -> (o1_dtype, out_code, padded_lse)
TARGETS = {
    "tier0_stride1": ("bf16", 0, False),   # 既有 Tier0 热路径（bf16 o1 / bf16 out / compact lse）
    "tier1_fp32out": ("fp32", 2, False),   # Tier1 probe 形态（fp32 o1 / fp32 out）
    "tier0_stride8": ("bf16", 0, True),    # 生产集成形态（catlass padded lse，走 CompactLse）
}


def make_inputs(o1_dtype, padded, seed=0):
    dt = torch.float32 if o1_dtype == "fp32" else torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(seed)
    o1 = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(dt).to("npu")
    o2 = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to("npu")
    l1 = torch.randn(T, H, generator=g, dtype=torch.float32).to("npu")
    l2 = torch.randn(T, H, generator=g, dtype=torch.float32).to("npu")
    l1c, l2c = l1, l2
    if padded:   # catlass FAInferBf16Fp32Out: 每 (t,h) 行 8 个复制 fp32
        l1c = l1.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
        l2c = l2.unsqueeze(-1).expand(T, H, 8).contiguous().reshape(T, H * 8)
    return o1, o2, l1c, l2c


def mk_fn(o1, o2, l1, l2, out_code):
    def fn():
        return torch.ops.npu.lse_merge(o1, o2, l1, l2, out_code)
    return fn


def phase_wall(args):
    res = {}
    for name, (o1_dtype, out_code, padded) in TARGETS.items():
        o1, o2, l1, l2 = make_inputs(o1_dtype, padded)
        fn = mk_fn(o1, o2, l1, l2, out_code)
        for _ in range(WARMUP):
            fn()
        torch.npu.synchronize()
        ts = []
        for _ in range(WALL_ITERS):
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        gm = (T * H * D * (4 if o1_dtype == "fp32" else 2)      # o1
              + T * H * D * 2                                    # o2
              + T * H * (8 if padded else 1) * 4 * 2)            # lse1+lse2
        res[name] = {"wall_us": {"median": round(statistics.median(ts), 3),
                                 "min": round(min(ts), 3),
                                 "max": round(max(ts), 3)},
                     "gm_read_bytes": gm}
        print(f"WALL {name:16s} median={res[name]['wall_us']['median']:8.3f}us "
              f"min={res[name]['wall_us']['min']:8.3f}us", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("written", args.out, flush=True)
    return 0


def phase_prof(args):
    from torch_npu.profiler import (
        AiCMetrics,
        ExportType,
        ProfilerActivity,
        ProfilerLevel,
        _ExperimentalConfig,
        profile,
        tensorboard_trace_handler,
    )
    o1_dtype, out_code, padded = TARGETS[args.target]
    o1, o2, l1, l2 = make_inputs(o1_dtype, padded)
    fn = mk_fn(o1, o2, l1, l2, out_code)
    out = os.path.join(args.prof_dir, args.target)
    os.makedirs(out, exist_ok=True)
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    cfg = _ExperimentalConfig(export_type=ExportType.Text,
                              profiler_level=ProfilerLevel.Level1,
                              msprof_tx=False, aic_metrics=AiCMetrics.PipeUtilization,
                              l2_cache=False, op_attr=False, data_simplification=True,
                              record_op_args=False, gc_detect_threshold=None)
    p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                with_stack=False, with_modules=False, experimental_config=cfg,
                on_trace_ready=tensorboard_trace_handler(out, worker_name=args.target))
    p.start()
    for _ in range(PROF_ITERS):
        fn()
    torch.npu.synchronize()
    p.stop()
    print(f"PROF_STOP_OK {args.target}", flush=True)
    time.sleep(1.0)
    sys.stdout.flush()
    os._exit(0)   # 跳过进程内 analyse（本机 segfault）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["wall", "prof"])
    ap.add_argument("--target", default="tier0_stride1", choices=list(TARGETS))
    ap.add_argument("--prof-dir", default="/tmp/lsemerge/prof")
    ap.add_argument("--out", default="/tmp/lsemerge/wall.json")
    args = ap.parse_args()
    return phase_wall(args) if args.mode == "wall" else phase_prof(args)


if __name__ == "__main__":
    sys.exit(main())
