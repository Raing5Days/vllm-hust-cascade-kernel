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

"""直接观测 Brcb 的真实产物（配合 LSE_FAST_ROWSCALE==2 的诊断构建），并验证
"一个 src 元素 = 一行" 的标定（dstRepStride = dim/16）在 dim ∈ {16,64,128} 上是否成立。

构建侧把行级折算后的权重（s1[r] = w1·s⁻¹，行行互异）经 Brcb 铺进输出张量，
本脚本第 r 行的所有元素都应等于 s1[r]。

输入取 lse1=0、lse2[r] 互异，于是 s1[r] = 1/(1+exp(lse2[r])) 可解析反解。
"""

import sys

import ascend_kernel  # noqa: F401
import torch
import torch_npu  # noqa: F401

SHAPES = [("D=128", 4, 8, 128), ("D=64", 4, 8, 64), ("D=16", 4, 8, 16)]


def check(T, H, D):
    rows = T * H
    lse2_vals = [-(r + 1) * 0.1 for r in range(rows)]
    lse2 = torch.tensor(lse2_vals, dtype=torch.float32, device="npu").reshape(T, H)
    o1 = torch.ones(T, H, D, dtype=torch.float32, device="npu")
    o2 = torch.zeros(T, H, D, dtype=torch.bfloat16, device="npu")
    l1 = torch.zeros(T, H, dtype=torch.float32, device="npu")

    out = torch.ops.npu.lse_merge(o1, o2, l1, lse2, 2)
    torch.npu.synchronize()
    dump = out.cpu().float().reshape(rows, D)

    exp_l2 = torch.exp(torch.tensor(lse2_vals, dtype=torch.float64))
    want = 1.0 / (1.0 + exp_l2)

    spread = (dump.max(dim=1).values.double() - dump.min(dim=1).values.double()).abs()
    diff = (dump[:, 0].double() - want).abs()
    n_ok = int((diff <= 1e-6).sum())
    ok = bool((diff <= 1e-6).all() and (spread <= 1e-9).all())
    return ok, n_ok, rows, spread, diff, dump, want


def main():
    all_ok = True
    for name, T, H, D in SHAPES:
        ok, n_ok, rows, spread, diff, dump, want = check(T, H, D)
        all_ok &= ok
        bad_rows = (diff > 1e-6).nonzero().flatten().tolist()
        print(f"{name:6s} dim={D:3d} rows={rows:3d}  映射正确={n_ok}/{rows}  "
              f"行内全同={bool((spread <= 1e-9).all())}  "
              f"不等期望的行={bad_rows[:6]}{' ...' if len(bad_rows) > 6 else ''}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            for r in bad_rows[:4]:
                print(f"      r={r:2d} 实测={dump[r,0].item():.6f} 期望={want[r].item():.6f}")
    print(f"\n标定（dstRepStride=dim/16）整体判定: {'全部成立' if all_ok else '有 shape 不成立'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
