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

"""逐元素 bit 比对：当前装机的 lse_merge 输出 vs golden 锚点。

判据：同一 (mode, shape) 用例的输出张量字节的 sha256 必须完全一致。
输入构造与 golden 生成脚本共用同一 seed 派生规则（见 lse_merge_cases / test 用例），
保证比对的是"同输入下的输出"。

用法：
  python verify_golden.py --golden-dir <dir> [--expect LSE_ROW_SRC值]
退出码 0 = 全等；1 = 有不一致。
"""

import argparse
import hashlib
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))

import ascend_kernel  # noqa: F401  import 即注册 torch.ops.npu.lse_merge
import torch
import torch_npu  # noqa: F401  设备后端注册
from lse_merge_cases import MODES, SHAPES, make_inputs


def out_sha256(t):
    b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    man_path = os.path.join(args.golden_dir, "MANIFEST.json")
    with open(man_path) as f:
        man_raw = json.load(f)
    man = man_raw.get("cases", man_raw)

    rows = []
    for mode, (o1_dtype, _o2_dtype, out_code, padded, *_rest) in MODES.items():
        for cat, (T, H, D) in SHAPES:
            key = f"{mode}__{cat}"
            if key not in man:
                rows.append({"case": key, "status": "MISSING_IN_MANIFEST"})
                continue
            seed = zlib.crc32(f"{cat}/{mode}".encode()) % 2**31
            o1, o2, l1c, l2c, _l1, _l2 = make_inputs(T, H, D, o1_dtype, padded, seed=seed)
            out = torch.ops.npu.lse_merge(o1, o2, l1c, l2c, out_code)
            torch.npu.synchronize()
            got = out_sha256(out)
            exp = man[key]["sha256"]
            rows.append({"case": key, "status": "OK" if got == exp else "MISMATCH",
                         "got": got[:16], "exp": exp[:16],
                         "dtype_match": str(out.dtype).replace("torch.", "") == man[key]["dtype"],
                         "shape_match": list(out.shape) == list(man[key]["shape"])})

    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_bad = len(rows) - n_ok
    for r in rows:
        if r["status"] != "OK":
            print(f"  FAIL {r['case']:34s} {r['status']} got={r.get('got')} exp={r.get('exp')} "
                  f"dtype={r.get('dtype_match')} shape={r.get('shape_match')}")
    print(f"BIT_COMPARE {n_ok}/{len(rows)} identical, {n_bad} mismatch")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "ok": n_ok, "bad": n_bad}, f, indent=1)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
