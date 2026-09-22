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

"""离线解析器：profiler 捕获目录 -> lse_merge 器件时间 + pipe 占比。

在纯 CPU 进程运行（不 import torch_npu），规避本机进程内 analyse 段 segfault。
范式照抄 profiles/.../f2-kernel/raw/d3/parse_prof.py。
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

PIPES = ["aiv_time(us)", "aiv_vec_time(us)", "aiv_scalar_time(us)",
         "aiv_mte2_time(us)", "aiv_mte3_time(us)",
         "aiv_vec_total_cflt_ratio", "aiv_icache_miss_rate"]
RATIOS = ["aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]
EXTRA = ["aicore_time(us)", "aiv_total_cycles", "Block Num",
         "aic_mte2_ratio", "aic_scalar_ratio"]


def num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def rows_of(path):
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof-dir", default="/tmp/lsemerge/prof")
    ap.add_argument("--out", default="/tmp/lsemerge/prof.json")
    args = ap.parse_args()

    res = {}
    for cap in sorted(glob.glob(os.path.join(args.prof_dir, "*"))):
        tag = os.path.basename(cap)
        csvs = glob.glob(os.path.join(cap, "**", "ASCEND_PROFILER_OUTPUT",
                                      "kernel_details.csv"), recursive=True)
        if not csvs:
            res[tag] = {"error": "no kernel_details.csv"}
            continue
        allrows = list(rows_of(csvs[0]))
        agg = {}
        for r in allrows:
            nm = r.get("Name") or ""
            agg[nm] = agg.get(nm, 0.0) + (num(r.get("Duration(us)")) or 0.0)
        top = max(agg, key=agg.get)
        rowset = [r for r in allrows if (r.get("Name") or "") == top]
        dur = [d for d in (num(r.get("Duration(us)")) for r in rowset) if d is not None]
        elem = {"capture": tag, "count": len(rowset),
                "other_kernels_total_us": {k: round(v, 2) for k, v in
                                           sorted(agg.items(), key=lambda kv: -kv[1])
                                           if k != top},
                "duration_us": {"median": round(statistics.median(dur), 3),
                                "min": round(min(dur), 3), "max": round(max(dur), 3)},
                "name": rowset[0].get("Name"), "type": rowset[0].get("Type"),
                "accelerator": rowset[0].get("Accelerator Core"),
                "block_num": rowset[0].get("Block Num")}
        for col in PIPES + RATIOS + EXTRA:
            vals = [v for v in (num(r.get(col)) for r in rowset) if v is not None]
            if vals:
                elem[col] = round(statistics.median(vals), 4)
        res[tag] = elem

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    hdr = f"{'capture':<18}{'n':>4}{'dur_med':>9}{'aiv':>8}{'vec':>8}{'vec_r':>8}{'scal':>8}{'sc_r':>7}{'mte2':>8}{'m2_r':>7}{'mte3':>8}"
    print(hdr)
    for tag, e in res.items():
        if "error" in e:
            print(f"{tag:<18} ERROR {e['error']}")
            continue
        print(f"{tag:<18}{e['count']:>4}{e['duration_us']['median']:>9.2f}"
              f"{e.get('aiv_time(us)', 0):>8.2f}{e.get('aiv_vec_time(us)', 0):>8.2f}"
              f"{e.get('aiv_vec_ratio', 0):>8.3f}{e.get('aiv_scalar_time(us)', 0):>8.2f}"
              f"{e.get('aiv_scalar_ratio', 0):>7.3f}{e.get('aiv_mte2_time(us)', 0):>8.2f}"
              f"{e.get('aiv_mte2_ratio', 0):>7.3f}{e.get('aiv_mte3_time(us)', 0):>8.2f}")
        if e.get("other_kernels_total_us"):
            print(f"{'':<18}其他 kernel: {e['other_kernels_total_us']}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
