#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Offline parser: profiler capture dir -> per-target device time + pipe breakdown.

Runs in a plain CPU process (no torch_npu import), so the known in-process
analyse() segfault of this box cannot touch it.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

PIPES = ["aiv_time(us)", "aiv_vec_time(us)", "aiv_scalar_time(us)",
         "aiv_mte2_time(us)", "aiv_mte3_time(us)", "aiv_icache_miss_rate"]
RATIOS = ["aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]


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
    ap.add_argument("--prof-dir", default="/tmp/f2bw/prof")
    ap.add_argument("--out", default="/tmp/f2bw/prof.json")
    ap.add_argument("--only-name", default="")  # "" = all rows; the analyser then
    # reports the per-capture dominant AI_VECTOR_CORE kernel
    args = ap.parse_args()

    res = {}
    for cap in sorted(glob.glob(os.path.join(args.prof_dir, "*"))):
        tag = os.path.basename(cap)
        csvs = glob.glob(os.path.join(cap, "**", "ASCEND_PROFILER_OUTPUT",
                                      "kernel_details.csv"), recursive=True)
        if not csvs:
            res[tag] = {"error": "no kernel_details.csv", "files": sorted(
                os.path.relpath(p, cap) for p in glob.glob(os.path.join(cap, "**", "*"), recursive=True))[:20]}
            continue
        allrows = list(rows_of(csvs[0]))
        rowset = [r for r in allrows if args.only_name in (r.get("Name") or "")]
        if not args.only_name:
            # pick the AI_VECTOR_CORE kernel with the largest total duration: the
            # capture is one op repeated N times, so that is our kernel, and any
            # helper kernels of the same launch show up in 'other_rows' below.
            agg = {}
            for r in allrows:
                nm = r.get("Name") or ""
                d = num(r.get("Duration(us)")) or 0.0
                agg[nm] = agg.get(nm, 0.0) + d
            keep = [r for r in allrows if (r.get("Name") or "") == max(agg, key=agg.get)]
            rest = {k: round(v, 2) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])
                    if k != max(agg, key=agg.get)}
            rowset = keep
            extra = {"other_kernels_total_us": rest}
        if not rowset:
            names = sorted({(r.get("Name") or "") for r in rows_of(csvs[0])})
            res[tag] = {"error": "kernel name not found", "names": names[:30]}
            continue
        dur = [num(r.get("Duration(us)")) for r in rowset]
        dur = [d for d in dur if d is not None]
        elem = {"capture": tag, "count": len(rowset), **(extra if not args.only_name else {}),
                "duration_us": {"median": round(statistics.median(dur), 3),
                                "min": round(min(dur), 3), "max": round(max(dur), 3)},
                "name": rowset[0].get("Name"), "type": rowset[0].get("Type"),
                "accelerator": rowset[0].get("Accelerator Core"),
                "block_num": rowset[0].get("Block Num")}
        for col in PIPES + RATIOS + ["aicore_time(us)", "aiv_total_cycles"]:
            vals = [num(r.get(col)) for r in rowset]
            vals = [v for v in vals if v is not None]
            if vals:
                elem[col] = round(statistics.median(vals), 4)
        res[tag] = elem

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for tag, e in res.items():
        if "error" in e:
            print(f"{tag:28s} ERROR {e['error']} {str(e.get('names'))[:120]}")
            continue
        print(f"{tag:28s} n={e['count']:<4} dur={e['duration_us']['median']:9.3f}us  "
              f"blocks={e['block_num']:<4} mte2={e.get('aiv_mte2_ratio')} "
              f"vec={e.get('aiv_vec_ratio')} mte3={e.get('aiv_mte3_ratio')} "
              f"scalar={e.get('aiv_scalar_ratio')} aiv={e.get('aiv_time(us)')}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
