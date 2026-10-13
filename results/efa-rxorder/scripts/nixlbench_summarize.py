#!/usr/bin/env python3
"""Median nixlbench bandwidth and latency per mode and block size, from nixlbench.sbatch logs.

Usage: nixlbench_summarize.py <log>... [--sizes 4096,65536,...]
"""
import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s")
MODE = re.compile(r"^=== mode=(\S+) rep=(\d+)")


def parse(paths):
    # data[mode][size] -> list of (GB/s, latency us), one per repetition
    data = defaultdict(lambda: defaultdict(list))
    order = []
    for path in paths:
        mode = None
        for line in open(path, errors="replace"):
            m = MODE.match(line)
            if m:
                mode = m.group(1)
                if mode not in order:
                    order.append(mode)
                continue
            r = ROW.match(line)
            if r and mode:
                data[mode][int(r.group(1))].append((float(r.group(3)), float(r.group(4))))
    return data, order


def fmt_size(n):
    for unit, div in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n // div} {unit}"
    return f"{n} B"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sizes = None
    if "--sizes" in sys.argv:
        sizes = [int(s) for s in sys.argv[sys.argv.index("--sizes") + 1].split(",")]
        args = [a for a in args if a != sys.argv[sys.argv.index("--sizes") + 1]]
    data, order = parse(args)
    all_sizes = sorted({s for m in data.values() for s in m})
    sizes = sizes or all_sizes
    print("| Block | " + " | ".join(f"{m} GB/s" for m in order) + " | "
          + " | ".join(f"{m} us" for m in order) + " |")
    print("|---" * (1 + 2 * len(order)) + "|")
    for s in sizes:
        bw, lat = [], []
        for m in order:
            vals = data[m].get(s, [])
            bw.append(f"{statistics.median(v[0] for v in vals):.2f}" if vals else "-")
            lat.append(f"{statistics.median(v[1] for v in vals):.1f}" if vals else "-")
        print(f"| {fmt_size(s)} | " + " | ".join(bw) + " | " + " | ".join(lat) + " |")
    reps = {m: max((len(v) for v in data[m].values()), default=0) for m in order}
    print("runs per mode: " + ", ".join(f"{m}={n}" for m, n in reps.items()))


if __name__ == "__main__":
    main()
