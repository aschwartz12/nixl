#!/usr/bin/env python3
"""Sum EFA hw counter deltas (after - before) per nixl_ep run; mean per mode.

Usage: hwc_summarize.py <EP_HWC_DIR> [counter ...]
Files: <host>-<mode>-<rep>-{before,after}.txt with "<device> <counter> <value>" lines.
"""
import collections
import os
import re
import statistics
import sys

DEFAULT = ["rdma_write_wrs", "rdma_write_bytes", "rdma_write_recv_bytes", "send_wrs",
           "recv_wrs", "rx_pkts", "tx_pkts", "rx_drops", "retrans_pkts",
           "retrans_timeout_events", "unresponsive_remote_events",
           "impaired_remote_conn_events"]


def read(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[2].lstrip("-").isdigit():
                out[(parts[0], parts[1])] = int(parts[2])
    return out


def main():
    d = sys.argv[1]
    counters = sys.argv[2:] or DEFAULT
    runs = collections.defaultdict(lambda: collections.Counter())  # (mode, rep) -> counter sums
    for name in sorted(os.listdir(d)):
        m = re.match(r"^(?P<host>.+?)-(?P<mode>[a-z0-9]+)-(?P<rep>\d+)-after\.txt$", name)
        if not m:
            continue
        before = os.path.join(d, name.replace("-after.txt", "-before.txt"))
        if not os.path.exists(before):
            continue
        a, b = read(os.path.join(d, name)), read(before)
        for (dev, c), v in a.items():
            runs[(m["mode"], m["rep"])][c] += v - b.get((dev, c), v)
    modes = sorted({k[0] for k in runs})
    print("| counter (sum over nodes, per run) | " + " | ".join(modes) + " |")
    print("|---|" + "---|" * len(modes))
    for c in counters:
        cells = []
        for mode in modes:
            vals = [runs[k][c] for k in runs if k[0] == mode]
            cells.append(f"{statistics.mean(vals):.4g} (n={len(vals)})" if vals else "-")
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
