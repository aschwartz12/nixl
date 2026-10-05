#!/usr/bin/env python3
"""Summarize rxorder.sbatch and ep-elastic logs: per run the mean over its trials,
per (mode, metric) the median over repetitions (with min..max).

Usage: rxorder_summarize.py <rxorder run dir>... [--ep <ep-elastic log>...]
       rxorder_summarize.py --sweep <run dir>...   (step 2 per-write cost table)
"""
import collections
import glob
import os
import re
import statistics
import sys


def csv_rows(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 6:
                    try:
                        rows.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3]),
                                     float(parts[4]), float(parts[5])))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return rows


def scalar(path, key):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key + ","):
                    return float(line.split(",")[1])
    except FileNotFoundError:
        pass
    return None


def mean_of(rows, test, size, field):
    vals = [r[field] for r in rows if r[0] == test and r[1] == size]
    return statistics.mean(vals) if vals else None


def busy_cores(out_path, rank):
    """Sum over a rank's proxy threads of busy time / their active window."""
    err = out_path.replace("-rank-0.out", f"-rank-{rank}.err")
    total, found = 0.0, False
    try:
        with open(err) as f:
            for line in f:
                m = re.search(r"EFA proxy cpu: thread \d+: .*busy_pct=([0-9.]+)", line)
                if m:
                    total += float(m.group(1)) / 100.0
                    found = True
    except FileNotFoundError:
        pass
    return total if found else None


# metric -> (suite, extractor(out_path) -> value, unit)
METRICS = [
    ("ping-pong 8 B (us)", "c1", lambda o: mean_of(csv_rows(o), "PUT-ordered-signal-pingpong", 8, 4)),
    ("put 8 B (us)", "c1", lambda o: mean_of(csv_rows(o), "PUT-target", 8, 4)),
    ("put+signal 8 B at sender (us)", "c1", lambda o: mean_of(csv_rows(o), "PUT-ordered-signal", 8, 4)),
    ("put+signal 64 KiB at sender (us)", "c1", lambda o: mean_of(csv_rows(o), "PUT-ordered-signal", 65536, 4)),
    ("pipelined 1 MiB, 1 ch (Gbit/s)", "c1", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 1048576, 5)),
    ("pipelined 64 KiB, 1 ch (Gbit/s)", "c1", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 65536, 5)),
    ("pipelined 8 KiB, 1 ch (Gbit/s)", "c1", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 8192, 5)),
    ("batch-signal 64 KiB, 1 ch (Gbit/s)", "c1", lambda o: mean_of(csv_rows(o), "PUT-batch-signal", 65536, 5)),
    ("pipelined 1 MiB, 4 ch (Gbit/s)", "c4", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 1048576, 5)),
    ("pipelined 64 KiB, 4 ch (Gbit/s)", "c4", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 65536, 5)),
    ("pipelined 8 KiB, 4 ch (Gbit/s)", "c4", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 8192, 5)),
    ("ordering check, 4 ch (signals/s)", "ord", lambda o: scalar(o, "ordering_signals_per_s")),
    ("ordering check, profiled (signals/s)", "ordp", lambda o: scalar(o, "ordering_signals_per_s")),
    ("target cores busy, ordering check", "ordp", lambda o: busy_cores(o, 1)),
    ("sender cores busy, ordering check", "ordp", lambda o: busy_cores(o, 0)),
    ("target cores busy, 16 KiB puts + host writes", "conc", lambda o: busy_cores(o, 1)),
    ("host 1 MiB writes, proxy idle (Gbit/s)", "host", lambda o: mean_of(csv_rows(o), "HOST-write", 1048576, 5)),
    ("host 64 KiB writes, proxy idle (Gbit/s)", "host", lambda o: mean_of(csv_rows(o), "HOST-write", 65536, 5)),
    ("host 1 MiB writes during device 16 KiB puts (Gbit/s)", "conc", lambda o: mean_of(csv_rows(o), "HOST-concurrent", 1048576, 5)),
    ("device 16 KiB puts, 4 ch, during host writes (Gbit/s)", "conc", lambda o: mean_of(csv_rows(o), "PUT-pipelined", 16384, 5)),
]

RUN_RE = re.compile(r"^(?P<mode>[a-z0-9]+)-(?P<suite>c1|c4|ordp|ord|host|conc)-(?P<rep>\d+)-rank-0\.out$")


def fmt(vals):
    if not vals:
        return "-"
    med = statistics.median(vals)
    digits = 0 if med >= 1000 else 1 if med >= 10 else 2
    if len(vals) == 1:
        return f"{med:.{digits}f}"
    return f"{med:.{digits}f} ({min(vals):.{digits}f}-{max(vals):.{digits}f}, n={len(vals)})"


def summarize(dirs):
    data = collections.defaultdict(list)  # (mode, metric) -> [run values]
    modes = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*-rank-0.out"))):
            m = RUN_RE.match(os.path.basename(path))
            if not m:
                continue
            if m["mode"] not in modes:
                modes.append(m["mode"])
            for name, suite, fn in METRICS:
                if suite == m["suite"]:
                    v = fn(path)
                    if v is not None:
                        data[(m["mode"], name)].append(v)
    print("| Metric | " + " | ".join(modes) + " |")
    print("|---|" + "---|" * len(modes))
    for name, _, _ in METRICS:
        if any(data.get((mode, name)) for mode in modes):
            print(f"| {name} | " + " | ".join(fmt(data.get((mode, name), [])) for mode in modes) + " |")


def ep_summary(paths):
    for path in paths:
        runs = []
        label = None
        cur = []
        with open(path) as f:
            for line in f:
                if line.startswith("=== install=") or line.startswith("=== mode="):
                    if label is not None:
                        runs.append((label, cur))
                    label, cur = re.sub(r" rep=\d+", "", line.strip()), []
                m = re.search(r"Dispatch \+ combine bandwidth: ([0-9.]+) GB/s", line)
                if m:
                    cur.append(float(m.group(1)))
        if label is not None:
            runs.append((label, cur))
        by_label = collections.defaultdict(list)
        failed = collections.Counter()
        for label, vals in runs:
            if vals:
                by_label[label].append(statistics.mean(vals))
            else:
                failed[label] += 1  # hung (step time limit) or failed before any result
        for label in dict.fromkeys(l for l, _ in runs):
            means = by_label[label]
            median = f" median {statistics.median(means):.2f} GB/s" if means else ""
            print(f"{os.path.basename(path)} {label}: per-run means {['%.2f' % m for m in means]}"
                  f"{median}; runs without a result: {failed[label]}")


SWEEP_RE = re.compile(
    r"^(?P<mode>[a-z0-9]+)-(?P<suite>sweepc?)-t(?P<t>\d+)-(?P<size>\d+)-(?P<rep>\d+)-rank-0\.out$")
RX_RE = re.compile(r"EFA proxy rx: thread (\d+): (.*)$")


def sweep_summary(dirs):
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*-sweep*-rank-0.out"))):
            m = SWEEP_RE.match(os.path.basename(path))
            if not m:
                continue
            key = (m["mode"] + ("+host" if m["suite"] == "sweepc" else ""), int(m["t"]), int(m["size"]))
            r = csv_rows(path)
            size = int(m["size"])
            mean_us = mean_of(r, "PUT-pipelined", size, 4)
            gbit = mean_of(r, "PUT-pipelined", size, 5)
            if mean_us:
                rows[key]["writes/s (M)"].append(1.0 / mean_us)
                rows[key]["Gbit/s"].append(gbit)
            host = mean_of(r, "HOST-concurrent", 1048576, 5)
            if host is not None:
                rows[key]["host Gbit/s"].append(host)
            # Target-side rx stats (rank 1 log), summed over its threads.
            err = path.replace("-rank-0.out", "-rank-1.err")
            stats = collections.defaultdict(float)
            threads = 0
            try:
                with open(err) as f:
                    for line in f:
                        mm = RX_RE.search(line)
                        if not mm:
                            continue
                        kv = dict(item.split("=") for item in mm.group(2).split())
                        threads += 1
                        for k, v in kv.items():
                            stats[k + "_list"] += 0  # keep keys
                            stats[k] += float(v)
                        stats["busy_max"] = max(stats["busy_max"], float(kv["busy_pct"]))
            except FileNotFoundError:
                pass
            if threads and stats["entries"]:
                e = stats["entries"]
                # Weighted by entries: the per-entry costs are sums over threads of ns / entries.
                rows[key]["sweep ns/entry"].append(stats["sweep_ns_per_entry"] / threads)
                rows[key]["read ns/entry"].append(stats["read_ns_per_entry"] / threads)
                rows[key]["decode ns/entry"].append(stats["decode_ns_per_entry"] / threads)
                rows[key]["entries/read"].append(stats["entries_per_read"] / threads)
                rows[key]["rx threads"].append(threads)
                rows[key]["busy % (sum)"].append(stats["busy_pct"])
                rows[key]["busy % (max)"].append(stats["busy_max"])
                rows[key]["consumed rx"].append(stats["consumed_rx"])
                rows[key]["entries"].append(e)
    cols = ["writes/s (M)", "Gbit/s", "host Gbit/s", "sweep ns/entry", "read ns/entry", "decode ns/entry",
            "entries/read", "busy % (sum)", "busy % (max)", "rx threads", "consumed rx"]
    print("| mode | threads | size | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for key in sorted(rows):
        vals = rows[key]
        cells = []
        for c in cols:
            v = vals.get(c)
            cells.append(f"{statistics.median(v):.3g}" if v else "-")
        print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--sweep":
        sweep_summary(args[1:])
        sys.exit(0)
    eps = []
    if "--ep" in args:
        i = args.index("--ep")
        eps = args[i + 1:]
        args = args[:i]
    if args:
        summarize(args)
    if eps:
        ep_summary(eps)
