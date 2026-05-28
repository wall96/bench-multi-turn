#!/usr/bin/env python3
"""Read SGLang's prom metrics directly and compute histogram quantiles
exactly like grafana's:

    histogram_quantile(0.9,
        sum by (le) (rate(sglang:inter_token_latency_seconds_bucket[$__rate_interval])))

Why this exists: client-side TPOT/ITL measurements drift from prom under
load (network buffering, chunk coalescing). Reading the server's own
histograms is the ground truth and bypasses both prom and the client.

Algorithm:
  1. Take a snapshot of /metrics from each server before the test.
  2. Take a second snapshot after the test.
  3. Subtract bucket counters per (server, label-set, le). The diff is
     exactly the count of observations that fell in each bucket during
     the test window — equivalent to integrating `rate(..._bucket[r])`
     over the window.
  4. Sum the diffs across all servers and all non-le labels for each le
     value. This is `sum by (le)`.
  5. Apply standard prom histogram_quantile linear interpolation.

Three subcommands:
  snapshot  — fetch /metrics from one or more SGLang servers, write JSON.
  diff      — given two snapshots, compute count + quantiles, append CSV.
  wrap      — snapshot, run a command, snapshot, diff, append CSV. Use
              this in sweep scripts so each rate point gets a paired row.

Examples:
  # one shot wrap (typical sweep usage):
  python3 server_metrics.py wrap \\
      --metrics-urls http://10.51.10.32:30000/metrics,http://10.51.10.33:30000/metrics \\
      --metric ITL=sglang:inter_token_latency_seconds \\
      --metric TTFT=sglang:time_to_first_token_seconds \\
      --metric E2E=sglang:e2e_request_latency_seconds \\
      --label-filter model_name=DeepSeek-V3.2-4decode-only \\
      --label-filter engine_type=decode \\
      --quantiles 0.5,0.9,0.99 \\
      --case-name r0.4 \\
      --summary-csv server_metrics.csv \\
      --post-sleep 5 \\
      -- python3 bench_multi_turn.py ...

  # explicit two-shot if you want the snapshots on disk:
  python3 server_metrics.py snapshot --metrics-urls ... --output before.json
  python3 bench_multi_turn.py ...
  python3 server_metrics.py snapshot --metrics-urls ... --output after.json
  python3 server_metrics.py diff --before before.json --after after.json \\
      --metric ITL=sglang:inter_token_latency_seconds \\
      --case-name r0.4 --summary-csv server_metrics.csv

No external deps — pure stdlib.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen


# OpenMetrics line: name{label="val",...} value [timestamp]
_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'      # metric name
    r'(?:\{([^}]*)\})?'                  # optional {labels}
    r'\s+([^\s]+)'                       # value
    r'(?:\s+[0-9.eE+\-]+)?\s*$'          # optional timestamp
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_metrics_text(text):
    """Yield (name, labels_dict, value) for each non-comment sample line."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, labelstr, val = m.group(1), m.group(2) or "", m.group(3)
        if val in ("NaN", "+Inf", "-Inf"):
            v = float(val)
        else:
            try:
                v = float(val)
            except ValueError:
                continue
        labels = dict(_LABEL_RE.findall(labelstr))
        yield name, labels, v


def fetch_one(url, timeout=10.0):
    with urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def snapshot(urls, timeout=10.0):
    """Return list of {url, ts, samples: [{name, labels, value}, ...]}."""
    out = []
    for url in urls:
        text = fetch_one(url, timeout=timeout)
        samples = []
        for name, labels, val in parse_metrics_text(text):
            samples.append({"name": name, "labels": labels, "value": val})
        out.append({"url": url, "ts": time.time(), "samples": samples})
    return out


def _key_excl_le(labels):
    return tuple(sorted((k, v) for k, v in labels.items() if k != "le"))


def diff_histogram(before, after, metric_name, label_filter=None):
    """Return dict le_str -> cumulative_count_in_window.

    Mimics `sum by (le)(rate(metric_name_bucket[r]))` integrated over
    the window. label_filter narrows which series participate (AND).
    """
    bucket = metric_name + "_bucket"
    label_filter = label_filter or {}

    def matches(labels):
        return all(labels.get(k) == v for k, v in label_filter.items())

    def index(snap):
        idx = {}
        for srv in snap:
            url = srv["url"]
            for s in srv["samples"]:
                if s["name"] != bucket:
                    continue
                if not matches(s["labels"]):
                    continue
                le = s["labels"].get("le")
                if le is None:
                    continue
                idx[(url, _key_excl_le(s["labels"]), le)] = s["value"]
        return idx

    a = index(before)
    b = index(after)
    by_le = defaultdict(float)
    for k in set(a.keys()) | set(b.keys()):
        delta = b.get(k, 0.0) - a.get(k, 0.0)
        if delta < 0:
            # Counter went backwards — server restart for this series.
            # Skip rather than poison the aggregate.
            continue
        by_le[k[2]] += delta
    return dict(by_le)


def _le_sort_key(le):
    return float("inf") if le == "+Inf" else float(le)


def histogram_quantile(by_le, q):
    """Standard prom linear interpolation. Returns NaN if no observations."""
    if not by_le:
        return float("nan")
    items = sorted(by_le.items(), key=lambda x: _le_sort_key(x[0]))
    total = items[-1][1]  # +Inf bucket holds the cumulative total
    if total <= 0:
        return float("nan")
    target = q * total
    prev_le = 0.0
    prev_cum = 0.0
    for le_str, cum in items:
        if cum >= target:
            if le_str == "+Inf":
                # target falls in the open-top bucket; can't interpolate,
                # return the previous le (matches prom behavior at q=1.0 ish)
                return prev_le if prev_le > 0 else float("inf")
            le = float(le_str)
            if cum == prev_cum:
                return le
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_le + (le - prev_le) * frac
        if le_str != "+Inf":
            prev_le = float(le_str)
        prev_cum = cum
    return float("nan")


def total_count(by_le):
    if not by_le:
        return 0.0
    items = sorted(by_le.items(), key=lambda x: _le_sort_key(x[0]))
    return items[-1][1]


def parse_metric_specs(specs):
    """spec format: alias=metric_name. metric_name should NOT include
    the _bucket / _count / _sum suffixes."""
    out = []
    for spec in specs:
        alias, _, name = spec.partition("=")
        if not name:
            name = alias
        out.append((alias, name))
    return out


def parse_label_filters(filters):
    out = {}
    for f in filters:
        k, _, v = f.partition("=")
        if k and v:
            out[k.strip()] = v.strip()
    return out


def _qkey(q):
    if q * 100 == int(q * 100):
        return f"p{int(q * 100)}"
    # non-integer percentile (e.g. 0.999) — use the literal
    return "q" + str(q).replace(".", "")


def _format_row(case_name, before, after, metric_specs, label_filter, quantiles,
                debug_buckets=False):
    duration = (after[0]["ts"] - before[0]["ts"]) if before and after else 0.0
    row = {"case_name": case_name, "duration_s": round(duration, 3)}
    for alias, name in metric_specs:
        by_le = diff_histogram(before, after, name, label_filter)
        cnt = total_count(by_le)
        row[f"{alias}_count"] = cnt
        for q in quantiles:
            v = histogram_quantile(by_le, q)
            v_ms = (v * 1000.0
                    if (isinstance(v, float) and not math.isnan(v)
                        and not math.isinf(v))
                    else v)
            row[f"{alias}_{_qkey(q)}_ms"] = v_ms
        if debug_buckets:
            print(f"--- {alias} ({name}) ---", file=sys.stderr)
            for le_str, c in sorted(by_le.items(), key=lambda x: _le_sort_key(x[0])):
                print(f"    le<={le_str:>14s}  cum={c}", file=sys.stderr)
    return row


def _write_csv(path, row):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _print_row(row):
    print()
    for k, v in row.items():
        if isinstance(v, float):
            print(f"  {k:<32s} {v:.4f}")
        else:
            print(f"  {k:<32s} {v}")


# --- subcommands -----------------------------------------------------

def cmd_snapshot(args):
    urls = [u.strip() for u in args.metrics_urls.split(",") if u.strip()]
    snap = snapshot(urls, timeout=args.timeout)
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(snap, f)
    n_samples = sum(len(s["samples"]) for s in snap)
    print(f"snapshot written: {args.output} ({len(urls)} servers, "
          f"{n_samples} samples)", file=sys.stderr)


def cmd_diff(args):
    with open(args.before) as f:
        before = json.load(f)
    with open(args.after) as f:
        after = json.load(f)

    metric_specs = parse_metric_specs(args.metric)
    label_filter = parse_label_filters(args.label_filter)
    quantiles = [float(q) for q in args.quantiles.split(",")]

    row = _format_row(args.case_name, before, after, metric_specs,
                      label_filter, quantiles, debug_buckets=args.print_buckets)
    _print_row(row)
    if args.summary_csv:
        _write_csv(args.summary_csv, row)
        print(f"appended row -> {args.summary_csv}", file=sys.stderr)


def cmd_wrap(args):
    urls = [u.strip() for u in args.metrics_urls.split(",") if u.strip()]
    metric_specs = parse_metric_specs(args.metric)
    label_filter = parse_label_filters(args.label_filter)
    quantiles = [float(q) for q in args.quantiles.split(",")]

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("ERROR: no command given (place after --)", file=sys.stderr)
        sys.exit(2)

    print(f"[server_metrics] before-snapshot ({len(urls)} servers) ...",
          file=sys.stderr)
    before = snapshot(urls, timeout=args.timeout)

    print(f"[server_metrics] running: {' '.join(cmd)}", file=sys.stderr)
    rc = subprocess.call(cmd)
    print(f"[server_metrics] command exited rc={rc}", file=sys.stderr)

    if args.post_sleep > 0:
        # Let in-flight observations land before sealing the window.
        # SGLang updates these histograms only at request finalization,
        # so a short sleep avoids missing the tail of the test.
        print(f"[server_metrics] sleeping {args.post_sleep}s before "
              f"after-snapshot ...", file=sys.stderr)
        time.sleep(args.post_sleep)

    print(f"[server_metrics] after-snapshot ...", file=sys.stderr)
    after = snapshot(urls, timeout=args.timeout)

    row = _format_row(args.case_name, before, after, metric_specs,
                      label_filter, quantiles, debug_buckets=args.print_buckets)
    _print_row(row)
    if args.summary_csv:
        _write_csv(args.summary_csv, row)
        print(f"appended row -> {args.summary_csv}", file=sys.stderr)

    sys.exit(rc)


def main():
    p = argparse.ArgumentParser(
        prog="server_metrics",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("snapshot", help="fetch /metrics and dump to JSON")
    sp.add_argument("--metrics-urls", required=True,
                    help="comma-separated list of /metrics URLs")
    sp.add_argument("--output", required=True, help="JSON file to write")
    sp.add_argument("--timeout", type=float, default=10.0)
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("diff", help="compute quantiles between two snapshots")
    sp.add_argument("--before", required=True)
    sp.add_argument("--after", required=True)
    sp.add_argument("--metric", action="append", default=[], required=True,
                    help="alias=metric_name (no _bucket suffix); repeatable")
    sp.add_argument("--label-filter", action="append", default=[],
                    help="key=value; AND-combined narrowing of label set")
    sp.add_argument("--quantiles", default="0.5,0.9,0.99",
                    help="comma-separated quantiles, e.g. 0.5,0.9,0.99")
    sp.add_argument("--case-name", default="unnamed")
    sp.add_argument("--summary-csv", help="append a row to this CSV")
    sp.add_argument("--print-buckets", action="store_true",
                    help="dump full per-le bucket counts to stderr")
    sp.set_defaults(func=cmd_diff)

    sp = sub.add_parser("wrap",
                        help="snapshot, run cmd, snapshot, diff, write CSV")
    sp.add_argument("--metrics-urls", required=True)
    sp.add_argument("--metric", action="append", default=[], required=True)
    sp.add_argument("--label-filter", action="append", default=[])
    sp.add_argument("--quantiles", default="0.5,0.9,0.99")
    sp.add_argument("--case-name", default="unnamed")
    sp.add_argument("--summary-csv")
    sp.add_argument("--timeout", type=float, default=10.0)
    sp.add_argument("--post-sleep", type=float, default=5.0,
                    help="seconds to wait after cmd before final snapshot, "
                         "so the last few requests' metrics land "
                         "(default 5; bump if the test ends with a long-tail)")
    sp.add_argument("--print-buckets", action="store_true")
    sp.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command to run (place after --)")
    sp.set_defaults(func=cmd_wrap)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
