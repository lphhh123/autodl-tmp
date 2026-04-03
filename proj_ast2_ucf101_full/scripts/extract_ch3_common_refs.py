#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

VAL_PATTERN = re.compile(
    r"\[val\]\s*epoch=(?P<epoch>\d+).*?end_lat=(?P<lat>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    r".*?end_mem=(?P<mem>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    r".*?end_comm=(?P<comm>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Chapter-3 common refs from warmup stdout.log")
    parser.add_argument("--log_path", required=True, help="Path to warmup stdout.log")
    parser.add_argument("--epoch_min", type=int, default=0, help="Minimum epoch (inclusive)")
    parser.add_argument("--epoch_max", type=int, default=4, help="Maximum epoch (inclusive)")
    parser.add_argument("--format", choices=("text", "json", "shell"), default="text", help="Output format")
    return parser.parse_args()


def collect_values(log_path: Path, epoch_min: int, epoch_max: int):
    lats = []
    mems = []
    comms = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = VAL_PATTERN.search(line)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        if epoch < epoch_min or epoch > epoch_max:
            continue
        lats.append(float(m.group("lat")))
        mems.append(float(m.group("mem")))
        comms.append(float(m.group("comm")))
    return lats, mems, comms


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_path)
    if not log_path.is_file():
        print(f"[extract_ch3_common_refs] log file not found: {log_path}", file=sys.stderr)
        return 2

    epoch_min = int(args.epoch_min)
    epoch_max = int(args.epoch_max)
    if epoch_min > epoch_max:
        epoch_min, epoch_max = epoch_max, epoch_min

    lats, mems, comms = collect_values(log_path, epoch_min, epoch_max)
    if not lats:
        print(
            f"[extract_ch3_common_refs] no valid [val] lines with end_lat/end_mem/end_comm in epoch range [{epoch_min}, {epoch_max}]",
            file=sys.stderr,
        )
        return 2

    ref_lat = float(statistics.median(lats))
    ref_mem = float(statistics.median(mems))
    ref_comm = float(statistics.median(comms))

    if args.format == "json":
        payload = {
            "ref_latency_ms": float(ref_lat),
            "ref_mem_mb": float(ref_mem),
            "ref_comm_ms": float(ref_comm),
            "epoch_min": int(epoch_min),
            "epoch_max": int(epoch_max),
            "num_samples": int(len(lats)),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.format == "shell":
        print(f"export HW_REF_LAT_MS={ref_lat:.8f}")
        print(f"export HW_REF_MEM_MB={ref_mem:.8f}")
        print(f"export HW_REF_COMM_MS={ref_comm:.8f}")
        return 0

    print(f"ref_latency_ms={ref_lat:.8f}")
    print(f"ref_mem_mb={ref_mem:.8f}")
    print(f"ref_comm_ms={ref_comm:.8f}")
    print(f"epoch_range=[{epoch_min},{epoch_max}] samples={len(lats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
