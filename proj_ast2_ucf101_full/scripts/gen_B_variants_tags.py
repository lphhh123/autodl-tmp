#!/usr/bin/env python3
"""Generate B-grid variant tags for E/C/D/S/K/M axes."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="emit a reduced high-priority subset")
    args = p.parse_args()

    tags = []
    e_vals = [2, 3] if args.fast else [0, 1, 2, 3]
    c_vals = [1] if args.fast else [0, 1]
    d_vals = [1, 2, 3] if args.fast else [0, 1, 2, 3]
    s_vals = [1, 2] if args.fast else [0, 1, 2]
    k_vals = [0, 1]
    m_vals = [1] if args.fast else [0, 1]

    for e in e_vals:
        for c in c_vals:
            for d in d_vals:
                for s in s_vals:
                    for k in k_vals:
                        for m in m_vals:
                            tags.append(f"V_E{e}_C{c}_D{d}_S{s}_K{k}_M{m}")

    print("\n".join(tags))


if __name__ == "__main__":
    main()
