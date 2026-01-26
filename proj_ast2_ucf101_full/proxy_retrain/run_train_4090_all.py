# proxy_retrain/run_train_4090_all.py
from __future__ import annotations

import os
import subprocess
import sys


def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    # 你给的数据（4090）
    csv_ms_mem = "/root/autodl-tmp/proxy_retrain/layer_dataset_4090.csv"  # 字段包含ms、peak_mem_mb
    csv_power = "/root/autodl-tmp/proxy_retrain/layer_power_dataset_4090.csv"  # 字段包含ms_event、energy_mj、avg_power_w

    out_dir = "./proxy_ckpts_4090"
    os.makedirs(out_dir, exist_ok=True)

    py = sys.executable

    # 1) latency（修改点：csv_ms_mem里的耗时列是ms，不是ms_event）—— log-space
    run([
        py, "-m", "proxy_retrain.train_one_proxy",
        "--csv", csv_ms_mem,
        "--out_dir", out_dir,
        "--target_col", "ms",  # 关键修改：ms_event → ms
        "--target_mode", "log",
        "--seed", "2024",
    ])

    # 2) peak mem (peak_mem_mb) —— log-space（这个列名是对的，无需修改）
    run([
        py, "-m", "proxy_retrain.train_one_proxy",
        "--csv", csv_ms_mem,
        "--out_dir", out_dir,
        "--target_col", "peak_mem_mb",
        "--target_mode", "log",
        "--seed", "2024",
    ])

    # 3) energy (energy_mj) —— log-space（这个列名是对的，无需修改）
    run([
        py, "-m", "proxy_retrain.train_one_proxy",
        "--csv", csv_power,
        "--out_dir", out_dir,
        "--target_col", "energy_mj",
        "--target_mode", "log",
        "--seed", "2024",
    ])

    # 可选：如果你也想单独训 avg_power_w（功耗）（列名正确，保留）
    # run([
    #     py, "-m", "proxy_retrain.train_one_proxy",
    #     "--csv", csv_power,
    #     "--out_dir", out_dir,
    #     "--target_col", "avg_power_w",
    #     "--target_mode", "log",
    #     "--seed", "2024",
    # ])

    print("\n[ALL DONE] check out_dir:", out_dir)
    print("Expected files:")
    print("  proxy_ms.pt, report_ms.json")  # 对应修改：ms_event → ms
    print("  proxy_peak_mem_mb.pt, report_peak_mem_mb.json")
    print("  proxy_energy_mj.pt, report_energy_mj.json")


if __name__ == "__main__":
    main()