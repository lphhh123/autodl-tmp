#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
measure_baseline_and_apply.py

作用（一步到位）：
  1) 在 GPU 空载时，通过 nvidia-smi 连续采样多次功率，求平均，作为该设备在某精度下的“基础功率”（P_base）。
  2) 把测出来的 P_base 写入/更新到 baseline YAML 文件（例如 power_baseline.yaml）。
  3) 从 layer_power_dataset_*.csv 读取逐层功率数据（power_w），用：
         power_dyn_w = max(power_w - P_base(device, prec), 0)
     得到“动态功率”列，写入新的 CSV（例如 layer_power_dataset_3090_dyn.csv）。

注意事项：
  - 请在测基础功率前，确保当前 GPU 基本空载（无大算子在跑）。
  - 文件里的 device 和 prec 需要跟你 profiler 写入 CSV 时一致，例如 device=RTX3090, prec=fp16/fp32。
  - 这个脚本只负责“测基础功率 + 生成带 power_dyn_w 的 CSV”，训练代理模型你继续用之前的 train_layerwise_power_proxy.py 即可。
"""

import argparse
import csv
import os
import subprocess
import time
from collections import defaultdict

import yaml


# ----------------- 第 1 部分：测基础功率 ----------------- #

def measure_baseline_power(gpu_index: int, samples: int = 50, interval: float = 0.5):
    """
    使用 nvidia-smi 连续采样多次功率，返回平均值（W）。
    要求：运行期间尽量不要有其他大负载任务。
    """
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    powers = []

    print(f"[measure] start measuring baseline power on GPU {gpu_index} ...")
    print(f"[measure] samples={samples}, interval={interval}s")

    for i in range(samples):
        try:
            out = subprocess.check_output(cmd, encoding="utf-8")
            val_str = out.strip().split("\n")[0].strip()
            val = float(val_str)
            powers.append(val)
            print(f"  sample {i+1:3d}/{samples}: {val:.3f} W")
        except Exception as e:
            print(f"[warn] nvidia-smi failed at sample {i+1}: {e}")
        time.sleep(interval)

    if not powers:
        raise RuntimeError("no valid power samples, please check nvidia-smi and GPU index")

    avg = sum(powers) / len(powers)
    # 简单算一下标准差方便你观察
    mean = avg
    var = sum((p - mean) ** 2 for p in powers) / len(powers)
    std = var ** 0.5

    print(f"[measure] baseline power avg = {avg:.3f} W, std = {std:.3f} W, n = {len(powers)}")
    return avg, std


# ----------------- 第 2 部分：更新 baseline YAML ----------------- #

def load_baseline_yaml(path: str):
    if not os.path.exists(path):
        print(f"[info] baseline yaml {path} not found, will create a new one")
        return {"baselines": {}, "meta": {}}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "baselines" not in data:
        data["baselines"] = {}
    if "meta" not in data:
        data["meta"] = {}
    return data


def update_baseline_yaml(yaml_path: str, device_name: str, prec: str, baseline_w: float, std_w: float):
    """
    在 YAML 中写入：
      baselines:
        <device_name>:
          <prec>: baseline_w
    """
    prec = prec.lower()
    data = load_baseline_yaml(yaml_path)

    if device_name not in data["baselines"]:
        data["baselines"][device_name] = {}

    data["baselines"][device_name][prec] = float(baseline_w)

    # 顺便把噪声信息记录一下（可选）
    note_key = f"{device_name}_{prec}_std"
    data["meta"][note_key] = float(std_w)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[yaml] updated baseline for ({device_name}, {prec}) in {yaml_path}: {baseline_w:.3f} W")


def load_baseline_map(yaml_path: str):
    """
    展平成 (device, prec) -> P_base 的字典。
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    baselines = data.get("baselines", {})
    base_map = {}
    for dev, prec_dict in baselines.items():
        if prec_dict is None:
            continue
        for prec, val in prec_dict.items():
            base_map[(str(dev), str(prec).lower())] = float(val)

    if not base_map:
        raise RuntimeError(f"no baselines found in {yaml_path}")
    print("[yaml] loaded baselines:")
    for (dev, prec), P in base_map.items():
        print(f"  - ({dev}, {prec}) → P_base = {P:.3f} W")
    return base_map


# ----------------- 第 3 部分：生成带 power_dyn_w 的 CSV ----------------- #

def load_rows(path: str):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def write_dyn_csv(rows, base_map, out_path: str):
    if not rows:
        raise RuntimeError("no rows to write")

    fieldnames = list(rows[0].keys())
    if "power_dyn_w" not in fieldnames:
        fieldnames.append("power_dyn_w")

    missing_keys = defaultdict(int)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for row in rows:
            device = row.get("device", "UNKNOWN")
            prec = row.get("prec", "fp16").lower()
            key = (device, prec)

            power_str = row.get("power_w", "")
            P_dyn = ""

            if power_str != "":
                try:
                    P = float(power_str)
                except ValueError:
                    P = None

                if P is not None:
                    if key in base_map:
                        P_base = base_map[key]
                        P_dyn = max(P - P_base, 0.0)
                    else:
                        # baseline 里没有，则直接用总功率当作动态功率，并记一条警告
                        missing_keys[key] += 1
                        P_dyn = P

            row["power_dyn_w"] = f"{P_dyn:.6f}" if isinstance(P_dyn, float) else ""
            w.writerow(row)

    print(f"[write] dynamic power csv → {out_path}")

    if missing_keys:
        print("[warn] some (device, prec) not found in baseline yaml, "
              "these rows 使用 power_w 作为 power_dyn_w：")
        for key, cnt in missing_keys.items():
            print(f"  - {key}: {cnt} rows")


# ----------------- 主入口 ----------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    # 基础功率测量参数
    ap.add_argument("--gpu_index", type=int, default=0,
                    help="用于测基础功率的 GPU index（nvidia-smi -i 的编号）")
    ap.add_argument("--device_name", required=True,
                    help="写入 CSV 和 YAML 时使用的设备名，例如 RTX3090 / RTX2080Ti")
    ap.add_argument("--prec", choices=["fp16", "fp32"], required=True,
                    help="当前测的是哪种精度的基础功率（只用于做键的区分）")
    ap.add_argument("--samples", type=int, default=50,
                    help="采样次数")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="采样间隔（秒）")

    # 文件相关
    ap.add_argument("--baseline_yaml", default="power_baseline.yaml",
                    help="基础功率 YAML 文件路径（会自动创建/更新）")
    ap.add_argument("--in_csv", required=True,
                    help="原始功率数据 CSV，比如 layer_power_dataset_3090.csv")
    ap.add_argument("--out_csv", required=True,
                    help="输出带 power_dyn_w 的 CSV，比如 layer_power_dataset_3090_dyn.csv")

    return ap.parse_args()


def main():
    args = parse_args()

    # 1) 测基础功率
    avg_w, std_w = measure_baseline_power(
        gpu_index=args.gpu_index,
        samples=args.samples,
        interval=args.interval,
    )

    # 2) 写入/更新 YAML
    update_baseline_yaml(
        yaml_path=args.baseline_yaml,
        device_name=args.device_name,
        prec=args.prec,
        baseline_w=avg_w,
        std_w=std_w,
    )

    # 3) 载入 YAML，生成动态功率 CSV
    base_map = load_baseline_map(args.baseline_yaml)
    rows = load_rows(args.in_csv)
    print(f"[data] loaded {len(rows)} rows from {args.in_csv}")
    write_dyn_csv(rows, base_map, args.out_csv)


if __name__ == "__main__":
    main()
