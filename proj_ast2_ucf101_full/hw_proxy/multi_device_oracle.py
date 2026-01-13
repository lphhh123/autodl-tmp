
from typing import Dict, List, Any

import yaml

from .layer_hw_proxy import LayerHwProxy


class MultiDeviceHwOracle:
    """Wrap multiple LayerHwProxy instances for different chip types.

    This class is used by the mapper and layout code.
    """

    def __init__(self, gpu_yaml: str, chip_types: List[str], weight_dir: str = "proxy_weights"):
        self.gpu_yaml = gpu_yaml
        self.chip_types = chip_types
        self.weight_dir = weight_dir
        with open(gpu_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and "chip_types" in data:
            self.chip_map = {entry["name"]: entry for entry in data["chip_types"]}
        else:
            self.chip_map = data or {}
        self.defaults = {}

        self.proxies: Dict[str, LayerHwProxy] = {}
        for dev in chip_types:
            self.proxies[dev] = LayerHwProxy(dev, gpu_yaml=gpu_yaml, weight_dir=weight_dir)

    def predict_layer_on_device(self, layer_row: Dict[str, Any], device_name: str) -> Dict[str, float]:
        proxy = self.proxies[device_name]
        pred = proxy.predict_layers_batch([layer_row])
        return {
            "ms": float(pred["lat_ms"][0]),
            "mem_mb": float(pred["mem_mb"][0]),
            "power_w": float(pred["power_w"][0]),
        }

    # Convenience helpers for mapping / layout

    def _chip_entry(self, device_name: str):
        if device_name in self.chip_map:
            return self.chip_map.get(device_name, None)
        key = f"{device_name}_FP16"
        return self.chip_map.get(key, None)

    def device_area(self, device_name: str) -> float:
        chip = self._chip_entry(device_name)
        if chip is None:
            return 400.0  # dummy
        return float(chip.get("area_mm2", 400.0))

    def device_peak_flops(self, device_name: str) -> float:
        chip = self._chip_entry(device_name)
        if chip is None:
            return 1e12
        return float(chip.get("peak_flops", 1e12))

    def device_mem_size_mb(self, device_name: str) -> float:
        chip = self._chip_entry(device_name)
        if chip is None:
            return 24_000.0
        if "mem_gb" in chip:
            return float(chip.get("mem_gb", 24.0)) * 1024.0
        return float(chip.get("mem_size_mb", 24_000.0))

    def device_link_bw(self, device_name: str) -> float:
        chip = self._chip_entry(device_name)
        if chip is None:
            return 100e9  # bytes/s
        if "peak_bw" in chip:
            return float(chip.get("peak_bw", 100e9))
        return float(chip.get("link_bw", 100e9))


def estimate_segment_hw_for_device(
    oracle: MultiDeviceHwOracle,
    segment_layers: List[Dict[str, Any]],
    device_name: str,
    bytes_per_param: int = 2,
) -> Dict[str, float]:
    """Estimate HW metrics of a segment on one device.

    Returns:
        dict with keys: ms, energy_mj, mem_mb, traffic_bytes
    """
    total_ms = 0.0
    total_energy_mj = 0.0
    peak_mem_mb = 0.0
    traffic_bytes = 0.0

    for row in segment_layers:
        pred = oracle.predict_layer_on_device(row, device_name=device_name)
        ms = pred["ms"]
        mem_mb = pred["mem_mb"]
        power_w = pred["power_w"]

        total_ms += ms
        peak_mem_mb = max(peak_mem_mb, mem_mb)
        energy_j = power_w * (ms / 1000.0)
        total_energy_mj += energy_j * 1e3  # mJ

        # Rough activation size as communication volume between segments
        L_eff = row["L_eff"]
        embed_dim = row["embed_dim"]
        bs = row["bs"]
        traffic_bytes += L_eff * embed_dim * bs * bytes_per_param

    return dict(
        ms=total_ms,
        energy_mj=total_energy_mj,
        mem_mb=peak_mem_mb,
        traffic_bytes=traffic_bytes,
    )
