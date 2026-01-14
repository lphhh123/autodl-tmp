# proj_ast2_ucf101_full/utils/config_validate.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.config import AttrDict


def _ensure_attrdict(x: Any) -> AttrDict:
    if isinstance(x, AttrDict):
        return x
    if isinstance(x, dict):
        return AttrDict({k: _ensure_attrdict(v) if isinstance(v, dict) else v for k, v in x.items()})
    return AttrDict({})


def _ensure_section(cfg: AttrDict, key: str) -> AttrDict:
    if key not in cfg or cfg[key] is None:
        cfg[key] = AttrDict({})
    if not isinstance(cfg[key], AttrDict):
        cfg[key] = _ensure_attrdict(cfg[key])
    return cfg[key]


def _get(cfg: AttrDict, path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, (dict, AttrDict)) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set(cfg: AttrDict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = cfg
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = AttrDict({})
        if not isinstance(cur[p], AttrDict):
            cur[p] = _ensure_attrdict(cur[p])
        cur = cur[p]
    cur[parts[-1]] = value


def validate_and_fill_defaults(cfg: AttrDict, mode: str) -> Tuple[AttrDict, Dict[str, Any]]:
    """
    宽松校验 + defaults 注入。
    mode: "single" | "version_c"
    返回：(cfg, meta) 其中 meta 记录做了哪些迁移/补默认。
    """
    meta: Dict[str, Any] = {"filled": [], "migrated": []}

    # --- sections ---
    train = _ensure_section(cfg, "train")
    loss = _ensure_section(cfg, "loss")
    data = _ensure_section(cfg, "data")
    model = _ensure_section(cfg, "model")
    hw = _ensure_section(cfg, "hw")
    training = _ensure_section(cfg, "training")

    # --- generic defaults ---
    if "seed" not in train:
        train.seed = 0
        meta["filled"].append("train.seed")
    if "device" not in train:
        train.device = "cuda"
        meta["filled"].append("train.device")
    if "amp" not in train:
        train.amp = True
        meta["filled"].append("train.amp")
    if "lr" not in train:
        train.lr = 1e-4
        meta["filled"].append("train.lr")
    if "weight_decay" not in train:
        train.weight_decay = 0.05
        meta["filled"].append("train.weight_decay")
    if "epochs" not in train:
        train.epochs = 1
        meta["filled"].append("train.epochs")

    if "lambda_AST" not in loss:
        loss.lambda_AST = 1.0
        meta["filled"].append("loss.lambda_AST")

    # --- migrate lambda_hw: loss -> hw ---
    if _get(cfg, "hw.lambda_hw", None) is None:
        old = _get(cfg, "loss.lambda_hw", None)
        if old is not None:
            _set(cfg, "hw.lambda_hw", float(old))
            meta["migrated"].append("loss.lambda_hw -> hw.lambda_hw")
        else:
            _set(cfg, "hw.lambda_hw", 0.0)
            meta["filled"].append("hw.lambda_hw")

    # --- hw defaults (used by both modes; version_c needs more) ---
    if _get(cfg, "hw.device_name", None) is None:
        _set(cfg, "hw.device_name", "RTX4090_FP16")
        meta["filled"].append("hw.device_name")
    if _get(cfg, "hw.gpu_yaml", None) is None:
        _set(cfg, "hw.gpu_yaml", "configs/gpu_data.yaml")
        meta["filled"].append("hw.gpu_yaml")
    if _get(cfg, "hw.proxy_weight_dir", None) is None:
        _set(cfg, "hw.proxy_weight_dir", "proxy_weights")
        meta["filled"].append("hw.proxy_weight_dir")

    # --- data defaults ---
    if "batch_size" not in data:
        data.batch_size = 1
        meta["filled"].append("data.batch_size")
    if "num_workers" not in data:
        data.num_workers = 2
        meta["filled"].append("data.num_workers")
    if "num_frames" not in data:
        # 尽量从 model.num_frames 兜底
        data.num_frames = int(_get(cfg, "model.num_frames", 16) or 16)
        meta["filled"].append("data.num_frames")

    # --- model minimal checks (仍然给出明确错误) ---
    if _get(cfg, "model.num_classes", None) is None:
        raise ValueError("cfg.model.num_classes is required")
    if _get(cfg, "model.img_size", None) is None:
        _set(cfg, "model.img_size", 224)
        meta["filled"].append("model.img_size")
    if _get(cfg, "model.num_frames", None) is None:
        _set(cfg, "model.num_frames", int(data.num_frames))
        meta["filled"].append("model.num_frames")

    # --- mode-specific defaults ---
    if mode == "single":
        if _get(cfg, "training.model_type", None) is None:
            _set(cfg, "training.model_type", "video")
            meta["filled"].append("training.model_type")
    elif mode == "version_c":
        # training loop knobs
        if _get(cfg, "training.outer_epochs", None) is None:
            _set(cfg, "training.outer_epochs", 1)
            meta["filled"].append("training.outer_epochs")
        if _get(cfg, "training.inner_steps_ast", None) is None:
            _set(cfg, "training.inner_steps_ast", 10)
            meta["filled"].append("training.inner_steps_ast")
        if _get(cfg, "training.inner_steps_alpha", None) is None:
            _set(cfg, "training.inner_steps_alpha", 0)
            meta["filled"].append("training.inner_steps_alpha")
        if _get(cfg, "training.inner_steps_layout", None) is None:
            _set(cfg, "training.inner_steps_layout", 0)
            meta["filled"].append("training.inner_steps_layout")
        if _get(cfg, "training.twostage", None) is None:
            _set(cfg, "training.twostage", False)
            meta["filled"].append("training.twostage")

        # mapping defaults
        mapping = _ensure_section(cfg, "mapping")
        if "strategy" not in mapping:
            mapping.strategy = "greedy_local"
            meta["filled"].append("mapping.strategy")
        if "mem_limit_factor" not in mapping:
            mapping.mem_limit_factor = 0.9
            meta["filled"].append("mapping.mem_limit_factor")

        # chiplet defaults
        chiplet = _ensure_section(cfg, "chiplet")
        if "candidate_types" not in chiplet:
            chiplet.candidate_types = []
            meta["filled"].append("chiplet.candidate_types")
        if "tau_init" not in chiplet:
            chiplet.tau_init = 5.0
            meta["filled"].append("chiplet.tau_init")
        if "tau_min" not in chiplet:
            chiplet.tau_min = 0.5
            meta["filled"].append("chiplet.tau_min")
        if "tau_decay" not in chiplet:
            chiplet.tau_decay = 0.98
            meta["filled"].append("chiplet.tau_decay")

        # hw layout defaults
        if _get(cfg, "hw.num_slots", None) is None:
            _set(cfg, "hw.num_slots", 16)
            meta["filled"].append("hw.num_slots")
        if _get(cfg, "hw.wafer_radius_mm", None) is None:
            _set(cfg, "hw.wafer_radius_mm", 150.0)
            meta["filled"].append("hw.wafer_radius_mm")

        # hw objective weights (must exist to avoid attribute errors)
        for k, dv in [
            ("lambda_T", 1.0),
            ("lambda_E", 0.0),
            ("lambda_mem", 0.0),
            ("lambda_area", 0.0),
            ("lambda_chip", 0.0),
            ("area_limit_mm2", 1e12),
            ("lambda_boundary", 0.0),
            ("lambda_overlap", 0.0),
            ("lambda_comm_extra", 0.0),
            ("lambda_thermal", 0.0),
        ]:
            if _get(cfg, f"hw.{k}", None) is None:
                _set(cfg, f"hw.{k}", dv)
                meta["filled"].append(f"hw.{k}")

        # objective (for export_layout_input)
        layout = _ensure_section(cfg, "layout")
        if _get(cfg, "layout.sigma_mm", None) is None:
            _set(cfg, "layout.sigma_mm", 20.0)
            meta["filled"].append("layout.sigma_mm")
        scalar = _ensure_section(layout, "scalar_weights")
        if _get(cfg, "layout.scalar_weights.w_comm", None) is None:
            scalar.w_comm = 0.7
            meta["filled"].append("layout.scalar_weights.w_comm")
        if _get(cfg, "layout.scalar_weights.w_therm", None) is None:
            scalar.w_therm = 0.3
            meta["filled"].append("layout.scalar_weights.w_therm")
        if _get(cfg, "layout.scalar_weights.w_penalty", None) is None:
            scalar.w_penalty = 1000.0
            meta["filled"].append("layout.scalar_weights.w_penalty")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cfg, meta


# 兼容旧调用名
def validate_cfg(cfg: AttrDict, mode: str = "version_c") -> AttrDict:
    cfg2, _ = validate_and_fill_defaults(cfg, mode=mode)
    return cfg2
