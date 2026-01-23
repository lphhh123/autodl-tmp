"""Entry for Version-C full training."""
# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

import argparse
import json
import uuid
from typing import Union

from omegaconf import OmegaConf

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything
from trainer.trainer_version_c import train_version_c


def _str2bool(value: Union[str, bool]) -> bool:
    """Parse flexible boolean flags.

    Accepts common textual forms (true/false/yes/no/1/0) so users can run either
    `--export_layout_input` or `--export_layout_input true` without argparse
    rejecting the value.
    """

    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"true", "t", "yes", "y", "1"}:
        return True
    if value_lower in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected for export_layout_input, got {value!r}")


def _cfg_get_path(cfg, keypath: str, default="__MISSING__"):
    cur = cfg
    for part in keypath.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
        if cur is default:
            return default
    return cur


def _inject_baseline_stats_path(cfg, baseline_stats_path: str):
    from pathlib import Path

    from omegaconf import OmegaConf

    p = str(Path(baseline_stats_path).expanduser())

    # Ensure stable_hw exists + enabled (Version-C + StableHW contract)
    if getattr(cfg, "stable_hw", None) is None:
        cfg.stable_hw = OmegaConf.create({})
    if not hasattr(cfg.stable_hw, "enabled"):
        cfg.stable_hw.enabled = True

    # Find the *active* locked_acc_ref (root preferred by StableHW resolver)
    locked_cfg = None
    if getattr(cfg, "locked_acc_ref", None) is not None and bool(getattr(cfg.locked_acc_ref, "enabled", True)):
        locked_cfg = cfg.locked_acc_ref
    elif getattr(cfg.stable_hw, "locked_acc_ref", None) is not None and bool(
        getattr(cfg.stable_hw.locked_acc_ref, "enabled", True)
    ):
        locked_cfg = cfg.stable_hw.locked_acc_ref
    else:
        # If neither exists, create under stable_hw (avoid creating both root + stable_hw)
        cfg.stable_hw.locked_acc_ref = OmegaConf.create({"enabled": True})
        locked_cfg = cfg.stable_hw.locked_acc_ref

    # Inject where it will actually be read
    locked_cfg.baseline_stats_path = p

    # Keep a readable alias (optional, but harmless)
    cfg.stable_hw.baseline_stats_path = p

    # Also help NoDrift stats_path if it exists but unset
    try:
        if getattr(cfg, "no_drift", None) is not None and not getattr(cfg.no_drift, "stats_path", None):
            cfg.no_drift.stats_path = p
        if getattr(cfg.stable_hw, "no_drift", None) is not None and not getattr(cfg.stable_hw.no_drift, "stats_path", None):
            cfg.stable_hw.no_drift.stats_path = p
    except Exception:
        pass

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/vc_phase3_full_ucf101.yaml",
        help="v5.4 default config (explicit stable_hw + reproducible settings)",
    )
    # ---- argparse: v5.4 uses --out_dir; keep --out as alias ----
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (v5.4 contract)")
    parser.add_argument("--out", type=str, default=None, help="Alias of --out_dir (backward compat)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--export_layout_input",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Export layout_input.json per SPEC v5.4 (accepts flag alone or true/false)",
    )
    parser.add_argument(
        "--baseline_stats",
        type=str,
        default=None,
        help="Path to dense baseline metrics/stats json for LockedAccRef",
    )
    parser.add_argument("--export_dir", type=str, default=None, help="Directory to write layout artifacts")
    args = parser.parse_args()
    cli_out = args.out_dir or args.out
    if args.export_dir is None and cli_out is not None:
        args.export_dir = str(Path(cli_out) / "exports" / "layout_input")
    cfg = load_config(args.cfg)
    cfg.cfg_path = args.cfg
    seed_everything(int(args.seed))
    if hasattr(cfg, "train"):
        cfg.train.seed = int(args.seed)
    if hasattr(cfg, "training"):
        cfg.training.seed = int(args.seed)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")
    if args.baseline_stats:
        cfg = _inject_baseline_stats_path(cfg, args.baseline_stats)

        # Enforce NoDrift: disable legacy direct-sum entrances
        try:
            if hasattr(cfg, "hw") and hasattr(cfg.hw, "lambda_hw"):
                cfg.hw.lambda_hw = 0.0
            if hasattr(cfg, "loss") and hasattr(cfg.loss, "lambda_hw"):
                cfg.loss.lambda_hw = 0.0
        except Exception:
            pass

    rid = OmegaConf.select(cfg, "train.run_id")
    if not rid:
        OmegaConf.update(cfg, "train.run_id", uuid.uuid4().hex, merge=True)

    # ---- out_dir resolution (CLI highest priority) ----
    auto_out = f"outputs/version_c_auto"
    out_dir = Path(cli_out) if cli_out else Path(getattr(cfg.train, "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ---- v5.4 contract: BOTH cfg.out_dir and cfg.train.out_dir must be set ----
    cfg.out_dir = str(out_dir)
    cfg.train.out_dir = str(out_dir)

    out_dir_path = Path(out_dir)

    export_layout_input = bool(args.export_layout_input)
    export_dir = args.export_dir or str(out_dir / "exports" / "layout_input")

    # ---- dump resolved config ----
    try:
        import omegaconf

        with open(out_dir_path / "config_resolved.yaml", "w", encoding="utf-8") as f:
            f.write(omegaconf.OmegaConf.to_yaml(cfg))
        resolved_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        with open(out_dir_path / "config_resolved.yaml", "w", encoding="utf-8") as f:
            f.write(str(cfg))
        resolved_cfg = cfg

    with (out_dir_path / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, ensure_ascii=False, default=str)

    # ---- cfg_hash for audit ----
    resolved_text = (out_dir_path / "config_resolved.yaml").read_text(encoding="utf-8")
    from utils.stable_hash import stable_hash

    print("[CFG] mode =", "version_c")
    print("[CFG] stable_hw.enabled =", bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False)))
    print(
        "[CFG] stable_hw.lambda_hw_schedule.lambda_hw_max =",
        float(getattr(getattr(getattr(cfg, "stable_hw", None), "lambda_hw_schedule", None), "lambda_hw_max", -1.0)),
    )
    print("[CFG] loss.lambda_hw =", float(getattr(getattr(cfg, "loss", None), "lambda_hw", -1.0)))
    print("[CFG] hw.lambda_hw =", float(getattr(getattr(cfg, "hw", None), "lambda_hw", -1.0)))
    keypaths = [
        "stable_hw.accuracy_guard.enabled",
        "stable_hw.locked_acc_ref.enabled",
        "stable_hw.no_drift.enabled",
        "stable_hw.no_double_scale",
        "stable_hw.min_latency_ms",
    ]
    for kp in keypaths:
        print("[CFG]", kp, "=", _cfg_get_path(cfg, kp, default="__MISSING__"))

    cfg_hash = stable_hash({"cfg": resolved_text})
    cfg.train.cfg_hash = cfg_hash
    cfg.train.cfg_path = str(args.cfg)

    # ---- run meta ----
    meta = {
        "argv": sys.argv,
        "out_dir": str(out_dir_path),
        "validation": {},
    }
    with open(out_dir_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    train_version_c(cfg, export_layout_input=export_layout_input, layout_export_dir=export_dir)


if __name__ == "__main__":
    main()
