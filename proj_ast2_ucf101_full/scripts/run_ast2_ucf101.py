"""Entry for AST2.0-lite single device training (SPEC)."""
import argparse
import json
import uuid
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from omegaconf import OmegaConf

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything
from trainer.trainer_single_device import train_single_device


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/ast2_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (v5.4 contract)")
    parser.add_argument("--out", type=str, default=None, help="Alias of --out_dir (backward compat)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_legacy", action="store_true", help="Allow running legacy AST2 script.")
    args = parser.parse_args()
    if not args.allow_legacy:
        raise SystemExit(
            "[v5.4] This script is LEGACY (AST2). Refusing to run by default.\n"
            "Use scripts/run_version_c.py for v5.4 experiments.\n"
            "If you really want legacy, pass --allow_legacy."
        )
    cfg = load_config(args.cfg)
    cfg.cfg_path = args.cfg
    seed = int(args.seed)
    rid = OmegaConf.select(cfg, "train.run_id")
    if not rid:
        OmegaConf.update(cfg, "train.run_id", uuid.uuid4().hex, merge=True)
    cli_out = args.out_dir or args.out
    auto_out = f"outputs/ast2_auto"
    out_dir = Path(cli_out) if cli_out else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = str(out_dir)
    if not hasattr(cfg, "train") or cfg.train is None:
        cfg.train = {}
    if hasattr(cfg, "train"):
        cfg.train.out_dir = str(out_dir)
        cfg.train.seed = seed
    if hasattr(cfg, "training"):
        cfg.training.seed = seed
    if not hasattr(cfg, "_contract") or cfg._contract is None:
        cfg._contract = {}
    overrides = cfg._contract.get("overrides", []) or []
    overrides.append(
        {
            "path": "train.mode",
            "requested": "ast2",
            "effective": "single",
            "reason": "mode_alias_ast2_to_single",
        }
    )
    cfg._contract["overrides"] = overrides
    cfg = validate_and_fill_defaults(cfg, mode="single")
    seed_everything(seed)
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        f.write(Path(args.cfg).read_text(encoding="utf-8"))
    try:
        import omegaconf

        (out_dir / "config_resolved.yaml").write_text(omegaconf.OmegaConf.to_yaml(cfg), encoding="utf-8")
        resolved_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        (out_dir / "config_resolved.yaml").write_text(str(cfg), encoding="utf-8")
        resolved_cfg = cfg

    (out_dir / "resolved_config.json").write_text(
        json.dumps(resolved_cfg, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ---- cfg_hash for audit ----
    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    from utils.stable_hash import stable_hash

    print("[CFG] mode =", "ast2")
    print("[CFG] stable_hw.enabled =", bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False)))
    print(
        "[CFG] stable_hw.lambda_hw_schedule.lambda_hw_max =",
        float(getattr(getattr(getattr(cfg, "stable_hw", None), "lambda_hw_schedule", None), "lambda_hw_max", -1.0)),
    )
    print("[CFG] loss.lambda_hw =", float(getattr(getattr(cfg, "loss", None), "lambda_hw", -1.0)))
    print("[CFG] hw.lambda_hw =", float(getattr(getattr(cfg, "hw", None), "lambda_hw", -1.0)))
    keypaths = [
        "stable_hw.enabled",
        "stable_hw.no_double_scale",
        "stable_hw.accuracy_guard.enabled",
        "stable_hw.accuracy_guard.metric",
        "stable_hw.locked_acc_ref.enabled",
        "stable_hw.locked_acc_ref.source",
        "stable_hw.no_drift.enabled",
    ]
    for kp in keypaths:
        print("[CFG]", kp, "=", _cfg_get_path(cfg, kp, default="__MISSING__"))

    cfg_hash = stable_hash({"cfg": resolved_text})
    train_single_device(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
