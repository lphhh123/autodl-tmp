"""Entry for AST2.0-lite single device training (SPEC)."""
import argparse
import sys
import time
from pathlib import Path

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything
from trainer.trainer_single_device import train_single_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/ast2_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="ast2")
    seed = int(args.seed)
    seed_everything(seed)
    if hasattr(cfg, "train"):
        cfg.train.seed = seed
    if hasattr(cfg, "training"):
        cfg.training.seed = seed
    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/ast2") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "train"):
        cfg.train.out_dir = str(out_dir)
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        f.write(Path(args.cfg).read_text(encoding="utf-8"))
    try:
        import omegaconf

        (out_dir / "config_resolved.yaml").write_text(omegaconf.OmegaConf.to_yaml(cfg), encoding="utf-8")
    except Exception:
        (out_dir / "config_resolved.yaml").write_text(str(cfg), encoding="utf-8")

    # ---- run_manifest (v5.4) ----
    from utils.run_manifest import write_run_manifest

    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    extra = {"repo_root": str(Path(__file__).resolve().parents[1]), "dataset_id": getattr(getattr(cfg, "data", {}), "dataset_id", None)}
    manifest = write_run_manifest(
        out_dir=out_dir,
        cfg_resolved_text=resolved_text,
        cfg_path=str(args.cfg),
        argv=sys.argv,
        seed=int(seed),
        spec_version="v5.4",
        extra=extra,
    )
    cfg.run_id = manifest["run_id"]
    train_single_device(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
