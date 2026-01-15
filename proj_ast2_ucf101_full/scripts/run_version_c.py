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
import time
from typing import Union

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/version_c_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--export_layout_input",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Export layout_input.json per SPEC v4.3.2 (accepts flag alone or true/false)",
    )
    parser.add_argument("--export_dir", type=str, default=None, help="Directory to write layout artifacts")
    args = parser.parse_args()
    if args.export_dir is None and args.out_dir is not None:
        args.export_dir = str(Path(args.out_dir) / "exports" / "layout_input")
    cfg = load_config(args.cfg)
    seed_everything(int(args.seed))
    if hasattr(cfg, "train"):
        cfg.train.seed = int(args.seed)
    if hasattr(cfg, "training"):
        cfg.training.seed = int(args.seed)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    # out_dir: CLI 优先；否则 cfg.train.out_dir；否则自动生成一个
    cfg_name = Path(args.cfg).stem
    auto_out = Path("outputs/version_c") / f"{cfg_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(getattr(cfg.train, "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.out_dir = str(out_dir)

    out_dir_path = Path(out_dir)

    export_layout_input = bool(args.export_layout_input)
    export_dir = args.export_dir or str(out_dir / "exports" / "layout_input")

    # ---- dump resolved config ----
    try:
        import omegaconf

        with open(out_dir_path / "config_resolved.yaml", "w", encoding="utf-8") as f:
            f.write(omegaconf.OmegaConf.to_yaml(cfg))
    except Exception:
        with open(out_dir_path / "config_resolved.yaml", "w", encoding="utf-8") as f:
            f.write(str(cfg))

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
