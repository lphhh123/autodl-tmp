"""Entry for Version-C full training."""
import argparse
from typing import Union

from utils.config import load_config
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
    cfg = load_config(args.cfg)
    train_version_c(cfg, export_layout_input=args.export_layout_input, export_dir=args.export_dir)


if __name__ == "__main__":
    main()
