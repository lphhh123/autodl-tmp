import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.data_ucf101 import UCF101Dataset
from utils.trace_contract_v54 import compute_effective_cfg_digest_v54


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/ast2_ucf101.yaml")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="single")

    d0 = compute_effective_cfg_digest_v54(cfg)
    train_ds = UCF101Dataset(cfg, split="train")
    val_ds = UCF101Dataset(cfg, split="val")
    d1 = compute_effective_cfg_digest_v54(cfg)

    print(f"[CHECK] seal_digest_before={d0}")
    print(f"[CHECK] seal_digest_after={d1}")
    print(f"[CHECK] len(train_ds)={len(train_ds)}")
    print(f"[CHECK] len(val_ds)={len(val_ds)}")

    if d0 != d1:
        raise RuntimeError(
            "cfg mutated after building UCF101Dataset; "
            f"seal_digest_before={d0} seal_digest_after={d1}"
        )


if __name__ == "__main__":
    main()
