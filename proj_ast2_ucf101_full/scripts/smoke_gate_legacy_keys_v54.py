import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import AttrDict
from utils.config_validate import validate_and_fill_defaults


def main() -> None:
    bad = AttrDict(
        {
            "train": {"mode": "version_c"},
            "no_drift": {"enabled": True},  # legacy root-level (forbidden)
            "stable_hw": {"enabled": True, "no_drift": {"enabled": True}, "locked_acc_ref": {"enabled": True}},
        }
    )
    try:
        validate_and_fill_defaults(bad, mode="version_c")
        raise AssertionError("Expected fail-fast on legacy root-level key, but passed.")
    except Exception:
        print("[SMOKE] PASS: legacy keys rejected in strict mode")


if __name__ == "__main__":
    main()
