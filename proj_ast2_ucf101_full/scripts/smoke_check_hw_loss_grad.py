# NOTE: This smoke is bound to SPEC_E v5.4; do not change fields without updating SPEC_E + trace_contract_v54.py
# SPEC_E S3 entrypoint: delegates to smoke_check_hw_grad.main
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.smoke_check_hw_grad import main  # noqa: F401

if __name__ == "__main__":
    main()
