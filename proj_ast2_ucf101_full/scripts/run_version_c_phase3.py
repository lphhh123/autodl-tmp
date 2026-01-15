# proj_ast2_ucf101_full/scripts/run_version_c_phase3.py
# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

raise RuntimeError(
    "run_version_c_phase3.py 已废弃（会调用旧 trainers/version_c）。请使用：scripts/run_version_c.py"
)
