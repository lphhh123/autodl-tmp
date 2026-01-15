# proj_ast2_ucf101_full/tests/conftest.py
import sys
from pathlib import Path

# Ensure project root (proj_ast2_ucf101_full/) is importable so `import layout.*` works anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
