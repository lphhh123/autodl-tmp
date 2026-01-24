from pathlib import Path
import tempfile
import subprocess
import sys
import textwrap

cfg_bad = textwrap.dedent(
    """
train:
  mode: version_c
no_drift:
  enabled: true
"""
).strip()

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "bad.yaml"
    p.write_text(cfg_bad, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "scripts/run_version_c.py", "--cfg", str(p)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "legacy key 'no_drift'" in (r.stderr + r.stdout)
print("OK: strict legacy gate works")
