import json
import os
from pathlib import Path

from utils.io_compat import cleanup_utils, init_utils


def test_fsync_forward(tmp_path: Path):
    log_path = tmp_path / "io_log.jsonl"
    init_utils({"log_path": log_path})
    try:
        f = Path(tmp_path / "trace.csv").open("w", newline="")
        f.write("x\n")
        f.flush()
        os.fsync(f.fileno())
        f.close()
    finally:
        cleanup_utils()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["basename"] == "trace.csv" for e in events)
