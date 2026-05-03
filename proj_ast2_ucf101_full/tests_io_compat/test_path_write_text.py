import json
from pathlib import Path

from utils.io_compat import cleanup_utils, init_utils


def test_path_write_text(tmp_path: Path):
    log_path = tmp_path / "io_log.jsonl"
    init_utils({"log_path": log_path})
    try:
        Path(tmp_path / "budget.json").write_text("{}", encoding="utf-8")
    finally:
        cleanup_utils()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["basename"] == "budget.json" and e["op_kind"] == "path.write_text" for e in events)
