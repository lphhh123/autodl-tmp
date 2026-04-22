import json
from pathlib import Path

from utils.io_compat import cleanup_utils, init_utils


def test_path_open_write(tmp_path: Path):
    log_path = tmp_path / "io_log.jsonl"
    init_utils({"log_path": log_path})
    try:
        with Path(tmp_path / "report.json").open("w", encoding="utf-8") as fp:
            fp.write("{}")
    finally:
        cleanup_utils()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["basename"] == "report.json" and e["op_kind"] == "file.write" for e in events)
