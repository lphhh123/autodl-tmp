import json
from pathlib import Path

from utils.io_compat import cleanup_utils, init_utils


def test_open_write(tmp_path: Path):
    log_path = tmp_path / "io_log.jsonl"
    init_utils({"log_path": log_path})
    try:
        with open(tmp_path / "trace.csv", "w", encoding="utf-8") as fp:
            fp.write("a,b\n")
    finally:
        cleanup_utils()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["basename"] == "trace.csv" and e["op_kind"] == "file.write" for e in events)
