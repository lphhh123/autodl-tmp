from pathlib import Path

from utils.io_compat import cleanup_utils, init_utils


def test_cleanup_restore(tmp_path: Path):
    log_path = tmp_path / "io_log.jsonl"

    init_utils({"log_path": log_path})
    cleanup_utils()

    with open(tmp_path / "trace.csv", "w", encoding="utf-8") as fp:
        fp.write("a\n")

    assert (tmp_path / "trace.csv").exists()
    if log_path.exists():
        assert log_path.read_text(encoding="utf-8") == ""
