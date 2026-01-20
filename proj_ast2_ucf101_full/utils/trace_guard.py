import os
import json
from datetime import datetime


def ensure_trace_file(path: str, header: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "trace_header",
                        "time": datetime.utcnow().isoformat(),
                        **header,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_trace_event(path: str, event: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
