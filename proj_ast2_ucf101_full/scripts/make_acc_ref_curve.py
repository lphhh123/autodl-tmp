"""Build a deterministic LockedAccRef curve snapshot from an experiment stdout log."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


_VAL_RE = re.compile(
    r"\[val\]\s+epoch=(?P<ep>\d+)\s+mode=(?P<mode>\w+)\s+acc_clip=(?P<acc_clip>[0-9]*\.?[0-9]+)\s+acc_video=(?P<acc_video>[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def _ema_smooth(xs: List[float], alpha: float) -> List[float]:
    if not xs:
        return []
    alpha = min(1.0, max(0.0, float(alpha)))
    out: List[float] = []
    m: Optional[float] = None
    for v in xs:
        m = float(v) if m is None else alpha * float(v) + (1.0 - alpha) * float(m)
        out.append(float(m))
    return out


def parse_stdout_curve(stdout_path: Path, prefer: str = "fast") -> Dict[int, float]:
    prefer = (prefer or "fast").lower().strip()
    best: Dict[int, float] = {}
    alt: Dict[int, float] = {}

    for line in stdout_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _VAL_RE.search(line)
        if not m:
            continue
        ep = int(m.group("ep"))
        mode = str(m.group("mode") or "").lower().strip()
        # Do NOT mix final test-set evaluation into the reference curve.
        # LockedAccRef is meant to reflect the validation protocol used for guard.
        if mode == "test":
            continue
        acc = float(m.group("acc_video"))
        if mode == "fast":
            best.setdefault(ep, acc)
        elif mode == "full":
            alt.setdefault(ep, acc)
        else:
            alt.setdefault(ep, acc)

    primary, secondary = (alt, best) if prefer == "full" else (best, alt)
    out: Dict[int, float] = {}
    all_eps = set(primary.keys()) | set(secondary.keys())
    for ep in sorted(all_eps):
        a = primary.get(ep)
        b = secondary.get(ep)
        if prefer == "min" and a is not None and b is not None:
            out[ep] = float(min(a, b))
        elif a is not None:
            out[ep] = float(a)
        elif b is not None:
            out[ep] = float(b)
    return out


def build_curve(acc_map: Dict[int, float]) -> List[float]:
    if not acc_map:
        return []
    max_ep = max(acc_map.keys())
    out: List[float] = []
    last: Optional[float] = None
    for ep in range(max_ep + 1):
        if ep in acc_map:
            last = float(acc_map[ep])
        if last is None:
            last = 0.0
        out.append(float(last))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefer", default="fast", choices=["fast", "full", "min"])
    ap.add_argument("--ema-alpha", type=float, default=0.2)
    ap.add_argument("--curve-margin", type=float, default=0.0)
    args = ap.parse_args()

    stdout_path = Path(args.stdout)
    if not stdout_path.exists():
        raise SystemExit(f"stdout not found: {stdout_path}")

    acc_map = parse_stdout_curve(stdout_path, prefer=args.prefer)
    if not acc_map:
        raise SystemExit(f"No [val] lines parsed from: {stdout_path}")

    curve_raw = build_curve(acc_map)
    curve_ema = _ema_smooth(curve_raw, alpha=args.ema_alpha) if len(curve_raw) > 1 else list(curve_raw)
    curve = list(curve_ema)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_stdout": str(stdout_path),
        "prefer": str(args.prefer),
        "ema_alpha": float(args.ema_alpha),
        "curve_margin": float(args.curve_margin),
        "curve_raw": curve_raw,
        "curve_ema": curve_ema,
        "curve": curve,
        "max_epoch": int(len(curve) - 1),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote acc_ref_curve: {out_path} (len={len(curve)})")


if __name__ == "__main__":
    main()
