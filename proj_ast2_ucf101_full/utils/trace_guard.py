import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import OmegaConf

from utils.trace_contract_v54 import (
    ALLOWED_EVENT_TYPES_V54,
    REQUIRED_EVENT_PAYLOAD_KEYS_V54,
    SCHEMA_VERSION_V54,
    compute_effective_cfg_digest_v54,
)
from utils.config_utils import get_nested
from utils.trace_signature_v54 import build_signature_v54
from .trace_schema import TRACE_FIELDS
from .stable_hash import stable_hash

FINALIZED_FLAG = "finalized.flag"


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_exception_json(trace_dir: Path, exc: Exception, stage: str = "") -> None:
    trace_dir = Path(trace_dir)
    payload = {
        "stage": str(stage),
        "type": type(exc).__name__,
        "message": str(exc),
        "repr": repr(exc),
        "timestamp": float(time.time()),
    }
    _write_json(trace_dir / "exception.json", payload)


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _trace_csv_candidates(trace_dir: Path) -> list[Path]:
    """
    v5.4 contract: trace.csv must live under trace_dir,
    while trace_events.jsonl/summary.json live under out_dir/trace/.
    """
    return [
        trace_dir / "trace.csv",
        trace_dir.parent / "trace.csv",
    ]


def _pick_trace_csv_path(trace_dir: Path) -> Path:
    for p in _trace_csv_candidates(trace_dir):
        if p.exists():
            return p
    # 默认写到 trace_dir/trace.csv（v5.4 contract）
    return trace_dir / "trace.csv"


def _make_min_row(
    stage: str,
    op: str,
    seed_id: int,
    iter_id: int,
    op_args: Dict[str, Any],
    signature: str,
    total_scalar: float = 0.0,
    comm_norm: float = 0.0,
    therm_norm: float = 0.0,
    objective_hash: str = "",
    time_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Minimal trace row that will NOT break csv writing and downstream int/float casts.
    Must only use keys in TRACE_FIELDS.
    """
    row = {k: "" for k in TRACE_FIELDS}
    row["iter"] = int(iter_id)
    row["stage"] = str(stage)
    row["op"] = str(op)
    row["op_args_json"] = json.dumps(op_args or {}, ensure_ascii=False)
    row["accepted"] = 1 if stage == "init" else 0
    row["seed_id"] = int(seed_id)
    row["signature"] = str(signature)

    row["total_scalar"] = float(total_scalar)
    row["comm_norm"] = float(comm_norm)
    row["therm_norm"] = float(therm_norm)

    row["delta_total"] = 0.0
    row["delta_comm"] = 0.0
    row["delta_therm"] = 0.0
    row["d_total"] = row.get("d_total", row.get("delta_total", 0.0))
    row["d_comm"] = row.get("d_comm", row.get("delta_comm", 0.0))
    row["d_therm"] = row.get("d_therm", row.get("delta_therm", 0.0))

    row["pareto_added"] = 0
    row["duplicate_penalty"] = 0.0
    row["boundary_penalty"] = 0.0

    row["tabu_hit"] = 0
    row["tabu_accept_override"] = 0
    row["repeat_signature"] = 0
    row["undo_signature"] = 0
    row["policy_switch"] = ""
    row["cache_hit"] = 0
    row["cache_enabled"] = 0
    row["eval_calls_cum"] = 0
    row["cache_hit_cum"] = 0
    row["accepted_steps_cum"] = 0
    row["score_lookahead"] = 0.0
    row["score_tts"] = 0.0
    row["score_llm"] = 0.0
    row["selection_id"] = ""
    row["step_id"] = ""
    row["heuristic_name"] = ""

    row["time_ms"] = float(time_ms)
    row["time_s"] = float(time_ms) / 1000.0

    row["objective_hash"] = str(objective_hash)
    row["cache_key"] = stable_hash({"signature": signature, "seed_id": int(seed_id), "objective_hash": str(objective_hash)})

    return row


def _read_last_csv_row(csv_path: Path) -> Optional[dict]:
    if not csv_path.exists():
        return None
    last = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            last = r
    return last


def _ensure_trace_csv_init(trace_dir: Path, seed_id: int) -> Path:
    trace_csv = _pick_trace_csv_path(trace_dir)
    trace_csv.parent.mkdir(parents=True, exist_ok=True)
    if not trace_csv.exists():
        with trace_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, restval="", extrasaction="ignore")
            w.writeheader()
            w.writerow(
                _make_min_row(
                    stage="init",
                    op="init",
                    seed_id=seed_id,
                    iter_id=-1,
                    op_args={"op": "init"},
                    signature="assign:unknown",
                    total_scalar=0.0,
                    comm_norm=0.0,
                    therm_norm=0.0,
                    objective_hash="",
                    time_ms=0.0,
                )
            )
    return trace_csv


def _ensure_trace_csv_finalize(trace_dir: Path) -> None:
    trace_csv = _pick_trace_csv_path(trace_dir)
    if not trace_csv.exists():
        _ensure_trace_csv_init(trace_dir, seed_id=0)

    last = _read_last_csv_row(trace_csv)
    if last and last.get("stage") == "finalize":
        return

    iter_id = int(last.get("iter", 0)) + 1 if last else 0
    seed_id = int(last.get("seed_id", 0)) if last else 0
    objective_hash = str(last.get("objective_hash", "")) if last else ""

    with trace_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, restval="", extrasaction="ignore")
        w.writerow(
            _make_min_row(
                stage="finalize",
                op="finalize",
                seed_id=seed_id,
                iter_id=iter_id,
                op_args={"op": "finalize"},
                signature="assign:unknown",
                total_scalar=float(last.get("total_scalar", 0.0)) if last else 0.0,
                comm_norm=float(last.get("comm_norm", 0.0)) if last else 0.0,
                therm_norm=float(last.get("therm_norm", 0.0)) if last else 0.0,
                objective_hash=objective_hash,
                time_ms=float(last.get("time_ms", 0.0)) if last else 0.0,
            )
        )


def init_trace_dir(
    trace_dir: Path,
    signature: Dict[str, Any],
    run_meta: Optional[Dict[str, Any]] = None,
    required_signature_keys: Optional[list[str]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
    requested_config: Optional[Dict[str, Any]] = None,
    contract_overrides: Optional[list[Dict[str, Any]]] = None,
    requested_cfg_yaml: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Create/initialize a trace directory with v5.4 required artifacts.

    SPEC_E requirements:
    - trace_header.json must include resolved effective config (after defaults/overrides).
    - eval_config_snapshot.yaml must contain the resolved config.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # 新 run：清掉 finalized 标记
    flag = trace_dir / FINALIZED_FLAG
    if flag.exists():
        try:
            flag.unlink()
        except Exception:
            pass

    if required_signature_keys:
        missing = [k for k in required_signature_keys if k not in signature]
        if missing:
            raise KeyError(f"trace signature missing keys: {missing}")

    run_meta = {} if run_meta is None else dict(run_meta)

    header_path = trace_dir / "trace_header.json"
    manifest_path = trace_dir / "manifest.json"
    snapshot_path = trace_dir / "eval_config_snapshot.yaml"
    requested_path = trace_dir / "config_requested.yaml"
    requested_snapshot_path = trace_dir / "requested_config_snapshot.yaml"
    effective_snapshot_path = trace_dir / "effective_config_snapshot.yaml"
    events_path = trace_dir / "trace_events.jsonl"

    # 0) Write resolved config snapshot (YAML)
    resolved_cfg_obj = None
    if resolved_config is not None:
        resolved_cfg_obj = dict(resolved_config)
        with snapshot_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved_cfg_obj, f, sort_keys=False, allow_unicode=True)
    else:
        if not snapshot_path.exists():
            snapshot_path.write_text("# eval_config_snapshot placeholder (v5.4)\n", encoding="utf-8")

    requested_cfg_obj = {}
    if requested_config is not None:
        requested_cfg_obj = dict(requested_config)
        with requested_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(requested_cfg_obj, f, sort_keys=False, allow_unicode=True)
    else:
        if not requested_path.exists():
            requested_path.write_text("# config_requested placeholder (v5.4)\n", encoding="utf-8")

    # ---- v5.4 contract: requested/effective snapshots + sha256 + seal_digest ----
    def _strip_seal_digest(obj: dict) -> dict:
        if not isinstance(obj, dict):
            return obj
        o = dict(obj)
        c = o.get("contract", None)
        if isinstance(c, dict) and "seal_digest" in c:
            c2 = dict(c)
            c2.pop("seal_digest", None)
            o["contract"] = c2
        return o

    if requested_config is None and requested_cfg_yaml:
        requested_snapshot_text = str(requested_cfg_yaml)
    else:
        requested_snapshot_text = yaml.safe_dump(
            requested_cfg_obj or {},
            sort_keys=False,
            allow_unicode=True,
        )
    requested_snapshot_path.write_text(requested_snapshot_text, encoding="utf-8")
    requested_config_sha256 = hashlib.sha256(requested_snapshot_text.encode("utf-8")).hexdigest()

    effective_snapshot_obj = _strip_seal_digest(resolved_cfg_obj or {})
    effective_snapshot_text = yaml.safe_dump(
        effective_snapshot_obj or {},
        sort_keys=False,
        allow_unicode=True,
    )
    effective_snapshot_path.write_text(effective_snapshot_text, encoding="utf-8")
    computed_seal_digest = compute_effective_cfg_digest_v54(effective_snapshot_obj)
    cfg_seal = str(get_nested(resolved_cfg_obj, "contract.seal_digest", "") or "").strip()
    if cfg_seal and cfg_seal != str(computed_seal_digest):
        raise RuntimeError(
            "[v5.4 P0][HardGate-C] cfg.contract.seal_digest mismatches computed seal. "
            f"cfg={cfg_seal} computed={computed_seal_digest}. Refuse to write non-auditable trace."
        )

    seal_digest = str(computed_seal_digest)

    # 1) trace_header.json（结构化元信息）
    # ---- contract_overrides: v5.4 合同证据，必须可审计 ----
    if contract_overrides is not None:
        overrides = list(contract_overrides)
    else:
        overrides = []
        if isinstance(resolved_cfg_obj, dict):
            c = resolved_cfg_obj.get("_contract", None)
            if isinstance(c, dict):
                overrides = c.get("overrides", []) or []
    requested_snap = requested_config
    effective_snap = effective_snapshot_obj

    trace_header_payload = build_trace_header_payload_v54(
        signature=signature,
        requested_config=requested_snap,
        effective_config=effective_snap,
        contract_overrides=overrides,
        requested=run_meta.get("requested", {}),
        effective=run_meta.get("effective", {}),
        no_drift_enabled=bool(signature.get("no_drift_enabled", True)),
        acc_ref_source=str(signature.get("acc_ref_source", "locked")),
        seal_digest=seal_digest,
    )
    header_payload = dict(trace_header_payload)
    header_payload.update(
        {
            "schema": SCHEMA_VERSION_V54,
            "run_meta": run_meta,
            "resolved_config": resolved_cfg_obj,
            "ts_ms": int(time.time() * 1000),
            "requested_config_sha256": requested_config_sha256,
            "effective_config_sha256": seal_digest,
            "requested_config": requested_cfg_obj,
            "effective_config": resolved_cfg_obj,
        }
    )
    header_payload.setdefault("requested_config", requested_snap)
    header_payload.setdefault("effective_config", effective_snap)

    # HardGate-B：trace_header.json 必须满足可审计 schema（缺字段/类型漂移直接 fail-fast）
    _assert_event("trace_header", header_payload)
    _write_json(header_path, header_payload)

    # 2) manifest.json
    manifest = {
        "schema": "v5.4",
        "signature": signature,
        "run_meta": run_meta,
        "resolved_config_path": str(snapshot_path),
        "requested_config_path": str(requested_path),
        "created_ts_ms": int(time.time() * 1000),
    }
    manifest["requested_config_snapshot"] = requested_snapshot_path.name
    manifest["effective_config_snapshot"] = effective_snapshot_path.name
    manifest["requested_config_sha256"] = requested_config_sha256
    manifest["effective_config_sha256"] = seal_digest
    manifest["seal_digest"] = seal_digest
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json(manifest_path, manifest)

    # 3) trace_events.jsonl：仅确保文件存在（trace_header 由调用方写入）
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")

    # 4) trace.csv：保证至少有 init 行（训练场景也不会因缺 trace.csv 而在收尾崩溃）
    seed_id = int(run_meta.get("seed_id", 0) or 0)
    _ensure_trace_csv_init(trace_dir, seed_id=seed_id)

    return {
        "trace_dir": trace_dir,
        "trace_header": header_path,
        "manifest": manifest_path,
        "eval_config_snapshot": snapshot_path,
        "config_requested": requested_path,
        "trace_events": events_path,
        "trace_csv": _pick_trace_csv_path(trace_dir),
    }


def init_trace_dir_v54(
    base_dir: Path,
    run_id: str,
    cfg: Any,
    signature: Dict[str, Any],
    signature_v54: Dict[str, Any],
    required_signature_fields: Optional[list[str]] = None,
    run_meta: Optional[Dict[str, Any]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    global _DEFAULT_RUN_ID
    _DEFAULT_RUN_ID = str(run_id)
    trace_dir = Path(base_dir) / str(run_id)
    resolved_config = None
    if cfg is not None:
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved_config, dict):
            raise RuntimeError("[P0][v5.4] resolved_config must be a dict for auditable trace_header.")
        seal_digest = compute_effective_cfg_digest_v54(resolved_config)
        if not hasattr(cfg, "contract") or getattr(cfg, "contract", None) is None:
            try:
                cfg.contract = {}
            except Exception:
                pass
        prev_seal = getattr(getattr(cfg, "contract", None), "seal_digest", None)
        if prev_seal != seal_digest:
            try:
                if not hasattr(cfg, "_contract") or getattr(cfg, "_contract", None) is None:
                    cfg._contract = {}
                ov = getattr(cfg._contract, "overrides", None)
                if ov is None:
                    cfg._contract.overrides = []
                cfg._contract.overrides.append(
                    {
                        "path": "contract.seal_digest",
                        "requested": prev_seal,
                        "effective": seal_digest,
                        "reason": "sync_seal_digest_to_effective_snapshot_v5.4",
                    }
                )
            except Exception:
                pass
        try:
            cfg.contract.seal_digest = seal_digest
        except Exception:
            pass
    requested_cfg_yaml = None
    if cfg is not None:
        requested_cfg_yaml = getattr(getattr(cfg, "train", None), "requested_cfg_yaml", None)
    requested_config = None
    if cfg is not None:
        contract = cfg.get("_contract", {}) if isinstance(cfg, dict) else getattr(cfg, "_contract", None)
        if contract is None and hasattr(cfg, "get"):
            contract = cfg.get("_contract", {})
        if contract is None:
            contract = {}
        snapshot = contract.get("requested_config_snapshot", {}) if isinstance(contract, dict) else getattr(
            contract, "requested_config_snapshot", {}
        )
        if snapshot is None:
            requested_config = None
        elif OmegaConf.is_config(snapshot):
            requested_config = OmegaConf.to_container(snapshot, resolve=False)
        elif isinstance(snapshot, dict):
            requested_config = snapshot
        else:
            # 最保守：强制转成 dict，避免 header 缺字段
            requested_config = {"_unsupported_requested_snapshot_type": str(type(snapshot))}
        if requested_config is not None and not isinstance(requested_config, dict):
            raise RuntimeError("[P0][v5.4] requested_config must be a dict for auditable trace_header.")
    contract_overrides = get_nested(cfg, "_contract.overrides", [])
    if contract_overrides is None:
        contract_overrides = []
    meta = init_trace_dir(
        trace_dir=trace_dir,
        signature=signature_v54 or signature,
        run_meta=run_meta,
        required_signature_keys=required_signature_fields,
        extra_manifest=extra_manifest,
        resolved_config=resolved_config,
        requested_config=requested_config,
        contract_overrides=contract_overrides,
        requested_cfg_yaml=requested_cfg_yaml,
    )
    snapshot_path = Path(meta["eval_config_snapshot"])
    if cfg is not None:
        try:
            snapshot_txt = OmegaConf.to_yaml(cfg)
        except Exception as exc:
            raise RuntimeError(
                "[P0][v5.4] Failed to snapshot eval_config; trace would be non-auditable."
            ) from exc
        snapshot_path.write_text(snapshot_txt, encoding="utf-8")
    return meta


_DEFAULT_RUN_ID: Optional[str] = None


def _assert_event(event_type: str, payload: dict) -> None:
    if event_type not in ALLOWED_EVENT_TYPES_V54:
        raise ValueError(f"Unknown v5.4 event_type={event_type}")

    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")

    req_keys = REQUIRED_EVENT_PAYLOAD_KEYS_V54.get(event_type, ())
    for k in req_keys:
        if k not in payload:
            raise KeyError(f"{event_type}.payload missing '{k}'")

    if event_type == "trace_header":
        if not isinstance(payload["signature"], dict):
            raise TypeError("trace_header.payload.signature must be dict")
        for field in ("requested_config_snapshot", "effective_config_snapshot"):
            val = payload.get(field)
            if not isinstance(val, (dict, str)) or (isinstance(val, str) and not val.strip()):
                raise ValueError(f"trace_header.payload.{field} must be non-empty")
        seal_val = payload.get("seal_digest")
        if not isinstance(seal_val, str) or not seal_val.strip():
            raise ValueError("trace_header.payload.seal_digest must be non-empty str")
        # ---- SPEC_E: contract_overrides must be auditable entries ----
        ov = payload.get("contract_overrides", [])
        if not isinstance(ov, list):
            raise ValueError("trace_header.contract_overrides must be a list")
        for i, it in enumerate(ov):
            if not isinstance(it, dict):
                raise ValueError(f"contract_overrides[{i}] must be a dict")
            for k in ("path", "requested", "effective", "reason"):
                if k not in it:
                    raise ValueError(f"contract_overrides[{i}] missing key: {k}")
            if not isinstance(it["path"], str) or not it["path"].strip():
                raise ValueError(f"contract_overrides[{i}].path must be non-empty str")
            if not isinstance(it["reason"], str) or not it["reason"].strip():
                raise ValueError(f"contract_overrides[{i}].reason must be non-empty str")
        return

    if event_type == "gating":
        _assert_gating_event(payload)
        return

    if event_type == "proxy_sanitize":
        _assert_proxy_sanitize_event(payload)
        return

    if event_type == "ref_update":
        return

    if event_type == "finalize":
        return


def _assert_gating_event(payload: Dict[str, Any]) -> None:
    req = set(REQUIRED_EVENT_PAYLOAD_KEYS_V54.get("gating", ()))
    missing = [k for k in req if k not in payload]
    assert not missing, f"gating payload missing keys: {missing}"
    assert payload["gate"] in ("allow_hw", "reject_hw"), f"bad gate={payload['gate']}"
    float(payload["acc_drop"])
    float(payload["acc_drop_max"])


def _assert_proxy_sanitize_event(payload: Dict[str, Any]) -> None:
    req = set(REQUIRED_EVENT_PAYLOAD_KEYS_V54.get("proxy_sanitize", ()))
    missing = [k for k in req if k not in payload]
    assert not missing, f"proxy_sanitize payload missing keys: {missing}"
    assert isinstance(payload["metric"], str) and payload["metric"], "metric must be non-empty str"
    float(payload["raw_value"])
    float(payload["used_value"])
    float(payload["penalty_added"])


def append_trace_event_v54(
    path: Path,
    event_type: str,
    payload: Optional[dict] = None,
    run_id: Optional[str] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    outer_iter: Optional[int] = None,
    strict: bool = True,
):
    if payload is None:
        payload = {}

    if strict:
        _assert_event(event_type, payload)

    path = Path(path)
    flag = path.parent / FINALIZED_FLAG
    if flag.exists() and event_type != "finalize":
        raise RuntimeError(f"trace_events is finalized; refusing to append event_type={event_type}")

    if run_id is None:
        run_id = _DEFAULT_RUN_ID or "unknown"
    if step is None:
        if outer_iter is not None:
            step = int(outer_iter)
        elif epoch is not None:
            step = int(epoch)
        else:
            step = 0

    _append_jsonl(
        path,
        {
            "ts_ms": int(time.time() * 1000),
            "run_id": str(run_id),
            "step": int(step),
            "event_type": str(event_type),
            "payload": payload,
        },
    )


def finalize_trace_events(path: Path, payload: dict, run_id: str, step: int):
    append_trace_event_v54(path, "finalize", payload=payload, run_id=run_id, step=step)


def update_trace_summary(
    trace_dir: Path,
    ok: Any,
    reason: Optional[str] = None,
    steps_done: Optional[int] = None,
    best_solution_valid: Optional[bool] = None,
) -> None:
    """
    v5.4 compat:
    - Accept both legacy signature:
        update_trace_summary(trace_dir, ok, reason, steps_done, best_solution_valid)
    - And new call style used across this repo:
        update_trace_summary(trace_dir, {"reason":..., "steps_done":..., "best_solution_valid":..., ...})
    """
    trace_dir = Path(trace_dir)

    extra = {}
    if isinstance(ok, dict) and reason is None and steps_done is None and best_solution_valid is None:
        payload = ok
        reason = str(payload.get("reason", "unknown"))
        steps_done = int(payload.get("steps_done", 0) or 0)
        best_solution_valid = bool(payload.get("best_solution_valid", False))
        ok_val = payload.get("ok", None)
        if ok_val is None:
            ok_val = reason in ("done", "steps0", "ok", "success")
        ok = bool(ok_val)

        extra = {
            k: v
            for k, v in payload.items()
            if k not in ("ok", "reason", "steps_done", "best_solution_valid")
        }
    else:
        ok = bool(ok)
        reason = str(reason if reason is not None else "unknown")
        steps_done = int(steps_done if steps_done is not None else 0)
        best_solution_valid = bool(best_solution_valid if best_solution_valid is not None else False)

    trace_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": "v5.4",
        "ok": ok,
        "reason": reason,
        "steps_done": steps_done,
        "best_solution_valid": best_solution_valid,
        "ts_ms": int(time.time() * 1000),
    }
    if extra:
        out.update(extra)

    _write_json(trace_dir / "summary.json", out)


def _get_cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        if OmegaConf.is_config(cfg):
            val = OmegaConf.select(cfg, path)
            return default if val is None else val
    except Exception:
        pass
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def build_baseline_trace_summary(cfg: Any, stable_hw_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    strict = bool(_get_cfg_value(cfg, "contract.strict", False) or _get_cfg_value(cfg, "_contract.strict", False))
    locked_path = _get_cfg_value(cfg, "stable_hw.locked_acc_ref.baseline_stats_path", None)
    if strict:
        # LEGAL: strict mode forbids legacy/root-level sources
        if _get_cfg_value(cfg, "locked_acc_ref", None) is not None:
            raise ValueError("P0: strict forbids legacy root key locked_acc_ref (should have failed in validate).")
    elif locked_path is None:
        locked_path = _get_cfg_value(cfg, "locked_acc_ref.baseline_stats_path", None)
    hw_path = _get_cfg_value(cfg, "stable_hw.baseline_stats_path", None)
    baseline_path = str(locked_path or hw_path or "").strip()

    acc_ref_source = ""
    hw_ref_source = ""
    if isinstance(stable_hw_state, dict):
        acc_ref_source = str(stable_hw_state.get("acc_ref_source", "") or "")
        hw_ref_source = str(stable_hw_state.get("hw_ref_source", "") or "")
    if not acc_ref_source:
        acc_ref_source = str(
            _get_cfg_value(cfg, "stable_hw.locked_acc_ref.source", None)
            or _get_cfg_value(cfg, "locked_acc_ref.source", "")
            or ""
        )
    if not hw_ref_source:
        hw_ref_source = str(_get_cfg_value(cfg, "stable_hw.hw_ref_source", "") or "")

    is_placeholder = None
    if baseline_path:
        try:
            p = Path(baseline_path)
            if p.exists() and p.is_file():
                obj = json.loads(p.read_text(encoding="utf-8"))
                note = str(obj.get("note", "")).lower()
                if obj.get("is_placeholder") is True or ("placeholder" in note):
                    is_placeholder = True
                else:
                    is_placeholder = False
        except Exception:
            is_placeholder = None

    return {
        "baseline_stats_path": baseline_path,
        "hw_baseline_stats_path": str(hw_path or "").strip(),
        "acc_ref_source": str(acc_ref_source),
        "hw_ref_source": str(hw_ref_source),
        "is_placeholder": is_placeholder,
    }


def ensure_trace_events(trace_dir: Path) -> Path:
    """
    兼容旧逻辑：保证 trace_events.jsonl 至少存在；真正的 trace_header/finalize 由 init/finalize 管控。
    """
    trace_dir = Path(trace_dir)
    p = trace_dir / "trace_events.jsonl"
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def finalize_trace_dir(trace_events_path: Path, *, reason: str, steps_done: int, best_solution_valid: bool):
    """
    v5.4 合同收尾：
    - 确保 summary.json 存在（否则写一个最小的）
    - 确保 trace.csv 最后一行是 finalize（否则自动补一行）
    - 追加 trace_events.jsonl 的 finalize 事件，并写 finalized.flag，禁止后续再 append
    - 最后做 required files 校验
    """
    trace_events_path = Path(trace_events_path)
    trace_dir = trace_events_path.parent
    required = [
        "trace_header.json",
        "manifest.json",
        "eval_config_snapshot.yaml",
        "config_requested.yaml",
        "trace_events.jsonl",
    ]

    def _read_first_event_type(p: Path) -> Optional[str]:
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    return obj.get("event_type")
                except Exception:
                    return None
        return None

    first_type = _read_first_event_type(trace_events_path)
    if first_type != "trace_header":
        raise RuntimeError(
            "[v5.4 TRACE CONTRACT] trace_events.jsonl must start with trace_header, "
            f"got {first_type} (trace_events={trace_events_path})."
        )

    # 0) summary.json 兜底
    summary_path = trace_dir / "summary.json"
    if not summary_path.exists():
        update_trace_summary(
            trace_dir,
            ok=False,
            reason="missing_summary_autofill",
            steps_done=int(steps_done),
            best_solution_valid=bool(best_solution_valid),
        )

    summary = _read_json(summary_path, default={})
    manifest = _read_json(trace_dir / "manifest.json", default={})
    run_meta = manifest.get("run_meta", {}) or {}
    signature = manifest.get("signature", {}) or {}

    run_id = run_meta.get("run_id") or signature.get("run_id") or "unknown"
    seed_id = int(run_meta.get("seed_id", 0) or 0)
    step_hint = max(0, int(steps_done))

    # 1) trace.csv finalize 行兜底
    _ensure_trace_csv_finalize(trace_dir)

    # 2) finalize 事件：只允许写一次，并且必须是最后一条
    flag = trace_dir / FINALIZED_FLAG
    if not trace_events_path.exists():
        trace_events_path.write_text("", encoding="utf-8")

    if not flag.exists():
        finalize_trace_events(
            trace_events_path,
            payload={
                "reason": str(reason),
                "steps_done": int(steps_done),
                "best_solution_valid": bool(best_solution_valid),
                "status": "ok",
                "summary": {
                    "reason": str(reason),
                    "steps_done": int(steps_done),
                    "best_solution_valid": bool(best_solution_valid),
                },
            },
            run_id=str(run_id),
            step=int(step_hint),
        )
        flag.write_text(f"finalized_ts_ms={int(time.time() * 1000)}\n", encoding="utf-8")

    # 3) required files 校验
    for name in required:
        p = trace_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing required trace file: {p}")

    # trace.csv must exist in trace_dir (v5.4 contract)
    csv_path = trace_dir / "trace.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required trace.csv at {csv_path}")


def build_trace_signature_v54(
    *,
    cfg: Any,
    run_id: str,
    seal_digest: str,
    method_name: str = "ast2_single_device",
) -> dict:
    overrides = {"run_id": str(run_id)}
    if seal_digest:
        overrides["seal_digest"] = str(seal_digest)
    return build_signature_v54(cfg, method_name=method_name, overrides=overrides)


def build_trace_header_payload_v54(
    *,
    signature: dict,
    requested_config=None,
    effective_config=None,
    contract_overrides=None,
    requested: dict | None = None,
    effective: dict | None = None,
    no_drift_enabled: bool | None = None,
    acc_ref_source: str | None = None,
    seal_digest: str | None = None,
    cfg: Any | None = None,
    requested_config_snapshot=None,
    effective_config_snapshot=None,
) -> dict:
    if requested_config is None and requested_config_snapshot is not None:
        requested_config = requested_config_snapshot
    if effective_config is None and effective_config_snapshot is not None:
        effective_config = effective_config_snapshot
    payload = {}
    payload["signature"] = dict(signature or {})

    # canonical snapshots required by HardGate-B
    payload["requested_config_snapshot"] = requested_config
    payload["effective_config_snapshot"] = effective_config
    payload["contract_overrides"] = list(contract_overrides or [])

    # seal digest must be present in header evidence chain
    payload["seal_digest"] = str(payload["signature"].get("seal_digest", "") or seal_digest or "")
    if not payload["seal_digest"] and isinstance(effective_config, dict):
        payload["seal_digest"] = str(get_nested(effective_config, "contract.seal_digest", "") or "")

    # optional backward-compatible aliases
    payload["requested_config"] = requested_config
    payload["effective_config"] = effective_config

    payload["requested"] = dict(requested or {})
    payload["effective"] = dict(effective or {})

    # keep existing booleans used by SPEC_E validators
    if no_drift_enabled is None:
        no_drift_enabled = bool(payload["signature"].get("no_drift_enabled", False))
    if acc_ref_source is None:
        acc_ref_source = payload["signature"].get("acc_ref_source", "unknown")
    payload["no_drift_enabled"] = bool(no_drift_enabled)
    payload["no_double_scale_enabled"] = bool(payload["signature"].get("no_double_scale_enabled", False))
    payload["acc_ref_source"] = str(acc_ref_source)
    payload["locked_acc_ref_enabled"] = bool(payload["signature"].get("locked_acc_ref_enabled", False))
    payload["acc_first_hard_gating_enabled"] = bool(payload["signature"].get("acc_first_hard_gating_enabled", False))
    payload["git_commit_or_version"] = str(signature.get("git_commit_or_version", ""))
    payload["config_fingerprint"] = str(signature.get("config_fingerprint", ""))
    payload["seed_global"] = int(signature.get("seed_global", 0))
    payload["seed_problem"] = int(signature.get("seed_problem", 0))
    return payload
