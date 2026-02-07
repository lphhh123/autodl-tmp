"""Entry for AST2.0-lite single device training (SPEC)."""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from omegaconf import OmegaConf

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.check_cfg_integrity import check_cfg_integrity
from utils.seed import seed_everything
from utils.torch_backend import maybe_enable_tf32
from utils.trace_guard import init_trace_dir_v54
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS
from trainer.trainer_single_device import train_single_device


def _cfg_get_path(cfg, keypath: str, default="__MISSING__"):
    cur = cfg
    for part in keypath.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
        if cur is default:
            return default
    return cur


def _is_smoke() -> bool:
    v = str(os.environ.get("SMOKE", "0")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def apply_smoke_overrides_single(cfg):
    """Short preflight run to validate the full pipeline can finish."""
    if not _is_smoke():
        return cfg

    try:
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass

    if OmegaConf.select(cfg, "train.epochs") is not None:
        cfg.train.epochs = 1
    if OmegaConf.select(cfg, "train.warmup_epochs") is not None:
        cfg.train.warmup_epochs = 1

    # SMOKE: shorten AST dense warmup stage (default in configs/ast2_ucf101.yaml is 15)
    # This affects only SMOKE=1 runs and helps hit pruning code-path quickly.
    if OmegaConf.select(cfg, "ast.schedule.warmup_epochs") is not None:
        cfg.ast.schedule.warmup_epochs = 1

    # --- Aggressively reduce clip count for SMOKE (do NOT affect formal runs) ---
    # code semantics: stride = int(clip_len * stride_ratio), bigger => fewer clips
    if OmegaConf.select(cfg, "data.train_stride_ratio") is not None:
        cfg.data.train_stride_ratio = 10.0
    if OmegaConf.select(cfg, "data.eval_stride_ratio") is not None:
        cfg.data.eval_stride_ratio = 10.0

    if OmegaConf.select(cfg, "val.fast_max_batches") is not None:
        cfg.val.fast_max_batches = min(int(cfg.val.fast_max_batches), 20)
    if OmegaConf.select(cfg, "val.full_every_epochs") is not None:
        cfg.val.full_every_epochs = 1

    # --- FORCE final test in smoke (to reproduce late-stage crashes) ---
    if OmegaConf.select(cfg, "test.run_final_test") is not None:
        cfg.test.run_final_test = True
    # Try to bound final-test cost if supported by your code.
    # (If unused, harmless.)
    if OmegaConf.select(cfg, "test.fast_max_batches") is None:
        try:
            cfg.test.fast_max_batches = 50
        except Exception:
            pass
    else:
        cfg.test.fast_max_batches = min(int(cfg.test.fast_max_batches), 50)

    print(
        "[SMOKE] single-device overrides applied: epochs=1, warmup=1, ast_warmup=1, "
        "train_stride_ratio=10.0, eval_stride_ratio=10.0, "
        "fast_val<=20, full_every=1, final_test=ON (test.fast_max_batches<=50 if supported)"
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/ast2_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (v5.4 contract)")
    parser.add_argument("--out", type=str, default=None, help="Alias of --out_dir (backward compat)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    cfg.cfg_path = args.cfg
    seed = int(args.seed)
    # ensure train node exists even if YAML has train: null
    if not hasattr(cfg, "train") or OmegaConf.select(cfg, "train") is None:
        OmegaConf.update(cfg, "train", {}, merge=True)
    # Ensure `cfg.train.run_id` is set before seal_digest
    if not hasattr(cfg.train, "run_id") or not cfg.train.run_id:
        cfg.train.run_id = uuid.uuid4().hex
    cli_out = args.out_dir or args.out
    auto_out = f"outputs/ast2_auto"
    out_dir = Path(cli_out) if cli_out else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = str(out_dir)
    if hasattr(cfg, "train"):
        cfg.train.out_dir = str(out_dir)
        cfg.train.seed = seed
    if hasattr(cfg, "training"):
        cfg.training.seed = seed
    # IMPORTANT: apply SMOKE overrides BEFORE validation/contract stamping,
    # otherwise cfg.contract.seal_digest will mismatch computed seal.
    cfg = apply_smoke_overrides_single(cfg)
    cfg = validate_and_fill_defaults(cfg, mode="single")
    signature = None
    # ---- v5.4 quick contract self-check (fail-fast before training) ----
    try:
        from utils.config_validate import get_nested
        from utils.trace_guard import build_trace_header_payload_v54
        from utils.trace_contract_v54 import TraceContractV54

        requested_config = get_nested(cfg, "_contract.requested_config_snapshot", {}) or {}
        effective_config = OmegaConf.to_container(cfg, resolve=True)
        contract_overrides = get_nested(cfg, "_contract.overrides", []) or []
        signature = build_signature_v54(cfg, method_name="ast2_single_device")
        payload = build_trace_header_payload_v54(
            signature=signature,
            requested_config=requested_config,
            effective_config=effective_config,
            contract_overrides=contract_overrides,
            requested={"method": "ast2_single_device"},
            effective={"method": "ast2_single_device"},
            no_drift_enabled=bool(get_nested(cfg, "stable_hw.no_drift.enabled", False)),
            acc_ref_source=str(get_nested(cfg, "stable_hw.locked_acc_ref.source", "none")),
            seal_digest=str(get_nested(cfg, "contract.seal_digest", "")),
        )
        TraceContractV54.validate_event("trace_header", payload)
    except Exception as exc:
        raise RuntimeError(f"v5.4 contract self-check failed: {exc}") from exc
    if signature is None:
        raise RuntimeError("v5.4 contract self-check failed: signature not generated")
    # v5.4: 进入训练运行语义就必须有可审计证据链（trace_header）
    trace_base = out_dir / "trace"
    run_id = str(getattr(getattr(cfg, "train", None), "run_id", "") or "")
    if not run_id:
        # v5.4: cfg has been sealed; DO NOT mutate cfg here.
        # Use a local run_id only for trace directory naming.
        run_id = uuid.uuid4().hex
    trace_meta = init_trace_dir_v54(
        base_dir=trace_base,
        run_id=run_id,
        cfg=cfg,
        signature=signature,
        signature_v54=signature,
        required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
        run_meta={"mode": "single_device_train", "seed_id": int(seed), "run_id": str(run_id)},
        extra_manifest={"task": "single_device", "out_dir": str(out_dir)},
    )
    trace_dir = Path(trace_meta["trace_dir"])
    trace_events_path = Path(trace_meta["trace_events"])
    run_id = trace_dir.name
    seed_everything(seed)
    # Optional speed-up for Ampere/Ada (A-experiments set ENABLE_TF32=1).
    maybe_enable_tf32()
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        f.write(Path(args.cfg).read_text(encoding="utf-8"))
    try:
        import omegaconf

        (out_dir / "config_resolved.yaml").write_text(omegaconf.OmegaConf.to_yaml(cfg), encoding="utf-8")
        resolved_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        (out_dir / "config_resolved.yaml").write_text(str(cfg), encoding="utf-8")
        resolved_cfg = cfg

    (out_dir / "resolved_config.json").write_text(
        json.dumps(resolved_cfg, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ---- cfg_hash for audit ----
    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    from utils.stable_hash import stable_hash

    print("[CFG] mode =", "single")
    print("[CFG] stable_hw.enabled =", bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False)))
    print(
        "[CFG] stable_hw.lambda_hw_schedule.lambda_hw_max =",
        float(getattr(getattr(getattr(cfg, "stable_hw", None), "lambda_hw_schedule", None), "lambda_hw_max", -1.0)),
    )
    print("[CFG] loss.lambda_hw =", float(getattr(getattr(cfg, "loss", None), "lambda_hw", -1.0)))
    print("[CFG] hw.lambda_hw =", float(getattr(getattr(cfg, "hw", None), "lambda_hw", -1.0)))
    keypaths = [
        "stable_hw.enabled",
        "stable_hw.no_double_scale",
        "stable_hw.accuracy_guard.enabled",
        "stable_hw.accuracy_guard.metric",
        "stable_hw.locked_acc_ref.enabled",
        "stable_hw.locked_acc_ref.source",
        "stable_hw.no_drift.enabled",
    ]
    for kp in keypaths:
        print("[CFG]", kp, "=", _cfg_get_path(cfg, kp, default="__MISSING__"))

    cfg_hash = stable_hash({"cfg": resolved_text})
    seal_digest = check_cfg_integrity(cfg)
    train_single_device(
        cfg,
        trace_events_path=str(trace_events_path),
        run_id=run_id,
        seal_digest=seal_digest,
    )


if __name__ == "__main__":
    main()
