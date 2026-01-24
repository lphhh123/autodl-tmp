from pathlib import Path
import tempfile

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.trace_contract_v54 import compute_effective_cfg_digest_v54
from utils.trace_guard import append_trace_event_v54, init_trace_dir_v54
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS

with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    cfg = load_config("configs/vc_phase3_full_ucf101.yaml")
    # out_dir must be set before validate (seal depends on effective config)
    cfg.out_dir = str(out)
    if not hasattr(cfg, "train") or cfg.train is None:
        cfg.train = {}
    cfg.train.out_dir = str(out)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    seal = str(cfg.contract.seal_digest)
    signature = build_signature_v54(cfg, method_name="smoke_check_seal_digest_mutation")
    trace_meta = init_trace_dir_v54(
        base_dir=out / "trace",
        run_id="smoke_check_seal_digest_mutation",
        cfg=cfg,
        signature=signature,
        signature_v54=signature,
        required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
        run_meta={"mode": "smoke_check_seal_digest_mutation", "seed_id": 0, "run_id": "smoke_check_seal_digest_mutation"},
        extra_manifest={"task": "smoke_check_seal_digest_mutation", "out_dir": str(out)},
    )
    trace_events = trace_meta["trace_events"]

    # seal after mutation
    cfg.train.lr = float(getattr(cfg.train, "lr", 3e-4)) * 0.5
    cur = compute_effective_cfg_digest_v54(cfg)
    assert cur != seal

    append_trace_event_v54(
        trace_events,
        "contract_violation",
        {
            "reason": "cfg_mutated_after_seal",
            "expected_seal_digest": seal,
            "actual_seal_digest": cur,
        },
    )
    print("OK: seal drift detected + contract_violation written")
