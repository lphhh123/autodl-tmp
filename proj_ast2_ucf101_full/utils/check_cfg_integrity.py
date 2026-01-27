from utils.trace_contract_v54 import compute_effective_cfg_digest_v54


def check_cfg_integrity(cfg):
    seal_digest = str(getattr(getattr(cfg, "contract", None), "seal_digest", "") or "")
    cur = compute_effective_cfg_digest_v54(cfg)
    if cur != seal_digest:
        raise RuntimeError(
            f"[v5.4 P0] cfg mutated before training start. "
            f"cur_digest={cur} seal_digest={seal_digest}. "
            f"Do not write cfg.* after validate_and_fill_defaults()."
        )
    return seal_digest
