"""Deprecated v5.4 entrypoint (dummy)."""

raise RuntimeError(
    "[v5.4 P0][HardGate-D] This script is DEPRECATED / UNSAFE for v5.4 contract runs. "
    "It bypasses contract_bootstrap/trace_init/seal. "
    "Use: python -m scripts.run_version_c --cfg <...> (SPEC_D OneCommand)."
)


if __name__ == "__main__":
    raise RuntimeError(
        "DEPRECATED (v5.4 strict): this entrypoint is forbidden. "
        "Use: python -m scripts.run_version_c --cfg <...>"
    )
