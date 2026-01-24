"""Deprecated v5.4 entrypoint (phase4)."""


if __name__ == "__main__":
    raise RuntimeError(
        "DEPRECATED (v5.4 strict): this entrypoint is forbidden. "
        "Use: python -m scripts.run_version_c --cfg <...>"
    )
