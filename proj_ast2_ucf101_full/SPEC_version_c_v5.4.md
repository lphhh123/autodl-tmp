# SPEC_version_c_v5.4 (Canonical Pointer)

This repository enforces the v5.4 contracts in code and configuration validation. The
canonical requirements are implemented in:

- `utils/config_validate.py` (v5.4 contract validation and defaults).
- `utils/stable_hw.py` (StableHW semantics, LockedAccRef, NoDrift, NoDoubleScale).
- `utils/trace_schema.py` and `utils/trace_signature_v54.py` (trace schema/signature).

Older SPEC files have been archived under `docs/legacy/spec_archive/` to avoid ambiguity.
If you need the full narrative spec, consult the project documentation that accompanies
this codebase and keep this file as the authoritative entry point for v5.4.
