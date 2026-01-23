# Wafer Layout Dataset Notes

This problem setup does **not** use the standard HeurAgenix dataset pipeline (train/val/test splits, etc.).
The wafer layout instances are injected by the surrounding research codebase at runtime, and the files
under this directory are only a lightweight container for prompts/assets used by that external runner.

If you are looking for the canonical HeurAgenix dataset layout, please refer to the upstream examples;
this folder intentionally keeps a minimal structure to avoid accidental coupling to that pipeline.
