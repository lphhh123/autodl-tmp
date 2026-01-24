# DEPRECATED: NOT v5.4. Do NOT use for v5.4 runs.
# Experiment commands (SPEC v4.3.2)

This file lists ready-to-run commands for Phase3 (training/export) and Phase-L (offline EDA-Agent) experiments using the provided configs.

## Phase3 (training + layout export)

Each command exports `layout_input.json` to the specified directory after training. Adjust `--export_dir` to your preferred output location.

```bash
# A0: no layout loss, grid seed only
python -m scripts.run_version_c --cfg configs/_deprecated_version_c_experiments/phase3_A0_nolayout.yaml \
  --export_layout_input true --export_dir outputs/P3/A0

# A1: grid seed baseline (layout step on, no micro-place)
python -m scripts.run_version_c --cfg configs/_deprecated_version_c_experiments/phase3_A1_grid.yaml \
  --export_layout_input true --export_dir outputs/P3/A1

# A2: traffic-aware seed (no micro-place)
python -m scripts.run_version_c --cfg configs/_deprecated_version_c_experiments/phase3_A2_seed.yaml \
  --export_layout_input true --export_dir outputs/P3/A2

# A3: traffic-aware seed + micro-place (default)
python -m scripts.run_version_c --cfg configs/_deprecated_version_c_experiments/phase3_A3_seed_micro.yaml \
  --export_layout_input true --export_dir outputs/P3/A3
```

Outputs for each run (per SPEC §12.1):
- `outputs/P3/Ax/layout_input.json`
- `outputs/P3/Ax/train_log.jsonl` (training log)

## Phase-L (offline EDA-Agent)

Use the `layout_input.json` from A3 unless otherwise noted. The commands below write results to `outputs/P3/A3/L*` by default.

```bash
# L0: legalize only (sanity)
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L0_legalize.yaml --out_dir outputs/P3/A3/L0

# L1: heuristic scalar SA without regions/pareto
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L1_scalar_noRegion.yaml --out_dir outputs/P3/A3/L1

# L2: heuristic scalar SA with regions
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L2_region_scalar.yaml --out_dir outputs/P3/A3/L2

# L3: heuristic with regions + Pareto (knee selection)
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L3_region_pareto.yaml --out_dir outputs/P3/A3/L3

# L4: mixed LLM + regions + Pareto
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L4_region_pareto_llm_mixed.yaml --out_dir outputs/P3/A3/L4

# L5: mixed LLM + regions + Pareto + alt-opt
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L5_region_pareto_llm_mixed_altopt.yaml --out_dir outputs/P3/A3/L5

# L6: heuristic + regions + Pareto + alt-opt
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L6_region_pareto_heur_altopt.yaml --out_dir outputs/P3/A3/L6

# L7: heuristic + regions + Pareto (no thermal term)
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L7_region_pareto_noTherm.yaml --out_dir outputs/P3/A3/L7
```

Optional: rerun L3/L6 using the A1 grid baseline seed to validate recovery from weaker seeds.

```bash
python -m scripts.run_layout_agent --layout_input outputs/P3/A1/layout_input.json \
  --cfg configs/layout_agent/layout_L3_region_pareto.yaml --out_dir outputs/P3/A1/L3
python -m scripts.run_layout_agent --layout_input outputs/P3/A1/layout_input.json \
  --cfg configs/layout_agent/layout_L6_region_pareto_heur_altopt.yaml --out_dir outputs/P3/A1/L6
```

Artifacts produced per run (SPEC §8):
- `layout_best.json` with Pareto summary and knee selection
- `trace.csv`, `pareto_points.csv`, and `report.json`
- `llm_usage.jsonl` when planner type is `mixed` or `llm`

To quickly assert L4/L5 produced non-empty LLM logs:

```bash
for L in L4 L5; do
  f="outputs/P3/A3/$L/llm_usage.jsonl"
  test -f "$f" || { echo "MISS $f"; exit 1; }
  test -s "$f" || { echo "EMPTY $f"; exit 1; }
done
echo "OK: LLM usage logs exist and non-empty."
```
