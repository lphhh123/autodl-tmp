# Experiment Execution Commands

This document lists the shell commands to run the Phase3 (A-series) and offline EDA-Agent (L-series) experiments described in the v4.3.2 specification.

## Phase3 Layout Ablations (A0–A3)

Run these to export `layout_input.json` and training logs for each ablation:

```bash
python -m scripts.run_version_c --cfg configs/vc_p3_A0_nolayout.yaml  --export_layout_input true --export_dir outputs/P3/A0
python -m scripts.run_version_c --cfg configs/vc_p3_A1_grid.yaml      --export_layout_input true --export_dir outputs/P3/A1
python -m scripts.run_version_c --cfg configs/vc_p3_A2_seed.yaml      --export_layout_input true --export_dir outputs/P3/A2
python -m scripts.run_version_c --cfg configs/vc_p3_A3_seed_micro.yaml --export_layout_input true --export_dir outputs/P3/A3
```

Each run should produce `outputs/P3/Ax/layout_input.json` and `outputs/P3/Ax/train_log.jsonl`.

## Offline EDA-Agent Comparisons (L0–L7)

Use the `layout_input.json` exported by **A3** as the baseline input:

```bash
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L0_legalize.yaml --out_dir outputs/P3/A3/L0
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L1_scalar_noRegion.yaml --out_dir outputs/P3/A3/L1
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L2_region_scalar.yaml --out_dir outputs/P3/A3/L2
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L3_region_pareto.yaml --out_dir outputs/P3/A3/L3
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L4_region_pareto_llm_mixed.yaml --out_dir outputs/P3/A3/L4
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L5_region_pareto_llm_mixed_altopt.yaml --out_dir outputs/P3/A3/L5
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L6_region_pareto_heur_altopt.yaml --out_dir outputs/P3/A3/L6
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L7_region_pareto_noTherm.yaml --out_dir outputs/P3/A3/L7
```

Optional sanity checks using the grid seed (A1) as input:

```bash
python -m scripts.run_layout_agent --layout_input outputs/P3/A1/layout_input.json --cfg configs/layout_L3_region_pareto.yaml --out_dir outputs/P3/A1/L3
python -m scripts.run_layout_agent --layout_input outputs/P3/A1/layout_input.json --cfg configs/layout_L6_region_pareto_heur_altopt.yaml --out_dir outputs/P3/A1/L6
```

## Plotting Pareto Fronts

Generate Pareto scatter plots for selected runs:

```bash
python -m scripts.plot_pareto --best outputs/P3/A3/L3/layout_best.json --out outputs/P3/A3/L3/pareto_points.csv
python -m scripts.plot_pareto --best outputs/P3/A3/L5/layout_best.json --out outputs/P3/A3/L5/pareto_points.csv
```
