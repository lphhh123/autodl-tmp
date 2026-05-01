# Run C/D Minimal Experiments

1) Run the minimal C/D experiments:

```bash
cd proj_ast2_ucf101_full
RUN_BASELINES=0 INSTANCES="cluster4" BUDGETS="160k 240k" SEEDS_MAIN="0 1 2" MAX_JOBS=24 bash scripts/launch_B_cd_minimal.sh
```

If baselines should be rerun too:

```bash
RUN_BASELINES=1 INSTANCES="cluster4" BUDGETS="160k 240k" SEEDS_MAIN="0 1 2" MAX_JOBS=24 bash scripts/launch_B_cd_minimal.sh
```

2) Generate process evidence:

```bash
python scripts/plot_B_cd_process_evidence.py --instance cluster4 --budgets 160k 240k
```
