# EXPERIMENTS VERSION C

```bash
python -m scripts.run_version_c --cfg configs/smoke_version_c_ucf101.yaml
```

---

## Phase B — Layout（预算感知 / 选择式超启发，Innovation B）

### 推荐入口（网格并行）

主表（论文对比，建议固定 seed=0、wT=0.3/wC=0.7）：

```bash
RUN_CTL_EVIDENCE=0 \
RUN_ABLATIONS=0 \
EXPS_MAIN="EXP-B1 EXP-B2-mpvs-only EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B3" \
INSTANCES="cluster4 chain_skip chain_skip_randw" \
BUDGETS="50k 100k 150k 200k" \
WEIGHT_PAIRS="0.3,0.7" \
SEEDS_MAIN="0" \
PACK_AFTER=1 PACK_TRACE_CSV=0 KEEP_B_OUTPUTS=0 \
MAX_JOBS=32 \
bash scripts/launch_B_grid_parallel.sh
```

Headroom 前置（controller=0，用于构造 Oracle 与 sign-flip 证据）：

```bash
RUN_HEADROOM=1 \
EXPS_HEADROOM="EXP-B2-naive-atomiconly EXP-B2-naive-macroonly EXP-B2-naive-chainonly EXP-B2-naive-ruinonly EXP-B2-naive-blockonly" \
INSTANCES_HEADROOM="cluster4 chain_skip chain_skip_randw" \
BUDGETS_HEADROOM="50k 200k" \
WEIGHT_PAIRS_HEADROOM="0.3,0.7" \
SEEDS_HEADROOM="0" \
PACK_AFTER=1 KEEP_B_OUTPUTS=0 \
MAX_JOBS=24 \
bash scripts/launch_B_grid_parallel.sh
```

### 结果分析（Oracle / Regret）

```bash
python scripts/analyze_B_oracle_regret.py --root outputs/B --out_dir outputs/B/_analysis
```

会生成：

- `outputs/B/_analysis/run_table.csv`：每个 run 的关键指标 + macro utilization
- `outputs/B/_analysis/oracle_arm.csv`：基于 headroom 的 Oracle（arm-level）
- `outputs/B/_analysis/regret.csv` / `regret_summary.csv`：主表方法相对 Oracle 的 regret

### 重要说明：LEGACY mixed_pick

旧版 mixed_pick 相关配置已迁入 `configs/layout_agent/_legacy_mixed_pick/`，并在 `experiments_version_c.sh` 中通过
`ALLOW_LEGACY_MIXED_PICK=1` 显式解锁，避免与 MPVS/controller 体系混淆。
