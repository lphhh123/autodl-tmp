# SPEC_version_c_agent_eda_v4.3.2
[DEPRECATED] v5.4 SPEC is canonical; do not implement from this file.

**Version-C（s+α+m） + 离线 EDA-Agent（Region Global Place + Pareto + Alt-Opt）总规范（可直接驱动代码编写）**

## 0. 安全与密钥（必须遵守）

### 0.1 不要把 Key 写进 spec / yaml / 代码仓库

* 你贴在 curl 里的 `Authorization: Bearer ...` **不要再出现在任何文本/仓库中**（包括发给 Codex 的 spec）。
* 正确做法：只在运行机上用环境变量注入。

### 0.2 环境变量约定（强制）

* `VOLC_ARK_API_KEY`
* `VOLC_ARK_ENDPOINT`（例：`https://ark.cn-beijing.volces.com/api/v3`，以控制台为准）
* `VOLC_ARK_MODEL`（例：`doubao-seed-1-6-251015`）
* 可选：`VOLC_ARK_TIMEOUT_SEC`、`VOLC_ARK_MAX_RETRY`

> **结论：你现在已有 Ark chat/completions 的可用 curl，就够了**。本 spec 不需要企业版/Coze 平台才能落地；企业版只在你要“私有化/团队权限/集中计费管控”时再考虑。

---

## 1. v4.3.2 与最早 spec 的“强制差异”与迁移清单（Codex 必看）

> 目标：让 Codex **按 diff 改造**，避免大范围推倒重写。

### 1.1 旧 spec（最早）常见实现（Codex 可能已写成这样）

* `layout` 直接是连续坐标 `pos[s,2]`，要么固定 grid，要么做几步梯度下降；
* 训练后只有“跑一次布局”或“没有正式离线流程”；
* 没有 **Region global placement**、没有 **Pareto front**、没有 **m↔L alt-opt**。

### 1.2 v4.3.2 强制变化（必须改）

1. **布局主变量从连续 pos → 离散 sites + assign（离散赋值）**

* `sites_xy_mm: [Ns,2]` 预生成，训练内/离线共用
* `assign: [S]` 是 slot→site 的离散映射（**主布局变量**）
* `pos_xy_mm = sites_xy_mm[assign]` 只是派生

2. **训练内 layout 只做“轻量 micro_place”，且触发受控**

* 只在 mapping 变化明显时做几十/几百步 SA/贪心，不允许每 step 全局重算导致 loss 抖动

3. **训练后必须有离线 EDA-Agent 多阶段流程**

* `coarsen → region_global_place → expand_in_region → legalize → detailed_place(SA/LLM) → Pareto → alt_opt(m↔L)`
* 输出必须包含 Pareto 前沿 + knee 选择

4. **Pareto（comm_norm, therm_norm）成为“论文级核心产物”**

* 不允许只输出一个 scalar best
* 必须保存 Pareto 集合，并用 knee/weighted 等策略选最终解

### 1.3 文件级改造清单（必须新增/修改）

**必须新增：**

```
layout/evaluator.py
layout/sites.py
layout/regions.py
layout/coarsen.py
layout/global_place_region.py
layout/expand.py
layout/legalize.py
layout/detailed_place.py
layout/pareto.py
layout/llm_provider.py
scripts/run_layout_agent.py
scripts/plot_pareto.py           # 可选但强烈建议
tests/test_pareto.py
tests/test_regions.py
tests/test_layout_agent_smoke.py
```

**必须修改：**

```
layout/wafer_layout.py           # 若已有：改为 sites+assign 的薄封装（训练内使用）
scripts/run_version_c.py         # Phase3 导出 layout_input.json（schema见第6章）
utils/config.py                  # 解析 layout_agent / regions / pareto / llm
mapping/mapping_solver.py        # 导出 traffic_matrix（或提供汇总函数）
```

---

## 2. 总体目标与输出（你论文里要写清楚的“EDA贡献点”）

### 2.1 目标（两阶段 + 闭环）

* 训练内：用粗布局给 `L_hw` 提供几何意义（距离矩阵稳定）
* 训练后：用离线 EDA-Agent 做更强的多阶段优化，并通过 **Pareto + alt-opt** 抑制“两阶段非最优”

### 2.2 离线 Agent 的论文贡献点（你要强调“这本身就是一篇 EDA”）

1. **Region-aware global placement**（ring/sector 区域化 + 容量约束）
2. **Multi-objective Pareto optimization**（comm vs therm）+ knee selection
3. **Local detailed placement with structured moves**（swap/relocate/cluster_move）
4. **Alt-Opt closure**：m↔L 交替闭环，最终 remap 再评估（避免 layout 只对旧 mapping 有效）
5. **Traceable EDA pipeline**：trace.csv + pareto.csv + report.json，复现实验

---

## 3. 核心对象与约定（统一口径）

* `S`：slot 数（固定，如 16/32）
* `Ns`：site 数（≥S）
* `assign[s] in [0,Ns)`：slot→site（不可重复）
* `pos[s]=sites_xy[assign[s]]`
* `traffic_bytes[i,j]`：slot i→slot j 的通信（bytes），允许有向；用于 comm cost 时用对称化
* `chip_tdp_w[s]`：slot 上芯粒 TDP，empty=0
* `L_comm`：通信成本
* `L_therm`：热 proxy
* `penalty`：重复 site / 越界（本版默认 sites 间距足够，不做 overlap penalty；若未来 pitch 更密再加）

---

## 4. 共享 Layout Space：sites 生成（训练内/离线必须同一实现）

### 4.1 build_sites（强制接口）

文件：`layout/sites.py`

```python
def build_sites(
    wafer_radius_mm: float,
    chip_max_width_mm: float,
    chip_max_height_mm: float,
    margin_mm: float,
    method: str,
    grid_pitch_mm: float | None,
    seed: int,
) -> np.ndarray:
    """
    return sites_xy_mm: [Ns,2] float32
    """
```

### 4.2 safe_spacing 固化规则

* `max_diag = sqrt(chip_max_width_mm^2 + chip_max_height_mm^2)`
* `safe_spacing = max_diag + 2*margin_mm`
* 若 `grid_pitch_mm is None`：`grid_pitch_mm = safe_spacing`

### 4.3 square_grid_in_circle（v4.3.2 固化算法）

* x,y ∈ [-R, R] 按 pitch 取网格点
* 保留 `sqrt(x^2+y^2) <= R - 0.5*max_diag - margin_mm`
* 输出顺序固定（先 y 后 x 或先 x 后 y，但要写死）

> 这保证“放到 site 上基本不重叠”，legalize 极简，工程更稳。

---

## 5. LayoutEvaluator（训练内/离线必须共用，避免两套口径）

文件：`layout/evaluator.py`

### 5.1 LayoutState（强制数据结构）

```python
@dataclass
class LayoutState:
    S: int
    Ns: int
    wafer_radius_mm: float
    sites_xy_mm: np.ndarray          # [Ns,2]
    assign: np.ndarray               # [S]
    chip_tdp_w: np.ndarray           # [S]
    traffic_bytes: np.ndarray        # [S,S]
    meta: dict                       # mapping_id, stage, iter, seed...
```

### 5.2 cost 分解（固化公式）

令 `pos = sites_xy_mm[assign]`，`d_ij = ||pos[i]-pos[j]||2`

**(1) L_comm（对称 traffic）**

* `T_sym = traffic + traffic.T`
* `L_comm = sum_{i<j} T_sym[i,j] * d_ij`

**(2) L_therm（静态 TDP proxy）**

* `L_therm = sum_{i<j} (tdp[i]*tdp[j]) * exp(-d_ij / sigma_mm)`

**(3) penalty**

* `duplicate_site`: assign 有重复 site（强罚，legalize 应该修掉）
* `boundary`: pos 超出 wafer（理论不应发生，仍保底）

### 5.3 baseline 归一化（强制）

* `comm_norm = L_comm / (L_comm_baseline + 1e-9)`
* `therm_norm = L_therm / (L_therm_baseline + 1e-9)`

### 5.4 evaluate 接口（强制）

```python
class LayoutEvaluator:
    def __init__(self, sigma_mm: float, baseline: dict, scalar_w: dict):
        """
        baseline: {"L_comm_baseline":..., "L_therm_baseline":...}
        scalar_w: {"w_comm":..., "w_therm":..., "w_penalty":...}
        """

    def evaluate(self, st: LayoutState) -> dict:
        """
        return {
          "L_comm": float,
          "L_therm": float,
          "comm_norm": float,
          "therm_norm": float,
          "penalty": {"duplicate":float,"boundary":float},
          "total_scalar": float
        }
        """
```

---

## 6. traffic_matrix 的构建（必须写清楚，避免“怎么汇总通信量”歧义）

> 你现在已有：segment 列表、mapping_result、每段的 traffic_in/out_bytes。
> v4.3.2 要求你导出 `traffic_matrix_bytes[S,S]` 给布局使用。

### 6.1 最小可行的 traffic 汇总（v4.3.2 固化）

假设 segments 是顺序 pipeline（segment k 输出喂给 k+1），则：

对每个 k from 0..K-2：

* `a = mapped_slot[k]`
* `b = mapped_slot[k+1]`
* 若 `a != b`，则 `traffic[a,b] += segments[k].traffic_out_bytes`

如果你有更精细的 DAG（多个边），则统一用：

* 遍历边 `(u->v, bytes)`
* `traffic[slot(u), slot(v)] += bytes`

### 6.2 必须在 layout_input.json 中导出 traffic_matrix

见第10章 schema。

---

## 7. 训练内布局（粗布局 + micro_place）：稳定优先

> 训练内只要“足够好 + 足够稳”，把大优化留给离线 Agent。

### 7.1 assign_seed 初始化（两种必须实现）

文件：`layout/wafer_layout.py` 或 `layout/sites.py`/`layout/init.py`

* `grid_baseline`：按 site_id 从小到大填 slot
* `traffic_aware_greedy`（固化算法）

  1. 找 topK 热通信对（从 `T_sym` 取最大 K 对）
  2. 预计算所有 site pair 距离，取最短的 M 对
  3. 依次把“热对”匹配到“短距离 site 对”，避免 site 冲突
  4. 剩余 slot 用空 site 填

### 7.2 micro_place（触发受控）

文件：`layout/wafer_layout.py` 或 `layout/detailed_place.py` 复用一套 SA 内核

**触发条件（固化）**

* `changed_ratio = mean(mapping_k != prev_mapping_k)`
* 若 `changed_ratio >= trigger_changed_ratio` 且距上次触发已超过 `min_steps_between_triggers`，才触发

**训练内动作集（最小集）**

* swap(i,j)
* relocate(i, empty_site)（只在邻域 radius 内选）

**训练内预算（建议默认）**

* steps=80（或 100）
* 温度 `T0=1.0`，`alpha=0.995`
* relocate 邻域半径 `r=2~3` 个 pitch

### 7.3 训练内输出必须落盘到 layout_input.json

* `seed.assign_seed`
* `seed.micro_place_stats`

---

## 8. 离线 EDA-Agent：多阶段流程（核心）

入口脚本：`scripts/run_layout_agent.py`
输入：`layout_input.json`
输出：`layout_best.json` + `trace.csv` + `pareto.csv` + `report.json` + （可选）图

### 8.1 Stage0：加载与 baseline 评估

* 读取 `layout_input.json`
* 构造 evaluator（baseline + sigma + scalar weights）
* 评估 `assign_grid` 与 `assign_seed`
* 初始化 ParetoSet（把 baseline 和 seed 都加入）

### 8.2 Stage1：coarsen（聚簇，communication graph clustering）

文件：`layout/coarsen.py`

**输入**

* `T_sym[S,S]`（对称 traffic）
* `slot_mask_used`（empty slot 可忽略）

**输出**

```python
@dataclass
class Cluster:
    cluster_id: int
    slots: list[int]
    tdp_sum: float
```

以及 `W_cluster[C,C]`：簇间通信权重（对称）

**固化算法：Greedy Agglomerative Merge**

* 初始每个 slot 一个 cluster
* 定义簇间权重：簇 A 与 B 的权重 = sum_{i in A, j in B} T_sym[i,j]
* 迭代合并权重最大的两簇
* 停止：

  * 达到 `target_num_clusters`
  * 或最大权重 < `min_merge_traffic`

### 8.3 Stage2：Region Global Place（论文级“簇→区域”）

文件：`layout/regions.py`, `layout/global_place_region.py`

#### 8.3.1 build_regions：ring/sector 区域化

输入：

* `ring_edges_ratio`（如 `[0.0,0.45,0.75,1.0]`）
* `sectors_per_ring`（如 `[4,8,12]`）
  输出：
* `regions: list[Region]`
* `site_to_region[Ns]`

Region 必须包含：

* `centroid_xy_mm`
* `site_ids`
* `capacity`（默认=site_ids 数，或乘一个 `capacity_ratio<=1`）

#### 8.3.2 cluster→region assignment（带容量约束）

目标函数（固化）：
[
J =
\lambda_{graph}\sum_{c1<c2}W_{c1c2}\cdot d(\text{centroid}(r(c1)), \text{centroid}(r(c2)))

* \lambda_{ring}\sum_c (tdp(c)\cdot ring_score(r(c)))
* \lambda_{cap}\cdot \text{violation}
  ]

求解（必须实现，易写且有效）：

* greedy：按 `tdp_sum` 降序放簇，选增量 J 最小且容量足够的 region
* refine：随机 swap 两个 cluster 的 region，若 J 降低则接受（可用 SA）

输出：

* `cluster_to_region[C]`

### 8.4 Stage3：expand_in_region（簇内展开到 sites）

文件：`layout/expand.py`

对每个 cluster：

* 从所属 region 的空 site 集合里选 `|cluster|` 个 sites
* 初始：按 site_id 顺序选
* 簇内 refine：只在 cluster slots 内做 swap，优化 **簇内 L_comm_intra**
* 写回全局 assign
* 剩余未放置 slot 用剩余空 sites 填充（就近优先）

### 8.5 Stage4：legalize（修复重复/越界）

文件：`layout/legalize.py`

* 若发现重复 site：

  * 对冲突 slot 逐个搬到最近的空 site
* 若越界（理论不应发生）：

  * 搬到最近合法 site

输出合法 assign。

### 8.6 Stage5：detailed_place（SA + 结构化动作 + Pareto）

文件：`layout/detailed_place.py`

#### 8.6.1 动作集合（必须实现）

* `swap(i,j)`
* `relocate(i, site_id)`
* `cluster_move(cluster_id, region_id)`：把整个 cluster 尝试搬到目标 region（优先外圈），通过“选空 sites + 局部微调”实现

统一 Action Schema（LLM/规则共用）：

```json
{"actions":[{"op":"swap","i":3,"j":7},{"op":"relocate","i":5,"site_id":120},{"op":"cluster_move","cluster_id":2,"region_id":11}]}
```

#### 8.6.2 region-aware 采样策略（固化默认）

* relocate：

  * 80% 在同 region 内找空 site
  * 20% 允许跨 region
* swap：

  * 优先从 top_hot_pairs（通信最热的 slot 对）采样 i/j
* cluster_move：

  * 优先把高 tdp cluster 往外圈 region 尝试移动（探索 therm trade-off）

#### 8.6.3 SA 接受（固化）

* `delta = new_total_scalar - cur_total_scalar`
* 若 `delta<0` 接受，否则以 `exp(-delta/T)` 接受
* `T <- T*alpha`

#### 8.6.4 Pareto 更新（必须）

* 每次评估都尝试加入 ParetoSet：

  * objectives = (comm_norm, therm_norm)
  * penalties 记录
* ParetoSet 需实现：

  * 非支配判定
  * ε-dominance 压缩（避免爆炸）
  * knee-point 选择（默认输出 best）

### 8.7 Stage6：alt_opt（m↔L 交替闭环，必须）

文件：`layout/alt_opt.py`（可在 `run_layout_agent.py` 内实现，但建议独立）

参数：`rounds=3~5`（默认 3）

每轮：

1. fix L：调用 mapping_solver 做一次 remap

   * **限制改动规模**：只允许 topX% bottleneck segments 重映射（如 20%），防止漂移过大
2. fix m：从 ParetoSet 取 topM 点作为起点（如 5 个），各自跑 `refine_steps_each_round` 的 SA，合并 Pareto
3. 记录本轮 Pareto size、knee objectives

**最后必须 remap 一次**，最终报告用 `(m_final, L_final)`。

---

## 9. LLM Planner（火山方舟 Ark）——只做“提案”，不做核心优化

文件：`layout/llm_provider.py`

### 9.1 统一接口（强制）

```python
class LLMProvider(ABC):
    def propose_actions(self, state_summary: dict, k: int) -> list[dict]:
        ...
```

### 9.2 VolcArkProvider（强制实现）

* 读取 env：`VOLC_ARK_API_KEY/ENDPOINT/MODEL`
* 请求：`POST {endpoint}/chat/completions`
* 强制 JSON 输出：

  * 优先使用 `response_format={"type":"json_object"}`（若可用）
  * 否则 prompt 强制“只输出 JSON”，并做解析失败重试（最多 2 次）

### 9.3 state_summary（强制压缩字段，固化）

```json
{
  "comm_norm": 0.93,
  "therm_norm": 1.10,
  "top_hot_pairs": [{"i":2,"j":9,"traffic":3.2e8,"dist_mm":58.2}],
  "top_hot_slots": [{"i":7,"tdp":350,"region":3}],
  "violations": {"duplicate":0,"boundary":0},
  "hint": "prefer swap on hot pairs; push high-tdp outward; avoid duplicate sites"
}
```

### 9.4 Planner 模式（必须支持三种）

* `heuristic`：纯规则（默认可跑通）
* `llm`：全 LLM 提案动作（会更贵）
* `mixed`：多数 step 用 heuristic，间隔 N 步调用一次 llm（推荐默认）

### 9.5 用量统计（必须落盘）

输出 `llm_usage.jsonl`，每条包含：

* timestamp, model, prompt_tokens, completion_tokens, total_tokens（若服务返回）
* 若不返回 usage：至少记录 request_bytes/response_bytes 与 call_id

---

## 10. 数据契约：layout_input.json / layout_best.json（强制 schema）

### 10.1 layout_input.json（训练后导出，离线输入）

路径示例：`outputs/P3/layout_input.json`

必须包含：

* `layout_version="v4.3.2"`
* wafer 参数（radius, margin）
* sites（method, pitch, sites_xy）
* slots（S、tdp）
* mapping（mapping_id、segments、traffic_matrix）
* baseline（assign_grid + baseline costs）
* seed（assign_seed + micro_place_stats）
* objective_cfg（sigma、scalar weights）

> 字段名建议与你之前 v4.3.1 一致；重要的是“全齐 + 可复现”。

### 10.2 layout_best.json（离线输出）

必须包含：

* `best`（assign、pos、objectives、raw、penalty、meta）
* `pareto_front`（一组点的简表）
* `selection`（knee 方法、pareto_size）
* `region_plan`（clusters、cluster_to_region、J）
* `artifacts`（trace_csv、pareto_csv、llm_usage_jsonl）
* `report.json`（见第12章）

---

## 11. trace / pareto / report（必须固定字段，方便论文画图）

### 11.1 trace.csv（每次评估/接受都记录）

字段（固化顺序）：

* iter
* stage（micro_place / coarsen / global_place / expand / legalize / detailed / alt_opt_round_k）
* op（swap/relocate/cluster_move/none）
* op_args_json
* accepted（0/1）
* total_scalar
* comm_norm
* therm_norm
* pareto_added（0/1）
* duplicate_penalty
* boundary_penalty
* seed_id
* time_ms

### 11.2 pareto_points.csv（每次加入 Pareto 记录）

字段：

* solution_id
* comm_norm
* therm_norm
* total_scalar
* stage
* iter
* seed_id
* assign_hash（用于去重）

### 11.3 report.json（每次实验必须产出，论文直接引用）

至少包含：

* baseline comm_norm=1, therm_norm=1
* best(knee) comm_norm/therm_norm
* pareto_size
* runtime_sec
* SA accept_rate
* 若启用 LLM：total_tokens / total_calls
* 关键 config 摘要（regions/pareto/alt_opt）

---

## 12. 实验设计（对比 + 消融）——你要的“写满 + 命令全给”

> 你最终至少要跑两大块：
> A) **训练阶段（Phase3）布局消融**：证明训练内 seed 稳定有效
> B) **离线阶段（Phase-L）EDA-Agent 对比/消融**：证明“这就是一套 EDA 论文级流程”

### 12.1 Phase3（Version-C 训练内）布局消融（A0~A3）

**A0**：no_layout（不算 comm/therm；layout 为零）
**A1**：grid baseline（assign_grid 固定）
**A2**：traffic_aware seed（无 micro_place）
**A3**：traffic_aware + micro_place（v4.3.2 默认）

命令（示例；cfg 名按你仓库实际落地；`--export_layout_input` 可单独写或显式传 `true/false`）：

```bash
python -m scripts.run_version_c --cfg configs/vc_p3_A0_nolayout.yaml  --export_layout_input --export_dir outputs/P3/A0
python -m scripts.run_version_c --cfg configs/vc_p3_A1_grid.yaml      --export_layout_input --export_dir outputs/P3/A1
python -m scripts.run_version_c --cfg configs/vc_p3_A2_seed.yaml      --export_layout_input --export_dir outputs/P3/A2
python -m scripts.run_version_c --cfg configs/vc_p3_A3_seed_micro.yaml --export_layout_input --export_dir outputs/P3/A3
```

输出检查点（每个都必须有）：

* `outputs/P3/Ax/layout_input.json`
* `outputs/P3/Ax/train_log.jsonl`（你现有的训练日志）

### 12.2 离线 EDA-Agent 主对比（L0~L7）

以 **A3 的 layout_input.json** 为统一输入（先做主线），再换 A0/A1 做附加证明。

#### L0：仅 legalize（几乎不优化， sanity）

* regions=off, pareto=off, detailed=off, alt_opt=off
  目的：验证 I/O 与 evaluator 正确

#### L1：Heuristic + noRegion + scalar SA（最弱 baseline）

* regions=off, pareto=off, planner=heuristic, detailed=on, alt_opt=off

#### L2：Heuristic + Region global place + scalar SA（证明 region 有用）

* regions=on, pareto=off, planner=heuristic, detailed=on

#### L3：Heuristic + Region + Pareto（knee 输出）（核心）

* regions=on, pareto=on, planner=heuristic, detailed=on

#### L4：Mixed LLM + Region + Pareto（证明 LLM 仅做提案也能提升）

* regions=on, pareto=on, planner=mixed(每 N 步调用 llm)

#### L5：Mixed LLM + Region + Pareto + Alt-Opt（闭环）

* regions=on, pareto=on, planner=mixed, alt_opt=on(rounds=3)

#### L6：Heuristic + Region + Pareto + Alt-Opt（证明闭环不是靠 LLM）

* regions=on, pareto=on, planner=heuristic, alt_opt=on

#### L7：NoTherm（热项消融）

* regions=on, pareto=on, enable_therm=false
  目的：展示 comm 最优 vs 加热约束后的 trade-off（论文好写）

命令（全部给齐）：

```bash
# 输入统一用 A3
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L0_legalize.yaml --out_dir outputs/P3/A3/L0
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L1_scalar_noRegion.yaml --out_dir outputs/P3/A3/L1
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L2_region_scalar.yaml --out_dir outputs/P3/A3/L2
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L3_region_pareto.yaml --out_dir outputs/P3/A3/L3
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L4_region_pareto_llm_mixed.yaml --out_dir outputs/P3/A3/L4
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L5_region_pareto_llm_mixed_altopt.yaml --out_dir outputs/P3/A3/L5
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L6_region_pareto_heur_altopt.yaml --out_dir outputs/P3/A3/L6
python -m scripts.run_layout_agent --layout_input outputs/P3/A3/layout_input.json --cfg configs/layout_L7_region_pareto_noTherm.yaml --out_dir outputs/P3/A3/L7
```

附加（可选但建议）：用 A1(grid) 输入重复跑 L3/L6（证明离线能“救回”差 seed，但 A3 仍更好）

```bash
python -m scripts.run_layout_agent --layout_input outputs/P3/A1/layout_input.json --cfg configs/layout_L3_region_pareto.yaml --out_dir outputs/P3/A1/L3
python -m scripts.run_layout_agent --layout_input outputs/P3/A1/layout_input.json --cfg configs/layout_L6_region_pareto_heur_altopt.yaml --out_dir outputs/P3/A1/L6
```

### 12.3 论文图表建议（必须由脚本产出）

* Pareto 散点图（comm_norm vs therm_norm），标注 knee 点
* L1/L2/L3/L6/L7 的条形对比（comm_norm、therm_norm、runtime、pareto_size）
* trace 收敛曲线（total_scalar 随 iter）
* region plan 可视化（cluster→region）

命令：

```bash
python -m scripts.plot_pareto --best outputs/P3/A3/L3/layout_best.json --out outputs/P3/A3/L3/pareto_points.csv
python -m scripts.plot_pareto --best outputs/P3/A3/L5/layout_best.json --out outputs/P3/A3/L5/pareto_points.csv
```

---

## 13. 配置文件（必须字段，Codex 直接照写 YAML）

下面给出 `configs/layout_L3_region_pareto.yaml` 的字段规范（其他 Lx 只是在此基础上开关项不同）：

```yaml
layout_agent:
  version: v4.3.2
  seed_list: [0, 1, 2, 3, 4]        # multi-start
  max_runtime_sec: 1800
  export_trace: true

sites:
  # 通常从 layout_input.json 读取；这里允许覆盖
  allow_override: false

objective:
  enable_therm: true
  sigma_mm: 20.0
  scalar_weights:
    w_comm: 0.7
    w_therm: 0.3
    w_penalty: 1000.0

regions:
  enabled: true
  ring_edges_ratio: [0.0, 0.45, 0.75, 1.0]
  sectors_per_ring: [4, 8, 12]
  ring_score: [1.0, 0.7, 0.4]
  capacity_ratio: 1.0

coarsen:
  target_num_clusters: 8
  min_merge_traffic: 1.0e6

global_place_region:
  lambda_graph: 1.0
  lambda_ring: 1.0
  lambda_cap: 10000.0
  refine:
    enabled: true
    steps: 400
    sa_T0: 1.0
    sa_alpha: 0.995

expand:
  intra_refine_steps: 200

legalize:
  enabled: true

pareto:
  enabled: true
  eps_comm: 0.01
  eps_therm: 0.01
  max_points: 2000
  selection: knee_point_v1

detailed_place:
  enabled: true
  steps: 8000
  sa_T0: 1.0
  sa_alpha: 0.9995
  action_probs:
    swap: 0.55
    relocate: 0.35
    cluster_move: 0.10
  relocate:
    same_region_prob: 0.8
    neighbor_k: 30        # 从候选空site里取最近K个再随机
  hot_sampling:
    top_pairs_k: 20
    top_slots_k: 20

planner:
  type: heuristic          # L4/L5 改成 mixed 或 llm
  mixed:
    every_n_steps: 50
    k_actions: 6

llm:                       # planner=llm/mixed 时才启用
  provider: volc_ark
  max_calls_per_run: 200
  timeout_sec: 30
  max_retry: 2
  # 不允许写 key；只读 env: VOLC_ARK_API_KEY/ENDPOINT/MODEL

alt_opt:
  enabled: false           # L5/L6 打开
  rounds: 3
  remap:
    enabled: true
    allow_top_segment_ratio: 0.2
  refine_each_round:
    steps: 2000
```

---

## 14. 单元测试与 smoke（必须写，不然后期很难改）

### 14.1 tests/test_pareto.py

* dominance 判定正确
* ε-dominance 压缩有效（点数不爆）
* knee 选择稳定（固定输入输出固定）

### 14.2 tests/test_regions.py

* site_to_region 映射全覆盖、无越界
* capacity 计算正确
* centroid 在 wafer 内

### 14.3 tests/test_layout_agent_smoke.py

* 构造 toy traffic（S=8），跑完整 pipeline（L3）输出 layout_best.json + trace.csv

---

## 15. 你现在“还需要获取什么吗？”（就你给的 curl 来看）

就实现本 spec 而言：**不需要再获取额外权限/账号信息**。你还需要做的是工程上的三件事：

1. **把 Key 从任何文本中移除**，只留 env 注入
2. 确认模型输出能稳定 JSON（不行就走“严格 JSON prompt + 解析失败重试”兜底）
3. 给 LLM 调用加预算：`max_calls_per_run`，避免 multi-start + alt_opt 把 token 拉爆

---

## 16. 最终验收标准（你一跑就知道对不对）

* Phase3 能导出 `layout_input.json`（含 baseline+seed+traffic_matrix）
* L3 能输出：

  * `layout_best.json`（含 pareto_front+selection+region_plan）
  * `trace.csv`、`pareto_points.csv`、`report.json`
* L2 vs L3：Pareto 打开后能看到 trade-off（点云不是一条线）
* L6 vs L3：alt_opt 后最终 remap 评估更稳（至少不变差，通常更好）
* tests 全过

---

如果你接下来一句话告诉我：“**我现在 S=16 还是 S=32、wafer 半径你打算用多少 mm（150/200/300）**”，我还能把上面 YAML 里的默认数值（pitch、sigma、cluster 数、steps）再给你做一次“更贴近你实际硬件库尺寸”的校准版；但就算你不补这句话，这份 spec 已经足够让 Codex 直接开写并且写出来不会走偏。
