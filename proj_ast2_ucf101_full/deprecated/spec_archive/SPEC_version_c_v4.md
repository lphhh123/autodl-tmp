# SPEC_version_c_v4.md
[DEPRECATED] v5.4 SPEC is canonical; do not implement from this file.

Version-C 实现说明书（带统一任务划分与可选细粒度拆分）

## 0. 总目标 & 核心思想

你要实现的系统：

* Backbone：Video Transformer（VideoViT），单模态 UCF-101 视频分类；
* **AST2.0-lite v2**：

  * 多级时间窗口 + 多级空间窗口的时空熵；
  * Voronoi 风格 patch 区域；
  * token / head / channel / block 多粒度稀疏；
* **硬件代理 (HW Proxy)**：layer/segment 级 latency / mem / power 预测；
* **可学习芯粒配置 α**：多个槽位，每个槽位通过 Gumbel-Softmax 选择芯粒类型或空；
* **晶圆布局 L**：芯粒在圆形晶圆上的 (x, y) 位置，考虑面积 / 重叠 / 通信 / 热；
* **统一任务划分策略 (PartitionPlanner)**：
  在**划分阶段**就同时考虑：

  * 粗粒度分段（按层 / 层组）；
  * 映射到多芯粒；
  * 通信成本与晶圆几何；
  * **可选细粒度拆分 + G/G⁻¹ 重排**：只有当**带重排的“细粒度划分方案”整体 objective 更好**，才对部分 segment / layer 做细粒度拆分并生成 GraphRewrite 计划，否则不拆、不触发这个 stage。

> 关键：**GraphRewrite（插 G / G⁻¹）不再是训练后期的“额外 Stage”，而是任务划分策略中的一个可选动作**。
> 划分器在比较“粗粒度 vs 带重排细粒度”时统一做决策。

---

## 1. 工程结构建议

```text
project_root/
  configs/
    ast2_ucf101.yaml
    version_c_full.yaml
    gpu_data.yaml

  scripts/
    run_ast2_ucf101.py         # 单卡 AST2.0-lite v2
    run_version_c.py           # Version-C 联合优化主脚本

  models/
    video_vit.py               # VideoViT backbone
    ast2_pruner.py             # AST2.0-lite v2 剪枝模块

  hw_proxy/
    layer_proxy_model.py       # MLP proxy
    layer_hw_proxy.py          # LayerHwProxy 封装

  mapping/
    segments.py                # LayerNode / Segment / SegmentGraph
    mapping_solver.py          # MappingSolver: 给定 segments 做映射
    partitioner.py             # PartitionPlanner: 统一任务划分策略（粗分 + 细分 + rewrite）

  layout/
    wafer_layout.py            # WaferLayout: 晶圆布局与损失

  trainer/
    trainer_single_device.py   # Phase2 单设备 AST 实验
    trainer_version_c.py       # Version-C 联合优化（调用 PartitionPlanner）

  utils/
    config.py
    logging.py
    metrics.py
```

---

## 2. 任务 & 数据（UCF-101）

* 输入视频预处理后：

```text
x_raw: [T, H, W, 3]
```

* 模型内部 patch 化后：

```text
x_token: [B, T, N, C]
  B: batch
  T: 帧数
  N: 每帧 patch 数（H_p×W_p）
  C: token embedding dim
```

* 分类输出：

```text
logits: [B, num_classes]  # num_classes=101
```

---

## 3. VideoViT 接口

```python
class VideoViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        num_frames: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        patch_size: int,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        use_ast_prune: bool = False,
        ast_cfg: Optional[Dict] = None,
    ):
        ...

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        x: [B, T, C, H, W]
        return_intermediate:
          False -> 只返回 logits
          True  -> 返回 (logits, info_dict)
            info_dict 至少包含:
              - "token_feat": [B, T, N, C] 某层的 token 特征
              - "ast_stats": dict, 包含:
                  * "L_AST": 标量 AST 稀疏损失
                  * "sparsity_token/head/ch/block": 若干标量
        """
```

---

## 4. AST2.0-lite v2：时空熵 + Voronoi + 多粒度稀疏

定义 ASTPruner，含多级时间/空间熵、Voronoi、token/head/channel/block gating，token gating 采用 min-max 归一化后 top-ρ soft mask。

---

## 5. 硬件代理 (HW Proxy)

LayerProxyModel + build_layer_features + LayerHwProxy：输入 layer/segment 配置，输出 lat/ms、mem/MB、power/W；特征包含 log(flops/bytes/peak)、layer type one-hot、embed_dim、num_heads、mlp_ratio、seq_len、precision。

---

## 6. 芯粒类型 & 槽位 α

芯粒类型字段：name, peak_flops, peak_bw, mem_gb, area_mm2, tdp_w。ChipSlotConfig 使用 Gumbel-Softmax 输出 alpha 和期望 eff_specs，芯粒数量正则：`lambda_chip * (1-alpha[:,-1]).sum()`。

---

## 7. LayerNode / Segment / SegmentGraph

LayerNode 描述单层：id, layer_type, flops, bytes, seq_len, embed_dim, num_heads, mlp_ratio, precision, traffic_in/out_bytes, splittable。
Segment 聚合若干 layer_ids 的统计；SegmentGraph 包含 segments + edges (src, dst, traffic_bytes)。

---

## 8. MappingSolver

build_cost_matrix -> cost dict [K,S]; solve_mapping 采用贪心局部搜索，遵守 mem_limit，返回 mapping、per_slot_time、total_latency_ms、comm_ms；pipeline latency balanced/serial；通信可结合布局距离。

---

## 9. WaferLayout

与 v3 一致：边界、重叠、通信、热 penalty，forward 返回 (loss_layout, stats)。

---

## 10. PartitionPlanner（统一划分 + 可选细分）

PartitionPlanner.plan：

1. 生成 LayerNode 列表；
2. 构造粗粒度 segments，评估 objective_base（由 MappingSolver+WaferLayout 得到 latency/comm）；
3. 若 use_fine_split=False，直接返回 coarse；
4. 选择候选可拆 LayerNode（基于 splittable、flops/traffic 比例等阈值）；
5. 对每个候选层模拟“拆 vs 不拆”，估计 gain_ratio = (obj_no_split - obj_split)/obj_no_split；
6. 只接受 gain_ratio >= min_split_gain_ratio 的若干层（不超 max_split_layers / max_new_segments），生成 rewrite_plan；
7. 若无收益则不拆分，rewrite_plan=None。

Objective 示例：`w_latency*total_latency_ms + w_comm*comm_ms + w_balance*imbalance`。

---

## 11. compute_hw_loss（调用 PartitionPlanner）

调用 ChipSlotConfig 得 eff_specs -> PartitionPlanner.plan 得 segments/mapping/rewrite_plan/hw_stats；再结合 MappingSolver/WaferLayout 估计总 latency/energy/mem/area/layout，组合 L_hw = λ_T*lat + λ_E*energy + λ_mem*mem + L_area + L_chip + L_layout。

---

## 12. 训练循环

Version-C 训练使用 PartitionPlanner 在每轮（或每 N 轮）决定划分与是否细分；只有当细粒度方案 objective 更好时才触发重排，否则保持粗粒度。

---

## 13. 配置 & 日志

新增 partition 配置：w_latency, w_comm, w_balance, min_split_gain_ratio, flops_ratio_thresh, traffic_ratio_thresh, max_split_layers, max_groups_per_layer, max_new_segments。日志需记录 coarse/最终 objective、被拆分层及 gain_ratio、rewrite_plan 统计，若无拆分需显式提示。

---

## 14. 提示

实现时严格按本文件接口和约定，不自行更改输入输出。细粒度拆分只有在 objective 提升超过阈值时才执行，否则保持粗粒度，不触发 GraphRewrite。
