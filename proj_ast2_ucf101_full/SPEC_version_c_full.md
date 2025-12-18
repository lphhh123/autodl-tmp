# SPEC_version_c_full — Version-C 实现 & 实验说明书（给代码助手看的硬规范）

> 你（代码助手）要在现有 PyTorch 项目基础上，实现一个
> **AST2.0-lite + 时空熵剪枝 v2 + 可学习芯粒 + 映射 + 晶圆布局 + 可选计算图重构** 的系统。
>
> 所有算法细节、函数输入输出、实验命令都在本文件写明，
> **禁止随意发挥、改公式或改接口**。
> 若本 SPEC 和旧代码冲突，以本 SPEC 为准。

---

## 0. 环境 & 路径假设

* 项目根目录（已存在）：
  `/root/autodl-tmp/proj_ast2_ucf101_full`

* 统一默认工作目录：

  ```bash
  cd /root/autodl-tmp/proj_ast2_ucf101_full
  ```

* UCF101 数据集（已抽帧）结构假定为：

```text
data/ucf101/
  frames/
    ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/....jpg
    ...
  splits/
    trainlist01.txt
    testlist01.txt
```

* 所有脚本命令都从项目根目录运行。

---

## 1. 工程目录结构（建议 & 约定）

```text
project_root/
  configs/
    ast2_ucf101_dense.yaml
    ast2_ucf101_ast_only.yaml
    ast2_ucf101_ast_hw.yaml
    vc_phase2_fixed4_big.yaml
    vc_phase3_full_ucf101.yaml
    vc_phase3_nolayout_ucf101.yaml
    smoke_ast_ucf101.yaml
    smoke_version_c_ucf101.yaml
    proxy_ms_mem.yaml
    proxy_power.yaml
    gpu_data.yaml            # 芯粒类型库参数（含 area / w / h）

  scripts/
    run_proxy_ms_mem.py
    run_proxy_power.py
    run_ast2_ucf101.py
    run_version_c.py
    gen_experiment_cmds.py   # 自动生成 EXPERIMENTS_VERSION_C.md

  models/
    video_vit.py             # 基础 VideoViT
    ast2_pruner.py           # AST2.0-lite + 时空熵 v2 + Voronoi 稀疏模块

  hw_proxy/
    layer_proxy_model.py     # MLP proxy
    layer_hw_proxy.py        # LayerHwProxy 封装

  mapping/
    segments.py              # Segment 数据结构 & 构建
    mapping_solver.py        # MappingSolver + 重构相关逻辑

  layout/
    wafer_layout.py          # WaferLayout（布局优化，可关）

  trainer/
    trainer_single_device.py # 单卡 AST 训练
    trainer_version_c.py     # Version-C 外层交替训练

  utils/
    config.py
    logging.py
    metrics.py
    flops_estimator.py       # 层/segment FLOPs & Bytes 估计
```

---

## 2. 任务 & 数据（以 UCF101 为例）

### 2.1 输入输出形状

* 单个样本抽帧后：`[T, H, W, 3]`，例如 `T=8/16`, `H=W=224`。
* 经过 patch embedding 后：

  ```text
  x_token: [B, T, N, C]
    B: batch size
    T: 帧数
    N: 每帧 patch 数 (H/patch_size * W/patch_size)
    C: embedding dim (如 768)
  ```
* 分类输出：

  ```text
  logits: [B, num_classes]  # num_classes = 101 (UCF101)
  ```

---

## 3. 基础模型：VideoViT（不剪枝时）

在 `models/video_vit.py` 中实现：

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
        ast_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        - 标准 ViT backbone + 时间维度处理（T×N token）。
        - 当 use_ast_prune=True 时，在指定层调用 ASTPruner 的 gating。
        """
        ...

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        x: [B, T, C, H, W] 或 [B, C, T, H, W]（统一在前处理里转换）
        return_intermediate:
          False: 返回 logits
          True: 返回 (logits, info_dict)
            info_dict 至少包含：
              - "token_feat": 某一参考层的 [B, T, N, C] 特征 (给 AST 用)
              - "ast_stats": dict，包含 sparsity_token/head/ch/block
              - （必要时）每层 FLOPs/Bytes的统计信息
        """
```

---

## 4. AST2.0-lite v2：时空熵多级窗口 + Voronoi 稀疏

本节是你要求“严格对齐原文思路”的升级版 AST 部分（**跨模态堆积不要**，**TensorCore 利用率不要**）。

### 4.1 ASTPruner 顶层接口

在 `models/ast2_pruner.py`：

```python
class ASTPruner(nn.Module):
    """
    Spatio-temporal entropy based multi-granularity sparsity:
    - multi-level window entropy (time & space)
    - Voronoi-like spatial regions
    - token gating
    - head gating
    - channel gating
    - block gating
    """
    def __init__(self, cfg, embed_dim: int, num_heads: int, depth: int, num_patches: int):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_patches = num_patches

        # 1) 时序窗口层级，例如 [1, 2, 4]
        self.time_window_levels = cfg.get("time_window_levels", [1, 2, 4])
        # 2) 空间窗口层级，例如 [1, 2, 4]
        self.space_window_levels = cfg.get("space_window_levels", [1, 2, 4])
        self.entropy_tau = cfg.get("entropy_tau", 1.0)
        self.entropy_eps = 1e-6

        # 3) Voronoi 区域划分（在 init 时构造）
        H_p = cfg.get("patch_grid_h", 14)
        W_p = cfg.get("patch_grid_w", 14)
        self.region_ids, self.num_regions = build_voronoi_regions(num_patches, (H_p, W_p))

        # 4) 各粒度 gating 参数
        self._init_head_gates()
        self._init_channel_gates()
        self._init_block_gates()

    def forward_token_gating(self, token_feat: torch.Tensor):
        """
        token_feat: [B, T, N, C]
        返回:
          - mask_token: [B, T, N, 1] in [0,1]
          - sparsity_token: scalar
        """
        ...

    def get_head_gates(self):
        ...
    def get_channel_gates(self):
        ...
    def get_block_gates(self):
        ...

    def compute_L_AST(self, sparsity_token, sparsity_head, sparsity_ch, sparsity_block):
        """
        按 cfg 组合稀疏损失。
        """
```

### 4.2 多级窗口时空熵（严格定义）

定义 softmax/entropy，多级时间/空间窗口熵，token gating 逻辑按规范实现（略）。

---

## 5. 硬件代理模块 (HW Proxy)

定义 LayerProxyModel、build_layer_features、LayerHwProxy predict_layer/predict_segment（略）。

---

## 6. 芯粒库 & 可学习芯粒配置 α

芯粒类型包含 name/peak_flops/peak_bw/mem_gb/area_mm2/width_mm/height_mm/tdp_w；Gumbel-Softmax 槽位输出 alpha & eff_specs，并加芯粒数量正则（略）。

---

## 7. Segment & 计算图（含可选细粒度拆分 + 重排 stage）

定义 Segment 结构（含 can_split_fine/fine_groups），初始划分规则、可选细粒度拆分与 permutation 重排的决策逻辑（略，按规范）。

---

## 8. 映射模块 MappingSolver

构建成本矩阵、估计 pipeline latency、在贪心搜索中可调用 try_fine_split_segment，仅当 fine 方案更优才拆分，返回 segments/mapping/rewire_meta（略）。

---

## 9. WaferLayout

边界/重叠/通信/热 penalty；可关闭优化但类需存在。

---

## 10. 硬件损失 L_hw 组合

调用 ChipSlotConfig、MappingSolver、WaferLayout，合成 L_hw = λ_T*lat + λ_E*energy + λ_mem*mem + L_area + L_chip + L_layout（略）。

---

## 11. 训练与交替优化

支持 dense_baseline/single_device_ast/multi_chip_fixed/version_c_full；Version-C 外层循环按规范调用 compute_hw_loss（略）。

---

## 12. 实验设计 & 命令

列出 P0–P4 的所有实验 ID/目的/config/命令（略，可通过 gen_experiment_cmds.py 自动生成）。

---

## 13. gen_experiment_cmds.py 规范

生成 EXPERIMENTS_VERSION_C.md，包含所有实验说明与命令，支持 --out 参数（略）。

---

## 14. 对 Codex 的最终指令

阅读本文件，按定义实现所有模块与脚本，确保列举命令可运行；仅当细粒度方案 objective 更优时才触发重排，否则保持粗粒度（略）。
