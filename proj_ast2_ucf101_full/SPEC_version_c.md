# Version-C 实现总规范（给代码助手看的硬性需求）

> 你现在要在现有的 `proj_ast2_ucf101_full` 工程中，完全按照本 `SPEC_version_c.md` 的规范实现 Version-C 系统。
> 所有算法细节（时空熵、token/head/channel/block gating、芯粒槽位 Gumbel-Softmax、segment 划分规则、映射局部搜索、晶圆布局损失、硬件损失组合、交替优化训练流程）都以本规范为唯一准则，不需要自行设计新方法。
> 实现时：严格遵守本规范中的输入输出维度、公式和损失定义；所有超参数从配置文件读取；在代码注释中标明对应规范章节（例如 `# SPEC §5.4`）。

## 0. 总目标 & 顶层逻辑

你要在现有 `proj_ast2_ucf101_full` 项目上实现一个完整的 **AST2.0-lite + 硬件代理 + 多芯粒映射 + 晶圆布局联合优化系统（Version-C）**。

**联合优化变量：**

* 模型参数：θ（VideoViT 权重）
* 剪枝结构：s（时空 token / head / channel / block 的稀疏 gate）
* 芯粒配置：α（N 个槽位里每个槽位用哪种芯粒 or 为空，决定芯粒类型和数量）
* 映射方案：m（每个网络 segment 映射到哪个槽位）
* 晶圆布局：L（每个槽位上芯粒在圆形晶圆中的位置坐标）

**总损失：**

\[
L_total = L_task(θ, s) + λ_AST L_AST(s) + λ_hw L_hw(s, α, m, L)
\]

其中：

* `L_task`：分类 / 动作识别任务损失（交叉熵）
* `L_AST`：AST2.0-lite 的时空熵 + 多粒度稀疏损失
* `L_hw`：基于代理模型的硬件损失（时延 / 显存 / 能耗 / 面积 / 热 / 通信 / 芯粒数量等）

训练框架是**交替优化**：

1. 固定 α、m、L，更新 θ 和稀疏 gate s。
2. 固定 θ、m、L，更新 α（学习芯粒类型和数量）。
3. 固定 θ、s、α，重新求 m（优化映射）。
4. 固定 θ、s、α、m，更新 L（优化晶圆布局）。

所有模块的行为、输入输出、公式都在下面明确规范。

---

## 1. 工程结构约定

在现有项目基础上，按下列结构组织新模块（目录名可以稍有不同，但建议一致）：

```text
project_root/
  configs/
    ast2_ucf101.yaml
    gpu_data.yaml           # 芯粒库配置（必须包含面积和长宽信息）
  models/
    video_vit.py            # VideoViT 主干
    ast2_pruner.py          # AST2.0-lite 稀疏模块
  hw_proxy/
    layer_proxy_model.py    # 通用 MLP 代理
    layer_hw_proxy.py       # LayerHwProxy 封装
  chiplet/
    chiplet_lib.py          # 芯粒类型、槽位、可学习配置 α
  mapping/
    segments.py             # Segment 定义与构建
    mapping_solver.py       # Segment→Slot 映射策略
  layout/
    wafer_layout.py         # 晶圆布局 L 与布局损失
  trainer/
    trainer_single_device.py # 单设备 AST2.0-lite 训练
    trainer_version_c.py     # 完整 Version-C 训练
  scripts/
    run_ast2_ucf101.py      # 单设备 baseline 入口
    run_version_c.py        # Version-C 入口
  utils/
    config.py
    logging.py
    metrics.py
```

> 要求：**模块之间用清晰的接口交互**，不要把所有逻辑塞进一个脚本。

---

## 2. 数据 & 基础任务设定（以 UCF101 为例）

* 抽帧后的 UCF101 数据路径固定为（相对项目根目录）：

```text
data/ucf101/frames/<class_name>/<video_id>/*.jpg
data/ucf101/splits/trainlist01.txt
data/ucf101/splits/testlist01.txt
```

* DataLoader 输出的每个 batch 形状：

```python
# 推荐统一为
x: [B, T, C, H, W]   # B: batch; T: 帧数; C: 3; H,W: 224
y: [B]               # 类别索引
```

* 模型内部 patch 化后主要特征形状：

```text
x_token: [B, T, N, C]
  N : 每帧 patch 数（例如 14×14=196）
  C : token embedding 维度（如 768）
```

* 输出 logits：

```text
logits: [B, num_classes]  # num_classes=101 对 UCF101
```

---

## 3. 配置文件规范

### 3.1 UCF101 主配置 `configs/ast2_ucf101.yaml`

必须包含以下部分（可以合并到你的原 YAML 中，但字段含义必须一致）：

```yaml
model:
  type: "VideoViT"
  img_size: 224
  num_frames: 8
  num_classes: 101
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  patch_size: 16
  in_chans: 3
  drop_rate: 0.0
  attn_drop: 0.0
  drop_path_rate: 0.0
  use_ast_prune: true

data:
  dataset: "ucf101"
  root: "./data/ucf101"
  train_split: "./data/ucf101/splits/trainlist01.txt"
  val_split: "./data/ucf101/splits/testlist01.txt"
  clip_len: 8
  img_size: 224
  num_workers: 8

train:
  device: "cuda:0"
  epochs: 50
  batch_size: 8
  lr: 3e-4
  weight_decay: 0.05
  warmup_epochs: 5
  amp: true

ast:
  use_ast_prune: true
  tau_entropy: 1.0          # 熵 softmax 温度
  rho_token_target: 0.5     # 目标 token 保留比例
  lambda_token: 0.1
  lambda_head: 0.1
  lambda_ch: 0.1
  lambda_block: 0.05
  lambda_AST: 1.0           # 总稀疏 loss 权重

hw:
  mode: "version_c_full"    # ["none","single_device","version_c_full"]
  device_name: "RTX3090_FP16"    # 仅单设备模式用
  gpu_yaml: "./configs/gpu_data.yaml"
  proxy_weight_dir: "./proxy_weights"

  # 硬件损失项权重
  lambda_T: 1e-3            # latency
  lambda_E: 1e-4            # energy
  lambda_mem: 1e-4          # peak mem
  lambda_area: 1e-6         # 晶圆总面积上限 penalty
  lambda_chip: 1e-3         # 芯粒数量
  lambda_boundary: 1e-3     # 布局越界
  lambda_overlap: 1e-3      # 芯粒重叠
  lambda_comm_extra: 1e-6   # 布局模块内部额外通信
  lambda_thermal: 1e-4      # 热 penalty
  area_limit_mm2: 70000.0   # 晶圆总面积上限

  num_slots: 16
  wafer_radius_mm: 150.0
  latency_mode: "balanced"  # ["serial","balanced"]

mapping:
  strategy: "greedy_local"
  mem_limit_factor: 0.9

version_c:
  enabled: true
  outer_epochs: 10
  inner_steps_ast: 100
  inner_steps_alpha: 20
  inner_steps_layout: 20

chiplet:
  candidate_types: ["RTX4090_FP16", "RTX3090_FP16", "RTX2080Ti_FP16"]
  tau_init: 5.0
  tau_min: 0.5
  tau_decay: 0.9
```

### 3.2 芯粒库配置 `configs/gpu_data.yaml`

**必须包含面积和长宽信息**，用统一规则生成长宽：

```yaml
RTX4090_FP16:
  peak_flops_tflops: 82.6
  peak_bw_gbps: 1000.0
  mem_gb: 24.0
  sm_count: 128
  die_area_mm2: 600.0       # 物理面积
  aspect_ratio: 1.0         # width/height

RTX3090_FP16:
  peak_flops_tflops: 35.6
  peak_bw_gbps: 936.0
  mem_gb: 24.0
  sm_count: 82
  die_area_mm2: 500.0
  aspect_ratio: 1.0

RTX2080Ti_FP16:
  peak_flops_tflops: 13.4
  peak_bw_gbps: 616.0
  mem_gb: 11.0
  sm_count: 68
  die_area_mm2: 400.0
  aspect_ratio: 1.0
```

**宽高计算规则必须固定：**

```text
width_mm  = sqrt(die_area_mm2 * aspect_ratio)
height_mm = sqrt(die_area_mm2 / aspect_ratio)
```

以后如果你新增虚拟芯粒，只要给出 `die_area_mm2` 和 `aspect_ratio` 即可，不需要手写宽高。

---

## 4. VideoViT 主干模型规范

`models/video_vit.py` 中定义：

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

    def forward(self, x, return_intermediate: bool = False):
        """
        x: [B, T, C, H, W]
        return_intermediate:
          False -> 返回 logits
          True  -> 返回 (logits, info_dict)
        info_dict 至少包含：
          - "token_feat": 某层或多层的 [B, T, N, C] 特征
          - "gates": 一个 dict，里边包含:
              "token_mask": [B, T, N]
              "head_weights": [L, num_heads]
              "ch_weights": [L, hidden_dim]
              "block_weights": [L]
              以及各自的 sparsity 标量
          - "L_AST": 一个标量 Tensor（稀疏损失）
        """
```

要求：

* 所有 AST2.0-lite 逻辑在 `use_ast_prune=True` 时启用。
* 不使用剪枝时，模型行为和普通 VideoViT 完全一致。

---

## 5. AST2.0-lite 稀疏模块规范（继承原 AST2.0 创新点）

### 5.1 ASTPruner 总体接口

`models/ast2_pruner.py`：

```python
class ASTPruner(nn.Module):
    """
    时空熵驱动 + 多粒度 gate：
      - token gating (时空 token)
      - head gating (注意力头)
      - channel gating (MLP hidden)
      - block gating (layer-level)
    """
    def __init__(self, cfg_ast, depth: int, num_heads: int, embed_dim: int, mlp_ratio: float):
        ...

    def forward(self, token_feat: torch.Tensor, blocks_info: Optional[Dict] = None):
        """
        token_feat: [B, T, N, C]，来自某一层或多层
        blocks_info: 包含各层中间特征（可选）

        返回:
          - gates: dict，同 VideoViT.info_dict["gates"]
          - L_AST: 稀疏损失标量
        """
```

### 5.2 时空熵计算（严格公式）

实现函数：

```python
def compute_spatiotemporal_entropy(x: torch.Tensor, tau: float = 1.0):
    """
    x: [B, T, N, C]
    返回:
      H_time: [B, N]
      H_space: [B, T]
    步骤:
      1) p[b,t,n,c] = softmax_c( x[b,t,n,c] / tau )
      2) μ[b,n,c] = mean_t p[b,t,n,c]
         H_time[b,n] = - sum_c μ[b,n,c] * log(μ[b,n,c] + 1e-6)
      3) ν[b,t,c] = mean_n p[b,t,n,c]
         H_space[b,t] = - sum_c ν[b,t,c] * log(ν[b,t,c] + 1e-6)
    """
```

这一块直接继承 AST2.0 关于时空熵的核心思想：**时间维上看每个 token 的信息量，空间维上看每帧整体的信息量**。

### 5.3 Voronoi 风格空间区域划分（规则化定义）

函数：

```python
def build_voronoi_regions(num_patches: int, grid_hw: Tuple[int,int], num_regions: int):
    """
    输入:
      num_patches = N
      grid_hw = (H_p, W_p)，例如 patch=16 时，224x224 -> (14,14)
      num_regions = R（例如 4 或 8）
    输出:
      region_ids: LongTensor [N], 每个 patch 一个 region id (0..R-1)

    规则:
      - 假设 patch 顺序为 i*W_p + j (行优先)
      - 把 H_p×W_p 划分成 R 个大格，每格对应一个“区域中心”。
        例如:
          R=4 -> 2x2 大格
          R=8 -> 4x2 或 2x4，可以固定为 4x2
      - 对每个 patch (i,j)，找到其大格 index，作为 region_ids[n]。
      - 这个映射是固定的（与 batch 无关），初始化时算一次并缓存。
    """
```

**目的**：保留 AST2.0 中“Voronoi 区域级稀疏”的思想，但用规则格子代替复杂 Voronoi，实现简单且可复现空间结构感知。

### 5.4 Token gating 规则（重要）

实现函数：

```python
def token_gating(
    H_time: torch.Tensor,     # [B, N]
    H_space: torch.Tensor,    # [B, T]
    region_ids: torch.Tensor, # [N]
    rho_target: float,
    a: float = 0.5,
    b: float = 0.5,
    temperature: float = 0.1,
):
    """
    目标：基于时空熵 + 区域重要性，决定保留哪些 token。

    步骤:
      1) 对 H_time[b,:] 做 min-max 归一到 [0,1]:
         h_t_norm[b,n] = (H_time[b,n] - min_n) / (max_n - min_n + eps)

      2) 对 H_space[b,:] 做 min-max 归一到 [0,1]:
         h_s_norm[b,t] 同理

      3) 区域重要性（每个 region r）:
         region_importance[b,r] = mean_t h_s_norm[b,t]    # 简化：所有 region 共用同一帧重要性

      4) patch 得分:
         score[b,n] = a * h_t_norm[b,n]
                    + b * region_importance[b, region_ids[n]]

      5) 对每个样本 b:
         - 按 score[b,:] 从大到小排序
         - k = int(rho_target * N)
         - 第 k 大的得分记作 threshold[b]
         - 软 mask:
             mask[b,n] = sigmoid( (score[b,n] - threshold[b]) / temperature )

      输出:
        mask_token: [B, N] ∈ (0,1)
        sparsity_token = 1 - mask_token.mean()
    """
```

**必须遵守以上逻辑，不能省略 soft 阶段**。推理时可以把 `mask_token` 改成 hard top-k。

### 5.5 Head / Channel / Block gating

**Head gating：**

* 对每层 l，维护 logit 向量 `g_head[l, num_heads]`
* `w_head[l,h] = sigmoid(g_head[l,h])`
* 在多头注意力输出处乘：

```python
# attn_out: [B, num_heads, T*N, head_dim]
attn_out = attn_out * w_head[l].view(1, num_heads, 1, 1)
```

* `sparsity_head = 1 - w_head.mean()`（对所有层取平均）

**Channel gating：**

* 对每层 MLP hidden dim，维护 `g_ch[l, hidden_dim]`
* `w_ch = sigmoid(g_ch)`
* 对 MLP 输出做逐通道缩放：

```python
# mlp_out: [B, T*N, hidden_dim]
mlp_out = mlp_out * w_ch[l].view(1, 1, hidden_dim)
```

* `sparsity_ch = 1 - w_ch.mean()`（对所有层平均）

**Block gating：**

* 每层维护标量 `g_block[l]`
* `w_block[l] = sigmoid(g_block[l])`
* 残差结构：

```python
x = x + w_block[l] * block_fn(x)
```

* `sparsity_block = 1 - w_block.mean()`

### 5.6 稀疏损失 L_AST（严格配方）

在 ASTPruner 内部计算：

```python
L_AST = (
    λ_token  * sparsity_token +
    λ_head   * sparsity_head  +
    λ_ch     * sparsity_ch    +
    λ_block  * sparsity_block
)
```

这些 λ 来自 `cfg.ast.*`。
VideoViT forward 时把 `L_AST` 放进 `info_dict["L_AST"]` 返回。

---

## 6. 硬件代理模块规范（LayerHwProxy）

### 6.1 特征构建

`hw_proxy/layer_hw_proxy.py` 中实现函数：

```python
def build_layer_features(layer_cfg: Dict, device_cfg: Dict) -> np.ndarray:
    """
    layer_cfg 必须包含:
      - layer_type: int (0=patch_embed,1=attn,2=mlp,...)
      - flops: float
      - bytes: float
      - embed_dim: int
      - num_heads: int
      - mlp_ratio: float
      - seq_len: int
      - precision: int (0=fp32,1=fp16)

    device_cfg 必须包含:
      - peak_flops_tflops: float
      - peak_bw_gbps: float

    输出向量维度固定为:
      [
        log10(flops + 1),
        log10(bytes + 1),
        log10(peak_flops_tflops * 1e12),
        log10(peak_bw_gbps * 1e9),
        layer_type_one_hot...,    # e.g. length 4
        embed_dim / 1024,
        num_heads / 16,
        mlp_ratio / 4,
        seq_len / 1024,
        precision
      ]
    """
```

要求：**训练 proxy 用的特征构建规则必须和这里一致**。

### 6.2 LayerProxyModel + LayerHwProxy 接口

```python
class LayerProxyModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, out_dim: int = 1):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        # return: [N, out_dim]
        ...
```

`LayerHwProxy` 封装：

```python
class LayerHwProxy:
    def __init__(self, device_name: str, gpu_yaml: str, weight_dir: str):
        """
        从 gpu_yaml 读取 device_cfg
        从 weight_dir 加载:
          - latency_proxy.pth
          - mem_proxy.pth
          - power_proxy.pth
        """
        ...

    def predict_layers_batch(self, layers_cfg: List[Dict]) -> Dict[str, np.ndarray]:
        """
        输入: 多个 layer_cfg
        输出:
          {
            "lat_ms": [N],
            "mem_mb": [N],
            "power_w": [N],
          }
        """
```

Segment 的预测在后面通过多层合成。

---

## 7. 芯粒库 & 可学习槽位 α（chiplet_lib.py）

### 7.1 ChipletType & ChipletLibrary

```python
@dataclass
class ChipletType:
    name: str
    peak_flops_tflops: float
    peak_bw_gbps: float
    mem_gb: float
    die_area_mm2: float
    aspect_ratio: float

    @property
    def width_mm(self) -> float:
        return math.sqrt(self.die_area_mm2 * self.aspect_ratio)

    @property
    def height_mm(self) -> float:
        return math.sqrt(self.die_area_mm2 / self.aspect_ratio)
```

`ChipletLibrary` 从 `gpu_data.yaml` 加载：

```python
class ChipletLibrary:
    def __init__(self, yaml_path: str):
        ...

    def get(self, name: str) -> ChipletType:
        ...
```

### 7.2 ChipletSlots：N 个槽位 + Gumbel-Softmax α

```python
class ChipletSlots(nn.Module):
    """
    N_slot 个槽位，每个槽位:
      - 在 M 种芯粒类型中选择一种
      - 或选择“空槽位”（不放芯片）
    通过 Gumbel-Softmax 实现可学习的概率分布 α。
    """
    def __init__(self, library: ChipletLibrary, candidate_names: List[str],
                 num_slots: int, tau_init: float):
        ...

    def set_tau(self, tau: float):
        self.tau = tau

    def forward(self, hard: bool = False):
        """
        返回:
          alpha:    [N_slot, M+1]  概率分布 (最后一维 index=M 是“空”)
          eff_specs: dict, 每个 slot 的有效硬件参数 (期望值):
            {
              "peak_flops": [N_slot],
              "peak_bw":    [N_slot],
              "mem_gb":     [N_slot],
              "area_mm2":   [N_slot],
              "width_mm":   [N_slot],
              "height_mm":  [N_slot],
              "tdp_w":      [N_slot],  # 如需要可从峰值功耗估
            }

        规范:
          - logits: [N_slot, M+1] 为可训练参数
          - 使用 Gumbel-Softmax:
              g ~ Gumbel(0,1)
              y = softmax( (logits + g) / tau )
            训练时可选 straight-through hard one-hot。
          - eff_specs[j,*] = sum_t alpha[j,t] * spec[t,*]
            其中 "空" 类型的 spec = 0。
        """
```

### 7.3 芯粒数量正则

在硬件损失中使用：

```python
chip_used_prob = 1.0 - alpha[:, -1]  # 去掉“空”的概率
L_chip_count = λ_chip * chip_used_prob.sum()
```

---

## 8. Segment 定义与 FLOPs / Bytes 统计（segments.py）

### 8.1 Segment 数据结构

```python
@dataclass
class Segment:
    id: int
    layer_indices: List[int]
    flops: float
    bytes: float
    seq_len: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    precision: int
    traffic_in_bytes: float
    traffic_out_bytes: float
```

### 8.2 Segment 构建规则

函数：

```python
def build_segments_from_model(model: VideoViT, cfg) -> List[Segment]:
    """
    定义 segment 划分规则（必须固定）:

    规范：
      - 设模型有 L = depth 层 Transformer block，索引为 0..L-1。
      - 设一个整型参数 seg_block_size（可写死或从 cfg 读取，例如 2）。
      - 将 block 序列按顺序均匀划分:
          seg_0: [0,1]
          seg_1: [2,3]
          ...
      - 对每个 segment:
          - layer_indices: 对应 block index 列表
          - flops: = 所有层 FLOPs 之和
          - bytes: = 所有层 Bytes 之和
          - seq_len: 取当前 token 数（剪枝后可更新）
          - embed_dim, num_heads, mlp_ratio:
              取 segment 中任意一层（建议第一层）的配置
          - precision: 0 或 1，来自训练设置
          - traffic_in_bytes:
              = seq_len * embed_dim * bytes_per_element
          - traffic_out_bytes:
              = seq_len * embed_dim * bytes_per_element
    """
```

层的 FLOPs/Bytes 计算可以用近似公式（只要相对量级合理即可），例如：

* MSA FLOPs ≈ 4 * L_token * C^2
* MLP FLOPs ≈ 2 * L_token * C * (mlp_ratio*C)
* Bytes ≈ 读写参数 + 读写激活简单估计。

**要求**：Segment 构建逻辑固定且可重复，不能随训练阶段改变规则。

---

## 9. 映射模块：Segment → Slot（mapping_solver.py）

### 9.1 成本矩阵构造

```python
class MappingSolver:
    def __init__(self, strategy: str, mem_limit_factor: float):
        self.strategy = strategy
        self.mem_limit_factor = mem_limit_factor

    def build_cost_matrix(self, segments: List[Segment],
                          eff_specs: Dict, proxy: LayerHwProxy) -> Dict[str, torch.Tensor]:
        """
        对每个 segment k 和每个槽位 j:

          - 构造 segment_cfg:
              {
                "layer_type": ...,       # 可以统一为某个类型或细分
                "flops": segments[k].flops,
                "bytes": segments[k].bytes,
                "embed_dim": segments[k].embed_dim,
                "num_heads": segments[k].num_heads,
                "mlp_ratio": segments[k].mlp_ratio,
                "seq_len": segments[k].seq_len,
                "precision": segments[k].precision,
              }
            再叠加 slot j 的 device_cfg (从 eff_specs 中取)

          - 调用 proxy.predict_segment(segment_cfg) 得:
              lat_ms[k,j], mem_mb[k,j], power_w[k,j]

        输出:
          {
            "lat_ms": Tensor[K, S],
            "mem_mb": Tensor[K, S],
            "power_w": Tensor[K, S],
          }
        """
```

### 9.2 pipeline 时延估计（balanced 模式）

```python
def estimate_pipeline_latency(mapping: List[int],
                              cost_lat: torch.Tensor,
                              mode: str = "balanced") -> float:
    """
    mapping: len=K，mapping[k]=slot id
    cost_lat: [K,S]

    若 mode="serial":
      total_latency_ms = sum_k cost_lat[k, mapping[k]]

    若 mode="balanced":
      device_time[s] = sum_{k: mapping[k]=s} cost_lat[k,s]
      total_latency_ms = max_s device_time[s]

    返回 total_latency_ms (float/Tensor)
    """
```

### 9.3 映射求解策略 "greedy_local"

```python
def solve_mapping(
    self,
    segments: List[Segment],
    eff_specs: Dict,
    proxy: LayerHwProxy,
    layout_positions: Optional[torch.Tensor] = None,
) -> Dict:
    """
    步骤:
      1) 用 build_cost_matrix 得到 lat_ms, mem_mb, power_w
      2) 初始 mapping:
           mapping[k] = k % num_slots   # round-robin
      3) 计算当前 total_latency_ms (balanced 模式)

      4) 局部搜索:
         repeat:
           improved = False
           for k in range(K):
             curr_d = mapping[k]
             best_d = curr_d
             best_latency = current_latency

             for d in range(S):  # 尝试所有槽位
               if d == curr_d: continue
               如果把 seg k 从 curr_d 移到 d 会违反 mem 约束 -> 跳过

               临时改 mapping[k]=d
               new_latency = estimate_pipeline_latency(mapping, lat_ms, mode="balanced")
               若 new_latency + eps < best_latency:
                 记录 best_d = d, best_latency = new_latency

             如果 best_d != curr_d:
               mapping[k] = best_d
               current_latency = best_latency
               improved = True
           直到 improved=false

      5) 显存约束:
         mem_usage[s] = max_{k: mapping[k]=s} mem_mb[k,s]
         要求 mem_usage[s] <= eff_specs["mem_gb"][s]*1024 * mem_limit_factor
         （违反时该 move 直接无效）

      6) 通信时间估计（可选）:
         若 layout_positions 非空:
           对所有相邻 (k, k+1):
             d1, d2 = mapping[k], mapping[k+1]
             traffic = segments[k].traffic_out_bytes
             dist = ||pos[d1]-pos[d2]||
             comm_ms += traffic * dist * comm_scale  (comm_scale 从配置读取)

    返回:
      {
        "mapping": List[int],
        "per_slot_time_ms": List[float],
        "total_latency_ms": float,
        "comm_ms": float,
      }
    """
```

---

## 10. 晶圆布局模块规范（wafer_layout.py）

### 10.1 参数与状态

```python
class WaferLayout(nn.Module):
    def __init__(self, num_slots: int, wafer_radius_mm: float):
        super().__init__()
        # 可训练坐标
        self.pos = nn.Parameter(torch.zeros(num_slots, 2))
        self.wafer_radius_mm = wafer_radius_mm
```

初始化时可以把 pos 布成一个圆环或网格，但不在规范中强制；唯一要求：**后续所有计算都用 self.pos**。

### 10.2 边界 penalty

```python
def boundary_penalty(self, eff_specs, margin: float = 0.0):
    # eff_specs["area_mm2"]: [S]
    # 半径近似: r_chip = sqrt(area/pi)
    centers = self.pos  # [S,2]
    r_center = torch.sqrt(centers[:,0]**2 + centers[:,1]**2 + 1e-6)
    r_chip = torch.sqrt(eff_specs["area_mm2"] / math.pi + 1e-6)
    # 芯粒最外侧不能超出 wafer_radius
    violation = torch.relu(r_center + r_chip + margin - self.wafer_radius_mm)
    return (violation**2).sum()
```

### 10.3 重叠 penalty

```python
def overlap_penalty(self, eff_specs):
    centers = self.pos
    r_chip = torch.sqrt(eff_specs["area_mm2"] / math.pi + 1e-6)
    S = centers.shape[0]
    penalty = 0.0
    for i in range(S):
        for j in range(i+1, S):
            dist = torch.sqrt(((centers[i]-centers[j])**2).sum() + 1e-6)
            min_dist = r_chip[i] + r_chip[j]
            overlap = torch.relu(min_dist - dist)
            penalty = penalty + overlap**2
    return penalty
```

### 10.4 通信距离损失（layout 部分）

```python
def comm_loss(self, mapping: List[int], segments: List[Segment],
              eff_specs, distance_scale: float):
    centers = self.pos
    comm_cost = 0.0
    for k in range(len(segments)-1):
        d1, d2 = mapping[k], mapping[k+1]
        if d1 == d2:
            continue
        traffic = segments[k].traffic_out_bytes
        dist = torch.sqrt(((centers[d1]-centers[d2])**2).sum() + 1e-6)
        comm_cost = comm_cost + traffic * dist * distance_scale
    return comm_cost
```

### 10.5 热 penalty（简化径向核）

```python
def thermal_penalty(self, eff_specs, T_ambient=25.0, T_limit=85.0,
                    sigma_mm=50.0, alpha=0.01):
    centers = self.pos                  # [S,2]
    power = eff_specs["tdp_w"]          # [S]
    S = centers.shape[0]

    # 温度在每个芯粒中心采样
    temps = []
    for i in range(S):
        x_i = centers[i]
        # kernel: K(r) = exp( -r^2 / (2*sigma^2) )
        r2 = ((centers - x_i)**2).sum(dim=1)
        K = torch.exp(-r2 / (2.0 * sigma_mm**2))
        T_i = T_ambient + alpha * (K * power).sum()
        temps.append(T_i)
    temps = torch.stack(temps)  # [S]
    T_max = temps.max()
    return torch.relu(T_max - T_limit)**2
```

### 10.6 布局总损失接口

```python
def forward(self, mapping, segments, eff_specs,
            lambda_boundary, lambda_overlap,
            lambda_comm, lambda_thermal,
            distance_scale):
    L_boundary = self.boundary_penalty(eff_specs)
    L_overlap  = self.overlap_penalty(eff_specs)
    L_comm     = self.comm_loss(mapping, segments, eff_specs, distance_scale)
    L_thermal  = self.thermal_penalty(eff_specs)

    L_layout = (lambda_boundary * L_boundary +
                lambda_overlap  * L_overlap  +
                lambda_comm     * L_comm     +
                lambda_thermal  * L_thermal)

    stats = {
        "boundary": L_boundary.detach(),
        "overlap":  L_overlap.detach(),
        "comm":     L_comm.detach(),
        "thermal":  L_thermal.detach(),
    }
    return L_layout, stats
```

---

## 11. 硬件损失 L_hw 组合（trainer_version_c.py 内）

### 11.1 compute_hw_loss 规范

```python
def compute_hw_loss(
    segments: List[Segment],
    chiplet_slots: ChipletSlots,
    hw_proxy: LayerHwProxy,
    mapping_solver: MappingSolver,
    wafer_layout: WaferLayout,
    hw_cfg: Dict,
):
    """
    hw_cfg 需包含:
      lambda_T, lambda_E, lambda_mem,
      lambda_area, lambda_chip,
      lambda_boundary, lambda_overlap,
      lambda_comm_extra, lambda_thermal,
      area_limit_mm2, latency_mode
    """
```

**步骤：**

1. 获取 α 和 eff_specs：

```python
alpha, eff_specs_tensor = chiplet_slots(hard=False)
# eff_specs_tensor: [S,6] -> 拆出为 dict:
eff_specs = {
  "peak_flops": eff_specs_tensor[:,0],
  "peak_bw":    eff_specs_tensor[:,1],
  "mem_gb":     eff_specs_tensor[:,2],
  "area_mm2":   eff_specs_tensor[:,3],
  "width_mm":   eff_specs_tensor[:,4],
  "height_mm":  eff_specs_tensor[:,5],
  "tdp_w":      some_function_of_peak_flops(...)  # 或从 yaml 加载
}
chip_used_prob = 1.0 - alpha[:, -1]
L_chip_count = hw_cfg["lambda_chip"] * chip_used_prob.sum()
```

2. 利用 `MappingSolver.build_cost_matrix` 与 `solve_mapping`：

```python
cost = mapping_solver.build_cost_matrix(segments, eff_specs, hw_proxy)
mapping_result = mapping_solver.solve_mapping(
    segments, eff_specs, hw_proxy,
    layout_positions=wafer_layout.pos,
)
mapping = mapping_result["mapping"]
total_latency_ms = mapping_result["total_latency_ms"]   # compute 部分
comm_ms = mapping_result["comm_ms"]
```

3. 估计总能耗：

```python
K = len(segments)
S = eff_specs["area_mm2"].shape[0]

lat_ms = cost["lat_ms"]
power_w = cost["power_w"]

total_energy_j = 0.0
for k in range(K):
    d = mapping[k]
    lat_s = lat_ms[k, d] / 1e3
    p = power_w[k, d]
    total_energy_j += lat_s * p
total_energy_j = torch.tensor(total_energy_j, device=alpha.device, dtype=torch.float32)
```

4. 峰值显存：

```python
mem_mb = cost["mem_mb"]
mem_usage = torch.zeros(S, device=alpha.device)
for k in range(K):
    d = mapping[k]
    mem_usage[d] = torch.maximum(mem_usage[d], mem_mb[k, d])
peak_mem_mb = mem_usage.max()
```

5. 面积 penalty：

```python
total_area_mm2 = (eff_specs["area_mm2"] * chip_used_prob).sum()
area_limit = hw_cfg["area_limit_mm2"]
L_area = hw_cfg["lambda_area"] * torch.relu(total_area_mm2 - area_limit)**2
```

6. 布局损失：

```python
L_layout, layout_stats = wafer_layout(
    mapping,
    segments,
    eff_specs,
    lambda_boundary=hw_cfg["lambda_boundary"],
    lambda_overlap=hw_cfg["lambda_overlap"],
    lambda_comm=hw_cfg["lambda_comm_extra"],
    lambda_thermal=hw_cfg["lambda_thermal"],
    distance_scale=1e-9,  # 具体数值从配置调整
)
```

7. 总硬件损失：

```python
L_hw = (
    hw_cfg["lambda_T"]   * total_latency_ms +
    hw_cfg["lambda_E"]   * total_energy_j   +
    hw_cfg["lambda_mem"] * peak_mem_mb      +
    L_area + L_chip_count + L_layout
)
```

返回 `L_hw` 和一份 `hw_stats` dict，包含：

* `total_latency_ms`, `total_energy_j`, `peak_mem_mb`
* `total_area_mm2`, `chip_count`（即 `chip_used_prob.sum()`）
* `layout_stats` 中的 boundary/overlap/comm/thermal

---

## 12. 训练与交替优化规范

### 12.1 单设备 AST2.0-lite 模式（hw.mode="single_device"）

`trainer_single_device.py`：

* 只用一个物理设备，不使用 chiplet_slots / mapping / wafer_layout；
* `L_hw` 只包含单设备预测（可用 hw_proxy + 简化 flops→latency 映射）。

### 12.2 完整 Version-C 模式（hw.mode="version_c_full"）

`trainer_version_c.py` 训练主循环：

```python
for outer_epoch in range(cfg.version_c.outer_epochs):

    # ===== Step A: 更新 θ 和 s （AST2.0-lite + 硬件损失）=====
    for step in range(cfg.version_c.inner_steps_ast):
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        logits, info = model(x, return_intermediate=True)
        L_task = F.cross_entropy(logits, y)
        L_AST = info["L_AST"]

        # 根据当前模型构建 segments（可在外层缓存）
        segments = build_segments_from_model(model, cfg)

        L_hw, hw_stats = compute_hw_loss(
            segments,
            chiplet_slots,
            hw_proxy,
            mapping_solver,
            wafer_layout,
            hw_cfg=cfg.hw,
        )

        loss = L_task + cfg.ast.lambda_AST * L_AST + cfg.hw.lambda_Tot * L_hw  # lambda_Tot 可设1.0

        optimizer_model.zero_grad()
        optimizer_chip_slots.zero_grad()   # α 也可以参与这一步
        optimizer_layout.zero_grad()       # 若希望 layout 也随梯度更新
        loss.backward()
        optimizer_model.step()
        optimizer_chip_slots.step()
        optimizer_layout.step()

    # ===== Step B: 只针对芯粒 logits α 的专门优化（可选但推荐）=====
    for step in range(cfg.version_c.inner_steps_alpha):
        segments = build_segments_from_model(model, cfg)
        L_hw, hw_stats = compute_hw_loss( ... )
        optimizer_chip_slots.zero_grad()
        L_hw.backward()
        optimizer_chip_slots.step()

    # ===== Step C: 在当前 θ, s, α 下求最优离散映射 m（不参与梯度）=====
    segments = build_segments_from_model(model, cfg)
    alpha, eff_specs = chiplet_slots(hard=False)
    mapping_result = mapping_solver.solve_mapping(
        segments,
        eff_specs,
        hw_proxy,
        layout_positions=wafer_layout.pos,
    )
    # 可以缓存 mapping_result，用于下一轮 layout 优化或分析

    # ===== Step D: 使用布局模块对 L 单独再优化若干步（梯度法）=====
    for step in range(cfg.version_c.inner_steps_layout):
        segments = build_segments_from_model(model, cfg)
        alpha, eff_specs = chiplet_slots(hard=False)
        mapping = mapping_result["mapping"]
        L_layout, layout_stats = wafer_layout(
            mapping,
            segments,
            eff_specs,
            cfg.hw.lambda_boundary,
            cfg.hw.lambda_overlap,
            cfg.hw.lambda_comm_extra,
            cfg.hw.lambda_thermal,
            distance_scale=1e-9,
        )
        optimizer_layout.zero_grad()
        L_layout.backward()
        optimizer_layout.step()
```

同时，Gumbel-Softmax 温度 τ 随 outer_epoch 递减：

```python
tau = max(cfg.chiplet.tau_min,
          cfg.chiplet.tau_init * (cfg.chiplet.tau_decay ** outer_epoch))
chiplet_slots.set_tau(tau)
```

---

## 13. 日志与输出要求

训练过程中，需要定期（每 N step 或每 epoch）记录以下指标（打印 + 写日志文件）：

1. 任务部分：

   * train/val loss, train/val acc（UCF101 Top-1/Top-5）

2. AST2.0-lite 部分：

   * `sparsity_token`, `sparsity_head`, `sparsity_ch`, `sparsity_block`
   * token 保留率（=1 - sparsity_token）

3. 硬件部分（来自 compute_hw_loss 和 wafer_layout）：

   * `total_latency_ms`, `total_energy_j`, `peak_mem_mb`
   * `total_area_mm2`, `chip_count`
   * `layout_boundary`, `layout_overlap`, `layout_comm`, `layout_thermal`

4. 结构摘要（每个 checkpoint）：

   * 剪枝后全局结构：有效深度、平均 token 数、head 数、hidden dim 比例
   * α 的分布：每个芯粒类型期望使用数量（`chip_used_prob` 在各类型上的贡献）
   * 映射：每个 slot 上的 segment 数量与时间占比。

这些信息建议写入 `logs/version_c_stats.jsonl`，每行一个 JSON 记录。

---

## 14. 给 Codex 的一句话提示（可以放在文档顶部）

> 你现在要在现有的 `proj_ast2_ucf101_full` 工程中，完全按照本 `SPEC_version_c.md` 的规范实现 Version-C 系统。
> 所有算法细节（时空熵、token/head/channel/block gating、芯粒槽位 Gumbel-Softmax、segment 划分规则、映射局部搜索、晶圆布局损失、硬件损失组合、交替优化训练流程）都以本规范为唯一准则，不需要自行设计新方法。
> 实现时：严格遵守本规范中的输入输出维度、公式和损失定义；所有超参数从配置文件读取；在代码注释中标明对应规范章节（例如 `# SPEC §5.4`）。
