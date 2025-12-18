# SPEC_version_c_v2.md

**AST2.0-v2 + 多芯粒 + 晶圆布局联合优化实现规范**

> 本文档是给代码助手（如 VSCode Copilot / OpenAI Codex）看的**硬规范**。
> 所有算法细节、输入输出、伪代码和约定，以本文件为准。
> 不允许“自由发挥算法”，只能在这个框架下写实现。

---

## 0. 目标与总览

要实现的是一个 **AST2.0-v2 + 硬件感知 + 多芯粒映射 + 晶圆布局联合优化系统（Version-C）**。

系统包含四大核心模块：

1. **AST2.0-v2 剪枝模块（本版升级）**

   * 在 Video Transformer 上实现**多尺度时间窗口时空熵 + 真 Voronoi 几何**驱动的多粒度稀疏：

     * token（时空）
     * head
     * MLP hidden channel
     * block
   * 保持分类 / 动作识别精度（UCF-101 为例）。
2. **硬件代理模块 (HW Proxy)**

   * 给定剪枝后结构 + 某芯粒类型，预测每层/segment 的 **时延 ms、显存 MB、功耗 W**。
3. **多芯粒映射模块 (Mapping)**

   * 把模型切成一系列 segments，将它们映射到多个芯粒上，近似最小化 pipeline 推理时延，同时满足显存约束。
4. **晶圆布局模块 (Wafer Layout)**

   * 在圆形晶圆中放置这些芯粒（由若干槽位实例化而来），最小化通信代价和热 penalty，满足面积 / 边界 / 不重叠约束。

最外层有一个 **交替优化训练框架**，联合优化：

* 网络参数 θ
* 剪枝结构 s（各种 gates）
* 芯粒配置 α（每个槽位用什么芯粒 / 要不要用）
* 映射 m（segment→slot）
* 布局 L（各 slot 坐标）

---

## 1. 工程目录结构（建议）

```text
project_root/
  configs/
    ast2_ucf101.yaml          # 单 GPU AST2.0 训练配置
    version_c_full.yaml       # 完整 Version-C 配置
    gpu_data.yaml             # 芯粒类型库参数
  scripts/
    run_ast2_ucf101.py        # Phase 2: 单设备 AST2.0-v2 训练
    run_version_c.py          # Phase 5: 完整 Version-C 交替优化
  models/
    video_vit.py              # 基础 VideoViT
    ast2_pruner.py            # AST2.0-v2 稀疏模块 (token/head/ch/block)
  hw_proxy/
    layer_proxy_model.py      # 通用 MLP proxy
    layer_hw_proxy.py         # LayerHwProxy 封装类
  mapping/
    segments.py               # Segment 数据结构 & 构造
    mapping_solver.py         # MappingSolver 映射算法
  layout/
    wafer_layout.py           # WaferLayout 布局优化与损失
  trainer/
    trainer_single_device.py  # 单设备 AST2.0-v2 训练器
    trainer_version_c.py      # Version-C 交替优化训练器
  utils/
    config.py                 # YAML/args 加载
    logging.py                # 日志
    metrics.py                # 精度、FLOPs 等指标
```

> 要求：文档里的类名 / 函数名优先使用以上命名。已有项目结构可做适配，但接口约定必须保持一致。

---

## 2. 任务与数据（UCF-101 示例）

以 UCF-101 动作分类为例，输入输出形状约定如下：

* 每个样本是一段视频，抽帧后：

  ```text
  x_raw: [T, H, W, 3]
  ```

* 经过 patch embedding + flatten 后某一层 token 形状：

  ```text
  x_token: [B, T, N, C]
    B：batch size
    T：采样帧数（8, 16 等）
    N：每帧 patch 数（H_p * W_p）
    C：embedding dim（如 768）
  ```

* 分类输出：

  ```text
  logits: [B, num_classes]
  ```

数据预处理逻辑（如 UCF-101 的抽帧脚本）已有，这里只定义模型接口。

---

## 3. 基础模型：VideoViT（无剪枝时）

`models/video_vit.py` 中定义标准 VideoViT，支持插入 AST2.0-v2 模块：

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
        # AST2.0-v2
        use_ast_prune: bool = False,
        ast_cfg: Optional[Dict] = None,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ):
        """
        x: [B, T, C, H, W]  (可在 forward 内部转换为 [B,T,N,C])
        return_intermediate:
          - False: 返回 logits
          - True: 返回 (logits, info_dict)
            info_dict 至少包含:
              - "L_AST": 稀疏损失标量 (float tensor)
              - "sparsity_token", "sparsity_head", "sparsity_ch", "sparsity_block"
              - 可选: 中间特征供构造 segments 使用
        """
```

当 `use_ast_prune=True` 时，在合适层（如中间几层或所有 block）插入 AST2.0-v2 的 token/head/ch/block gating。

---

## 4. AST2.0-v2 稀疏模块（完整版规范）

这一节是 **本 v2 的核心更新**，替换掉旧版简化 AST2.0-lite。

### 4.1 输入形状 & patch 排布

在 AST 剪枝层使用的特征形状：

```text
x: [B, T, N, C]
  B: batch
  T: 时间帧数
  N: 总 patch 数 = H_p * W_p
  C: 通道维度 / embedding 维度
```

patch 序号与 2D 坐标映射：

* 行优先序：`n = i * W_p + j`
* `i ∈ [0, H_p-1]`, `j ∈ [0, W_p-1]`

实现函数（一次性生成）：

```python
def get_patch_coords(H_p: int, W_p: int, device=None) -> torch.Tensor:
    """
    返回:
      coords: [N, 2]，N=H_p*W_p
      coords[n] = (u, v) ∈ [0,1]^2
        u = (i + 0.5) / H_p
        v = (j + 0.5) / W_p
    """
```

---

### 4.2 ASTPruner 总体接口

`models/ast2_pruner.py` 中定义：

```python
class ASTPruner(nn.Module):
    """
    AST2.0-v2 稀疏模块:
      - 多尺度时间窗口时空熵
      - 多尺度 Voronoi 空间熵
      - token / head / channel / block 多粒度 gating
    """
    def __init__(
        self,
        cfg,                   # cfg.ast 子树
        embed_dim: int,
        num_heads: int,
        depth: int,
        H_p: int,
        W_p: int,
    ):
        super().__init__()
        ...
```

ASTPruner 中至少要做：

1. 存储 patch 坐标：`self.patch_coords: [N,2]`（buffer，不训练）
2. 初始化 Voronoi 中心（粗/细两个级别）
3. 初始化 head/channel/block gates
4. 提供接口：

   * `compute_spatiotemporal_scores(x)` → [B,T,N]
   * `forward_token_gating(x)` → (x_gated, stats)
   * head / channel / block gating函数
   * 统计 `L_AST` 和各种 sparsity

---

### 4.3 多尺度时间窗口熵（严格版）

实现函数：

```python
def compute_multi_scale_temporal_entropy(
    x: torch.Tensor,
    window_sizes: List[int],
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    输入:
      x: [B, T, N, C]
      window_sizes: 例如 [1,2,4,8]
      tau: softmax 温度
    输出:
      H_time_ms: [B, T, N] 多尺度时间熵
    """
```

规范步骤（必须满足语义等价）：

1. **通道 softmax：**

   ```python
   p = torch.softmax(x / tau, dim=-1)  # [B,T,N,C]
   ```

2. **对每个窗口大小 L 滑动平均 + 熵：**

   对于窗口 `L`：

   * 对时间轴做滑动平均（stride=1）：

     ```python
     B, T, N, C = p.shape

     # reshape -> [B*N, C, T]
     p_bt = p.permute(0,2,3,1).reshape(B*N, C, T)  # [B*N, C, T]

     # conv1d 方式实现滑动平均, kernel_size=L, stride=1
     # 可以用 groups=C 或 unfold 手写
     q = conv1d_mean_over_time(p_bt, L)  # [B*N, C, T-L+1]

     # reshape 回 [B,T',N,C]
     T_prime = T - L + 1
     q_bnct = q.reshape(B, N, C, T_prime).permute(0,3,1,2)  # [B,T',N,C]
     ```

   * 每个时间窗口中心位置的熵：

     ```python
     H_L_center = - (q_bnct * (q_bnct + eps).log()).sum(dim=-1)  # [B,T',N]
     ```

   * 将长度 `T'` 的序列插值回长度 T：

     ```python
     # [B,N,T'] -> [B*N,1,T']
     H_flat = H_L_center.permute(0,2,1).reshape(B*N,1,T_prime)
     H_interp = torch.nn.functional.interpolate(
         H_flat, size=T, mode="linear", align_corners=False
     )  # [B*N,1,T]
     H_L_full = H_interp.reshape(B,N,T).permute(0,2,1)  # [B,T,N]
     ```

3. **多尺度聚合：**

   * 对所有 L 的 `H_L_full` 求均值（或加权平均）：

     ```python
     H_time_ms = 0
     for H_L_full in H_list:
         H_time_ms = H_time_ms + H_L_full
     H_time_ms = H_time_ms / len(H_list)  # [B,T,N]
     ```

---

### 4.4 Voronoi 空间几何（两级，多尺度）

#### 4.4.1 初始化 Voronoi 中心

在 `ASTPruner.__init__` 中：

```python
coords = get_patch_coords(H_p, W_p)
self.register_buffer("patch_coords", coords)  # [N,2]

self.num_regions_coarse = cfg.ast.num_regions_coarse   # 譬如 4 或 8
self.num_regions_fine   = cfg.ast.num_regions_fine     # 譬如 8 或 16

self.centers_coarse = nn.Parameter(
    init_voronoi_centers(coords, self.num_regions_coarse, jitter=False)
)  # [R_c,2]
self.centers_fine = nn.Parameter(
    init_voronoi_centers(coords, self.num_regions_fine, jitter=True)
)  # [R_f,2]
```

实现：

```python
def init_voronoi_centers(
    coords: torch.Tensor,  # [N,2]
    num_regions: int,
    jitter: bool = False,
) -> torch.Tensor:
    """
    简化规范（允许 Codex按此写）:
      - 随机选择 num_regions 个 patch 作为初始中心:
          idx = torch.randperm(N)[:num_regions]
          centers = coords[idx].clone()
      - 如果 jitter=True:
          添加 N(0, 0.02) 扰动，然后 clamp 到 [0,1]
    """
```

> 允许以后替换为更复杂的 k-means++，但必须保证接口不变。

#### 4.4.2 Voronoi 区域分配

函数：

```python
def assign_voronoi_regions(
    coords: torch.Tensor,    # [N,2]
    centers: torch.Tensor,   # [R,2]
) -> torch.Tensor:
    """
    返回:
      region_ids: LongTensor [N], 每个值 ∈ [0,R-1]

    规则:
      对每个 patch n = 0..N-1:
        region_ids[n] = argmin_r ||coords[n] - centers[r]||_2
    """
```

该函数需要高效实现（用广播和 argmin）。

在每次需要计算时空熵前，可以重新计算 `region_ids_coarse/fine`，或者每 N 步更新一次。

---

### 4.5 多尺度空间熵（按 Voronoi 区域）

函数：

```python
def compute_multi_scale_spatial_entropy(
    x: torch.Tensor,
    region_ids_coarse: torch.Tensor,  # [N]
    region_ids_fine: torch.Tensor,    # [N]
    num_regions_coarse: int,
    num_regions_fine: int,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
      x: [B, T, N, C]
    输出:
      H_region_coarse: [B, T, num_regions_coarse]
      H_region_fine:   [B, T, num_regions_fine]
    """
```

规范：

1. 通道 softmax（可与时间熵共用）：

   ```python
   p = torch.softmax(x / tau, dim=-1)  # [B,T,N,C]
   ```

2. 按 region 聚合:

   对 coarse：

   ```python
   B, T, N, C = p.shape
   R_c = num_regions_coarse

   # p_region_coarse[b,t,r,c]
   p_region_coarse = torch.zeros(B, T, R_c, C, device=p.device)
   count_region = torch.zeros(B, T, R_c, 1, device=p.device)

   # 实现可以使用 for r in range(R_c) + mask 的方式，或者 scatter_add:
   for r in range(R_c):
       mask_n = (region_ids_coarse == r).to(p.dtype)  # [N]
       if mask_n.sum() == 0:
           continue
       mask = mask_n.view(1,1,N,1)      # [1,1,N,1]
       sel = p * mask                   # [B,T,N,C]
       p_sum = sel.sum(dim=2)           # [B,T,C]
       cnt = mask_n.sum().item()        # N_r
       p_region_coarse[:,:,r,:] = p_sum / (cnt + eps)
   ```

   细尺度类似，得到 `p_region_fine: [B,T,R_f,C]`。

3. 区域熵：

   ```python
   H_region_coarse = - (p_region_coarse * (p_region_coarse + eps).log()).sum(dim=-1)
   H_region_fine   = - (p_region_fine   * (p_region_fine   + eps).log()).sum(dim=-1)
   # 形状都是 [B,T,R_*]
   ```

---

### 4.6 综合时空多尺度熵打分

目标：得到 `score[b,t,n]`，综合：

* 多尺度时间熵：`H_time_ms: [B,T,N]`
* 多尺度空间熵映射到 token：`Hc_per_token`, `Hf_per_token`

#### 4.6.1 归一化

对 `H_time_ms`：

```python
Ht = H_time_ms              # [B,T,N]
Ht_min = Ht.view(B, -1).min(dim=-1, keepdim=True)[0].view(B,1,1)
Ht_max = Ht.view(B, -1).max(dim=-1, keepdim=True)[0].view(B,1,1)
Ht_norm = (Ht - Ht_min) / (Ht_max - Ht_min + eps)  # [B,T,N] in [0,1]
```

对 region 熵：

```python
def norm_region(H: torch.Tensor) -> torch.Tensor:
    B, T, R = H.shape
    H_min = H.view(B, -1).min(dim=-1, keepdim=True)[0].view(B,1,1)
    H_max = H.view(B, -1).max(dim=-1, keepdim=True)[0].view(B,1,1)
    return (H - H_min) / (H_max - H_min + eps)

Hc_norm = norm_region(H_region_coarse)  # [B,T,R_c]
Hf_norm = norm_region(H_region_fine)   # [B,T,R_f]
```

把区域熵映射到每个 token：

```python
Hc_per_token = torch.zeros(B, T, N, device=x.device)
Hf_per_token = torch.zeros(B, T, N, device=x.device)

for r in range(R_c):
    mask_n = (region_ids_coarse == r)  # [N]
    if mask_n.sum() == 0:
        continue
    Hc_per_token[:,:,mask_n] = Hc_norm[:,:,r:r+1]   # 广播到所有属于该 region 的 patch

for r in range(R_f):
    mask_n = (region_ids_fine == r)
    if mask_n.sum() == 0:
        continue
    Hf_per_token[:,:,mask_n] = Hf_norm[:,:,r:r+1]
```

#### 4.6.2 最终 score 公式

配置中提供：

```yaml
ast:
  alpha_time: 1.0
  beta_space_coarse: 0.5
  gamma_space_fine: 0.5
```

打分：

```python
score = (
    cfg.ast.alpha_time        * Ht_norm       # 多尺度时间熵
  + cfg.ast.beta_space_coarse * Hc_per_token  # 粗尺度 Voronoi 区域熵
  + cfg.ast.gamma_space_fine  * Hf_per_token  # 细尺度 Voronoi 区域熵
)  # [B,T,N]
```

---

### 4.7 Token Gating（soft/hard）

目标：根据 `score[b,t,n]` 做 top-ρ 保留。

#### 4.7.1 build_soft_token_mask

实现：

```python
def build_soft_token_mask(
    score: torch.Tensor,        # [B,T,N]
    rho: float,                 # 保留比例 0~1
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    输出:
      mask_soft: [B,T,N] ∈ (0,1)
    规则:
      - 对每个 b 单独计算阈值 threshold_b:
          - 将 score[b] 展平为 [T*N]
          - k = int(rho * (T*N))
          - 用 torch.kthvalue 找到第 (T*N-k) 个元素作为 threshold
      - mask_soft[b,t,n] = sigmoid((score[b,t,n] - threshold_b) / temperature)
    """
```

#### 4.7.2 ASTPruner.forward_token_gating

```python
def forward_token_gating(self, x: torch.Tensor):
    """
    x: [B,T,N,C]
    返回:
      x_gated: [B,T,N,C]
      stats: dict
    """
    score = self.compute_spatiotemporal_scores(x)  # [B,T,N]

    rho  = self.cfg.ast.rho_token_target
    temp = self.cfg.ast.token_temperature

    mask_soft = build_soft_token_mask(score, rho, temp)  # [B,T,N]
    x_gated = x * mask_soft.unsqueeze(-1)                # [B,T,N,C]

    sparsity_token = 1.0 - mask_soft.mean()

    return x_gated, {"sparsity_token": sparsity_token}
```

推理/导出结构时可实现 `build_hard_token_mask`：按同样 threshold 做 0/1 mask，用于真正剪枝。

---

### 4.8 Head / Channel / Block gating（保持原规范）

在 ASTPruner 中还要实现：

1. **Head gating**

   ```python
   # 初始化
   self.g_head = nn.Parameter(torch.zeros(depth, num_heads))
   ```

   前向：

   ```python
   # 以第 l 层 attention 输出 attn_out: [B, num_heads, L, head_dim]
   w_head = torch.sigmoid(self.g_head[l])       # [H]
   attn_out = attn_out * w_head.view(1,-1,1,1) # 广播

   # 统计
   sparsity_head = 1.0 - w_head.mean()
   ```

2. **Channel gating (MLP hidden)**

   ```python
   self.g_ch = nn.Parameter(torch.zeros(depth, mlp_hidden_dim))  # 每层一向量
   ```

   MLP 中：

   ```python
   w_ch = torch.sigmoid(self.g_ch[l])    # [D_hidden]
   x = x * w_ch.view(1,1,-1)             # [B,L,D_hidden]
   sparsity_ch = 1.0 - w_ch.mean()
   ```

3. **Block gating**

   ```python
   self.g_block = nn.Parameter(torch.zeros(depth))  # 每层一标量
   ```

   Block 残差：

   ```python
   w_block = torch.sigmoid(self.g_block[l])    # scalar
   x = x + w_block * block_fn(x)
   sparsity_block = 1.0 - w_block.mean()
   ```

4. **AST 稀疏损失 L_AST**

   在 ASTPruner 内部或 VideoViT 中组合：

   ```python
   L_AST = (
       lambda_token  * sparsity_token
     + lambda_head   * sparsity_head
     + lambda_ch     * sparsity_ch
     + lambda_block  * sparsity_block
   )
   ```

   然后在 `info_dict` 中返回 `L_AST` 和各项 sparsity。

---

## 5. 硬件代理 (HW Proxy) 模块

这一节沿用之前规范，只把核心再列一遍，方便 Codex 一起实现。

### 5.1 LayerProxyModel

`hw_proxy/layer_proxy_model.py`：

```python
class LayerProxyModel(nn.Module):
    """
    通用 MLP 回归器：
      in_dim -> hidden -> hidden -> out_dim
    用于预测:
      - latency_ms
      - peak_mem_mb
      - power_w
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, out_dim: int = 1):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,in_dim] -> [N,out_dim]
        ...
```

### 5.2 特征构建

```python
def build_layer_features(layer_cfg: Dict, device_cfg: Dict) -> np.ndarray:
    """
    输出 1D feature 向量，至少包含:
      - log10(flops)
      - log10(bytes)
      - log10(peak_flops)
      - log10(peak_bw)
      - layer_type one-hot (少数维)
      - embed_dim / 1024
      - num_heads / 16
      - mlp_ratio / 4
      - seq_len / 1024
      - precision (0/1)
    """
```

### 5.3 LayerHwProxy

`hw_proxy/layer_hw_proxy.py`：

```python
class LayerHwProxy:
    def __init__(
        self,
        device_name: str,
        gpu_yaml: str,
        weight_dir: str,
        use_latency: bool = True,
        use_mem: bool = True,
        use_power: bool = True,
    ):
        # 读 gpu_data.yaml, 加载 proxy 权重
        ...

    def predict_layer(self, layer_cfg: Dict) -> Dict[str, float]:
        """
        返回:
          {"lat_ms": float, "mem_mb": float, "power_w": float}
        """
        ...

    def predict_segment(self, segment_cfg: Dict) -> Dict[str, float]:
        """
        segment 内多层合成:
          - lat_ms: 可 sum 或 max
          - mem_mb: max
          - power_w: 时间加权平均
        """
        ...
```

---

## 6. 芯粒类型库 & 可学习芯粒配置 α

### 6.1 `gpu_data.yaml` 芯粒类型结构

示例：

```yaml
chip_types:
  - name: "big_core"
    peak_flops: 1.0e14
    peak_bw: 1.0e12
    mem_gb: 24
    area_mm2: 600
    tdp_w: 350
  - name: "small_core"
    peak_flops: 3.0e13
    peak_bw: 4.0e11
    mem_gb: 12
    area_mm2: 250
    tdp_w: 150
```

Python 对象：

```python
@dataclass
class ChipType:
    name: str
    peak_flops: float
    peak_bw: float
    mem_gb: float
    area_mm2: float
    tdp_w: float
```

### 6.2 槽位 + Gumbel-Softmax

```python
class ChipSlotConfig(nn.Module):
    """
    N_slot 个潜在槽位；每个槽位通过 Gumbel-Softmax 选择:
      - M 种芯粒类型之一
      - 或空槽位
    """
    def __init__(self, chip_types: List[ChipType], num_slots: int = 16, tau_init: float = 1.0):
        super().__init__()
        self.chip_types = chip_types
        self.num_types = len(chip_types)
        self.num_slots = num_slots
        self.logits = nn.Parameter(torch.zeros(num_slots, self.num_types + 1))  # +1 for "empty"
        self.tau = tau_init

    def forward(self, hard: bool = False) -> Dict[str, Any]:
        """
        返回:
          - alpha: [num_slots, num_types+1]
          - eff_specs: dict:
              {
                "peak_flops": [num_slots],
                "peak_bw":    [num_slots],
                "mem_gb":     [num_slots],
                "area_mm2":   [num_slots],
                "tdp_w":      [num_slots],
              }
        规范:
          1) 使用 Gumbel-Softmax:
               y = softmax((logits + gumbel_noise) / tau)
             若 hard=True，可做 straight-through。
          2) eff_specs 每槽位为各芯粒加权和，empty 那一维不贡献硬件参数。
        """
```

芯粒数量正则：

```python
chip_used_prob = 1.0 - alpha[:, -1]              # 去掉 empty 概率
L_chip_count = lambda_chip * chip_used_prob.sum()
```

---

## 7. Segment 数据结构 & 构造

`mapping/segments.py`：

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
    precision: int   # 0=FP32, 1=FP16
    traffic_in_bytes: float
    traffic_out_bytes: float
```

构造函数：

```python
def build_segments_from_model(model, cfg) -> List[Segment]:
    """
    规范:
      - 将 depth 层按顺序分割为 K 个连续 segment (如每 2 层一个，或由 cfg 指定边界)
      - 对每个 segment:
          · flops = sum(layer_flops)
          · bytes = sum(layer_bytes)
          · seq_len / embed_dim / num_heads / mlp_ratio 可取 segment 内代表层的参数
          · traffic_in/out_bytes ~ seq_len * embed_dim * bytes_per_element
      - FLOPs/Bytes 可用近似公式，只要相对大小正确。
    """
```

---

## 8. 映射模块 MappingSolver

`mapping/mapping_solver.py`：

```python
class MappingSolver:
    def __init__(self, mem_limit_factor: float = 0.9):
        self.mem_limit_factor = mem_limit_factor
```

### 8.1 成本矩阵

```python
def build_cost_matrix(
    self,
    segments: List[Segment],
    eff_specs: Dict[str, torch.Tensor],
    proxy: LayerHwProxy,
) -> Dict[str, torch.Tensor]:
    """
    返回:
      {
        "lat_ms": [K, S],
        "mem_mb": [K, S],
        "power_w": [K, S],
      }
      K: segment 数
      S: 槽位数
    规范:
      对每个 segment k 和 slot j:
        - 构造 segment_cfg (包含 flops/bytes/seq_len 等)
        - device_cfg 由 eff_specs 生成
        - 调用 proxy.predict_segment -> lat,mem,power
    """
```

### 8.2 pipeline latency 估计

```python
def estimate_pipeline_latency(
    mapping: List[int],      # len=K
    lat_ms: torch.Tensor,    # [K,S]
    mode: str = "balanced",  # "serial" 或 "balanced"
) -> float:
    """
    serial:
      total = sum(lat_ms[k, mapping[k]])
    balanced:
      对每个槽位 d:
        device_time[d] = sum(lat_ms[k,d] for k s.t. mapping[k]=d)
      total = max(device_time)
    返回 total (float)
    """
```

### 8.3 solve_mapping

```python
def solve_mapping(
    self,
    segments: List[Segment],
    eff_specs: Dict[str, torch.Tensor],
    proxy: LayerHwProxy,
    layout_positions: Optional[torch.Tensor] = None,
    strategy: str = "greedy_local",
) -> Dict[str, Any]:
    """
    返回:
      {
        "mapping": List[int],
        "per_slot_time": List[float],
        "total_latency_ms": float,
        "comm_ms": float,
      }
    """
```

`strategy="greedy_local"` 规范：

1. 使用 `build_cost_matrix` 获取 `lat_ms`, `mem_mb`

2. 初始化映射：

   ```python
   mapping[k] = k % num_slots
   ```

3. 局部搜索：

   ```python
   improved = True
   while improved:
       improved = False
       for k in range(K):
           curr = mapping[k]
           best = curr
           best_latency = current_total_latency

           for d in range(num_slots):
               if d == curr: continue
               if violates_mem_constraint_if_move(k, curr, d, mem_mb, eff_specs):
                   continue

               mapping[k] = d
               new_latency = estimate_pipeline_latency(mapping, lat_ms, mode)
               if new_latency + eps < best_latency:
                   best_latency = new_latency
                   best = d

           if best != curr:
               mapping[k] = best
               current_total_latency = best_latency
               improved = True
   ```

4. 通信时间（若 `layout_positions` 不为 None）按简单公式估计：

   ```python
   comm_ms = 0.0
   for k in range(K-1):
       d1, d2 = mapping[k], mapping[k+1]
       if d1 == d2: continue
       traffic = segments[k].traffic_out_bytes
       dist = torch.norm(layout_positions[d1] - layout_positions[d2])
       eff_bw = torch.min(eff_specs["peak_bw"][d1], eff_specs["peak_bw"][d2])  # bytes/s
       base_time = traffic / (eff_bw + 1e-9) * 1e3  # ms
       dist_penalty = dist * distance_scale_ms      # 配置项
       comm_ms += base_time + dist_penalty
   ```

---

## 9. WaferLayout 晶圆布局模块

`layout/wafer_layout.py`：

```python
class WaferLayout(nn.Module):
    def __init__(self, num_slots: int, wafer_radius_mm: float):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(num_slots, 2))  # [x,y] in mm
        self.wafer_radius_mm = wafer_radius_mm
```

### 9.1 边界约束

```python
def compute_boundary_penalty(self):
    # r_j = sqrt(x_j^2 + y_j^2)
    r = torch.norm(self.pos, dim=-1)                       # [S]
    penalty = torch.relu(r - self.wafer_radius_mm)**2
    return penalty.sum()
```

### 9.2 重叠约束（用等效圆）

```python
def compute_overlap_penalty(self, eff_specs: Dict[str, torch.Tensor]):
    area = eff_specs["area_mm2"]        # [S]
    radius = torch.sqrt(area / math.pi) # [S]

    S = self.pos.size(0)
    penalty = 0.0
    for i in range(S):
        for j in range(i+1, S):
            center_dist = torch.norm(self.pos[i] - self.pos[j])
            min_dist = radius[i] + radius[j]
            overlap = torch.relu(min_dist - center_dist)
            penalty = penalty + overlap**2
    return penalty
```

### 9.3 通信代价（空间相关）

```python
def compute_comm_loss(
    self,
    mapping: List[int],
    segments: List[Segment],
    eff_specs: Dict[str, torch.Tensor],
    distance_scale: float = 1.0,
) -> torch.Tensor:
    comm_cost = 0.0
    for k in range(len(segments)-1):
        d1, d2 = mapping[k], mapping[k+1]
        if d1 == d2:
            continue
        traffic = segments[k].traffic_out_bytes
        dist = torch.norm(self.pos[d1] - self.pos[d2])  # mm
        comm_cost = comm_cost + traffic * dist * distance_scale
    return comm_cost
```

### 9.4 简单热 penalty（径向核）

```python
def compute_thermal_penalty(
    self,
    eff_specs: Dict[str, torch.Tensor],
    T_ambient: float = 25.0,
    T_limit: float = 85.0,
    sigma_mm: float = 20.0,
    alpha: float = 0.01,
):
    """
    简化热模型:
      对每个 slot i:
        T_i = T_amb + sum_j K(||pos_i-pos_j||) * alpha * P_j
      K(r) = exp(- r^2 / (2*sigma^2))
    """
    pos = self.pos                                   # [S,2]
    P = eff_specs["tdp_w"]                           # [S]
    S = pos.size(0)

    T = torch.full((S,), T_ambient, device=pos.device)
    for i in range(S):
        for j in range(S):
            r = torch.norm(pos[i] - pos[j])
            K = torch.exp(- (r*r) / (2*(sigma_mm**2)))
            T[i] = T[i] + alpha * K * P[j]

    T_max = T.max()
    penalty = torch.relu(T_max - T_limit)**2
    return penalty, T_max
```

### 9.5 forward 组合

```python
def forward(
    self,
    mapping: List[int],
    segments: List[Segment],
    eff_specs: Dict[str, torch.Tensor],
    lambda_boundary: float,
    lambda_overlap: float,
    lambda_comm: float,
    lambda_thermal: float,
):
    L_boundary = self.compute_boundary_penalty()
    L_overlap  = self.compute_overlap_penalty(eff_specs)
    L_comm     = self.compute_comm_loss(mapping, segments, eff_specs)
    L_thermal, T_max = self.compute_thermal_penalty(eff_specs)

    loss_layout = (
        lambda_boundary * L_boundary
      + lambda_overlap  * L_overlap
      + lambda_comm     * L_comm
      + lambda_thermal  * L_thermal
    )

    stats = {
        "boundary": L_boundary.item(),
        "overlap":  L_overlap.item(),
        "comm_cost": L_comm.item(),
        "T_max":    T_max.item(),
    }
    return loss_layout, stats
```

---

## 10. 硬件损失 L_hw 组合

统一封装函数：

```python
def compute_hw_loss(
    segments: List[Segment],
    chip_slot_config: ChipSlotConfig,
    hw_proxy: LayerHwProxy,
    mapping_solver: MappingSolver,
    wafer_layout: WaferLayout,
    hw_loss_cfg: Dict[str, Any],
):
    """
    返回:
      L_hw: 标量 tensor
      hw_stats: dict，用于日志
    """
```

规范步骤：

1. 芯粒配置：

   ```python
   out = chip_slot_config(hard=False)
   alpha = out["alpha"]                       # [S,M+1]
   eff_specs = out["eff_specs"]               # 字典: "peak_flops"/"peak_bw"/"mem_gb"/"area_mm2"/"tdp_w"
   chip_used_prob = 1.0 - alpha[:, -1]
   L_chip_count = hw_loss_cfg["lambda_chip"] * chip_used_prob.sum()
   ```

2. 成本矩阵与映射：

   ```python
   cost = mapping_solver.build_cost_matrix(segments, eff_specs, hw_proxy)
   mapping_result = mapping_solver.solve_mapping(
       segments, eff_specs, hw_proxy,
       layout_positions=wafer_layout.pos,
       strategy=hw_loss_cfg.get("mapping_strategy", "greedy_local"),
   )
   total_latency_ms = mapping_result["total_latency_ms"]
   comm_ms = mapping_result["comm_ms"]
   mapping = mapping_result["mapping"]
   ```

3. 能耗 / 显存 / 面积：

   ```python
   K = len(segments)
   S = alpha.size(0)
   lat_ms = cost["lat_ms"]   # [K,S]
   mem_mb = cost["mem_mb"]   # [K,S]
   power_w = cost["power_w"] # [K,S]

   total_energy_j = 0.0
   per_slot_mem = torch.zeros(S, device=lat_ms.device)

   for k in range(K):
       d = mapping[k]
       lat_s = lat_ms[k, d] / 1e3           # 秒
       p_w   = power_w[k, d]
       total_energy_j += lat_s * p_w        # P * t

       per_slot_mem[d] = torch.max(per_slot_mem[d], mem_mb[k, d])

   peak_mem_mb = per_slot_mem.max()

   total_area_mm2 = (eff_specs["area_mm2"] * chip_used_prob).sum()
   area_limit = hw_loss_cfg["area_limit_mm2"]
   L_area = hw_loss_cfg["lambda_area"] * torch.relu(total_area_mm2 - area_limit)**2
   ```

4. 布局损失：

   ```python
   L_layout, layout_stats = wafer_layout(
       mapping,
       segments,
       eff_specs,
       lambda_boundary=hw_loss_cfg["lambda_boundary"],
       lambda_overlap=hw_loss_cfg["lambda_overlap"],
       lambda_comm=hw_loss_cfg["lambda_comm_extra"],
       lambda_thermal=hw_loss_cfg["lambda_thermal"],
   )
   ```

5. 总 L_hw：

   ```python
   lambda_T   = hw_loss_cfg["lambda_T"]
   lambda_E   = hw_loss_cfg["lambda_E"]
   lambda_mem = hw_loss_cfg["lambda_mem"]

   L_hw = (
       lambda_T   * total_latency_ms
     + lambda_E   * total_energy_j
     + lambda_mem * peak_mem_mb
     + L_area
     + L_chip_count
     + L_layout
   )
   ```

6. 统计信息：

   ```python
   hw_stats = {
       "lat_ms": float(total_latency_ms),
       "energy_j": float(total_energy_j),
       "peak_mem_mb": float(peak_mem_mb),
       "area_mm2": float(total_area_mm2),
       "chip_used_est": float(chip_used_prob.sum()),
       "comm_ms": float(comm_ms),
   }
   hw_stats.update(layout_stats)
   ```

---

## 11. 训练与交替优化框架

### 11.1 模式

配置中通过 `training.mode` 控制：

* `"proxy_only"`：只训练/评估 HW proxy
* `"single_device_ast"`：单 GPU AST2.0-v2（关闭多芯粒 & 布局）
* `"multi_chip_fixed"`：固定芯粒集合，学习映射和布局（不学习 α）
* `"version_c_full"`：完整 Version-C：α + s + m + L 联合优化

### 11.2 单设备 AST2.0-v2 训练（Phase 2）

`trainer/trainer_single_device.py`：

```python
for epoch in range(num_epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits, info = model(x, return_intermediate=True)
        L_task = F.cross_entropy(logits, y)

        L_AST = info["L_AST"]

        if use_hw_loss:
            segments = build_segments_from_model(model, cfg)  # 或提前构建
            L_hw = compute_hw_loss_single_device(segments, hw_proxy, cfg)
        else:
            L_hw = 0.0

        loss = L_task + cfg.ast.lambda_AST * L_AST + cfg.hw.lambda_hw * L_hw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

`compute_hw_loss_single_device` 是 `compute_hw_loss` 的简化：只用一个芯粒、不做 mapping 和 layout。

### 11.3 完整 Version-C 交替优化（Phase 5）

`trainer/trainer_version_c.py`：

```python
for outer_epoch in range(cfg.training.outer_epochs):

    # Step A: 训练 θ & 稀疏 gates s（AST 部分）
    for step in range(cfg.training.inner_steps_ast):
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        logits, info = model(x, return_intermediate=True)
        L_task = F.cross_entropy(logits, y)
        L_AST = info["L_AST"]

        segments = build_segments_from_model(model, cfg)

        L_hw, hw_stats = compute_hw_loss(
            segments,
            chip_slot_config,
            hw_proxy,
            mapping_solver,
            wafer_layout,
            cfg.hw,
        )

        loss = L_task + cfg.ast.lambda_AST * L_AST + cfg.hw.lambda_hw * L_hw

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

    # Step B: 更新芯粒 logits (α)
    for step in range(cfg.training.inner_steps_alpha):
        segments = build_segments_from_model(model, cfg)
        L_hw, hw_stats = compute_hw_loss(
            segments,
            chip_slot_config,
            hw_proxy,
            mapping_solver,
            wafer_layout,
            cfg.hw,
        )
        optimizer_alpha.zero_grad()
        L_hw.backward()
        optimizer_alpha.step()

    # Step C: 重新求映射 m（不参与梯度）
    segments = build_segments_from_model(model, cfg)
    out = chip_slot_config(hard=False)
    eff_specs = out["eff_specs"]
    mapping_result = mapping_solver.solve_mapping(
        segments,
        eff_specs,
        hw_proxy,
        layout_positions=wafer_layout.pos,
        strategy=cfg.mapping.strategy,
    )

    # Step D: 更新布局 L（仅优化 wafer_layout.pos）
    for step in range(cfg.training.inner_steps_layout):
        segments = build_segments_from_model(model, cfg)
        out = chip_slot_config(hard=False)
        eff_specs = out["eff_specs"]
        mapping = mapping_result["mapping"]

        L_layout, layout_stats = wafer_layout(
            mapping,
            segments,
            eff_specs,
            lambda_boundary=cfg.hw.lambda_boundary,
            lambda_overlap=cfg.hw.lambda_overlap,
            lambda_comm=cfg.hw.lambda_comm_extra,
            lambda_thermal=cfg.hw.lambda_thermal,
        )

        optimizer_layout.zero_grad()
        L_layout.backward()
        optimizer_layout.step()
        # 可加投影: 把 pos 限制在圆盘内

    # 可选 Step E: RL 调 λ_*，不强制
```

---

## 12. 配置字段规范（YAML）

示例（只列关键字段）：

```yaml
ast:
  use_ast_prune: true
  # 时空熵
  time_window_sizes: [1, 2, 4, 8]
  entropy_tau: 1.0
  rho_token_target: 0.5
  token_temperature: 0.1
  num_regions_coarse: 4
  num_regions_fine: 8
  alpha_time: 1.0
  beta_space_coarse: 0.5
  gamma_space_fine: 0.5
  # 多粒度稀疏
  lambda_token: 0.1
  lambda_head: 0.1
  lambda_ch: 0.1
  lambda_block: 0.05
  lambda_AST: 1.0   # L_AST 总权重

hw:
  mode: "version_c_full"
  device_name: "3090"
  gpu_yaml: "./configs/gpu_data.yaml"
  proxy_weight_dir: "./proxy_weights"

  lambda_hw: 1.0           # L_hw 在总loss中的权重
  lambda_T: 1e-3
  lambda_E: 1e-4
  lambda_mem: 1e-4
  lambda_area: 1e-6
  lambda_chip: 1e-3
  lambda_boundary: 1e-3
  lambda_overlap: 1e-3
  lambda_comm_extra: 1e-6
  lambda_thermal: 1e-4
  area_limit_mm2: 70000.0

  num_slots: 16
  wafer_radius_mm: 150.0
  latency_mode: "balanced"
  mapping_strategy: "greedy_local"

mapping:
  strategy: "greedy_local"
  mem_limit_factor: 0.9

training:
  mode: "version_c_full"
  outer_epochs: 20
  inner_steps_ast: 100
  inner_steps_alpha: 20
  inner_steps_layout: 20
```

---

## 13. 日志与输出要求

训练器需要定期记录：

* **任务指标**：train/val loss, acc/mAP
* **AST 稀疏指标**：

  * `sparsity_token`, `sparsity_head`, `sparsity_ch`, `sparsity_block`
* **硬件指标**（来自 `compute_hw_loss` 与 `WaferLayout`）：

  * `lat_ms`, `energy_j`, `peak_mem_mb`, `area_mm2`, `chip_used_est`
  * `boundary`, `overlap`, `comm_cost`, `T_max`
* **结构摘要**：

  * 剪枝后 ViT 参数：depth, embed_dim, num_heads, mlp_ratio
  * token/head/channel 的平均保留率
  * 芯粒使用情况：各 ChipType 的期望数量

建议写入 JSON / YAML，方便后续作图和论文整理。

---

## 14. 给代码助手的使用提示（可以直接贴）

在 VSCode 里让 Codex 读到这个文件后，可以这样提示：

> “请阅读 `SPEC_version_c_v2.md`。
> 按规范：
>
> 1. 在 `models/ast2_pruner.py` 中实现 `ASTPruner` 全部逻辑，尤其是第 4 章的多尺度时间窗口熵、多尺度 Voronoi 空间熵、token gating 和多粒度稀疏；
> 2. 在 `hw_proxy/`, `mapping/`, `layout/`, `trainer/` 中按照对应章节实现类和函数，保持输入输出形状与伪代码一致；
> 3. 不要更改函数签名和返回值约定，不要删减算法步骤；
> 4. 所有超参数（窗口大小、区域数、权重 λ 等）必须从配置读取，而非硬编码。”

---

你把这份 `SPEC_version_c_v2.md` 丢给 Codex，它就算看不到论文，也能按这份规范把**完整版时空熵 + Voronoi + 硬件联合优化**的代码一步步补全。
