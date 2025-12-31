# SPEC_version_c_full_latest — Version-C 实现 & 实验说明书

（多尺度滑窗 + 视频/音频跨模态堆积 + AST2.0-lite + Version-C）

> **原则：**
>
> 1. 以本 SPEC 为唯一“真相来源”。旧代码、旧 SPEC 与本文件冲突时，一律以本文件为准。
> 2. 优先保证下面列出的“主线入口脚本 + 配置 + 模块”完备、可运行。
> 3. 旧版 Dataset/Trainer 可以保留为 legacy，但**主线脚本不得再依赖它们**。

---

## 0. 环境 & 路径假设

项目根目录（当前仓库）：

```bash
/root/autodl-tmp/proj_ast2_ucf101_full
cd /root/autodl-tmp/proj_ast2_ucf101_full
```

### 0.1 UCF101 抽帧 & splits（用于训练 / 验证）

放在**当前仓库**下：

```text
project_root/
  data/
    ucf101/
      frames/
        ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg
        ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00002.jpg
        ...
      splits/
        trainlist01.txt
        testlist01.txt
      audio/
        v_ApplyEyeMakeup_g01_c01.npy    # 由 preprocess_ucf101_audio.py 生成
        ...
```

> **注意：Dataset 只访问这一棵目录：`data/ucf101/{frames,splits,audio}`。**

### 0.2 原始 UCF101 `.avi` 视频（仅音频预处理用）

在另一个大仓库：

```text
/root/autodl-tmp/proj_ast2_wafer_v2/
  data/
    UCF-101/
      ApplyLipstick/v_ApplyLipstick_g01_c01.avi
      ...
```

> 只有 `scripts/preprocess_ucf101_audio.py` 允许访问这里，用于提取音频再存回 `proj_ast2_ucf101_full/data/ucf101/audio/`。
> 任何 **Dataset / Trainer / 训练脚本都不能依赖这个路径**。

所有命令统一使用：

```bash
python -m scripts.xxx --cfg path/to/config.yaml
```

---

## 1. 工程目录结构（主线）

规范中的**主线文件**：

```text
project_root/
  configs/
    # 单模态视频
    ast2_ucf101_dense.yaml
    ast2_ucf101_dense_noscale.yaml
    ast2_ucf101_ast_only.yaml
    ast2_ucf101_ast_hw.yaml

    # 视频+音频
    ast2_ucf101_av_dense.yaml
    ast2_ucf101_av_ast_only.yaml
    ast2_ucf101_av_ast_hw.yaml

    # Version-C / 多芯粒
    vc_phase2_fixed4_big.yaml
    vc_phase3_full_ucf101.yaml
    vc_phase3_nolayout_ucf101.yaml

    # 烟雾测试
    smoke_ast_ucf101.yaml
    smoke_version_c_ucf101.yaml

    # Proxy & 芯粒库
    proxy_ms_mem.yaml
    proxy_power.yaml
    gpu_data.yaml

  scripts/
    run_proxy_ms_mem.py
    run_proxy_power.py
    run_ast2_ucf101.py
    run_version_c.py
    preprocess_ucf101_audio.py
    gen_experiment_cmds.py

  models/
    video_vit.py             # VideoViT & VideoAudioAST
    ast2_pruner.py           # AST2.0-lite + 时空熵 v2 + Voronoi

  hw_proxy/
    layer_proxy_model.py
    layer_hw_proxy.py

  mapping/
    segments.py              # Segment + ChipSlotConfig
    mapping_solver.py        # MappingSolver + 细粒度拆分 & 通道重排

  layout/
    wafer_layout.py          # WaferLayout

  trainer/
    trainer_single_device.py
    trainer_version_c.py

  utils/
    config.py
    logging.py
    metrics.py
    flops_estimator.py
    data_ucf101.py           # 唯一主线 UCF101 数据集实现
```

### 1.1 Legacy 文件（可以保留但不能再被主线使用）

以下文件视为 **legacy**：

```text
datasets/ucf101_dataset.py
trainers/version_c_full_trainer.py
trainers/version_c_phase3_trainer.py
scripts/run_version_c_full.py
# 其它旧 trainer/dataset 脚本如存在
```

要求：

* 这些文件可加大注释 `# LEGACY: not used by main pipeline`；
* 主线入口 `scripts/run_ast2_ucf101.py` 和 `scripts/run_version_c.py` **只能**依赖 `trainer/*.py + utils/data_ucf101.py` 这条新路径。

---

## 2. 配置：data & training 字段规范

### 2.1 data 字段（UCF101）

所有使用 UCF101Dataset 的 config 必须包含：

```yaml
data:
  dataset: ucf101                 # 显式声明
  frames_root: data/ucf101/frames
  splits_root: data/ucf101/splits
  audio_root: data/ucf101/audio   # 可选，但推荐统一写上
  use_audio: false | true         # 决定使用 VideoViT 还是 VideoAudioAST
  clip_lens: [8, 16]              # 覆盖窗口长度（多尺度）
  num_frames: 8                   # 模型输入固定帧数 T
  train_stride_ratio: 0.5
  eval_stride_ratio: 0.5
  clip_jitter: true               # train 时是否随机偏移起点
  # 其余如 batch_size, num_workers 等保持原样
```

允许兼容旧字段：

* 若 `clip_lens` 缺失但有 `clip_len: 16`，则自动转为：

  ```python
  clip_lens = [clip_len]
  num_frames 默认等于 clip_len
  ```

### 2.2 training / hw 字段（核心）

```yaml
training:
  mode: "dense_baseline"     # or "single_device_ast" / "multi_chip_fixed" / "version_c_full"
  use_audio: false           # 与 data.use_audio 一致；true 则用 VideoAudioAST
  num_epochs:  ...
  # 其它优化器、lr 等

hw:
  mode: "none" | "single_device" | "multi_chip_fixed" | "version_c_full"
  use_hw_loss: true/false
  # lambda_T, lambda_E, lambda_mem, lambda_chip, lambda_area 等
```

---

## 3. 数据管线：UCF101Dataset（多尺度滑窗 + 固定 num_frames）

文件：`utils/data_ucf101.py`
类：`class UCF101Dataset(Dataset)`

### 3.1 路径解析

```python
project_root = Path(__file__).resolve().parents[2]

frames_root = cfg.data.frames_root or "data/ucf101/frames"
splits_root = cfg.data.splits_root or "data/ucf101/splits"
audio_root  = cfg.data.audio_root or None

if not Path(frames_root).is_absolute():
    frames_root = project_root / frames_root
if not Path(splits_root).is_absolute():
    splits_root = project_root / splits_root
if audio_root is not None and not Path(audio_root).is_absolute():
    audio_root = project_root / audio_root
```

* `split="train"` → 读 `trainlist01.txt`
* `split="val"` 或 `"test"` → 读 `testlist01.txt`
* 若文件不存在，抛 `FileNotFoundError`，并打印 `splits_root`。

### 3.2 clip_lens + num_frames + 滑动窗口

在 `__init__` 中：

1. 解析：

   ```python
   self.clip_lens = cfg.data.clip_lens      # list[int]
   self.num_frames = cfg.data.num_frames    # int
   self.train_stride_ratio = cfg.data.train_stride_ratio
   self.eval_stride_ratio = cfg.data.eval_stride_ratio
   self.is_train = (split == "train")
   self.use_audio = cfg.data.use_audio
   ```

2. 对每个视频 `vid`，长度 `L_frames`，对每个 `cover_len` in `clip_lens`：

   * 训练集：

     ```python
     stride = max(1, int(cover_len * train_stride_ratio))
     # 第一次随机偏移
     offset = random.randint(0, max(0, stride-1)) if clip_jitter else 0
     starts = list(range(offset, max(L_frames - cover_len + 1, 1), stride))
     ```

   * 验证/测试集：

     ```python
     stride = max(1, int(cover_len * eval_stride_ratio))
     starts = list(range(0, max(L_frames - cover_len + 1, 1), stride))
     ```

   * 每个 `(vid, cover_len, start)` 形成一个 sample entry：

     ```python
     {
       "video_id": vid,
       "cover_len": cover_len,
       "start": start,
       "label": class_index,
     }
     ```

`__len__` 返回 sample 列表长度。

### 3.3 **getitem**：从覆盖窗口采样 num_frames

给定一个 sample：

* 窗口区间：`[start, start + cover_len)`，并裁剪到 `[0, L_frames)`；
* 窗口内实际帧数记为 `Lw`；

构造 `num_frames` 个索引：

```python
if Lw >= num_frames:
    # 在窗口内均匀采样 num_frames 个帧
    idx = np.linspace(0, Lw - 1, num_frames)
    idx = np.floor(idx).astype(int)
else:
    # 不足时：先用所有帧，再在两端重复补齐到 num_frames
    base = np.arange(Lw)
    pad_needed = num_frames - Lw
    pad = np.clip(base[-1:], 0, Lw-1).repeat(pad_needed)
    idx = np.concatenate([base, pad], axis=0)
assert len(idx) == num_frames
```

然后：

* 加上窗口起点偏移：`frame_ids = start + idx`；
* 再 clamp 到 `[0, L_frames-1]`；

加载图像为 `[3, H, W]`，stack 后得：

```python
video = torch.stack(frames_list, dim=0)   # [T=num_frames, 3, H, W]
```

如 `use_audio=True`，调用 `_load_audio_clip(video_id, frame_ids)` 得到：

```python
audio = torch.from_numpy(A[frame_ids])    # [T=num_frames, D_audio]
```

最终返回：

```python
sample = {
  "video": video,         # float32, [T,3,H,W]
  "label": class_idx,     # int64
  "video_id": video_id,
}
if self.use_audio and self.audio_root is not None:
  sample["audio"] = audio  # float32, [T,D_audio]
```

> **保证：所有样本的 video/audio 第一维都是 `num_frames`，因此默认 collate_fn 可以直接 `torch.stack`。**

---

## 4. 音频预处理：preprocess_ucf101_audio.py

文件：`scripts/preprocess_ucf101_audio.py`

### 4.1 配置字段

```yaml
data:
  raw_video_root: /root/autodl-tmp/proj_ast2_wafer_v2/data/UCF-101
  audio_root: data/ucf101/audio
  sample_rate: 16000
  n_mels: 64
```

* `raw_video_root`：**只在这个脚本用**；Dataset 不读它。
* `audio_root` 若非绝对路径，则相对 `proj_root` 解析并创建。

### 4.2 算法步骤

对每个 `.avi`：

1. 使用 `librosa` or `ffmpeg` 读 waveform（优先 librosa）：

   ```python
   y, sr = librosa.load(video_path, sr=sample_rate)
   ```

2. 提取 log-mel 频谱：

   ```python
   S = librosa.feature.melspectrogram(
         y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
   S_db = librosa.power_to_db(S, ref=np.max)        # [n_mels, T_spec]
   ```

3. 获取该视频的总帧数 `L_frames`：

   * 可以通过事先的 metadata 或用 OpenCV 读帧计数。
   * 若无法获取，则 fallback 为：

     ```python
     L_frames = T_spec   # 简化：一帧一个时间步
     ```

4. 将 time 轴均分为 `L_frames` 段：

   ```python
   # t in [0, T_spec)
   boundaries = np.linspace(0, T_spec, L_frames+1)
   for k in range(L_frames):
       t0, t1 = int(boundaries[k]), int(boundaries[k+1])
       if t1 <= t0: t1 = t0 + 1
       seg = S_db[:, t0:t1]
       a_k = seg.mean(axis=1)   # [n_mels]
   A = np.stack(a_list, axis=0)  # [L_frames, n_mels]
   ```

5. 保存：

   ```python
   np.save(audio_root / f"{video_id}.npy", A.astype("float32"))
   ```

6. 若读取或解码失败，则打印 warning，并保存同形状的全 0 特征。

---

## 5. 模型：VideoViT & VideoAudioAST

文件：`models/video_vit.py`

### 5.1 VideoViT（单模态 Video）

接口：

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
        ...

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        x: [B, T, C, H, W] 或 [B, C, T, H, W]
        """
```

**约束：**

1. 预处理把输入统一为 `[B, T, C, H, W]`。
2. 强制断言 `T == self.num_frames`，否则报错。

Patch embedding：

* 把每帧 `[C, H, W]` 变成 `N_v` 个 patch token，`N_v = (img_size/patch_size)^2`；
* 输出 `v_tokens: [B, T, N_v, C]`。

Transformer：

* 将 `[B, T, N_v, C]` reshape 为 `[B, T*N_v, C]` 送入标准 ViT block；
* 在某个指定层（或所有层）里调用 `ASTPruner` 的 token/head/ch/block gating：

  * 将 AST 产生的 `mask_token` 应用到对应层的 token 上。

分类头：

* 通常用 `[CLS]` 或所有 token 的 mean pool 输出 `cls_feat`；
* 分类器：`logits = Linear(cls_feat) → [B, num_classes]`。

`return_intermediate=True` 时，返回：

```python
return logits, {
  "token_feat": token_feat_ref,   # [B,T,N_v,C], 某层或 pooled token
  "ast_stats": {
    "sparsity_token": ...,
    "sparsity_head": ...,
    "sparsity_ch": ...,
    "sparsity_block": ...
  },
  "L_AST": L_AST,                 # 若 use_ast_prune=True
  "flops_bytes_per_layer": [...], # 可选：给 segments 使用
}
```

### 5.2 VideoAudioAST（视频 + 音频跨模态）

接口：

```python
class VideoAudioAST(nn.Module):
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
        audio_feat_dim: int,
        ...
        use_ast_prune: bool = True,
        ast_cfg: Optional[Dict[str, Any]] = None,
    ):
        ...

    def forward(
        self,
        x_video: torch.Tensor,   # [B,T,3,H,W]
        x_audio: torch.Tensor,   # [B,T,D_audio]
        return_intermediate: bool = False
    ):
        ...
```

实现细节：

1. 视频部分：与 `VideoViT` 相同，得到 `v_tokens: [B,T,N_v,C]`。

2. 音频部分：

   ```python
   a_tokens = self.audio_proj(x_audio)   # [B,T,D_audio] → [B,T,C]
   a_tokens = a_tokens.unsqueeze(2)      # [B,T,1,C]   （每帧一个 audio token）
   ```

3. 跨模态堆积：

   ```python
   tokens_all = torch.cat([v_tokens, a_tokens], dim=2)   # [B,T,N_v+1,C]
   ```

4. 构造 `modality_slices`：

   ```python
   num_patches_video = N_v
   num_patches_audio = 1
   modality_slices = {
     "video": (0, N_v),
     "audio": (N_v, N_v+1),
   }
   ```

5. ASTPruner 初始化：

   ```python
   if use_ast_prune:
       ast_cfg = ast_cfg or {}
       ast_cfg.update({
         "patch_grid_h": img_size // patch_size,
         "patch_grid_w": img_size // patch_size,
         "num_modalities": 2,
         "modalities": ["video", "audio"],
         "num_patches_video": N_v,
         "num_patches_audio": 1,
       })
       self.ast_pruner = ASTPruner(
           cfg=ast_cfg,
           embed_dim=embed_dim,
           num_heads=num_heads,
           depth=depth,
           num_patches=N_v+1,
       )
   ```

6. Forward 中，在某层调用：

   ```python
   mask_token, sparsity_token, modal_stats = self.ast_pruner.forward_token_gating(
       token_feat_layer, modality_slices=modality_slices
   )
   # mask_token: [B,T,N_total,1], 与 tokens_all 对齐
   tokens_all = tokens_all * mask_token   # 或乘在 attention output 上，具体位置保持与当前实现一致
   ```

7. 分类头 & info_dict 与 VideoViT 一致，只是多了：

   ```python
   info_dict["modality_slices"] = modality_slices
   info_dict["modal_stats"] = modal_stats
   ```

---

## 6. AST2.0-lite：时空熵 + Voronoi + 多模态 token gating

文件：`models/ast2_pruner.py`
类：`ASTPruner`

### 6.1 初始化

```python
class ASTPruner(nn.Module):
    def __init__(self, cfg, embed_dim: int, num_heads: int, depth: int, num_patches: int):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_patches = num_patches

        # 多级时间/空间窗口
        self.time_window_levels = cfg.get("time_window_levels", [1, 2, 4])
        self.space_window_levels = cfg.get("space_window_levels", [1, 2, 4])
        self.time_window_overlap = cfg.get("time_window_overlap", 0.5)
        self.entropy_tau = cfg.get("entropy_tau", 1.0)
        self.entropy_eps = 1e-6

        # patch 网格 + Voronoi 区域
        H_p = cfg.get("patch_grid_h", 14)
        W_p = cfg.get("patch_grid_w", 14)
        num_regions = cfg.get("num_regions", 8)
        self.region_ids, self.num_regions = build_voronoi_regions(
            num_patches, (H_p, W_p), num_regions
        )

        # 多模态信息
        self.num_modalities = cfg.get("num_modalities", 1)
        self.modalities = cfg.get("modalities", ["video"])
        self.modality_logit = nn.Parameter(torch.zeros(self.num_modalities))

        # gating 参数
        self._init_head_gates()
        self._init_channel_gates()
        self._init_block_gates()
```

辅助函数 `_softmax_over_channels`、`_entropy_from_prob` 按之前规格。

### 6.2 时间熵：multi-level + overlap

```python
def compute_multi_level_time_entropy(
    x: torch.Tensor,      # [B,T,N,C]
    levels: List[int],
    tau: float,
    eps: float,
    overlap: float,
) -> Dict[str, torch.Tensor]:
    p = _softmax_over_channels(x, tau)   # [B,T,N,C]
    B,T,N,C = p.shape
    H_time_level = {}
    for L in levels:
        window_size = math.ceil(T / L)
        if overlap <= 0.0:
            stride = window_size
        else:
            stride = max(1, int(window_size * (1 - overlap)))
        # 生成窗口起点
        starts = list(range(0, T, stride))
        windows = []
        for s in starts:
            e = min(s + window_size, T)
            if e <= s: continue
            # [B,e-s,N,C] -> mean over t
            pw = p[:, s:e]                           # [B,tw,N,C]
            pw_mean = pw.mean(dim=1)                 # [B,N,C]
            Hw = _entropy_from_prob(pw_mean, dim=-1, eps=eps)  # [B,N]
            windows.append(Hw)
        if not windows:
            H_level = torch.zeros(B,N, device=x.device)
        else:
            H_stack = torch.stack(windows, dim=-1)   # [B,N,num_win]
            H_level = H_stack.mean(dim=-1)           # [B,N]
        H_time_level[L] = H_level
    H_global = H_time_level[levels[0]]   # L=1 对应全局
    return {"H_time_global": H_global, "H_time_level": H_time_level}
```

### 6.3 空间熵：多级 patch 网格

按之前规格实现，输出：

```python
{
  "H_space_global": [B,T],
  "H_space_level": {L: [B,T]}
}
```

### 6.4 Voronoi 区域

`build_voronoi_regions(num_patches, (H_p,W_p), num_regions)` 按 grid 上均匀取 seed，分配最近区域，返回 `region_ids: [N]`。

### 6.5 单模态 token gating（score 计算）

1. 归一化并融合时间熵：

   * 对每个 L：`H = H_time_level[L]`；对 batch 维做 min-max 归一：得到 `[B,N]`；
   * 所有 L 的归一结果平均：`H_time_fused[b,n] ∈ [0,1]`。

2. 空间熵同理，得到 `H_space_fused[b,t] ∈ [0,1]`。

3. 区域重要性：

   ```python
   region_importance[b,r] = H_space_fused[b].mean(dim=0)  # [B,R]
   ```

4. token 打分：

   ```python
   score[b,t,n] =
        a_time   * H_time_fused[b,n]
      + b_space  * H_space_fused[b,t]
      + c_region * region_importance[b, region_ids[n]]
   ```

5. 对每个样本 b：

   * flatten 为 `[T*N]`；
   * 根据 `rho_target` 计算 keep 个数 `k = int(rho_target * T*N)`；
   * `threshold[b] = kth value`（第 k 大 score）。

6. mask：

   ```python
   mask_soft = sigmoid((score - threshold[b]) / temperature)
   mask_token = mask_soft.unsqueeze(-1)      # [B,T,N,1]
   sparsity_token = 1.0 - mask_soft.mean()
   ```

### 6.6 多模态 token gating（视频 + 音频）

`forward_token_gating(token_feat, modality_slices)`：

1. 拆模态：

   ```python
   tokens_by_modal = {...}
   ```

2. 视频模态：用完整时空熵（6.2 + 6.3）。
   音频模态（N_a=1）：只用时间熵，空间熵退化为：

   ```python
   H_space_audio_global[b,t] = H_time_audio["H_time_global"][b,0]
   ```

3. 构造全局 `H_time_token[b,t,n]`、`H_space_token[b,t,n]`：

   * 对属于视频的 n：用视频对应值；
   * 对属于音频的 n：用 audio 对应值。

4. 模态权重：

   ```python
   w_modal = torch.sigmoid(self.modality_logit)    # [M]
   modal_id_for_token[n] = m_idx
   ```

5. 最终打分：

   ```python
   score[b,t,n] =
        a_time   * H_time_token[b,t,n]
      + b_space  * H_space_token[b,t,n]
      + c_region * region_importance[b, region_ids[n_video_or_dummy]]
      + d_modal  * w_modal[modal_id_for_token[n]]
   ```

   * `d_modal` 从 cfg 中读取或设为 1。

6. 其余步骤（threshold → sigmoid → mask_token → sparsity_token）与单模态相同。
   额外统计：

   ```python
   modal_stats[m] = {
     "sparsity": 1.0 - mask_token[:,:,s:e,:].mean(),
     "H_time_mean": H_time_token[:,:,s:e].mean(),
   }
   ```

### 6.7 Head / Channel / Block gating & L_AST

保持之前形式：head_logit, ch_logit, block_logit + sigmoid；
`compute_L_AST` 按 λ 加权四个 sparsity。

---

## 7. 硬件代理 (HW Proxy)

文件：`hw_proxy/layer_proxy_model.py`, `hw_proxy/layer_hw_proxy.py`

### 7.1 LayerProxyModel

简单 MLP，in_dim → hidden → ... → out_dim；用于拟合 layer latency/mem/power。

### 7.2 build_layer_features

输入：

```python
layer_cfg: {
  "flops": ...,
  "bytes": ...,
  "layer_type": ...,
  "embed_dim": ...,
  "num_heads": ...,
  "mlp_ratio": ...,
  "seq_len": ...,
  "precision": 0 or 1,
}
device_cfg: 从 gpu_data.yaml 中读的该芯粒的峰值 flops/bw 等
```

输出顺序固定的 1D 向量：

```text
[ log10(flops+1),
  log10(bytes+1),
  log10(peak_flops+1),
  log10(peak_bw+1),
  one_hot(layer_type),
  embed_dim/1024,
  num_heads/16,
  mlp_ratio/4,
  seq_len/1024,
  precision ]
```

### 7.3 LayerHwProxy

* `predict_layer(layer_cfg)` → `{lat_ms, mem_mb, power_w}`
* `predict_segment(segment_cfg)`：多层合并：

  * `lat_ms`: sum 或 max（配置决定）
  * `mem_mb`: max
  * `power_w`: `sum(power_i * lat_i)/sum(lat_i)`。

---

## 8. 芯粒库 & ChipSlotConfig

文件：`mapping/segments.py`（或单独）。

### 8.1 gpu_data.yaml

每种芯粒：

```yaml
chip_types:
  - name: big_core
    peak_flops: 1.0e14
    peak_bw: 1.0e12
    mem_gb: 24
    area_mm2: 600
    width_mm: 25
    height_mm: 24
    tdp_w: 350
  - name: small_core
    ...
```

对应 dataclass：

```python
@dataclass
class ChipType:
    name: str
    peak_flops: float
    peak_bw: float
    mem_gb: float
    area_mm2: float
    width_mm: float
    height_mm: float
    tdp_w: float
```

### 8.2 ChipSlotConfig（可学习芯粒布置）

```python
class ChipSlotConfig(nn.Module):
    def __init__(self, chip_types: List[ChipType], num_slots: int = 16, tau_init: float = 1.0):
        self.logits = nn.Parameter(torch.zeros(num_slots, num_types+1))  # +1: empty
        self.tau = tau_init

    def forward(self, hard=False):
        # Gumbel-Softmax
        if self.training:
            y = F.gumbel_softmax(self.logits, tau=self.tau, hard=hard, dim=-1)
        else:
            y = F.softmax(self.logits, dim=-1)
        alpha = y                           # [S, T+1]
        eff_specs = {   # 各硬件参数对 chip_types 做加权和
          "peak_flops": ...,
          ...
        }
        return {"alpha": alpha, "eff_specs": eff_specs}
```

芯粒数量正则：

```python
chip_used_prob = 1.0 - alpha[:, -1]
L_chip_count = lambda_chip * chip_used_prob.sum()
```

---

## 9. Segment & 计算图重构（可选细粒度拆分 + 通道重排）

文件：`mapping/segments.py`, `mapping/mapping_solver.py`

### 9.1 Segment 结构

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
    can_split_fine: bool = False
    fine_groups: Optional[int] = None
```

`build_segments_from_model(model, cfg)`：

* 每 `segment_size` 层归成一个 segment；
* 用 `flops_estimator.py` 按层统计 flops & bytes，再求和；
* traffic_in/out 估算为 `seq_len * embed_dim * bytes_per_elem`。

### 9.2 标记可细粒度拆分的 Segment

```python
def mark_fine_splittable_segments(segments, cfg):
    for seg in segments:
        if seg.attn_flops_ratio > cfg["fine_split_attn_ratio_threshold"] \n           and seg.traffic_out_bytes > cfg["fine_split_traffic_threshold"]:
            seg.can_split_fine = True
            seg.fine_groups = cfg.get("fine_groups", 2)
```

（attn_flops_ratio 可由 flops_estimator 提供粗略估计；没有就先留 TODO。）

### 9.3 通道重排 & try_fine_split_segment

`try_fine_split_segment(segment, eff_specs, proxy, layout_positions, G, cfg)`：

* 构造 G 个子 segment，每个 flops/bytes/traffic 近似 1/G；

* 估算：

  ```python
  cost_coarse = compute_segment_cost(segment, mapping_coarse, proxy, eff_specs, layout)
  cost_fine   = compute_segment_cost(sub_segments, best_mapping_fine, proxy, eff_specs, layout)
  ```

* 若 `cost_fine + margin < cost_coarse`，则：

  * `use_fine=True`，返回子 segments 列表和 `rewire_meta`；

  * rewire_meta 包含 channel permutation：

    ```python
    perm = torch.randperm(C)
    inv_perm = torch.argsort(perm)
    ```

  * 运行时在 segment 出口插入 `ChannelPermute(perm)`，在下游 segment 入口插入 `ChannelPermute(inv_perm)`。

---

## 10. 映射模块：MappingSolver

文件：`mapping/mapping_solver.py`

### 10.1 build_cost_matrix

```python
def build_cost_matrix(self, segments, eff_specs, proxy):
    K = len(segments); S = num_slots
    cost_lat = torch.zeros(K,S)
    cost_mem = torch.zeros(K,S)
    cost_pow = torch.zeros(K,S)
    for k, seg in enumerate(segments):
        for j in range(S):
            seg_cfg = {..., "device": eff_specs_j}
            pred = proxy.predict_segment(seg_cfg)
            cost_lat[k,j] = pred["lat_ms"]
            cost_mem[k,j] = pred["mem_mb"]
            cost_pow[k,j] = pred["power_w"]
    return {"lat_ms": cost_lat, "mem_mb": cost_mem, "power_w": cost_pow}
```

### 10.2 estimate_pipeline_latency

两种模式：

* `"serial"`：按 segment 顺序累加 `lat_ms[k, mapping[k]]`；
* `"balanced"`：按设备聚合任务时间取 max。

### 10.3 solve_mapping（含可选细粒度拆分）

流程简述：

1. 调用 `mark_fine_splittable_segments` 标注 segments。
2. 初始不拆分，构建 cost_matrix，得到 baseline mapping（比如 greedy 把重 segment 分到空闲槽）。
3. 若 `strategy="greedy_local"`，进行局部搜索：尝试把单个 segment 从当前槽挪到其它槽，若总 cost 降低则接受。
4. 对 `can_split_fine=True` 的 segment，调用 `try_fine_split_segment`，若 `use_fine=True`，用子 segments 替换并更新 mapping。
5. 返回：

   ```python
   {
     "mapping": List[int],
     "per_slot_time": List[float],
     "total_latency_ms": float,
     "comm_ms": float,
     "segments": segments_new,
     "rewire_meta": rewire_meta,
   }
   ```

---

## 11. 晶圆布局：WaferLayout

文件：`layout/wafer_layout.py`

* learnable `self.pos: [num_slots, 2]`，代表芯粒中心坐标（mm）；
* 约束：

  * 边界 penalty：距离圆周超出部分平方；
  * 重叠 penalty：基于芯粒宽高近似半径 `r_i`，若两圆距 < r_i + r_j，则惩罚；
  * 通信 penalty：通信量 × 物理距离；
  * 热度 penalty：相邻高 TDP 芯粒距离太近时惩罚。

`__call__(mapping, segments, eff_specs, ...)` 返回 `L_layout` 和统计信息。
在大多数实验中 `layout.optimize_layout=false`，则不反向优化，只用规则格点放置，`L_layout=0`。

---

## 12. 硬件损失 L_hw 组合

文件：`hw_proxy/hw_loss.py` 或 trainer 内部函数。

```python
def compute_hw_loss(
    segments,
    chip_slot_config,
    hw_proxy,
    mapping_solver,
    wafer_layout,
    hw_loss_cfg,
):
    out = chip_slot_config(hard=False)
    alpha, eff_specs = out["alpha"], out["eff_specs"]

    # 芯粒数量正则
    chip_used_prob = 1.0 - alpha[:, -1]
    L_chip = hw_loss_cfg.lambda_chip * chip_used_prob.sum()

    # 映射与 cost
    mapping_result = mapping_solver.solve_mapping(
        segments, eff_specs, hw_proxy,
        layout_positions=wafer_layout.pos.detach(),
        strategy=hw_loss_cfg.mapping_strategy,
        cfg=hw_loss_cfg.mapping_cfg,
    )
    total_latency_ms = mapping_result["total_latency_ms"]
    # 能耗、峰值显存、面积
    total_energy_j = ...
    peak_mem_mb = ...
    total_area_mm2 = ...

    # 面积正则
    L_area = max(0, total_area_mm2 - hw_loss_cfg.area_budget_mm2) * hw_loss_cfg.lambda_area

    # 布局损失
    if hw_loss_cfg.optimize_layout:
        L_layout, layout_stats = wafer_layout(...)
    else:
        L_layout = torch.tensor(0.0, device=...)

    L_hw = (
        hw_loss_cfg.lambda_T   * total_latency_ms +
        hw_loss_cfg.lambda_E   * total_energy_j   +
        hw_loss_cfg.lambda_mem * peak_mem_mb      +
        L_chip + L_area + L_layout
    )

    stats = {...}
    return L_hw, stats
```

---

## 13. 训练与交替优化

文件：`trainer/trainer_single_device.py`, `trainer/trainer_version_c.py`

### 13.1 单卡训练：train_single_device

* 根据 `cfg.data.use_audio` 选择 `VideoViT` 或 `VideoAudioAST`；
* Dataloader 使用 `UCF101Dataset` 和 `batch_size` 等；
* 每 batch：

  ```python
  if not cfg.data.use_audio:
      x, y = batch["video"], batch["label"]
      logits, info = model(x.to(device), return_intermediate=True)
  else:
      x_v, x_a, y = batch["video"], batch["audio"], batch["label"]
      logits, info = model(x_v.to(device), x_a.to(device), return_intermediate=True)

  L_task = cross_entropy(logits, y.to(device))
  L_AST  = info.get("L_AST", torch.tensor(0.0, device=device))

  if cfg.hw.mode == "single_device" and cfg.hw.use_hw_loss:
      segments = build_segments_from_model(model, cfg)
      L_hw, _ = compute_hw_loss_single_device(segments, hw_proxy, cfg)
  else:
      L_hw = torch.tensor(0.0, device=device)

  loss = L_task + cfg.loss.lambda_AST * L_AST + cfg.loss.lambda_hw * L_hw
  loss.backward(); optimizer.step()
  ```

### 13.2 Version-C 交替优化：trainer_version_c

外循环：

1. **Step A：更新模型参数 θ & 稀疏结构 s**（AST gating）

   * 多个 inner steps：同上，但 `compute_hw_loss` 使用多芯粒模式（ChipSlotConfig + MappingSolver + WaferLayout）。

2. **Step B：更新芯粒 logits α**

   * 冻结模型参数，优化 `chip_slot_config.logits`，目标是直接最小化 `L_hw`。

3. **Step C：更新离散 mapping（不求梯度）**

   * 调用 `mapping_solver.solve_mapping` 得到新的 `mapping_result`，缓存供 Step A/B 中 `compute_hw_loss` 使用。

4. **Step D：可选布局优化**

   * 若 `layout.optimize_layout=true`，则多步梯度下降优化 `wafer_layout.pos`，损失为 `L_layout`。

---

## 14. 实验设计 & 命令（全部用 `--cfg`）

与之前版本一致，这里只列 ID + 命令：

### Phase 0 — Proxy

* `EXP-P0-1-ms-mem`: `python -m scripts.run_proxy_ms_mem --cfg configs/proxy_ms_mem.yaml`
* `EXP-P0-2-power`: `python -m scripts.run_proxy_power --cfg configs/proxy_power.yaml`

### Phase 1-V — 单模态视频

* `EXP-P1V-1-dense-multiscale`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_dense.yaml`
* `EXP-P1V-2-dense-noscale`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_dense_noscale.yaml`
* `EXP-P1V-3-ast-no-hw`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_ast_only.yaml`
* `EXP-P1V-4-ast-hw`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_ast_hw.yaml`

### Phase 1-AV — 视频 + 音频

* `EXP-P1AV-1-av-dense-late-fusion`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_av_dense.yaml`
* `EXP-P1AV-2-av-ast-no-hw`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_av_ast_only.yaml`
* `EXP-P1AV-3-av-ast-hw`: `python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_av_ast_hw.yaml`

### Phase 2 — Fixed 多芯粒

* `EXP-P2-1-fixed4-grid`: `python -m scripts.run_version_c --cfg configs/vc_phase2_fixed4_big.yaml`
* `EXP-P2-2-fixed4-mapping`: 同上 config，改 `mapping.strategy="greedy_local"`。
* `EXP-P2-3-fixed4-mapping-layout`: 同上 config，再开启 `layout.optimize_layout=true`。

### Phase 3 — 完整 Version-C

* `EXP-P3-1-full-vc`: `python -m scripts.run_version_c --cfg configs/vc_phase3_full_ucf101.yaml`
* `EXP-P3-2-full-vc-no-layout`: `python -m scripts.run_version_c --cfg configs/vc_phase3_nolayout_ucf101.yaml`
* `EXP-P3-3-full-vc-no-chip-regularizer`: 基于 `vc_phase3_full_ucf101.yaml` 把 `lambda_chip=0` 再跑一次。

### Phase 4 — 烟雾测试

* `EXP-SM-1-ast-smoke`: `python -m scripts.run_ast2_ucf101 --cfg configs/smoke_ast_ucf101.yaml`
* `EXP-SM-2-vc-smoke`: `python -m scripts.run_version_c --cfg configs/smoke_version_c_ucf101.yaml`

---

## 15. gen_experiment_cmds.py

* 根据上述实验列表，生成 `EXPERIMENTS_VERSION_C.md`，格式保持之前约定：phase 分组、每个实验包含“目的 / 配置文件 / 命令”。

---

你可以把这一整份直接丢给 Codex，当成**最新版 SPEC** 让它再跑一轮自检：
重点让它检查：

1. 所有主线代码是否严格遵守这里的接口 & 算法细节；
2. legacy 文件是否已经断开与主线入口的依赖，或者被改造成调用新 Dataset/Trainer。
