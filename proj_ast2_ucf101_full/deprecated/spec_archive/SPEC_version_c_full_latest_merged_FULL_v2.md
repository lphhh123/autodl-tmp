# SPEC_version_c_full_latest_merged_FULL_v2 — Version-C + Hyper-heuristic Wafer Layout (LLM Pick-ID) + Paper-grade Experiments (NO OMISSION)
[DEPRECATED] v5.4 SPEC is canonical; do not implement from this file.

（多尺度滑窗 + 视频/音频跨模态堆积 + AST2.0-lite 时空熵剪枝 + Version-C 多芯粒联合/交替优化 + L4/L5 超启发布局增强 + 震荡治理 + 论文级实验脚本生成）

> 原则：
> 1) 本 SPEC 为唯一真相来源；旧代码/旧 SPEC 冲突以此为准。
> 2) 先保证主线入口脚本与配置可跑通（Smoke），再保证完整实验可复现（Experiments）。
> 3) Legacy 可保留但主线不得依赖。
> 4) 离线可运行：无 Key / LLM 超时必须 fallback 完整跑完（ok:false 但不崩）。
> 5) 实验输出必须包含：metrics、trace、report、llm_usage.jsonl（无 key 允许 ok:false 但仍跑完）。
> 6) Layout 的 report.json 必须包含论文表格所需的震荡指标：oscillation_rate / repeat_signature_rate / undo_rate / objective_variance（详见第 17 节）。
> 7) 论文主表只要求“最小充分证据链”；其余实验归类为附录/扩展，但仍保留可一键跑（详见第 19.3）。

---

## 0. 环境 & 路径假设

项目根目录：
- /root/autodl-tmp/proj_ast2_ucf101_full

```bash
cd /root/autodl-tmp/proj_ast2_ucf101_full
```

### 0.1 UCF101 抽帧 & splits（训练/验证）

必须位于当前仓库：

```
project_root/
  data/ucf101/
    frames/...
    splits/trainlist01.txt
    splits/testlist01.txt
    audio/*.npy   # scripts/preprocess_ucf101_audio.py 生成
```

注意：Dataset **只访问** `data/ucf101/{frames,splits,audio}`

### 0.2 原始 UCF101 .avi（仅音频预处理用）

仅 `scripts/preprocess_ucf101_audio.py` 可访问：
- /root/autodl-tmp/proj_ast2_wafer_v2/data/UCF-101/...

### 0.3 命令统一入口

所有命令统一：

```bash
python -m scripts.xxx --cfg path/to/config.yaml
```

（如需要 out_dir/seed 等 CLI 参数，必须补齐并保持默认值兼容，见第 20/21 节脚本要求。）

---

## 1. 工程目录结构（主线）

```
project_root/
  configs/
    # 单模态视频（训练类）
    ast2_ucf101_dense.yaml
    ast2_ucf101_ast_only.yaml
    ast2_ucf101_ast_hw.yaml

    #（附录/扩展）dense_noscale：默认不进主表，但仍保留可跑
    ast2_ucf101_dense_noscale.yaml

    # 视频+音频（训练类）
    ast2_ucf101_av_dense.yaml
    ast2_ucf101_av_ast_hw.yaml

    #（附录/扩展）AV ast-only：默认不进主表，但仍保留可跑
    ast2_ucf101_av_ast_only.yaml

    # Version-C / 多芯粒（训练类）
    vc_phase2_fixed4_big.yaml                 # baseline / smoke / 或附录
    vc_phase3_full_ucf101.yaml                # Version-C full
    vc_phase3_nolayout_ucf101.yaml            # no-layout variant (mapping only)
    vc_phase3_twostage_ucf101.yaml            # NEW: two-stage baseline
    vc_phase3_mapping_only_ucf101.yaml        # NEW: mapping-only baseline
    vc_phase3_layout_only_ucf101.yaml         # NEW: layout-only baseline

    # Ablations（训练类）
    ablations/
      ast_no_time.yaml
      ast_no_space.yaml
      ast_no_voronoi.yaml
      ast_level1.yaml
      ast_no_modal.yaml
      ast_uniform_keep.yaml
      ast_random_keep.yaml

    # Proxy & 芯粒库
    proxy_ms_mem.yaml
    proxy_power.yaml
    gpu_data.yaml

    # Smoke
    smoke_ast_ucf101.yaml
    smoke_version_c_ucf101.yaml

    # Layout Agent (L0~L7)
    layout_agent/
      smoke_layout_agent.yaml
      layout_L0_heuristic.yaml
      layout_L3_region_pareto_sa.yaml
      layout_L4_region_pareto_llm_mixed_pick.yaml
      layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml
      layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml
      layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml
      layout_L5_region_pareto_llm_mixed_altopt.yaml

  scripts/
    run_proxy_ms_mem.py
    run_proxy_power.py
    run_ast2_ucf101.py
    run_version_c.py
    preprocess_ucf101_audio.py
    run_layout_agent.py
    gen_experiment_cmds.py

    # MUST GENERATE (executable)
    smoke_tests_version_c.sh
    experiments_version_c.sh

  models/
    video_vit.py
    ast2_pruner.py

  hw_proxy/
    layer_proxy_model.py
    layer_hw_proxy.py
    hw_loss.py

  mapping/
    segments.py
    mapping_solver.py

  layout/
    wafer_layout.py
    detailed_place.py
    llm_provider.py
    candidate_pool.py        # NEW: 候选池 + 多样性 + 单步试算 + 二次可行性检查

  trainer/
    trainer_single_device.py
    trainer_version_c.py

  utils/
    config.py
    logging.py
    metrics.py
    flops_estimator.py
    data_ucf101.py
```

### 1.1 Legacy（可保留但主线不能依赖）

```
datasets/ucf101_dataset.py
trainers/version_c_full_trainer.py
trainers/version_c_phase3_trainer.py
scripts/run_version_c_full.py
...
```

要求：
- 加注释：`# LEGACY: not used by main pipeline`
- 主线入口 `scripts/run_ast2_ucf101.py`、`scripts/run_version_c.py` **只能依赖** `trainer/*.py + utils/data_ucf101.py`

---

## 2. 配置字段规范（data / training / hw）

### 2.1 data（UCF101Dataset）

```yaml
data:
  dataset: ucf101
  frames_root: data/ucf101/frames
  splits_root: data/ucf101/splits
  audio_root: data/ucf101/audio
  use_audio: false | true
  clip_lens: [8, 16]
  num_frames: 8
  train_stride_ratio: 0.5
  eval_stride_ratio: 0.5
  clip_jitter: true
  batch_size: 8
  num_workers: 8
  img_size: 224
  patch_size: 16
```

兼容旧字段：
- 若缺 clip_lens 但有 clip_len: 16，则：
  - clip_lens = [clip_len]
  - 若 num_frames 缺失，则 num_frames = clip_len

### 2.2 training / hw

```yaml
training:
  mode: dense_baseline | single_device_ast | multi_chip_fixed | version_c_full
  use_audio: false | true
  num_epochs: ...
  lr: ...
  weight_decay: ...
  seed: 0
  log_every: 20
  save_every: 1

hw:
  mode: none | single_device | multi_chip_fixed | version_c_full
  use_hw_loss: true/false
  lambda_T: ...
  lambda_E: ...
  lambda_mem: ...
  lambda_chip: ...
  lambda_area: ...
  area_budget_mm2: ...
  mapping_strategy: greedy | greedy_local
  optimize_layout: false | true
```

---

## 3. 数据管线：UCF101Dataset（多尺度滑窗 + 固定 num_frames）

文件：utils/data_ucf101.py  
类：UCF101Dataset(Dataset)

### 3.1 路径解析（必须按此实现）

```python
project_root = Path(__file__).resolve().parents[2]

frames_root = cfg.data.frames_root
splits_root = cfg.data.splits_root
audio_root  = cfg.data.audio_root if hasattr(cfg.data, "audio_root") else None

if not Path(frames_root).is_absolute():
    frames_root = project_root / frames_root
if not Path(splits_root).is_absolute():
    splits_root = project_root / splits_root
if audio_root is not None and not Path(audio_root).is_absolute():
    audio_root = project_root / audio_root
```

split 规则：
- split="train" -> trainlist01.txt
- split="val"/"test" -> testlist01.txt  
找不到：抛 FileNotFoundError 并打印 splits_root

### 3.2 clip_lens + 滑窗采样（必须按此实现）

对每个视频 vid 的帧长 L_frames，对每个 cover_len in clip_lens：

- train：
  - stride = max(1, int(cover_len * train_stride_ratio))
  - offset = randint(0, stride-1) if clip_jitter else 0
  - starts = range(offset, max(L_frames - cover_len + 1, 1), stride)

- eval：
  - stride = max(1, int(cover_len * eval_stride_ratio))
  - offset = 0
  - starts = range(0, max(L_frames - cover_len + 1, 1), stride)

每个 (vid, cover_len, start) 生成 sample entry：
```python
{"video_id": vid, "cover_len": cover_len, "start": start, "label": class_idx}
```

### 3.3 __getitem__：窗口内均匀采样 num_frames（必须按此实现）

窗口：[start, start+cover_len) 裁剪到 [0, L_frames)

令窗口内实际帧数 Lw。

构造 num_frames 个索引 idx：
```python
if Lw >= num_frames:
    idx = np.linspace(0, Lw-1, num_frames)
    idx = np.floor(idx).astype(int)
else:
    base = np.arange(Lw)
    pad_needed = num_frames - Lw
    pad = np.clip(base[-1:], 0, max(Lw-1, 0)).repeat(pad_needed)
    idx = np.concatenate([base, pad], axis=0)
assert len(idx) == num_frames
frame_ids = start + idx
frame_ids = np.clip(frame_ids, 0, L_frames-1)
```

加载图像，返回：
```python
sample = {
  "video": FloatTensor [T=num_frames, 3, H, W],
  "label": LongTensor scalar,
  "video_id": str
}
```

如 use_audio=True：
- 读取 `audio_root/video_id.npy` 得 A[L_frames, D_audio]
- `audio = A[frame_ids]` -> FloatTensor [T, D_audio]
- sample["audio"] = audio

**硬保证**：video/audio 第一维都是 num_frames，默认 collate 可 stack。

---

## 4. 音频预处理：scripts/preprocess_ucf101_audio.py（必须按此实现）

cfg:
```yaml
data:
  raw_video_root: /root/autodl-tmp/proj_ast2_wafer_v2/data/UCF-101
  audio_root: data/ucf101/audio
  sample_rate: 16000
  n_mels: 64
```

步骤（必须可离线运行）：
1) librosa.load(video_path, sr=sample_rate) -> y
2) mel：librosa.feature.melspectrogram(y, sr, n_mels, n_fft=2048, hop_length=512)
3) S_db = librosa.power_to_db(S, ref=np.max)  # [n_mels, T_spec]
4) 获取该视频 L_frames（优先 OpenCV 计数；失败则 L_frames=T_spec）
5) time 轴均分到 L_frames 段，帧级特征：
   ```python
   boundaries = np.linspace(0, T_spec, L_frames+1)
   for k in range(L_frames):
       t0,t1=int(boundaries[k]),int(boundaries[k+1])
       if t1<=t0: t1=t0+1
       a_k = S_db[:, t0:t1].mean(axis=1)   # [n_mels]
   A = np.stack(a_list, axis=0).astype("float32")  # [L_frames,n_mels]
   ```
6) 保存：np.save(audio_root/f"{video_id}.npy", A)

失败处理：
- 任一视频解码失败：warning + 保存同 shape 的全 0（至少保证 npy 存在，Dataset 不崩）。

---

## 5. 模型：VideoViT & VideoAudioAST（models/video_vit.py）

### 5.1 VideoViT（单模态视频）

接口：
```python
class VideoViT(nn.Module):
    def forward(self, x: torch.Tensor, return_intermediate: bool=False):
        # x: [B,T,3,H,W] (若输入为 [B,3,T,H,W] 必须在内部转置到前者)
```

必须实现：
- 强制 assert：T == self.num_frames
- patch embed：每帧 (H,W) -> Nv=(img_size/patch_size)^2 个 patch token
- tokens reshape：[B,T,Nv,C] -> [B, T*Nv, C] 送入 ViT blocks
- 在 ViT blocks 中接入 ASTPruner gating（token/head/ch/block）
- 输出 logits：[B,num_classes]
- return_intermediate=True 时返回：
  ```python
  return logits, {
    "L_AST": L_AST,
    "ast_stats": {"sparsity_token":..., "sparsity_head":..., "sparsity_ch":..., "sparsity_block":...},
    "token_feat": token_feat_ref,         # 可选
    "flops_bytes_per_layer": [...],       # 可选（供 segments 构建）
  }
  ```

### 5.2 VideoAudioAST（视频+音频跨模态）

接口：
```python
class VideoAudioAST(nn.Module):
    def forward(self, x_video, x_audio, return_intermediate: bool=False):
        # x_video: [B,T,3,H,W]
        # x_audio: [B,T,D_audio]
```

必须实现：
- 视频 tokens：v_tokens [B,T,Nv,C]
- 音频 token：a_tokens = audio_proj(x_audio) -> [B,T,C] -> unsqueeze -> [B,T,1,C]
- concat：tokens_all = cat([v_tokens, a_tokens], dim=2) -> [B,T,Nv+1,C]
- modality_slices：
  - video: (0, Nv)
  - audio: (Nv, Nv+1)
- ASTPruner.forward_token_gating(token_feat_layer, modality_slices) 产生 mask_token，施加到 token 或 attention output（实现位置保持一致，但必须真的生效）
- 返回 info_dict 多包含：
  - "modality_slices"
  - "modal_stats"（每模态 sparsity/H_time_mean 等统计）

---

## 6. AST2.0-lite：时空熵 + Voronoi + 多模态 token gating（models/ast2_pruner.py）【完整不省略】

文件：models/ast2_pruner.py  
类：ASTPruner

### 6.1 初始化与公共工具函数（必须实现）

初始化参数（cfg）示例：
```yaml
ast:
  rho_target: 0.6
  temperature: 0.15
  entropy_tau: 1.0
  time_window_levels: [1,2,4]
  space_window_levels: [1,2,4]
  time_window_overlap: 0.5
  num_regions: 8
  patch_grid_h: 14
  patch_grid_w: 14
  num_modalities: 2
  modalities: ["video","audio"]
  lambda_token: 1.0
  lambda_head: 0.2
  lambda_ch: 0.2
  lambda_block: 0.1
  score_weights:
    a_time: 1.0
    b_space: 0.6
    c_region: 0.4
    d_modal: 0.2
```

工具函数（必须有，且行为一致）：

1) softmax over channels（温度）
```python
def _softmax_over_channels(x, tau):
    # x: [..., C]
    return torch.softmax(x / max(tau, 1e-6), dim=-1)
```

2) entropy
```python
def _entropy_from_prob(p, dim=-1, eps=1e-6):
    # p: prob
    return -(p * torch.log(p + eps)).sum(dim=dim)
```

3) per-sample minmax norm（避免跨样本尺度差）
```python
def minmax_norm_per_batch(x):
    # x: [B, ...]
    B = x.shape[0]
    x_flat = x.view(B, -1)
    mn = x_flat.min(dim=1).values.view(B, *([1]*(x.dim()-1)))
    mx = x_flat.max(dim=1).values.view(B, *([1]*(x.dim()-1)))
    return (x - mn) / (mx - mn + 1e-6)
```

4) kth threshold（每样本独立）
```python
def kth_threshold(scores, k_keep):
    # scores: [B, M] flattened
    # k_keep: int (#keep)
    # return thr: [B]
    B, M = scores.shape
    k_keep = max(1, min(k_keep, M))
    # topk returns largest; threshold is kth largest -> smallest among topk
    topk_vals, _ = torch.topk(scores, k_keep, dim=1, largest=True, sorted=True)
    thr = topk_vals[:, -1]
    return thr
```

### 6.2 Multi-level Time Entropy（带 overlap）【必须按此实现】

输入 token_feat: x [B,T,N,C]

步骤：
1) p = softmax(x, tau) -> [B,T,N,C]
2) 对每个 level L：
   - window_size = ceil(T / L)
   - stride = window_size if overlap<=0 else max(1, int(window_size*(1-overlap)))
   - starts = range(0, T, stride)
   - 对每个窗口 [s:e)：
     - pw = p[:, s:e] -> [B,tw,N,C]
     - pw_mean = mean over time -> [B,N,C]
     - Hw = entropy(pw_mean) -> [B,N]
   - H_level = mean over windows -> [B,N]
3) 返回：
```python
{
  "H_time_level": {L: [B,N]},
  "H_time_global": H_time_level[1]  # 若 levels 含 1，否则取 levels[0]
}
```

实现参考伪码：
```python
def compute_multi_level_time_entropy(x, levels, tau, overlap):
    p = softmax(x/tau)
    B,T,N,C = p.shape
    out = {}
    for L in levels:
        w = math.ceil(T/L)
        stride = w if overlap<=0 else max(1, int(w*(1-overlap)))
        windows=[]
        for s in range(0,T,stride):
            e=min(s+w,T)
            if e<=s: continue
            pw = p[:,s:e].mean(dim=1)       # [B,N,C]
            Hw = entropy(pw)                # [B,N]
            windows.append(Hw)
        H = torch.stack(windows,dim=-1).mean(dim=-1) if windows else torch.zeros(B,N,device=x.device)
        out[L]=H
    H_global = out[1] if 1 in out else out[levels[0]]
    return {"H_time_level": out, "H_time_global": H_global}
```

### 6.3 Multi-level Space Entropy（patch grid）【必须按此实现】

目标：输出空间熵随时间变化：H_space [B,T]，以及多 level 字典。

输入：
- token_feat x [B,T,N_total,C]
- 其中 video patch token 个数 Nv = patch_grid_h * patch_grid_w
- 若有 audio token（Na=1），它不参与空间网格，单独处理

步骤（视频部分）：
1) 取视频 token：xv = x[:,:,0:Nv,:] -> [B,T,Nv,C]
2) reshape 为网格：xg = xv.view(B,T,H_p,W_p,C)
3) 转 prob：p = softmax over C：pg [B,T,H,W,C]
4) 对每个 space level L（例如 1,2,4）：
   - 将网格划分为 LxL 区域（每区包含若干 patch）
   - 对每个 t、每个区域 r：
     - 区域内对 patch 维度平均得到 pr [B,T,L,L,C]
     - entropy -> Hr [B,T,L,L]
   - 将 Hr 在空间上平均：H_level[L] = Hr.mean(dim=(2,3)) -> [B,T]
5) 取 global：H_space_global = H_level[1]（若无则取 levels[0]）
6) 返回：
```python
{"H_space_level": {L:[B,T]}, "H_space_global":[B,T]}
```

实现要点：
- 区域划分可用 adaptive pooling（推荐）：
  - `pr = adaptive_avg_pool2d(pg.permute(0,1,4,2,3).reshape(B*T,C,H,W), (L,L))`
  - reshape 回 [B,T,C,L,L] 并在 C 上算 entropy。

### 6.4 Voronoi Region IDs（必须按此实现）

目的：把 video patch grid 划分为 num_regions 个区域，区域用于 region term 打分。

输入：
- grid (H_p, W_p)
- num_regions R

步骤：
1) 在 grid 上选 R 个 seed（要求尽量均匀）：
   - 可用规则均匀采样；或 farthest point sampling（实现可简化）
2) 对每个 patch (h,w) 计算到各 seed 的欧氏距离，分配给最近 seed
3) 输出 region_ids_video [Nv]，Nv=H_p*W_p
4) 对非视频 token（如 audio token），region_id 设为 0（或固定 dummy），确保 region_ids 长度 == N_total

必须提供：
```python
region_ids: LongTensor [N_total]
num_regions: int
```

### 6.5 Token Gating Score & Mask（单模态）【必须按此实现】

输入：
- token_feat x [B,T,N,C]
- time entropy：H_time_level[L] [B,N]
- space entropy：H_space_level[L] [B,T]
- region_ids [N]
- cfg: rho_target, temperature, score_weights(a_time,b_space,c_region)

融合与归一化（必须按此实现）：
1) 时间熵融合：
```python
Ht_list=[]
for L,H in H_time_level.items():         # [B,N]
    Hn = minmax_norm_per_batch(H)        # [B,N] -> [0,1]
    Ht_list.append(Hn)
H_time_fused = mean(stack(Ht_list), dim=0)   # [B,N]
```

2) 空间熵融合：
```python
Hs_list=[]
for L,H in H_space_level.items():        # [B,T]
    Hn = minmax_norm_per_batch(H)        # [B,T]
    Hs_list.append(Hn)
H_space_fused = mean(stack(Hs_list), dim=0)  # [B,T]
```

3) 区域重要性（推荐实现：区域内 token 的时间熵均值）：
```python
region_importance[b,r] = mean_{n in region r} H_time_fused[b,n]
```

4) score 计算（输出 [B,T,N]）：
```python
a = cfg.score_weights.a_time
b = cfg.score_weights.b_space
c = cfg.score_weights.c_region

Ht = H_time_fused.unsqueeze(1).expand(B,T,N)     # [B,T,N]
Hs = H_space_fused.unsqueeze(2).expand(B,T,N)    # [B,T,N]
Hr = region_importance[:, region_ids].unsqueeze(1).expand(B,T,N)  # [B,T,N]

score = a*Ht + b*Hs + c*Hr
```

5) keep ratio 与阈值：
- M = T*N，k_keep = int(rho_target * M)，clip 到 [1,M]
- per-sample threshold：
```python
score_flat = score.view(B, -1)
thr = kth_threshold(score_flat, k_keep)          # [B]
thr = thr.view(B,1,1)
```

6) mask：
```python
temp = max(cfg.temperature, 1e-6)
mask_soft = torch.sigmoid((score - thr) / temp)  # [B,T,N]
mask_token = mask_soft.unsqueeze(-1)             # [B,T,N,1]
sparsity_token = 1.0 - mask_soft.mean()          # scalar
```

### 6.6 Token Gating（多模态：video+audio）【必须按此实现】

输入：
- token_feat x [B,T,N_total,C]
- modality_slices: dict，例如 {"video":(0,Nv), "audio":(Nv,Nv+1)}
- cfg.score_weights 包含 d_modal
- self.modality_logit: learnable [M]（M=模态数）

步骤（必须按此实现）：
1) 拆模态 token（略）
2) 分别计算熵（音频空间熵可用时间熵广播）
3) 构造全局 fused：H_time_token/H_space_token [B,T,N_total]
4) 模态权重项：
```python
w_modal = torch.sigmoid(self.modality_logit)   # [M]
modal_id_for_token: LongTensor [N_total]       # video->0, audio->1
Wm = w_modal[modal_id_for_token].view(1,1,N_total).expand(B,T,N_total)
```
5) region term：audio token region_id 固定 0
6) 最终 score：
```python
score = a*H_time_token + b*H_space_token + c*Hr + d*Wm
```
7) threshold/mask/sparsity_token 同 6.5
8) modal_stats（必须输出）：
```python
modal_stats[name] = {
  "sparsity": float(1.0 - mask_soft[:,:,s:e].mean().item()),
  "H_time_mean": float(H_time_token[:,:,s:e].mean().item()),
  "mask_mean": float(mask_soft[:,:,s:e].mean().item())
}
```

### 6.7 Head / Channel / Block gating + L_AST（必须按此实现）

参数化（sigmoid logit）：
- head_logit: [depth, num_heads]
- ch_logit: [depth, embed_dim]（或 MLP hidden）
- block_logit: [depth]

施加方式（至少一种必须真正生效）：
- head gating：attention head 维乘 mask
- channel gating：MLP 输出通道乘 mask
- block gating：layer 输出乘 mask_block[l]

L_AST（必须输出到 info_dict）：
```python
L_AST = lambda_token*sparsity_token + lambda_head*sparsity_head + lambda_ch*sparsity_ch + lambda_block*sparsity_block
```

---

## 7. 硬件代理 HW Proxy（hw_proxy/*）

### 7.1 LayerProxyModel（MLP）

输入：layer feature vector  
输出：lat_ms / mem_mb / power_w（3-head 或 3 个独立模型）

### 7.2 build_layer_features（固定顺序，必须一致）

```
[ log10(flops+1),
  log10(bytes+1),
  log10(peak_flops+1),
  log10(peak_bw+1),
  one_hot(layer_type),          # attn/mlp/norm/other（至少4类）
  embed_dim/1024,
  num_heads/16,
  mlp_ratio/4,
  seq_len/1024,
  precision ]                  # 0 fp32, 1 fp16
```

### 7.3 LayerHwProxy

- predict_layer(layer_cfg, device_cfg) -> {lat_ms, mem_mb, power_w}
- predict_segment(segment_cfg):
  - lat_ms：sum（serial）或 max（balanced）
  - mem_mb：max
  - power_w：加权平均 `sum(power_i*lat_i)/sum(lat_i)`

---

## 8. 芯粒库 & ChipSlotConfig（mapping/segments.py）

gpu_data.yaml:
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

ChipSlotConfig（Gumbel-Softmax）：
- logits: [num_slots, num_types+1]（+1 empty）
- forward(hard=False) -> alpha [S,T+1]，eff_specs 为字段加权和
- chip count 正则：
```python
chip_used_prob = 1.0 - alpha[:, -1]
L_chip = lambda_chip * chip_used_prob.sum()
```

---

## 9. Segment & 计算图重构（mapping/segments.py + mapping/mapping_solver.py）

Segment dataclass（至少包含）：
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

构建 segments：
- 每 segment_size 层聚合
- flops/bytes 用 flops_estimator 估算并求和
- traffic_in/out 估算：seq_len*embed_dim*bytes_per_elem（可乘常数）

细粒度拆分（可选，但接口必须存在）：
- mark_fine_splittable_segments
- try_fine_split_segment（含 compute/comm 与 rewire_meta）

---

## 10. MappingSolver（mapping/mapping_solver.py）

必须实现：
1) build_cost_matrix
2) estimate_pipeline_latency（serial / balanced）
3) solve_mapping（greedy / greedy_local；可选 fine split）

返回 mapping_result：
```python
{
  "mapping": List[int],
  "per_slot_time": List[float],
  "total_latency_ms": float,
  "comm_ms": float,
  "segments": List[Segment],
  "rewire_meta": dict
}
```

---

## 11. WaferLayout（layout/wafer_layout.py）

- pos: learnable [num_slots,2]（mm）
- penalty：
  - boundary / overlap / comm / thermal
- optimize_layout=false：规则初始化，L_layout=0

---

## 12. L_hw 组合（hw_proxy/hw_loss.py）

compute_hw_loss(segments, chip_slot_config, hw_proxy, mapping_solver, wafer_layout, cfg):

1) alpha/eff_specs = chip_slot_config()
2) L_chip
3) mapping_result = mapping_solver.solve_mapping(..., layout_positions=wafer_layout.pos.detach(), strategy=cfg.mapping_strategy)
4) 得到：latency_ms / peak_mem_mb / energy_j / area_mm2
5) L_area
6) L_layout（按 optimize_layout）
7) 总损失：
```python
L_hw = lambda_T*T + lambda_E*E + lambda_mem*mem + L_chip + L_area + L_layout
```
8) 必须返回 stats（写入 hw_stats.json）：lat/energy/mem/area/chip_count/comm/layout_penalties...

---

## 13. Trainer（trainer/*）

### 13.1 单卡训练（trainer_single_device.py）

loss = L_task + lambda_AST*L_AST + lambda_hw*L_hw（按配置启用）

### 13.2 Version-C 交替优化（trainer_version_c.py）

外循环（可配置 inner steps）：
A) 更新 θ 与稀疏结构 s（AST gating）：loss 包含 multi-chip L_hw  
B) 冻结 θ，更新 chip_slot_config.logits：最小化 L_hw  
C) 更新离散 mapping：solve_mapping（缓存用于 A/B）  
D) 可选布局优化：optimize_layout=true 时更新 wafer_layout.pos（多步）

---

# PART-II：晶圆级布局 Layout Agent（L0~L7）+ LLM Pick-ID 超启发增强（完整不省略）

## 14. Layout Agent 运行入口与输出

入口：scripts/run_layout_agent.py  
必选参数：
- --layout_input <json>
- --cfg configs/layout_agent/xxx.yaml
- --out_dir <dir>
可选：
- --seed <int>（默认 0）

out_dir 输出必须包含：
1) trace.csv（每 step 一行，至少包含以下列）：
   - iter, stage, op, op_args_json, accepted, total_scalar, comm_norm, therm_norm,
   - pareto_added, duplicate_penalty, boundary_penalty, seed_id, time_ms,
   - signature（NEW 必须）, d_total（NEW 必须）, d_comm（建议）, d_therm（建议）

2) report.json（汇总，必须含第17节震荡指标）

3) llm_usage.jsonl（每次 LLM 调用一行 JSON，ok 可 false 但必须存在）

---

## 15. 震荡治理机制（必须实现且可消融）

治理机制必须落地：
1) Tabu/forbidden（短期禁忌）
2) Queue + refresh
3) 接受判据 hysteresis（防抖）
4) 多样性配额（防止 swap 模板化）
5) 二次可行性检查（pick-IDs -> sequential feasibility -> actions）

---

## 16. L4：候选池 + LLM 只 pick candidate ID（完整要求 + 代码骨架）

### 16.1 新增文件：layout/candidate_pool.py（必须实现，按此骨架扩展）

（此处骨架与函数定义 **不省略**；实现要点 **必须满足**）
——（内容同你提供的 FULL 版骨架：Candidate / signature / apply / eval / pack_diverse / build_state_summary / pick_ids_to_actions_sequential）——

【强制约束补充（必须实现，防止“只生成 swap”偷懒）】
- raw_candidates 必须覆盖至少 4 类：swap / relocate / cluster_move / explore
- pack_diverse_candidates 默认配额不得为 0（swap/relocate/cluster/explore 都要有）
- 同一 slot 的 relocate 候选数量上限 <= 6（避免 relocate 刷屏）
- 候选必须做单步试算 evaluate（不允许“估计值全靠规则”）
- 过滤几乎不变动作（flat filter）：|d_total|,|d_comm|,|d_therm| < eps_flat 则丢弃

### 16.2 修改 layout/detailed_place.py（必须按此接入候选池+queue）

- LLM 不直接产 action，而是 pick IDs
- pick_ids -> sequential feasibility -> actions -> push queue
- 每 step：优先 pop queue；否则 fallback heuristic action
- 写 trace.csv 必须含 signature/d_total（及可选 d_comm/d_therm）
- llm_usage.jsonl 必须写 n_queue_push / picked_types / best_d_total 等字段

### 16.3 修改 layout/llm_provider.py（LLM 只输出 pick IDs）

接口：
```python
class LLMProvider:
    def propose_pick(self, state_summary: Dict, k: int) -> List[int]:
        ...
```

HeuristicProvider：
- 不调用 LLM：按 d_total 最小排序取前 k，并保证至少 1 个非 swap（若存在）

LLM 输出 wrapper 与解析、repair、validate（同 FULL 版要求）

### 16.4 Prompt（逐字拷贝，必须一致）

（system_prompt + user_content 模板同 FULL 版）

---

## 17. Layout 震荡指标（必须写入 report.json，定义严格、可直接实现）

（repeat_signature_rate / undo_rate / objective_variance / objective_std_lastN / oscillation_rate / improve_step_ratio / flat_step_ratio 等定义同 FULL 版，不省略）

---

## 18. 工程鲁棒性（必须实现）

- trace.csv 写入禁止 f.write(a,b,c)
- heartbeat.json / checkpoint_state.json（同 FULL 版）

---

# PART-III：论文级实验设计（主表 + 消融 + 运行脚本）

## 19. 论文实验设计：主表 vs 附录（“最小充分证据链”落地）

### 19.1 创新 A（剪枝+多芯粒）：主表必须覆盖（推荐主表 EXP 集合）

主表（最小充分证据链）推荐只保留：
- **EXP-A1** Dense baseline（single-device）
- **EXP-A3** AST + single-device hw loss
- **EXP-A4** Ours（Version-C full：multi-chip aware + 交替优化）
- **EXP-A5** Two-stage baseline（先单卡剪好再后处理 mapping/layout，不回传剪枝）  ← A3 的硬对照
- **EXP-A-G2** 控制稀疏率公平对照：uniform / random / ours entropy（同 rho_target） ← A2 关键
- **EXP-A-G3** 关键组件消融（time/space/voronoi/1lvl/nomodal） ← A2 机制证据

> 若你主表行数受限，可将 A-G3 放附录，但至少保留 time/space/voronoi 三项。

### 19.2 创新 B（布局超启发）：主表必须覆盖（推荐主表 EXP 集合）

主表推荐：
- **EXP-B1** Heuristic only（candidate pool + heuristic pick）
- **EXP-B2** LLM Pick-ID enhanced（完整版：diversity+feasibility+queue+refresh）
- **EXP-B3** SA baseline（L3：纯 SA/局部搜索）
- **EXP-B2-ab**（至少 1 条放主文，建议 nofeas 或 noqueue；其余放附录）

### 19.3 可以删/合并的实验（已在脚本中默认归类为附录 EXP）

- dense_noscale：默认附录（保留 EXP-APP-A-DENSE-NOSCALE）
- AV 三套太重：主文只保留 `AV dense` + `AV ast-hw`（`AV ast-only` 默认附录）
- Phase2 fixed4 三套：主文只保留 mapping 与 mapping+layout 两个；grid/smoke 作为 smoke 或附录

---

## 20. 实验输出统一口径（写论文用）

训练类实验输出：
- metrics.json：top1/acc、loss、sparsity_token/head/ch/block、L_AST、L_hw 分解
- hw_stats.json：lat/energy/mem/area/chip_count/mapping 摘要

layout 类实验输出：
- trace.csv
- report.json（必须含第17节震荡指标）
- llm_usage.jsonl（含 ok/n_pick/pick_ids/...）

随机性：
- 训练类默认 seeds=[0,1,2]
- layout 类默认 seeds=[0,1,2,3,4]

---

## 21. EXP-ID 清单（必须实现，脚本逐条跑）

### 21.1 创新 A：面向多芯粒的时空熵剪枝（必须补齐）

A-Core：
- EXP-A1：Dense baseline（single device）
- EXP-A2：AST only（no hw loss）【可附录】
- EXP-A3：AST + single-device hw loss（主表）
- EXP-A4：Ours multi-chip aware（Version-C full：交替优化）（主表）
- EXP-A5：Two-stage baseline（主表）
- EXP-A6：Mapping-only（剪枝固定，只优化 mapping）【可附录】
- EXP-A7：Layout-only（剪枝固定，只优化 layout）【可附录】

A-G2（同稀疏率公平对照，主表推荐）：
- EXP-A-G2-uniform：uniform keep（同 rho_target）
- EXP-A-G2-random：random keep（同 rho_target）
- EXP-A-G2-ours：ours entropy keep（等价于 EXP-A4 或对应 cfg）

A-G3（关键组件消融，主表/附录）：
- EXP-Abl-time：去 time entropy
- EXP-Abl-space：去 space entropy
- EXP-Abl-vor：去 Voronoi
- EXP-Abl-1lvl：levels=[1]
- EXP-Abl-nomodal：去 modality weight
- EXP-Abl-uniform：uniform keep（等价于 G2-uniform）
- EXP-Abl-random：random keep（等价于 G2-random）

### 21.2 创新 B：超启发式晶圆布局（必须补齐）

B-Core：
- EXP-B1：Heuristic only（candidate pool + heuristic pick）
- EXP-B2：LLM Pick-ID enhanced（完整版）
- EXP-B3：SA baseline（L3）
- EXP-B4：（可选）LLM Direct actions（若旧版仍在）

B-Ablation（必须可跑完并出 report）：
- EXP-B2-ab-noqueue：去队列/refresh
- EXP-B2-ab-nofeas：去 sequential feasibility
- EXP-B2-ab-nodiverse：去多样性配额

---

## 22. 两份指令文件（必须生成，Smoke 与 Experiments 分离）

必须生成并可执行（chmod +x）：

1) scripts/smoke_tests_version_c.sh  
   覆盖：
   - proxy 两脚本
   - ast 单卡
   - version-c
   - layout agent（L0 与 L4 pick），无 key 时 fallback 跑完

2) scripts/experiments_version_c.sh  
   - 支持：`./scripts/experiments_version_c.sh EXP-A4 0`
   - 第一个参数 EXP_ID，第二个 seed（默认 0）
   - 每个 EXP_ID -> cfg + out_dir（含 EXP_ID 与 seed）

---

## 23. 必须生成的脚本内容（最终版，直接写入文件）

### 23.1 scripts/smoke_tests_version_c.sh（最终版）

```bash
#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[SMOKE] Proxy ms/mem"
python -m scripts.run_proxy_ms_mem --cfg configs/proxy_ms_mem.yaml

echo "[SMOKE] Proxy power"
python -m scripts.run_proxy_power --cfg configs/proxy_power.yaml

echo "[SMOKE] AST single-device"
python -m scripts.run_ast2_ucf101 --cfg configs/smoke_ast_ucf101.yaml

echo "[SMOKE] Version-C"
python -m scripts.run_version_c --cfg configs/smoke_version_c_ucf101.yaml

echo "[SMOKE] Layout agent L0"
python -m scripts.run_layout_agent \\n  --layout_input outputs/P3/A3/layout_input.json \\n  --cfg configs/layout_agent/layout_L0_heuristic.yaml \\n  --out_dir outputs/SMOKE/layout_L0_heuristic \\n  --seed 0

echo "[SMOKE] Layout agent L4 pick-ID (fallback if no key)"
python -m scripts.run_layout_agent \\n  --layout_input outputs/P3/A3/layout_input.json \\n  --cfg configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml \\n  --out_dir outputs/SMOKE/layout_L4_pick \\n  --seed 0

echo "[SMOKE DONE]"
```

### 23.2 scripts/experiments_version_c.sh（最终版：主表+附录都能跑）

```bash
#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_ID="${1:-}"
SEED="${2:-0}"

if [[ -z "$EXP_ID" ]]; then
  echo "Usage: $0 <EXP_ID> [SEED]"
  exit 1
fi

run_ast () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_ast2_ucf101 --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_vc () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_version_c --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_layout () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_layout_agent \\n    --layout_input outputs/P3/A3/layout_input.json \\n    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

case "$EXP_ID" in
  # -------------------------
  # Innovation A (Main/Core)
  # -------------------------
  EXP-A1) run_ast configs/ast2_ucf101_dense.yaml           "outputs/EXP-A1/seed${SEED}" ;;
  EXP-A2) run_ast configs/ast2_ucf101_ast_only.yaml        "outputs/EXP-A2/seed${SEED}" ;;
  EXP-A3) run_ast configs/ast2_ucf101_ast_hw.yaml          "outputs/EXP-A3/seed${SEED}" ;;
  EXP-A4) run_vc  configs/vc_phase3_full_ucf101.yaml       "outputs/EXP-A4/seed${SEED}" ;;
  EXP-A5) run_vc  configs/vc_phase3_twostage_ucf101.yaml   "outputs/EXP-A5_twostage/seed${SEED}" ;;
  EXP-A6) run_vc  configs/vc_phase3_mapping_only_ucf101.yaml "outputs/EXP-A6_mappingonly/seed${SEED}" ;;
  EXP-A7) run_vc  configs/vc_phase3_layout_only_ucf101.yaml  "outputs/EXP-A7_layoutonly/seed${SEED}" ;;

  # A-G2 fairness (same rho_target)
  EXP-A-G2-uniform) run_ast configs/ablations/ast_uniform_keep.yaml "outputs/EXP-A-G2-uniform/seed${SEED}" ;;
  EXP-A-G2-random)  run_ast configs/ablations/ast_random_keep.yaml  "outputs/EXP-A-G2-random/seed${SEED}" ;;
  EXP-A-G2-ours)    run_vc  configs/vc_phase3_full_ucf101.yaml      "outputs/EXP-A-G2-ours/seed${SEED}" ;;

  # A-G3 ablations
  EXP-Abl-time)     run_ast configs/ablations/ast_no_time.yaml      "outputs/EXP-Abl-time/seed${SEED}" ;;
  EXP-Abl-space)    run_ast configs/ablations/ast_no_space.yaml     "outputs/EXP-Abl-space/seed${SEED}" ;;
  EXP-Abl-vor)      run_ast configs/ablations/ast_no_voronoi.yaml   "outputs/EXP-Abl-vor/seed${SEED}" ;;
  EXP-Abl-1lvl)     run_ast configs/ablations/ast_level1.yaml       "outputs/EXP-Abl-1lvl/seed${SEED}" ;;
  EXP-Abl-nomodal)  run_ast configs/ablations/ast_no_modal.yaml     "outputs/EXP-Abl-nomodal/seed${SEED}" ;;
  EXP-Abl-uniform)  run_ast configs/ablations/ast_uniform_keep.yaml "outputs/EXP-Abl-uniform/seed${SEED}" ;;
  EXP-Abl-random)   run_ast configs/ablations/ast_random_keep.yaml  "outputs/EXP-Abl-random/seed${SEED}" ;;

  # -------------------------
  # Innovation B (Layout)
  # -------------------------
  EXP-B1) run_layout configs/layout_agent/layout_L0_heuristic.yaml "outputs/EXP-B1/seed${SEED}" ;;
  EXP-B2) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml "outputs/EXP-B2/seed${SEED}" ;;
  EXP-B3) run_layout configs/layout_agent/layout_L3_region_pareto_sa.yaml "outputs/EXP-B3/seed${SEED}" ;;
  EXP-B2-ab-noqueue)   run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml   "outputs/EXP-B2-ab-noqueue/seed${SEED}" ;;
  EXP-B2-ab-nofeas)    run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml    "outputs/EXP-B2-ab-nofeas/seed${SEED}" ;;
  EXP-B2-ab-nodiverse) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml "outputs/EXP-B2-ab-nodiverse/seed${SEED}" ;;

  # -------------------------
  # Appendix / Optional (kept but not required for main table)
  # -------------------------
  EXP-APP-A-DENSE-NOSCALE) run_ast configs/ast2_ucf101_dense_noscale.yaml "outputs/EXP-APP-A-DENSE-NOSCALE/seed${SEED}" ;;
  EXP-APP-AV-DENSE)        run_ast configs/ast2_ucf101_av_dense.yaml      "outputs/EXP-APP-AV-DENSE/seed${SEED}" ;;
  EXP-APP-AV-AST-HW)       run_ast configs/ast2_ucf101_av_ast_hw.yaml     "outputs/EXP-APP-AV-AST-HW/seed${SEED}" ;;
  EXP-APP-AV-AST-ONLY)     run_ast configs/ast2_ucf101_av_ast_only.yaml   "outputs/EXP-APP-AV-AST-ONLY/seed${SEED}" ;;
  EXP-APP-VC-PHASE2-FIXED4) run_vc configs/vc_phase2_fixed4_big.yaml      "outputs/EXP-APP-VC-PHASE2-FIXED4/seed${SEED}" ;;

  *) echo "Unknown EXP_ID=$EXP_ID"; exit 2 ;;
esac
```

### 23.3 脚本权限（必须执行一次）

```bash
chmod +x scripts/smoke_tests_version_c.sh
chmod +x scripts/experiments_version_c.sh
```

---

## 24. CLI 兼容性强约束（必须实现）

若当前 `scripts/run_ast2_ucf101.py` 与 `scripts/run_version_c.py` 不支持参数，必须补齐并保持默认兼容：

- `--cfg <path>`（必选）
- `--out_dir <path>`（可选，默认由 cfg 内决定或 outputs/default）
- `--seed <int>`（可选，默认 0）

要求：
- 不传 --out_dir 也能跑（保持旧行为）
- seed 必须真正影响：torch / numpy / random / dataloader worker seed

---

# END OF SPEC
