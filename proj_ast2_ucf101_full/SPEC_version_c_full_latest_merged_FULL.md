# SPEC_version_c_full_latest_merged_FULL — Version-C + Hyper-heuristic Wafer Layout (LLM Pick-ID) + Paper-grade Experiments (NO OMISSION)

（多尺度滑窗 + 视频/音频跨模态堆积 + AST2.0-lite 时空熵剪枝 + Version-C 多芯粒联合/交替优化 + L4/L5 超启发布局增强 + 震荡治理 + 论文级实验脚本生成）

> 原则：
> 1) 本 SPEC 为唯一真相来源；旧代码/旧 SPEC 冲突以此为准。
> 2) 先保证主线入口脚本与配置可跑通（Smoke），再保证完整实验可复现（Experiments）。
> 3) Legacy 可保留但主线不得依赖。
> 4) 离线可运行：无 Key / LLM 超时必须 fallback 完整跑完。
> 5) 实验输出必须包含：metrics、trace、report、llm_usage.jsonl（期望出现 ok:true；无 key 则 ok:false 但仍跑完）。
> 6) Layout 的 report.json 必须包含论文表格所需的震荡指标：oscillation_rate / repeat_signature_rate / undo_rate / objective_variance（详见第 17 节）。

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

注意：Dataset **只访问** data/ucf101/{frames,splits,audio}

### 0.2 原始 UCF101 .avi（仅音频预处理用）

仅 scripts/preprocess_ucf101_audio.py 可访问：
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

    # Proxy & 芯粒库
    proxy_ms_mem.yaml
    proxy_power.yaml
    gpu_data.yaml

    # Smoke
    smoke_ast_ucf101.yaml
    smoke_version_c_ucf101.yaml

    # Layout Agent (L0~L7)
    layout_agent/
      layout_L0_heuristic.yaml
      layout_L3_region_pareto_sa.yaml
      layout_L4_region_pareto_llm_mixed_pick.yaml
      layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml
      layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml
      layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml
      layout_L5_region_pareto_llm_mixed_altopt.yaml
      smoke_layout_agent.yaml

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
    hw_loss.py               # 推荐独立出来（也可放 trainer 内，但需统一出口）

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
- 主线入口 scripts/run_ast2_ucf101.py、scripts/run_version_c.py **只能依赖** trainer/*.py + utils/data_ucf101.py

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
- 区域划分可用 reshape/stride 方式实现；若 H,W 不能整除，用 adaptive pooling（推荐）：
  - 先把 pg 的 patch 维平均到 LxL：`pr = adaptive_avg_pool2d(pg.permute(0,1,4,2,3).reshape(B*T,C,H,W), (L,L))`
  - 再 reshape 回 [B,T,C,L,L] 并在 C 上算 entropy。

### 6.4 Voronoi Region IDs（必须按此实现）

目的：把 video patch grid 划分为 num_regions 个区域，区域用于 region term 打分。

输入：
- grid (H_p, W_p)
- num_regions R

步骤：
1) 在 grid 上选 R 个 seed（要求尽量均匀）：
   - 可用规则均匀采样：把网格分成 sqrt(R) x sqrt(R) 近似格点，取中心四舍五入
   - 或 farthest point sampling（更好，但实现可简化）
2) 对每个 patch (h,w) 计算到各 seed 的欧氏距离，分配给最近 seed
3) 输出 region_ids_video [Nv]，Nv=H_p*W_p
4) 对非视频 token（如 audio token），region_id 设为 0（或一个固定 dummy），确保 region_ids 长度 == N_total

必须提供：
```python
region_ids: LongTensor [N_total]
num_regions: int
```

### 6.5 Token Gating Score & Mask（单模态）【必须按此实现】

输入：
- token_feat x [B,T,N,C]（这里 N=token 数，单模态即 N=Nv）
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

3) 区域重要性：
- region_importance[b,r] = mean over time of H_space_fused[b,t]
  - 简化实现：region_importance = H_space_fused.mean(dim=1).unsqueeze(-1) broadcast 到 R
  - 更合理：可把空间熵按区域定义，但为简化可用全局时间均值映射到所有区域同值（至少可跑）
  - 若要更有意义：可以用“区域内 token 的时间熵均值”：
    ```python
    region_importance[b,r] = mean_{n in region r} H_time_fused[b,n]
    ```
  推荐用后者（更贴合 region 概念）。

4) score 计算（输出 [B,T,N]）：
```python
a = cfg.score_weights.a_time
b = cfg.score_weights.b_space
c = cfg.score_weights.c_region

# broadcast
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
- cfg.score_weights 里包含 d_modal
- self.modality_logit: learnable [M]（M=模态数）

步骤（必须按此实现）：

1) 拆模态 token：
```python
sv, ev = modality_slices["video"]
sa, ea = modality_slices["audio"]
xv = x[:,:,sv:ev,:]     # [B,T,Nv,C]
xa = x[:,:,sa:ea,:]     # [B,T,Na,C], Na=1
```

2) 分别计算熵：
- 视频：按 6.2 得 H_time_video [B,Nv]；按 6.3 得 H_space_video [B,T]
- 音频：
  - 时间熵：H_time_audio [B,Na]（按 6.2，但 N=Na）
  - 空间熵：对音频定义为其时间熵的广播：  
    `H_space_audio[b,t] = H_time_audio_fused[b,0]`（或 0，二者取其一；推荐广播以保持可比）

3) 构造全局 fused：
- H_time_token [B,T,N_total]：
  - video token：使用 H_time_video_fused broadcast 到 time
  - audio token：使用 H_time_audio_fused broadcast 到 time
- H_space_token [B,T,N_total]：
  - video token：使用 H_space_video_fused broadcast 到 N
  - audio token：使用 H_space_audio broadcast 到 N_a

4) 模态权重项：
```python
w_modal = torch.sigmoid(self.modality_logit)   # [M]
modal_id_for_token: LongTensor [N_total]       # video token->0, audio token->1
Wm = w_modal[modal_id_for_token].view(1,1,N_total).expand(B,T,N_total)
```

5) region term：
- region_ids 长度 N_total，audio token region_id 固定 0
- region_importance 推荐由 H_time_video_fused 聚合得到（或包含 audio）
- Hr broadcast 同 6.5

6) 最终 score：
```python
score = a*H_time_token + b*H_space_token + c*Hr + d*Wm
```

7) threshold/mask/sparsity_token 同 6.5

8) modal_stats（必须输出）：
对每个模态 slice (s,e)：
```python
modal_stats[name] = {
  "sparsity": float(1.0 - mask_soft[:,:,s:e].mean().item()),
  "H_time_mean": float(H_time_token[:,:,s:e].mean().item()),
  "mask_mean": float(mask_soft[:,:,s:e].mean().item())
}
```

### 6.7 Head / Channel / Block gating + L_AST（必须按此实现）

目的：结构稀疏不仅在 token，还包括 attention head、MLP channel、block（layer）开关。

参数化（建议可学习 logit + sigmoid）：

- head gating：每层每 head 一个 logit
  - head_logit: Parameter [depth, num_heads]
  - mask_head = sigmoid(head_logit) in (0,1)
  - sparsity_head = 1 - mean(mask_head)

- channel gating：每层对 embed_dim 一个 logit（或只对 MLP hidden）
  - ch_logit: Parameter [depth, embed_dim]
  - mask_ch = sigmoid(ch_logit)
  - sparsity_ch = 1 - mean(mask_ch)

- block gating：每层一个 logit
  - block_logit: Parameter [depth]
  - mask_block = sigmoid(block_logit)
  - sparsity_block = 1 - mean(mask_block)

施加方式（至少一种必须真正生效）：
- head gating：在 attention 输出上按 head 加权（实现可简化为对 attention 的 head 维乘 mask）
- channel gating：在 MLP 输出通道乘 mask（或在 token embedding 上乘，至少要影响计算）
- block gating：在 layer 输出乘 mask_block[l]（或用 skip-connect 调整）

L_AST（必须输出到 info_dict）：
```python
L_AST = lambda_token*sparsity_token + lambda_head*sparsity_head + lambda_ch*sparsity_ch + lambda_block*sparsity_block
```

---

## 7. 硬件代理 HW Proxy（hw_proxy/*）

### 7.1 LayerProxyModel（MLP）

输入：layer feature vector  
输出：lat_ms / mem_mb / power_w（可以 3 个 head，也可以 3 个独立模型）

### 7.2 build_layer_features（固定顺序，必须一致）

给定 layer_cfg 与 device_cfg，输出 1D feature（顺序固定）：

```
[ log10(flops+1),
  log10(bytes+1),
  log10(peak_flops+1),
  log10(peak_bw+1),
  one_hot(layer_type),          # layer_type: attn/mlp/norm/other（至少4类）
  embed_dim/1024,
  num_heads/16,
  mlp_ratio/4,
  seq_len/1024,
  precision ]                  # 0 fp32, 1 fp16
```

### 7.3 LayerHwProxy

- predict_layer(layer_cfg, device_cfg) -> {lat_ms, mem_mb, power_w}
- predict_segment(segment_cfg):
  - lat_ms：sum（serial）或 max（balanced）按配置
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
- logits: Parameter [num_slots, num_types+1]（+1 为 empty）
- forward(hard=False) -> alpha [S,T+1]，eff_specs 为 chip_types 各字段的加权和
- chip count 正则：
```python
chip_used_prob = 1.0 - alpha[:, -1]
L_chip = lambda_chip * chip_used_prob.sum()
```

---

## 9. Segment & 计算图重构（mapping/segments.py + mapping/mapping_solver.py）

Segment dataclass 必须至少包含：
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
- mark_fine_splittable_segments：基于 attn_flops_ratio 与 traffic_out_bytes 阈值标记
- try_fine_split_segment：
  - coarse cost vs fine cost（包含 compute/comm）
  - 若 fine 更优：返回子 segments + rewire_meta（ChannelPermute 的 perm/inv_perm）

---

## 10. MappingSolver（mapping/mapping_solver.py）

必须实现：
1) build_cost_matrix：对每 seg/slot 调 proxy 得 lat/mem/power 矩阵
2) estimate_pipeline_latency：
   - serial：按 seg 顺序累加
   - balanced：按 slot 聚合后取 max
3) solve_mapping：
   - baseline greedy
   - greedy_local 局部移动尝试（单 seg relocate 到其它 slot）
   - 可选 fine split 试探
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
- 约束项（penalty）：
  - boundary：||pos|| > wafer_radius 的超出平方
  - overlap：近似圆半径 r_i（由 width/height）若 dist < r_i+r_j 则惩罚
  - comm：sum(traffic_ij * dist_ij)
  - thermal：高 tdp 芯粒距离太近惩罚（如 tdp_i*tdp_j / (dist+eps)）
- optimize_layout=false：使用规则格点/圆环初始化，L_layout=0

---

## 12. L_hw 组合（hw_proxy/hw_loss.py）

compute_hw_loss(segments, chip_slot_config, hw_proxy, mapping_solver, wafer_layout, cfg):

1) alpha/eff_specs = chip_slot_config()
2) L_chip = lambda_chip * sum(1-alpha[:,empty])
3) mapping_result = mapping_solver.solve_mapping(segments, eff_specs, hw_proxy, layout_positions=wafer_layout.pos.detach(), strategy=cfg.mapping_strategy)
4) 得到：
   - total_latency_ms
   - peak_mem_mb
   - total_energy_j（可由 power_w*lat_ms 汇总）
   - total_area_mm2（由 alpha 对 area_mm2 加权并求和）
5) L_area = max(0, total_area_mm2 - area_budget_mm2) * lambda_area
6) L_layout：
   - optimize_layout=true：wafer_layout(...) 返回 layout loss
   - 否则 L_layout=0
7) 总损失：
```python
L_hw = lambda_T*T + lambda_E*E + lambda_mem*mem + L_chip + L_area + L_layout
```
8) 返回 stats（必须写入 metrics/hw_stats）：
- latency_ms, energy_j, peak_mem_mb, chip_count_expect, area_mm2, comm_ms, layout_penalties 等

---

## 13. Trainer（trainer/*）

### 13.1 单卡训练（trainer_single_device.py）

- 根据 use_audio 选择 VideoViT 或 VideoAudioAST
- 每 batch：
  - L_task = CE(logits,y)
  - L_AST = info["L_AST"]
  - 若 hw.mode==single_device 且 use_hw_loss：计算 L_hw_single_device
  - loss = L_task + lambda_AST*L_AST + lambda_hw*L_hw
- 输出 metrics.json：acc/loss/sparsity/L_hw 分解

### 13.2 Version-C 交替优化（trainer_version_c.py）

外循环（必须实现，可配置 inner steps）：
A) 更新 θ 与稀疏结构 s（AST gating）：多 inner steps，loss 包含 multi-chip L_hw  
B) 冻结 θ，更新 chip_slot_config.logits：最小化 L_hw  
C) 更新离散 mapping：solve_mapping（缓存用于 A/B）  
D) 可选布局优化：optimize_layout=true 时更新 wafer_layout.pos（多步）

---

# PART-II：晶圆级布局 Layout Agent（L0~L7）+ LLM Pick-ID 超启发增强（完整不省略）

## 14. Layout Agent 运行入口与输出

入口：scripts/run_layout_agent.py  
必选参数：
- --layout_input <json>（如 outputs/P3/A3/layout_input.json）
- --cfg configs/layout_agent/xxx.yaml
- --out_dir <dir>
可选：
- --seed <int>（默认 0）

out_dir 输出必须包含：
1) trace.csv（每 step 一行，至少包含以下列，顺序建议固定）：
   - iter
   - stage
   - op
   - op_args_json
   - accepted
   - total_scalar
   - comm_norm
   - therm_norm
   - pareto_added
   - duplicate_penalty
   - boundary_penalty
   - seed_id
   - time_ms
   - signature              # NEW 必须
   - d_total                # NEW 必须
   - d_comm                 # 建议
   - d_therm                # 建议

2) report.json（汇总）：
   - best_total/best_comm/best_therm
   - last_total/...
   - pareto_front_size
   - accept_rate_overall / accept_rate_lastN
   - runtime_s / evaluate_calls
   - oscillation metrics（第17节定义的字段，必须有）

3) llm_usage.jsonl：
   - 每次 LLM 调用写一行 JSON
   - 必须字段：ok, n_pick, pick_ids, error, raw_preview, prompt_tokens, completion_tokens, n_queue_push, picked_types, best_d_total

---

## 15. 震荡治理机制（必须实现且可消融）

定义“震荡/回跳”：
- undo（逆操作）与 repeat（重复 signature）是可量化的震荡表现（第17节给出严格计算）

治理机制（必须落地到代码）：

1) Tabu/forbidden（短期禁忌）：
- 维护 recent_action_signatures = deque(maxlen=Ksig, 默认 12)
- 候选池生成时：
  - 若 candidate.signature 在 recent_action_signatures：默认 **过滤**（ablation 可改为加罚）
- 维护 forbidden_ids（最近一次 LLM pick 的 id 集合）用于下一次 LLM 避免重复

2) Queue + refresh：
- llm_action_queue: FIFO actions
- recent_reject_count：若队列动作连续被拒绝累计 >= 6，则 force_refresh=True
- 若 queue 为空且未到 mixed_every 步，也允许强制触发一次 LLM（避免停滞）

3) 接受判据 hysteresis（防抖）：
- 若 SA 温度已低（T < T_low）且动作使 total_scalar 变差超过 tiny_eps（如 1e-4）：直接拒绝
- 对“几乎不变”的动作（|d_total|<eps 且 |d_comm|<eps 且 |d_therm|<eps）：直接过滤（不进入候选）

4) 多样性配额（防止 swap 模板化）：
- pack 时强制最小配额（见 candidate_pool.pack_diverse_candidates）

5) 二次可行性检查（避免 relocate 冲突导致全拒）：
- LLM pick 的 IDs 必须经 sequential feasibility check 过滤成可执行 actions（见 candidate_pool.pick_ids_to_actions_sequential）

---

## 16. L4：候选池 + LLM 只 pick candidate ID（完整要求 + 代码骨架）

### 16.1 新增文件：layout/candidate_pool.py（必须实现，按此骨架扩展）

要求：
- 候选 Candidate 必须包含：id, action, type, desc, est(单步试算), score, signature
- 生成 raw_candidates 150~220，过滤后 pack 到 final 60（默认）
- 必须做单步试算 evaluator.evaluate(new_assign)
- 必须做多样性配额（swap/relocate/cluster_move/explore）
- 必须做去重（signature）与限制“同一slot relocate数量”（<=6）
- 必须提供“按 pick 顺序 sequential feasibility check”函数：把 pick 映射成 actions，依次 apply，发现 duplicate/boundary/站点冲突就丢掉该条
- cluster_move 候选也必须有 est（因此要能“试算 cluster_move 后的 new_assign 并 evaluate”）

文件骨架（可直接拷贝实现）：

--- layout/candidate_pool.py ---
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class Candidate:
    id: int
    action: Dict
    type: str              # "swap"|"relocate"|"cluster_move"|"explore"
    desc: str              # <= 60 chars
    est: Dict              # {"total_new","comm_new","therm_new","d_total","d_comm","d_therm","pen_dup","pen_bnd"}
    score: float           # lower is better
    signature: str         # "swap:3-9", "rel:8->25", "cl:2->7"

def action_signature(action: Dict) -> str:
    op = action.get("op")
    if op == "swap":
        i, j = int(action["i"]), int(action["j"])
        a, b = (i, j) if i < j else (j, i)
        return f"swap:{a}-{b}"
    if op == "relocate":
        i, site_id = int(action["i"]), int(action["site_id"])
        return f"rel:{i}->{site_id}"
    if op == "cluster_move":
        cid, rid = int(action["cluster_id"]), int(action["region_id"])
        return f"cl:{cid}->{rid}"
    return f"exp:{op}"

def apply_action(assign: np.ndarray, action: Dict) -> np.ndarray:
    new_assign = assign.copy()
    op = action.get("op")
    if op == "swap":
        i, j = int(action["i"]), int(action["j"])
        new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
    elif op == "relocate":
        i, site_id = int(action["i"]), int(action["site_id"])
        new_assign[i] = site_id
    else:
        # cluster_move is handled in apply_cluster_move()
        pass
    return new_assign

def apply_cluster_move(assign: np.ndarray, cluster_id: int, region_id: int,
                       clusters: List[List[int]],
                       region_sites: List[List[int]]) -> Optional[np.ndarray]:
    """
    将一个 cluster 的所有 slot 搬到 region_id 里的空位上：
    - 要求 region 内空位数量 >= cluster_size
    - 选择空位策略：优先距离当前 cluster 质心最近（或随机但可复现）
    返回 new_assign；若不可行返回 None。
    """
    new_assign = assign.copy()
    slots = clusters[cluster_id]
    # 当前已占 sites
    used = set(int(x) for x in new_assign.tolist())
    empties = [sid for sid in region_sites[region_id] if sid not in used]
    if len(empties) < len(slots):
        return None
    # 简化：按 empties 前 len(slots) 个分配
    for k, slot in enumerate(slots):
        new_assign[int(slot)] = int(empties[k])
    return new_assign

def inside_wafer(site_xy: np.ndarray, wafer_radius_mm: float) -> bool:
    return float(np.linalg.norm(site_xy)) <= float(wafer_radius_mm) + 1e-9

def _score_candidate(d_total_z: float, d_comm_z: float, d_therm_z: float,
                     cand_type: str, touch_hot: bool, extra_bonus: float = 0.0) -> float:
    # lower is better; bias toward total
    wT, wC, wH = 0.70, 0.20, 0.10
    score = wT * d_total_z + wC * d_comm_z + wH * d_therm_z
    # diversity bonus
    if cand_type == "relocate":
        score -= 0.03
    elif cand_type == "cluster_move":
        score -= 0.02
    if touch_hot:
        score -= 0.02
    score -= float(extra_bonus)
    return float(score)

def _z_norm(x: float, scale: float) -> float:
    # robust normalization; scale should be >0
    return float(x) / float(scale if scale > 1e-9 else 1.0)

def _eval_make_candidate(
    *,
    cand_id: int,
    assign: np.ndarray,
    action: Dict,
    cand_type: str,
    evaluator,
    layout_state,
    sites_xy: np.ndarray,
    wafer_radius_mm: float,
    clusters,
    region_sites,
    cur_eval: Dict,
    scale_total: float,
    scale_comm: float,
    scale_therm: float,
    touch_hot: bool,
) -> Optional[Candidate]:
    # 1) apply
    op = action.get("op")
    if op == "cluster_move":
        cid, rid = int(action["cluster_id"]), int(action["region_id"])
        new_assign = apply_cluster_move(assign, cid, rid, clusters, region_sites)
        if new_assign is None:
            return None
    else:
        new_assign = apply_action(assign, action)

    # 2) boundary check (site inside wafer)
    # relocate / cluster_move must ensure new sites are inside wafer
    for sid in set(int(x) for x in new_assign.tolist()):
        if not inside_wafer(sites_xy[sid], wafer_radius_mm):
            return None

    # 3) duplicate check (injective assignment)
    if len(set(int(x) for x in new_assign.tolist())) != int(new_assign.shape[0]):
        return None

    # 4) one-step evaluate
    layout_state.assign = new_assign
    new_eval = evaluator.evaluate(layout_state)  # must return dict with total_scalar/comm_norm/therm_norm + penalties

    # penalties must be present (or default 0)
    pen_dup = float(new_eval.get("duplicate_penalty", 0.0))
    pen_bnd = float(new_eval.get("boundary_penalty", 0.0))
    if pen_dup > 0.0 or pen_bnd > 0.0:
        return None

    total_new = float(new_eval["total_scalar"])
    comm_new  = float(new_eval.get("comm_norm", 0.0))
    therm_new = float(new_eval.get("therm_norm", 0.0))

    total_cur = float(cur_eval["total_scalar"])
    comm_cur  = float(cur_eval.get("comm_norm", 0.0))
    therm_cur = float(cur_eval.get("therm_norm", 0.0))

    d_total = total_new - total_cur
    d_comm  = comm_new  - comm_cur
    d_therm = therm_new - therm_cur

    # z-normalize deltas for scoring
    d_total_z = _z_norm(d_total, scale_total)
    d_comm_z  = _z_norm(d_comm,  scale_comm)
    d_therm_z = _z_norm(d_therm, scale_therm)

    score = _score_candidate(d_total_z, d_comm_z, d_therm_z, cand_type, touch_hot)

    sig = action_signature(action)
    desc = sig
    if len(desc) > 60:
        desc = desc[:60]

    return Candidate(
        id=cand_id,
        action=action,
        type=cand_type,
        desc=desc,
        est={
            "total_new": total_new,
            "comm_new": comm_new,
            "therm_new": therm_new,
            "d_total": d_total,
            "d_comm": d_comm,
            "d_therm": d_therm,
            "pen_dup": pen_dup,
            "pen_bnd": pen_bnd,
        },
        score=float(score),
        signature=sig,
    )

def pack_diverse_candidates(cands: List[Candidate], final_n: int = 60,
                            min_swap: int = 18, min_reloc: int = 18,
                            min_cluster: int = 8, min_explore: int = 6) -> List[Candidate]:
    # sort by score ascending
    cands = sorted(cands, key=lambda x: x.score)
    buckets = {"swap": [], "relocate": [], "cluster_move": [], "explore": []}
    for c in cands:
        buckets[c.type].append(c)
    out = []
    def take(tp, k):
        nonlocal out
        out.extend(buckets[tp][:k])
        buckets[tp] = buckets[tp][k:]
    take("swap", min_swap)
    take("relocate", min_reloc)
    take("cluster_move", min_cluster)
    take("explore", min_explore)
    # fill remaining
    rest = buckets["swap"] + buckets["relocate"] + buckets["cluster_move"] + buckets["explore"]
    rest = sorted(rest, key=lambda x: x.score)
    out.extend(rest[: max(0, final_n - len(out))])
    # unique by id
    seen=set()
    final=[]
    for c in out:
        if c.id in seen: continue
        seen.add(c.id)
        final.append(c)
        if len(final) >= final_n: break
    return final

def build_state_summary_for_llm(
    *,
    step: int,
    K: int,
    cur_eval: Dict,
    candidates: List[Candidate],
    candidate_ids: List[int],
    forbidden_ids: List[int],
) -> Dict:
    # Keep it compact (NO huge matrices)
    # Provide only what LLM needs
    cand_rows=[]
    for c in candidates:
        cand_rows.append({
            "id": int(c.id),
            "type": c.type,
            "d_total": float(c.est["d_total"]),
            "d_comm": float(c.est["d_comm"]),
            "d_therm": float(c.est["d_therm"]),
            "desc": c.desc,
        })
    return {
        "step": int(step),
        "K": int(K),
        "cur_total": float(cur_eval["total_scalar"]),
        "cur_comm": float(cur_eval.get("comm_norm", 0.0)),
        "cur_therm": float(cur_eval.get("therm_norm", 0.0)),
        "candidate_ids": [int(x) for x in candidate_ids],
        "forbidden_ids": [int(x) for x in forbidden_ids],
        "candidates": cand_rows,
    }

def pick_ids_to_actions_sequential(
    *,
    pick_ids: List[int],
    candidates_by_id: Dict[int, Candidate],
    assign: np.ndarray,
    sites_xy: np.ndarray,
    wafer_radius_mm: float,
) -> List[Dict]:
    """
    二次可行性检查：按 pick 顺序逐个 apply，发现冲突就跳过该条。
    - relocate 必须落 empty site
    - 不能造成 duplicate site
    - boundary: site 必须在 wafer 内
    返回通过检查的 actions（保持原顺序）。
    """
    cur = assign.copy()
    S = int(cur.shape[0])
    actions=[]
    for pid in pick_ids:
        if int(pid) not in candidates_by_id:
            continue
        cand = candidates_by_id[int(pid)]
        act = cand.action
        op = act.get("op")
        # apply tentative
        if op == "swap":
            i, j = int(act["i"]), int(act["j"])
            if not (0 <= i < S and 0 <= j < S):
                continue
            new = cur.copy()
            new[i], new[j] = new[j], new[i]
        elif op == "relocate":
            i, sid = int(act["i"]), int(act["site_id"])
            if not (0 <= i < S and 0 <= sid < sites_xy.shape[0]):
                continue
            # must relocate to empty
            used = set(int(x) for x in cur.tolist())
            if sid in used:
                continue
            if not inside_wafer(sites_xy[sid], wafer_radius_mm):
                continue
            new = cur.copy()
            new[i] = sid
        else:
            # cluster_move feasibility handled earlier in candidate build; still keep simple guard
            actions.append(act)
            continue

        # duplicate check
        if len(set(int(x) for x in new.tolist())) != S:
            continue
        cur = new
        actions.append(act)
    return actions
```

### 16.2 修改 layout/detailed_place.py（必须按此接入候选池+queue）

必须做的改动点：

1) import：
```python
from layout.candidate_pool import (
  build_state_summary_for_llm,
  pack_diverse_candidates,
  pick_ids_to_actions_sequential,
  Candidate,
  action_signature,
  _eval_make_candidate,  # 若你拆成私有函数可不 import，但必须实现等价逻辑
)
```

2) run_detailed_place 主循环：把原 “LLM 直接产出 actions” 改为：
- 每次触发 LLM 前，先 build candidates（见 16.1 的 build 逻辑：raw 150~220 -> filter -> pack 60）
- state_summary = build_state_summary_for_llm(...)
- pick_ids = planner.propose_pick(state_summary, K)
- actions = pick_ids_to_actions_sequential(...)
- push actions into llm_action_queue（FIFO）
- 每 step 优先 pop queue 的 action；queue 为空才 fallback to heuristic random action

3) 队列刷新与禁忌：
- llm_action_queue: List[Dict] = []
- recent_action_signatures: deque(maxlen=12)
- forbidden_ids: List[int]（最近一次 pick 的 ids，下一次给 LLM 避免重复）
- recent_reject_count: int
- force_refresh: bool

逻辑：
- 若 queue 为空：force_refresh=True
- 若连续拒绝 >= 6：force_refresh=True
- 若 step % mixed_every == 0 或 force_refresh=True：触发一次 LLM（生成 queue），并将 forbidden_ids 更新为 pick_ids 的前若干（例如前 min(6,K) 个）
- actions 被 accept 后，把 signature 追加到 recent_action_signatures

4) trace.csv 写入必须单参 write 或 csv.writer（见第18节也要求）
- 并且必须写 signature、d_total（以及可选 d_comm/d_therm）

5) llm_usage.jsonl 增强字段（写入 last_usage 前附加）：
- n_queue_push = len(actions)
- picked_ids = pick_ids
- picked_types = {"swap":x,"relocate":y,"cluster_move":z,"explore":w}
- best_d_total = min(d_total among picked ids present) 或 None

### 16.3 修改 layout/llm_provider.py（LLM 只输出 pick IDs）

接口改为：
```python
class LLMProvider:
    def propose_pick(self, state_summary: Dict, k: int) -> List[int]:
        ...
```

HeuristicProvider：
- 不调用 LLM：从 state_summary["candidates"] 按 d_total 最小排序取前 k，并保证至少 1 个非 swap（若存在）。

VolcArkProvider（或任意 LLM provider）：
- 严格 3 行 wrapper 输出：
  1) BEGIN_JSON
  2) {"pick":[1,2,3]}
  3) END_JSON

解析流程（必须实现）：
- extract_wrapped_json（BEGIN_JSON/END_JSON 中间）
- recover_json：从 raw text 里用正则提取 {"pick":[...]} 或仅提取列表 [...]
- repair：将列表包装成 {"pick":[...]}
- validate：
  - 必须是 int
  - 必须在 candidate_ids 内
  - 必须不在 forbidden_ids
  - unique
  - 最多 k 个
- 若最终为空，返回 []

last_usage（必须包含）：
```json
{
  "ok": true/false,
  "n_pick": 3,
  "pick_ids": [..],
  "error": "...",
  "raw_preview": "...(<=300 chars)",
  "status_code": ...,
  "prompt_tokens": ...,
  "completion_tokens": ...,
  "latency_ms": ...
}
```

LLM 参数建议：
- max_completion_tokens: 96~128
- temperature: 0.1
- top_p: 0.9
- stop: ["END_JSON"]

### 16.4 Prompt（逐字拷贝，必须一致）

system_prompt（逐字拷贝）：

```
STRICT PICK MODE.
You MUST output EXACTLY 3 lines and NOTHING ELSE.
LINE1: BEGIN_JSON
LINE2: {"pick":[ID1,ID2,...]}
LINE3: END_JSON

HARD RULES:
- The FIRST characters MUST be "BEGIN_JSON".
- Output JSON must have ONLY key "pick".
- Each ID must be an integer.
- IDs MUST be chosen ONLY from candidate_ids.
- IDs MUST NOT appear in forbidden_ids.
- IDs must be UNIQUE. Max K IDs.
- If you cannot comply, output {"pick":[]}.

OPTIMIZATION GOAL:
- Prefer IDs with more negative d_total (best improvement).
- Secondary: improve d_comm and d_therm (more negative is better).
- DIVERSITY: If candidates include relocate/cluster_move, ensure at least ONE picked ID is NOT swap.

ANTI-TEMPLATE:
- Do NOT always pick the smallest IDs.
- Do NOT always pick the same pattern (e.g., 0-1,1-2,2-3 swaps).
- Use the provided candidate list only.
```

user_content 模板（必须生成紧凑 JSON，不含大矩阵）：

```
K=<k>
STATE_JSON=<json.dumps(state_summary,separators=(",",":"))>
REMINDER:
- Output ONLY 3 lines wrapper.
- Use ONLY candidate_ids and avoid forbidden_ids.
- Pick up to K IDs, unique.
```

---

## 17. Layout 震荡指标（必须写入 report.json，定义严格、可直接实现）

### 17.1 必须写入 report.json 的字段（最少集合）

report.json 必须包含（至少）：

- `accept_rate_overall`：accepted_steps / total_steps
- `accept_rate_lastN`：后 N 步接受率（N 默认 200，可配置）
- `repeat_signature_rate`：动作重复率（定义见 17.3）
- `undo_rate`：回跳率（定义见 17.4）
- `objective_variance`：目标函数波动方差（定义见 17.5）
- `objective_std_lastN`：后 N 步目标标准差
- `oscillation_rate`：综合震荡率（定义见 17.6）
- `improve_step_ratio`：有效改进步比例（accepted 且 d_total < 0）
- `flat_step_ratio`：平坦步比例（|d_total|,|d_comm|,|d_therm| 都很小）

此外建议写入：
- `best_total/best_comm/best_therm`
- `last_total/...`
- `pareto_front_size`
- `runtime_s`
- `evaluate_calls`
- `seed`

### 17.2 signature 规范（必须统一，用于 repeat/undo/禁忌）

每一步 action 必须生成 signature（字符串），并写入 trace.csv：

- swap：`swap:{min(i,j)}-{max(i,j)}`
- relocate：`rel:{i}->{site_id}`
- cluster_move：`cl:{cluster_id}->{region_id}`
- explore（如有）：`exp:{...}`（必须稳定、可重复）

### 17.3 repeat_signature_rate（重复动作率）定义（必须按此实现）

给定 steps=0..T-1 的 signature 序列 sig[t]（无动作则 sig[t]=""）：

**窗口版（推荐用于论文）**：在最近 W 步（W 默认 200，可配 `metrics_window_lastN`）中计算：

- 令窗口 index 集合为 t in [T-W, T-1]（clip 到有效范围）
- 定义重复：sig[t] 在窗口内此前出现过（t' < t 且 sig[t']==sig[t] 且 sig[t]非空）
- repeat_count = 这样的 t 的数量
- repeat_signature_rate = repeat_count / max(1, window_len)

实现参考：
```python
seen=set()
repeat=0
for t in window_indices:
    s = sig[t]
    if not s: 
        continue
    if s in seen:
        repeat += 1
    else:
        seen.add(s)
rate = repeat / max(1, len(window_indices))
```

同时建议在 report.json 里也写全程版本：
- `repeat_signature_rate_overall`：对全程 t=0..T-1 计算同样逻辑

### 17.4 undo_rate（回跳率）定义（必须按此实现）

undo 只看相邻两步 (t-1, t) 的动作是否互为“逆操作”。

必须支持三类：

**(1) swap undo**
- 若 sig[t] == sig[t-1] 且 sig[t] 以 "swap:" 开头，则 (t-1,t) 记为 undo

**(2) relocate undo**
- 需要记录“slot i 在前一步的 site”。
- 若 step t-1 是 relocate：rel:i->A
- step t   是 relocate：rel:i->B
- 且 B == site_of_slot_i_before_step_(t-1)（即撤销回原 site），则记为 undo

实现方式：
- 在迭代过程中，维护 `prev_assign`（执行动作前的 assign），以及 `cur_assign`
- 对每步 t：
  - 解析动作类型与参数
  - 判断 undo 时需要用到：
    - `assign_before_prev`（即 t-1 动作执行前的 assign）
    - `assign_before_curr`（即 t 动作执行前的 assign）
  - 推荐在写 trace 时也写入 `slot_site_before`（可选），但不是必须；最简单是报告阶段回放 trace + checkpoint 的 assign 序列（见 18.2）。

**(3) cluster_move undo（可选但建议实现）**
- 若 step t-1 是 cl:cid->R1，step t 是 cl:cid->R0，其中 R0 是 t-1 之前 cluster 所在 region，则记为 undo
- 需要 cluster->region 的映射：可由 cluster 内任一 slot 的 site_to_region[assign[slot]] 得到（如果 cluster 内混 region，则取多数或报错并跳过 undo 判定）

undo_rate 定义：
- undo_count = 满足 undo 的相邻对数量
- undo_rate = undo_count / max(1, window_len-1)（窗口内相邻对数）

同样建议写全程版本：
- `undo_rate_overall`

### 17.5 objective_variance（目标方差）定义（必须按此实现）

目标函数序列：obj[t] = trace.csv 里的 total_scalar（每步都写）

计算两种（建议都写入 report.json）：

1) 全程方差：
```python
objective_variance = var(obj[0:T])
```

2) 后 N 步方差与标准差：
```python
obj_last = obj[max(0,T-W):T]
objective_variance_lastN = var(obj_last)
objective_std_lastN = std(obj_last)
```

> 论文里一般更关心 lastN 的稳定性，所以 `objective_std_lastN` 必须有。

另外建议写“归一化方差”（可选）：
- `objective_cv_lastN = std_lastN / (abs(mean_lastN)+1e-9)`

### 17.6 oscillation_rate（综合震荡率）定义（必须按此实现）

oscillation_rate 需要把 “重复/回跳/波动” 合成一个 0~1 的指标，要求**可解释、可复现**。

推荐定义（必须实现，权重可配置但默认如下）：

- 先计算窗口指标：
  - r_rep = repeat_signature_rate_lastN
  - r_undo = undo_rate_lastN
  - v = objective_std_lastN
- 归一化波动项：
  - v_norm = min(1.0, v / (abs(best_total_lastN)+1e-9))  
    - best_total_lastN = min(obj_last)
- 综合：
```python
oscillation_rate = 0.4*r_undo + 0.4*r_rep + 0.2*v_norm
```

要求：
- 将 r_rep、r_undo、v_norm、best_total_lastN、mean_lastN 一并写入 report.json，保证可复核。

### 17.7 improve_step_ratio & flat_step_ratio（必须实现）

从 trace.csv 读取 d_total/d_comm/d_therm（若没写 d_comm/d_therm，可用 new-old 算）：

- improve_step_ratio（窗口版）：
  - 统计 accepted 且 d_total < 0 的步数 / window_len

- flat_step_ratio：
  - 定义 eps_flat（默认 1e-4，可配置）
  - 若 |d_total|<eps_flat 且 |d_comm|<eps_flat 且 |d_therm|<eps_flat，则记为 flat
  - flat_step_ratio = flat_count / window_len

---

## 18. 工程鲁棒性（必须实现）

### 18.1 trace.csv 写入修复（必须）

禁止：
- `f.write(a,b,c)`（会触发 TypeError）

必须使用：
- `csv.writer.writerow([...])`
或
- `f.write(f"{a},{b},{c}\n")`

并要求：
- 每 trace_flush_every（默认 20）行 flush：`f.flush()`
- 每 progress_every（默认 10）步 print 心跳
- 每 save_every（默认 50）步写 checkpoint_state.json（覆盖写）

### 18.2 heartbeat.json / checkpoint_state.json（必须）

每 progress_every steps 写 out_dir/heartbeat.json（覆盖写）：
```json
{
  "step": 120,
  "elapsed_s": 83.2,
  "cur_total": 0.9231,
  "best_total": 0.9012,
  "accept_rate": 0.31,
  "queue_len": 3,
  "last_op": "relocate",
  "temperature": 0.42
}
```

每 save_every steps 写 out_dir/checkpoint_state.json（覆盖写）：
```json
{
  "step": 120,
  "assign": [...],
  "best_assign": [...],
  "best_eval": {...},
  "cur_eval": {...},
  "temperature": 0.42,
  "recent_signatures": ["swap:3-9", ...]
}
```

---

# PART-III：论文级实验设计（主表 + 消融 + 运行脚本）

## 19. 实验输出统一口径（写论文用）

所有训练类实验输出：
- metrics.json：top1/acc、loss、sparsity_token/head/ch/block、L_AST、L_hw 分解
- hw_stats.json：lat/energy/mem/area/chip_count/mapping 摘要

所有 layout 类实验输出：
- trace.csv
- report.json（必须含第17节震荡指标）
- llm_usage.jsonl（包含 ok/n_pick/pick_ids/...）

随机性：
- 训练类默认 seeds=[0,1,2]
- layout 类默认 seeds=[0,1,2,3,4]

---

## 20. 论文实验清单（必须实现 EXP-ID，可按脚本逐条跑）

### 20.1 创新 A：面向多芯粒的时空熵剪枝（必须补齐）

A-Core（主表推荐）：
- EXP-A1：Dense baseline（single device）
- EXP-A2：AST only（no hw loss）
- EXP-A3：AST + single-device hw loss
- EXP-A4：Ours multi-chip aware（Version-C full：AST + L_hw(multi-chip) + mapping(+layout 可选) 交替）
- EXP-A5：Two-stage baseline（先 single-device AST+hw 训练好，再固定剪枝去做 multi-chip mapping/layout）
- EXP-A6：Mapping-only（剪枝固定，只优化 mapping）
- EXP-A7：Layout-only（剪枝固定，只优化 layout）

A-Ablation（消融表，至少这些）：
- EXP-Abl-time：去 time entropy（a_time=0 或直接返回 0）
- EXP-Abl-space：去 space entropy（b_space=0 或直接返回 0）
- EXP-Abl-vor：去 Voronoi（c_region=0 且 region_importance 不参与）
- EXP-Abl-1lvl：levels=[1]（去 multi-level）
- EXP-Abl-nomodal：去 modality weight（d_modal=0 且 modality_logit 不学/固定）
- EXP-Abl-uniform：uniform keep（不看 score，固定均匀 idx 保留）
- EXP-Abl-random：random keep（同 rho_target）

输出指标（必须在 metrics/hw_stats 中齐全，方便拉表）：
- top1/acc
- sparsity_token/head/ch/block
- latency_ms / energy_j / peak_mem_mb / chip_count_expect / area_mm2
- pareto（若做多 seed，可输出 mean±std）

### 20.2 创新 B：超启发式晶圆布局（候选池 + LLM pick-ID）（必须补齐）

B-Core（主表推荐）：
- EXP-B1：Heuristic only（candidate pool + heuristic pick，不调用 LLM）
- EXP-B2：LLM Pick-ID enhanced（完整版：diversity + feasibility + queue + refresh）
- EXP-B3：SA baseline（L3：纯 SA/局部搜索，无候选池/LLM）
- EXP-B4：（可选）LLM Direct actions（若旧版仍在，用来证明 pick-ID 更稳）

B-Ablation（至少三项，必须可跑完并出 report）：
- EXP-B2-ab-noqueue：去队列/refresh
- EXP-B2-ab-nofeas：去 sequential feasibility check
- EXP-B2-ab-nodiverse：去多样性配额（允许 swap 占满）

布局指标（report.json 必须包含）：
- best_total / best_comm / best_therm（或 norm）
- pareto_front_size
- accept_rate_overall / accept_rate_lastN
- repeat_signature_rate / undo_rate / objective_variance / objective_std_lastN / oscillation_rate
- runtime_s / evaluate_calls

---

## 21. 两份指令文件（必须生成，Smoke 与 Experiments 分离）

必须生成并可执行（chmod +x）：

1) scripts/smoke_tests_version_c.sh  
   只验证“能跑通”，用 smoke cfg 或短步数，覆盖：
   - proxy 两脚本
   - ast 单卡
   - version-c
   - layout agent（L0 与 L4 pick），无 key 时必须 fallback 跑完

2) scripts/experiments_version_c.sh  
   真正跑论文数据，要求：
   - 支持：`./scripts/experiments_version_c.sh EXP-A4 0`
   - 第一个参数 EXP_ID，第二个可选 seed（默认 0）
   - 每个 EXP_ID 对应一个 cfg + out_dir（包含 EXP_ID 与 seed）
   - 如脚本支持 --out_dir/--seed CLI，必须保持兼容；否则 seed 可写入 cfg，但脚本形态要保留

---

## 22. 推荐脚本内容（Codex 可直接生成）

### 22.1 scripts/smoke_tests_version_c.sh（示例）

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
python -m scripts.run_layout_agent \
  --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L0_heuristic.yaml \
  --out_dir outputs/SMOKE/layout_L0_heuristic

echo "[SMOKE] Layout agent L4 pick-ID (fallback if no key)"
python -m scripts.run_layout_agent \
  --layout_input outputs/P3/A3/layout_input.json \
  --cfg configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml \
  --out_dir outputs/SMOKE/layout_L4_pick

echo "[SMOKE DONE]"
```

### 22.2 scripts/experiments_version_c.sh（示例）

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
  python -m scripts.run_layout_agent \
    --layout_input outputs/P3/A3/layout_input.json \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

case "$EXP_ID" in
  # --- Innovation A ---
  EXP-A1) run_ast configs/ast2_ucf101_dense.yaml        "outputs/EXP-A1/seed${SEED}" ;;
  EXP-A2) run_ast configs/ast2_ucf101_ast_only.yaml     "outputs/EXP-A2/seed${SEED}" ;;
  EXP-A3) run_ast configs/ast2_ucf101_ast_hw.yaml       "outputs/EXP-A3/seed${SEED}" ;;
  EXP-A4) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A4/seed${SEED}" ;;
  EXP-A5) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A5_twostage/seed${SEED}" ;;  # cfg 内 twostage=true
  EXP-A6) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A6_mappingonly/seed${SEED}" ;; # cfg 内 mapping_only=true
  EXP-A7) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A7_layoutonly/seed${SEED}" ;;  # cfg 内 layout_only=true

  EXP-Abl-time)   run_ast configs/ablations/ast_no_time.yaml   "outputs/EXP-Abl-time/seed${SEED}" ;;
  EXP-Abl-space)  run_ast configs/ablations/ast_no_space.yaml  "outputs/EXP-Abl-space/seed${SEED}" ;;
  EXP-Abl-vor)    run_ast configs/ablations/ast_no_voronoi.yaml"outputs/EXP-Abl-vor/seed${SEED}" ;;
  EXP-Abl-1lvl)   run_ast configs/ablations/ast_level1.yaml    "outputs/EXP-Abl-1lvl/seed${SEED}" ;;
  EXP-Abl-nomodal)run_ast configs/ablations/ast_no_modal.yaml  "outputs/EXP-Abl-nomodal/seed${SEED}" ;;
  EXP-Abl-uniform)run_ast configs/ablations/ast_uniform_keep.yaml "outputs/EXP-Abl-uniform/seed${SEED}" ;;
  EXP-Abl-random) run_ast configs/ablations/ast_random_keep.yaml  "outputs/EXP-Abl-random/seed${SEED}" ;;

  # --- Innovation B ---
  EXP-B1) run_layout configs/layout_agent/layout_L0_heuristic.yaml "outputs/EXP-B1/seed${SEED}" ;;
  EXP-B2) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml "outputs/EXP-B2/seed${SEED}" ;;
  EXP-B3) run_layout configs/layout_agent/layout_L3_region_pareto_sa.yaml "outputs/EXP-B3/seed${SEED}" ;;

  EXP-B2-ab-noqueue)   run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml   "outputs/EXP-B2-ab-noqueue/seed${SEED}" ;;
  EXP-B2-ab-nofeas)    run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml    "outputs/EXP-B2-ab-nofeas/seed${SEED}" ;;
  EXP-B2-ab-nodiverse) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml "outputs/EXP-B2-ab-nodiverse/seed${SEED}" ;;

  *) echo "Unknown EXP_ID=$EXP_ID"; exit 2 ;;
esac
```

> 注：若当前 scripts/run_* 没有 --out_dir / --seed，必须补 CLI 参数并保持默认兼容（不传也能跑）。

---

# END OF SPEC
