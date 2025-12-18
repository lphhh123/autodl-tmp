# gen_pruned_cfg_grid.py
import itertools
import csv

"""
作用：
  在红线约束下，枚举大量 (depth, embed_dim, num_heads, mlp_ratio) 组合，
  写到 pruned_cfg_grid.csv，后续用来生成剪枝数据库和训练代理模型。

这里把结构搜索空间适当收紧，靠近 ViT-Huge：
  - baseline: d=32, e=1280, r=4.0
  - depth:    [16, 24, 32]
  - embed:    [896, 1024, 1152, 1280]
  - heads:    [12, 16]
  - mlp:      [3.0, 3.5, 4.0]

你如果觉得不够丰富，可以自己把这些列表扩一下。
"""

# ====== 基线 ViT-H，用来算剪枝强度 ======
BASE_D = 32
BASE_E = 1280
BASE_R = 4.0

# ====== 结构搜索空间（已压缩） ======
DEPTH_LIST = [16, 24, 32]
EMBED_LIST = [896, 1024, 1152, 1280]
HEADS_LIST = [12, 16]
MLP_LIST   = [3.0, 3.5, 4.0]

# ====== 模型并行相关约束 ======
TP_WORLD_SIZE = 4       # 张量并行份数
HEAD_DIM_ALIGN = 16     # head_dim 对齐 16，利好 Tensor Core

def complexity_ratio(d, e, r):
    """粗略复杂度 ~ d * e^2 * r，相对 ViT-H 的比值"""
    base = BASE_D * (BASE_E ** 2) * BASE_R
    cur  = d * (e ** 2) * r
    return cur / base

def valid_combo(d, e, h, r,
                min_ratio=0.25,
                max_ratio=1.1):
    # 1) 维度关系：d_model 可被 heads 整除
    if e % h != 0:
        return False
    head_dim = e // h

    # 2) 每头维度对齐 Tensor Core
    if head_dim % HEAD_DIM_ALIGN != 0:
        return False

    # 3) d_model 能被张量并行份数切平
    if e % TP_WORLD_SIZE != 0:
        return False

    # 4) 剪枝强度范围
    ratio = complexity_ratio(d, e, r)
    if not (min_ratio <= ratio <= max_ratio):
        return False

    return True

def fmt_ratio(r: float):
    """把 3.5 变成 '3_5'，用于命名"""
    ra = int(r)
    rb = int(round((r - ra) * 10))
    return f"{ra}_{rb}"

def main():
    cfgs = []
    for d, e, h, r in itertools.product(DEPTH_LIST, EMBED_LIST, HEADS_LIST, MLP_LIST):
        if not valid_combo(d, e, h, r):
            continue
        ratio = complexity_ratio(d, e, r)
        tag = "gridH"
        r_str = fmt_ratio(r)
        name = f"vith_{tag}_d{d}_e{e}_h{h}_r{r_str}"
        head_dim = e // h
        cfgs.append(dict(
            cfg=name,
            depth=d,
            embed_dim=e,
            num_heads=h,
            mlp_ratio=r,
            complexity_ratio=ratio,
            head_dim=head_dim,
            tp_world_size=TP_WORLD_SIZE,
        ))

    cfgs.sort(key=lambda x: x["complexity_ratio"], reverse=True)

    out_csv = "pruned_cfg_grid.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cfg", "depth", "embed_dim", "num_heads", "mlp_ratio",
                "complexity_ratio", "head_dim", "tp_world_size"
            ]
        )
        w.writeheader()
        w.writerows(cfgs)

    print(f"[gen] total valid cfgs: {len(cfgs)}")
    print(f"[gen] written to {out_csv}")
    for x in cfgs[:10]:
        print(x)

if __name__ == "__main__":
    main()
