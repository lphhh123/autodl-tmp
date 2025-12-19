"""Generate EXPERIMENTS_VERSION_C.md with commands per SPEC_version_c_full."""
from __future__ import annotations

import argparse
from pathlib import Path

EXPERIMENTS = [
    {
        "id": "EXP-P0-1-ms-mem",
        "purpose": "验证 layerwise 时延+显存 proxy",
        "config": "configs/proxy_ms_mem.yaml",
        "command": "python -m scripts.run_proxy_ms_mem --cfg configs/proxy_ms_mem.yaml",
    },
    {
        "id": "EXP-P0-2-power",
        "purpose": "验证 layerwise 功耗 proxy",
        "config": "configs/proxy_power.yaml",
        "command": "python -m scripts.run_proxy_power --cfg configs/proxy_power.yaml",
    },
    {
        "id": "EXP-P1V-1-dense-multiscale",
        "purpose": "单模态视频 dense baseline（多尺度滑窗）",
        "config": "configs/ast2_ucf101_dense.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_dense.yaml",
    },
    {
        "id": "EXP-P1V-2-dense-noscale",
        "purpose": "单模态视频 dense baseline（无多尺度滑窗）",
        "config": "configs/ast2_ucf101_dense_noscale.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_dense_noscale.yaml",
    },
    {
        "id": "EXP-P1V-3-ast-no-hw",
        "purpose": "单模态视频 AST2.0-lite，无硬件loss",
        "config": "configs/ast2_ucf101_ast_only.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_ast_only.yaml",
    },
    {
        "id": "EXP-P1V-4-ast-hw",
        "purpose": "单模态视频 AST2.0-lite + 单设备硬件loss",
        "config": "configs/ast2_ucf101_ast_hw.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_ast_hw.yaml",
    },
    {
        "id": "EXP-P1AV-1-av-dense-late-fusion",
        "purpose": "视频+音频 晚融合 baseline",
        "config": "configs/ast2_ucf101_av_dense.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_av_dense.yaml",
    },
    {
        "id": "EXP-P1AV-2-av-ast-no-hw",
        "purpose": "视频+音频 跨模态堆积 + AST，无硬件loss",
        "config": "configs/ast2_ucf101_av_ast_only.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_av_ast_only.yaml",
    },
    {
        "id": "EXP-P1AV-3-av-ast-hw",
        "purpose": "视频+音频 跨模态堆积 + AST + 单设备硬件loss",
        "config": "configs/ast2_ucf101_av_ast_hw.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/ast2_ucf101_av_ast_hw.yaml",
    },
    {
        "id": "EXP-P2-1-fixed4-grid",
        "purpose": "固定4大核，顺序均分",
        "config": "configs/vc_phase2_fixed4_big.yaml",
        "command": "python -m scripts.run_version_c --cfg configs/vc_phase2_fixed4_big.yaml",
    },
    {
        "id": "EXP-P2-2-fixed4-mapping",
        "purpose": "固定4大核，贪心映射",
        "config": "configs/vc_phase2_fixed4_big.yaml",
        "command": "python -m scripts.run_version_c --cfg configs/vc_phase2_fixed4_big.yaml",
    },
    {
        "id": "EXP-P3-1-full-vc",
        "purpose": "完整 Version-C",
        "config": "configs/vc_phase3_full_ucf101.yaml",
        "command": "python -m scripts.run_version_c --cfg configs/vc_phase3_full_ucf101.yaml",
    },
    {
        "id": "EXP-P3-2-full-vc-no-layout",
        "purpose": "Version-C 无布局优化",
        "config": "configs/vc_phase3_nolayout_ucf101.yaml",
        "command": "python -m scripts.run_version_c --cfg configs/vc_phase3_nolayout_ucf101.yaml",
    },
    {
        "id": "EXP-SM-1-ast-smoke",
        "purpose": "AST 烟雾测试",
        "config": "configs/smoke_ast_ucf101.yaml",
        "command": "python -m scripts.run_ast2_ucf101 --cfg configs/smoke_ast_ucf101.yaml",
    },
    {
        "id": "EXP-SM-2-vc-smoke",
        "purpose": "Version-C 烟雾测试",
        "config": "configs/smoke_version_c_ucf101.yaml",
        "command": "python -m scripts.run_version_c --cfg configs/smoke_version_c_ucf101.yaml",
    },
]


TEMPLATE_HEADER = """# EXPERIMENTS_VERSION_C\n\n"""


def render(out_path: Path):
    lines = [TEMPLATE_HEADER]
    phase_titles = {
        "P0": "Phase 0 — Proxy Sanity",
        "P1V": "Phase 1-V — 单模态视频",
        "P1AV": "Phase 1-AV — 视频+音频",
        "P2": "Phase 2 — 固定多芯粒",
        "P3": "Phase 3 — 完整 Version-C",
        "SM": "Phase 4 — 烟雾测试",
    }
    current_phase = None
    for exp in EXPERIMENTS:
        phase_key = exp["id"].split("-")[1]
        if phase_key != current_phase:
            current_phase = phase_key
            lines.append(f"## {phase_titles.get(phase_key, phase_key)}\n\n")
        lines.append(f"### {exp['id']}\n\n")
        lines.append(f"- 目的：{exp['purpose']}\n")
        lines.append(f"- 配置文件：`{exp['config']}`\n")
        lines.append("- 运行命令：\n\n")
        lines.append("```bash\n" + exp["command"] + "\n```\n\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[gen_experiment_cmds] written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="EXPERIMENTS_VERSION_C.md")
    args = parser.parse_args()
    out_path = Path(args.out)
    render(out_path)


if __name__ == "__main__":
    main()
