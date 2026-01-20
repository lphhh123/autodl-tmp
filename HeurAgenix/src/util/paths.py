import os
from pathlib import Path


def default_llm_config_path(filename: str = "azure_gpt_4o.json") -> str:
    """
    Resolve default LLM config path robustly.
    Priority:
      1) env HEURAGENIX_LLM_CONFIG (if points to existing file)
      2) data/llm_config/<filename> (legacy README default)
      3) configs/llm/<filename> (repo-shipped default)
    """
    root = Path(__file__).resolve().parents[2]
    envp = os.environ.get("HEURAGENIX_LLM_CONFIG")
    if envp:
        p = Path(envp)
        if p.is_file():
            return str(p)

    p1 = root / "data" / "llm_config" / filename
    if p1.is_file():
        return str(p1)

    p2 = root / "configs" / "llm" / filename
    if p2.is_file():
        return str(p2)

    return str(p1)
