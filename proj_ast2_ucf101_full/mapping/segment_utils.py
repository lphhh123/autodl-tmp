
from typing import List, Dict, Any


def group_layers_into_segments(
    layer_metas: List[Dict[str, Any]],
    num_segments: int,
) -> List[List[Dict[str, Any]]]:
    """Group consecutive layers into `num_segments` segments.

    This is a simple evenly-sized partition used for the dummy pipeline.
    Later you can plug in a more sophisticated graph partitioner.

    Returns a list of segments; each segment is a list of layer meta dicts.
    """
    assert len(layer_metas) >= num_segments, "num_segments must be <= num_layers"
    n = len(layer_metas)
    seg_size = (n + num_segments - 1) // num_segments  # ceil
    segments: List[List[Dict[str, Any]]] = []
    for i in range(num_segments):
        start = i * seg_size
        end = min(n, (i + 1) * seg_size)
        if start >= n:
            break
        segments.append(layer_metas[start:end])
    return segments


def segments_to_edges(segments: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Construct a simple chain-like communication graph over segments.

    Returns:
        dict with:
          - edges: list of (u, v, traffic_bytes)
    """
    edges = []
    for i in range(len(segments) - 1):
        src_layers = segments[i]
        dst_layers = segments[i + 1]
        # Use the last layer of src as activation size
        last_row = src_layers[-1]
        L_eff = last_row["L_eff"]
        embed_dim = last_row["embed_dim"]
        bs = last_row["bs"]
        traffic_bytes = L_eff * embed_dim * bs * 2  # assume fp16
        edges.append((i, i + 1, traffic_bytes))
    return {"edges": edges}
