import numpy as np

from layout.regions import build_regions


def test_site_to_region_full_coverage():
    sites = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]], dtype=np.float32)
    regions, site_to_region = build_regions(
        sites_xy_mm=sites,
        wafer_radius_mm=20.0,
        ring_edges_ratio=[0.0, 1.0],
        sectors_per_ring=[4],
        ring_score=[1.0],
        capacity_ratio=1.0,
    )
    assert site_to_region.shape[0] == sites.shape[0]
    assert (site_to_region >= 0).all()
    assert len(regions) == 4
    # capacity respects ratio
    for reg in regions:
        assert reg.capacity >= len(reg.site_ids)
