import pytest

np = pytest.importorskip("numpy")

from layout.regions import build_regions


def test_regions_cover_sites():
    sites = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]], dtype=np.float32)
    regions, site_to_region = build_regions(
        sites_xy_mm=sites,
        wafer_radius_mm=50.0,
        ring_edges_ratio=[0.0, 0.5, 1.0],
        sectors_per_ring=[2, 4],
        capacity_ratio=1.0,
    )
    assert site_to_region.shape[0] == len(sites)
    assert np.all(site_to_region >= 0)
    assert sum(len(r.site_ids) for r in regions) == len(sites)

