import pytest

np = pytest.importorskip("numpy")

from layout.pareto import ParetoSet


def test_pareto_dominance_and_knee():
    ps = ParetoSet(eps_comm=0.01, eps_therm=0.01, max_points=10)
    points = [
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.8, 1.2),
    ]
    for idx, (c, t) in enumerate(points):
        ps.try_add(c, t, total_scalar=c + t, meta={"id": idx})
    assert len(ps.points) == 4
    knee = ps.knee_point()
    assert knee is not None
    assert knee.comm_norm <= 1.1
    assert knee.therm_norm <= 1.2

