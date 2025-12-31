import numpy as np

from layout.pareto import ParetoSet


def test_pareto_dominance_and_knee():
    p = ParetoSet(eps_comm=0.0, eps_therm=0.0)
    assert p.add(1.0, 1.0, {"id": 1})
    assert not p.add(1.1, 1.1, {"id": 2})  # dominated
    assert p.add(0.9, 1.2, {"id": 3})
    assert p.add(1.2, 0.9, {"id": 4})
    arr = p.as_array()
    assert arr.shape[0] == 3
    comm, therm, payload = p.knee_point()
    assert payload  # ensure payload returned
    assert comm <= 1.2 and therm <= 1.2
