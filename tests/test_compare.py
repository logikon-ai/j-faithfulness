import math

import pytest

from j_faithfulness.compare import jsd, renormalize


def test_renormalize():
    assert renormalize({"A": 0.3, "B": 0.1}) == pytest.approx({"A": 0.75, "B": 0.25})


def test_jsd():
    a, b, half = {"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}, {"A": 0.5, "B": 0.5}
    assert jsd(a, a) == 0
    assert jsd(a, b) == pytest.approx(math.log(2))
    assert jsd(a, half) == pytest.approx(jsd(half, a))
    # m = (0.75, 0.25): (KL(a ‖ m) + KL(half ‖ m)) / 2
    expected = (math.log(4 / 3) + 0.5 * math.log(2 / 3) + 0.5 * math.log(2)) / 2
    assert jsd(a, half) == pytest.approx(expected)
    # Renormalized: mass off the options doesn't count.
    assert jsd({"A": 0.3, "B": 0.3}, half) == pytest.approx(0)
