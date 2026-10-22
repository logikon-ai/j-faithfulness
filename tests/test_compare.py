import math

import pytest

from j_faithfulness.compare import brier, kl, renormalize, sq_dist


def test_brier():
    assert brier({"A": 1.0, "B": 0.0}, "A") == 0
    assert brier({"A": 0.0, "B": 1.0}, "A") == 2
    assert brier({"A": 0.5, "B": 0.5}, "A") == pytest.approx(0.5)
    # Renormalized: mass off the options doesn't count.
    assert brier({"A": 0.4, "B": 0.4}, "A") == pytest.approx(0.5)


def test_renormalize():
    assert renormalize({"A": 0.3, "B": 0.1}) == pytest.approx({"A": 0.75, "B": 0.25})


def test_sq_dist_and_kl():
    p, q = {"A": 0.3, "B": 0.1}, {"A": 0.5, "B": 0.5}
    assert sq_dist(p, q) == pytest.approx(2 * 0.25**2)
    assert sq_dist(p, p) == 0
    assert kl(p, q) == pytest.approx(0.75 * math.log(1.5) + 0.25 * math.log(0.5))
    assert kl(p, p) == pytest.approx(0)
    assert kl({"A": 1.0, "B": 0.0}, q) == pytest.approx(math.log(2))
