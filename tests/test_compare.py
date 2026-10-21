import math

import pytest

from j_faithfulness.compare import brier, log_prob, renormalize


def test_brier():
    assert brier({"A": 1.0, "B": 0.0}, "A") == 0
    assert brier({"A": 0.0, "B": 1.0}, "A") == 2
    assert brier({"A": 0.5, "B": 0.5}, "A") == pytest.approx(0.5)
    # Renormalized: mass off the options doesn't count.
    assert brier({"A": 0.4, "B": 0.4}, "A") == pytest.approx(0.5)


def test_log_prob_and_renormalize():
    assert renormalize({"A": 0.3, "B": 0.1}) == pytest.approx({"A": 0.75, "B": 0.25})
    assert log_prob({"A": 0.3, "B": 0.1}, "A") == pytest.approx(math.log(0.75))
