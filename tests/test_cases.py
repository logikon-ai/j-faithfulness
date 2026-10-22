import math

import pytest

from j_faithfulness.cases import (
    A2_CONDITIONS,
    AGAIN,
    CASES,
    a2_prompts,
    noise_floor,
    run,
    stability,
    uptake,
    verdict,
)
from j_faithfulness.inject import EMPTY, FOLLOW, injected_decision_prompt
from j_faithfulness.protocols import decision_prompt

A, B, HALF = {"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}, {"A": 0.5, "B": 0.5}
LN2 = math.log(2)


def outcomes(own_j=A, follow_j=A, oppose_j=B, follow_empty=A, oppose_empty=A):
    return {
        "no injection": A,
        "own J": own_j,
        "own J · follow": follow_j,
        "own J · oppose": oppose_j,
        "empty · follow": follow_empty,
        "empty · oppose": oppose_empty,
    }


def test_stability():
    assert stability(outcomes()) == 0
    # Compared with the original distribution, not with a one-hot D: an
    # unsure model that stays unsure is stable.
    unsure = outcomes() | {"no injection": HALF, "own J": HALF}
    assert stability(unsure) == 0
    assert stability(outcomes(own_j=B)) == pytest.approx(LN2)


def test_uptake():
    # The instruction flips D only with own J: full uptake.
    assert uptake(outcomes()) == pytest.approx(LN2)
    # It moves D as much with the empty block: none.
    assert uptake(outcomes(oppose_empty=B)) == 0
    assert 0 < uptake(outcomes(oppose_j=HALF)) < LN2
    # The positive control, against the same empty arm.
    control = outcomes() | {"control · follow": A, "control · oppose": HALF}
    assert 0 < uptake(control, "control") < LN2


def test_verdict():
    assert verdict(0.0, 1.0, 0.1, 0.1) == "faithful"
    assert verdict(0.5, 1.0, 0.1, 0.1) == "unfaithful"
    assert verdict(0.0, 0.0, 0.1, 0.1) == "uninformative"


def test_unbuilt_cases_raise_with_their_milestone():
    for case in set(CASES) - {"A1", "A2"}:
        with pytest.raises(NotImplementedError, match=CASES[case].milestone):
            run(None, {}, case)


def test_a2_prompts():
    prompts = a2_prompts("Q?", "<latentthought>\na.\n</latentthought>")
    assert list(prompts) == A2_CONDITIONS
    original = decision_prompt("Q?", thinking=True)
    assert prompts["no injection"] == prompts[AGAIN] == original
    assert prompts["empty · follow"] == injected_decision_prompt(
        "Q?", EMPTY, FOLLOW, thinking=True
    )
    assert all(p.endswith("<think>\n") for p in prompts.values())


def test_noise_floor():
    assert noise_floor({"no injection": A, AGAIN: A}) == 0
    assert noise_floor({"no injection": A, AGAIN: B}) == pytest.approx(LN2)
