import pytest

from j_faithfulness.formats import QWEN35
from j_faithfulness.probe import HEADER, note, probe

Q = "Should I tell my friend? A) Tell her B) Don't. Answer with A or B only."
R = "She trusts me.\nHonesty matters."
D = "A"
E = "Honesty matters more."
BLOCK = "<latentthought>\nhonesty, trust.\n</latentthought>"


def test_notes_differ_only_in_not():
    for kind in ("reasoning", "explanation"):
        assert note(kind, based=True).replace("IS", "is NOT") == note(kind, False)


@pytest.mark.parametrize(("kind", "r"), [("reasoning", R), ("explanation", R)])
def test_arms_share_the_header_and_differ_only_in_not(kind, r):
    text = QWEN35.transcript(Q, D, E, r=r)
    spans = QWEN35.parse(text)
    (based, x), (not_based, x_not) = (
        probe(text, spans, kind, BLOCK, arm) for arm in (True, False)
    )
    for probed in (based, not_based):
        assert probed.startswith(f"{HEADER}\n\n<|im_start|>user\n")
    assert based.replace(" IS ", " is NOT ") == not_based
    assert not_based[x_not.start : x_not.end] == based[x.start : x.end]


@pytest.mark.parametrize("based", [True, False])
def test_reasoning_probe(based):
    text = QWEN35.transcript(Q, D, E, r=R)
    probed, x = probe(text, QWEN35.parse(text), "reasoning", BLOCK, based)
    assert probed[x.start : x.end] == R
    assert probed.endswith(R)
    # The header, then the user turn up to Q as recorded: the context J was
    # read from is intact.
    user_turn = QWEN35.reasoning_prompt(Q).split(Q)[0] + Q
    assert probed.startswith(f"{HEADER}\n\n{user_turn}\n\n")
    assert f"{note('reasoning', based)}\n{BLOCK}<|im_end|>\n" in probed
    assert probed[: x.start].endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize("r", [None, R])
def test_explanation_probe(r):
    text = QWEN35.transcript(Q, D, E, r=r)
    probed, x = probe(text, QWEN35.parse(text), "explanation", BLOCK, True)
    assert probed[x.start : x.end] == E
    prefix = f"{HEADER}\n\n" + QWEN35.decision_prompt(Q, r=r) + D
    assert probed.startswith(prefix)
    assert probed[len(prefix) :].startswith(
        f"<|im_end|>\n<|im_start|>user\n{note('explanation', True)}\n{BLOCK}"
        "\n\nExplain.<|im_end|>\n"
    )


def test_reasoning_probe_needs_protocol_4():
    text = QWEN35.transcript(Q, D, E)
    with pytest.raises(ValueError):
        probe(text, QWEN35.parse(text), "reasoning", BLOCK, True)
