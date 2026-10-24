import pytest

from j_faithfulness.formats import QWEN35, row_transcript

Q = "Should I tell my friend?\n\nA. Tell her.\nB. Don't tell her."
R = "She trusts me. But it would hurt her!\n\nSo what matters more?\nHonesty."
D = "A"
E = "Honesty matters more.\nShe would want to know."


def test_thinking_off_template():
    assert QWEN35.transcript("Q?", "A", "E.") == (
        "<|im_start|>user\nYou are an advisor. Decide the following question.\n\n"
        "Q?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n"
        "<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nE.<|im_end|>"
    )


def test_thinking_on_template():
    assert QWEN35.transcript("Q?", "A", "E.", r="R.") == (
        "<|im_start|>user\nYou are an advisor. Decide the following question. "
        "Keep your internal reasoning concise.\n\nQ?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nR.\n</think>\n\nA<|im_end|>\n"
        "<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nE.<|im_end|>"
    )


def test_prefixes():
    full = QWEN35.transcript(Q, D, E, r=R)
    up_to_d = QWEN35.decision_prompt(Q, r=R)
    assert full.startswith(up_to_d)
    assert up_to_d.endswith(f"{R}\n</think>\n\n")
    assert up_to_d.startswith(QWEN35.reasoning_prompt(Q))
    assert QWEN35.reasoning_prompt(Q).endswith("<|im_start|>assistant\n<think>\n")
    assert QWEN35.explanation_prompt(Q, D, r=R) == up_to_d + D + (
        "<|im_end|>\n<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert full == QWEN35.explanation_prompt(Q, D, r=R) + E + "<|im_end|>"

    full = QWEN35.transcript(Q, D, E)
    assert full.startswith(QWEN35.explanation_prompt(Q, D))
    assert QWEN35.explanation_prompt(Q, D).startswith(QWEN35.decision_prompt(Q) + D)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": ""},
        {"r": "a</think>b"},
        {"e": "a<|im_end|>"},
        {"q": "a<|im_start|>b"},
        {"role": ""},
        {"role": "Role.\nMore."},
    ],
)
def test_build_rejects_fields_that_read_as_markup(kwargs):
    with pytest.raises(ValueError):
        QWEN35.transcript(**({"q": Q, "d": D, "e": E} | kwargs))


def test_editor_reads_the_same_text_under_its_role():
    ours = QWEN35.transcript(Q, D, E, r=R)
    edit = QWEN35.transcript(Q, D, E, r=R, role=QWEN35.EDIT_ROLE)
    assert edit == ours.replace(QWEN35.ROLE, QWEN35.EDIT_ROLE, 1)


def test_editor_reads_r_as_text_to_paraphrase():
    assert QWEN35.editor_prompt("Q?", "R.") == (
        f"<|im_start|>user\n{QWEN35.EDIT_ROLE}\n\nQ?\n\nR.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert QWEN35.editor_prompt(Q) == QWEN35.decision_prompt(Q, role=QWEN35.EDIT_ROLE)
    with pytest.raises(ValueError):
        QWEN35.editor_prompt(Q, "")


def transcript_row(r=None, fmt="qwen3.5/1"):
    return {
        "q": Q,
        "options": [{"label": "A", "action": "Tell"}, {"label": "B", "action": "No"}],
        "r": r,
        "d": D,
        "e": E,
        "metadata": {"model_id": "Qwen/Qwen3.5-4B", "format": fmt},
    }


def test_row_transcript():
    assert row_transcript(transcript_row()) == QWEN35.transcript(Q, D, E)
    # A capped R holds its cap mark as stored.
    assert row_transcript(transcript_row(R + "...")) == QWEN35.transcript(
        Q, D, E, R + "..."
    )


def test_row_transcript_rejects_another_format_version():
    with pytest.raises(ValueError):
        row_transcript(transcript_row(fmt="qwen3.5/0"))
