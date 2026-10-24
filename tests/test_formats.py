import pytest

from j_faithfulness.formats import FORMATS, QWEN35, Span, TranscriptFormat

Q = "Should I tell my friend?\n\nA. Tell her.\nB. Don't tell her."
R = "She trusts me. But it would hurt her!\n\nSo what matters more?\nHonesty."
D = "A"
E = "Honesty matters more.\nShe would want to know."


def field(text: str, span: Span | None) -> str | None:
    return None if span is None else text[span.start : span.end]


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
    assert QWEN35.transcript("Q?", "A", "E.", r="R", capped=True) == QWEN35.transcript(
        "Q?", "A", "E.", r="R..."
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


@pytest.mark.parametrize("role", [None, QWEN35.ROLE, QWEN35.EDIT_ROLE])
@pytest.mark.parametrize(
    ("r", "capped", "parsed_r"),
    [(None, False, None), (R, False, R), (R, True, R + "..."), ("\n", False, "\n")],
)
def test_round_trip(role, r, capped, parsed_r):
    text = QWEN35.transcript(Q, D, E, r=r, capped=capped, role=role)
    spans = QWEN35.parse(text)
    assert field(text, spans.q) == Q
    assert field(text, spans.r) == parsed_r
    assert field(text, spans.d) == D
    assert field(text, spans.e) == E
    # The spans tile the transcript up to D, but for the end of the think block.
    after_prompt = spans.d if r is None else spans.r
    assert spans.opening.start == 0
    assert spans.opening.end == spans.q.start
    assert spans.q.end == spans.prompt.start
    assert spans.prompt.end == after_prompt.start
    if r is not None:
        assert text[spans.r.end : spans.d.start] == "\n</think>\n\n"
    assert text[spans.d.end : spans.e.start].startswith("<|im_end|>\n<|im_start|>u")
    assert text[spans.e.end :] == "<|im_end|>"
    assert field(text, spans.opening) == text[: text.index(Q)]


def test_prompt_span():
    text = QWEN35.transcript(Q, D, E, r=R)
    spans = QWEN35.parse(text)
    assert text[: spans.r.start] == QWEN35.reasoning_prompt(Q)
    assert text[: spans.d.start] == QWEN35.decision_prompt(Q, r=R)

    text = QWEN35.transcript(Q, D, E)
    spans = QWEN35.parse(text)
    assert text[: spans.d.start] == QWEN35.decision_prompt(Q)
    assert field(text, spans.prompt) == (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        QWEN35.transcript(Q, D, E)[:-1],
        QWEN35.transcript(Q, D, E) + "\n",
        QWEN35.transcript(Q, D, E).replace("Explain.", "Explain why."),
        QWEN35.transcript(Q, D, E, r=R).replace("\n</think>\n\nA", "</think>\n\nA"),
        QWEN35.transcript(Q, D, E).replace(
            "<|im_start|>user\n", "<|im_start|>user\n\n", 1
        ),
        # The role line needs the blank line before Q.
        "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>"
        "\n\nA<|im_end|>\n<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nE.<|im_end|>",
        # A second assistant turn inside E.
        QWEN35.transcript(Q, D, E) + "\n<|im_start|>assistant\nMore.<|im_end|>",
    ],
)
def test_parse_rejects_malformed(text):
    with pytest.raises(ValueError):
        QWEN35.parse(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": ""},
        {"r": None, "capped": True},
        {"r": "a</think>b"},
        {"e": "a<|im_end|>"},
        {"q": "a<|im_start|>b"},
        {"role": ""},
        {"role": "Role.\nMore."},
    ],
)
def test_build_rejects_unparsable_fields(kwargs):
    with pytest.raises(ValueError):
        QWEN35.transcript(**({"q": Q, "d": D, "e": E} | kwargs))


def test_qwen35_is_a_transcript_format():
    assert isinstance(QWEN35, TranscriptFormat)
    assert FORMATS["Qwen/Qwen3.5-4B"] is QWEN35
    assert FORMATS["Qwen/Qwen3.5-27B"] is QWEN35


def test_baseline_is_the_same_text_under_the_editor_role():
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
