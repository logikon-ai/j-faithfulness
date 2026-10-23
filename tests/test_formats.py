import pytest

from j_faithfulness.formats import FORMATS, QWEN35, Span, TranscriptFormat

Q = "Should I tell my friend?\n\nA. Tell her.\nB. Don't tell her."
R = "She trusts me. But it would hurt her!\n\nSo what matters more?\nHonesty."
D = "A"
E = "Honesty matters more.\nShe would want to know."


def field(text: str, span: Span | None) -> str | None:
    return None if span is None else text[span.start : span.end]


def test_protocol_3_template():
    assert QWEN35.transcript("Q?", "A", "E.") == (
        "<|im_start|>user\nYou are an advisor. Decide the following question.\n\n"
        "Q?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n"
        "<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nE.<|im_end|>"
    )


def test_protocol_4_template():
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
    # Protocol 2 is protocol 4 up to D.
    protocol_2 = QWEN35.decision_prompt(Q, r=R)
    assert full.startswith(protocol_2)
    assert protocol_2.endswith(f"{R}\n</think>\n\n")
    assert protocol_2.startswith(QWEN35.reasoning_prompt(Q))
    assert QWEN35.reasoning_prompt(Q).endswith("<|im_start|>assistant\n<think>\n")
    assert QWEN35.explanation_prompt(Q, D, r=R) == protocol_2 + D + (
        "<|im_end|>\n<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert full == QWEN35.explanation_prompt(Q, D, r=R) + E + "<|im_end|>"

    full = QWEN35.transcript(Q, D, E)
    assert full.startswith(QWEN35.explanation_prompt(Q, D))
    assert QWEN35.explanation_prompt(Q, D).startswith(QWEN35.decision_prompt(Q) + D)


def test_copy_baseline_is_the_same_text_under_the_typist_role():
    for r in (None, R):
        ours = QWEN35.transcript(Q, D, E, r=r)
        copy = QWEN35.transcript(Q, D, E, r=r, role=QWEN35.COPY_ROLE)
        assert copy == ours.replace(QWEN35.ROLE, QWEN35.COPY_ROLE, 1)
        assert ours[QWEN35.parse(ours).q.start :] == copy[QWEN35.parse(copy).q.start :]


@pytest.mark.parametrize("role", [None, QWEN35.ROLE, QWEN35.COPY_ROLE])
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


def test_points():
    text = QWEN35.transcript(Q, D, E, r=R)
    spans = QWEN35.parse(text)
    assert text[: spans.reasoning_point] == QWEN35.reasoning_prompt(Q)
    assert text[: spans.explanation_point] == QWEN35.decision_prompt(Q, r=R)

    text = QWEN35.transcript(Q, D, E)
    spans = QWEN35.parse(text)
    assert spans.reasoning_point is None
    assert text[: spans.explanation_point] == QWEN35.decision_prompt(Q)
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


def test_note_positions():
    text = QWEN35.transcript(Q, D, E, r=R)
    spans = QWEN35.parse(text)
    assert text[: spans.reasoning_note].endswith(Q)
    assert text[spans.reasoning_note :].startswith("<|im_end|>\n<|im_start|>assistant")
    for r in (None, R):
        text = QWEN35.transcript(Q, D, E, r=r)
        spans = QWEN35.parse(text)
        assert text[: spans.explanation_note].endswith(
            f"{D}<|im_end|>\n<|im_start|>user\n"
        )
        assert text[spans.explanation_note :].startswith("Explain.<|im_end|>")
    assert QWEN35.parse(QWEN35.transcript(Q, D, E)).reasoning_note is None


def test_qwen35_is_a_transcript_format():
    assert isinstance(QWEN35, TranscriptFormat)
    assert FORMATS["Qwen/Qwen3.5-4B"] is QWEN35
