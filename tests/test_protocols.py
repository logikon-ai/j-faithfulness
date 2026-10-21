from j_faithfulness.protocols import decision_prompt


def test_decision_prompt_matches_template():
    # apply_chat_template([user Q], add_generation_prompt=True,
    # enable_thinking=False) for Qwen3.5-4B.
    assert decision_prompt("Q?") == (
        "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
