"""MMLU-Pro reasoning variant: fewshot assistant messages use configurable thinking tokens."""

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def doc_to_text(example):
    """Return question + options only (no Answer, no CoT)."""
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        if i >= len(choices):
            break
        prompt += f"{choices[i]}. {opt.strip()}\n"
    return prompt + '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'



def fewshot_doc_to_target(
    example,
    think_start_token: str = "<think>",
    think_end_token: str = "</think>",
):
    """Assistant message for fewshot: thinking block (CoT) + end token + 'The answer is (X).'.

    think_start_token and think_end_token are passed by the harness when
    --think_start_token / --think_end_token (or model think_*_token) are set.
    """
    cot_content = example["cot_content"].replace(
        "A: Let's think step by step. ", think_start_token
    )
    ans = example["answer"]
    letter = choices[ans] if isinstance(ans, int) else ans
    return cot_content + think_end_token + " The answer is (" + letter + ")."
