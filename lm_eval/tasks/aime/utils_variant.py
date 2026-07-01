"""
Standardized variant formatters for aime25_variant.

This variant follows the complete standardization requirements:
- Uses "the final answer is (X)" pattern consistently
- Properly formatted prompts with Question structure
- Zero-shot evaluation with clear instructions
"""

# Import the original process_results and helper functions
from lm_eval.tasks.aime.utils import (
    process_results as original_process_results,
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
    strip_string
)
from typing import Dict, List


def doc_to_text(doc):
    """
    Format the math problem for the user message.

    Returns a prompt that includes:
    - Question text
    - Explicit instruction to think step by step and reply with "the final answer is (X)"
    """
    problem = doc["problem"]

    # Use the standardized instruction
    prompt = f"Question:\n{problem}\n\n"
    prompt += 'Think step by step and then finish your answer with "the final answer is (X)" where X is your answer. '
    prompt += 'You may use LaTeX formatting and \\boxed{{}} notation if needed.'

    return prompt


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """
    Process results with custom extraction for "the final answer is" format.

    First tries to extract from "the final answer is ..." pattern,
    then falls back to the original extraction logic (boxed, $...$, etc.)
    """
    retval = 0
    response = results[0]

    # Try to extract from "the final answer is ..." first
    answer = None

    # Look for "the final answer is" pattern (case insensitive)
    import re
    final_answer_pattern = re.compile(r'the final answer is[:\s]+(.+?)(?:\.|$|\n)', re.IGNORECASE | re.DOTALL)
    match = final_answer_pattern.search(response)

    if match:
        answer = match.group(1).strip()
        # Remove trailing punctuation
        answer = answer.rstrip('.')

        # If the answer contains \boxed{}, extract from that
        if '\\boxed{' in answer:
            boxed_answer = last_boxed_only_string(answer)
            if boxed_answer is not None:
                try:
                    boxed_content = remove_boxed(boxed_answer)
                    if boxed_content is not None:
                        answer = boxed_content
                except (AssertionError, IndexError):
                    pass

    # If we didn't find "the final answer is", fall back to original extraction
    if answer is None:
        # Try to extract answer from $...$ format first
        indices = [pos for pos, char in enumerate(response) if char == "$"]
        if len(indices) <= 1:
            answer = response
        else:
            answer = response[indices[0] + 1 : indices[-1]]

        # Extract from \\boxed{} if present
        boxed_answer = last_boxed_only_string(response)
        if boxed_answer is not None:
            try:
                boxed_content = remove_boxed(boxed_answer)
                if boxed_content is not None:
                    answer = boxed_content
            except (AssertionError, IndexError):
                pass

    # Check if answer matches target
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])
    if is_equiv(answer, target):
        retval = 1

    return {"exact_match": retval}
