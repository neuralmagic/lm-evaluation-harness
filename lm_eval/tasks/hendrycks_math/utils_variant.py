"""
Standardized variant formatters for hendrycks_math500_variant.

This variant follows the complete standardization requirements:
- Uses "the final answer is (X)" pattern consistently
- Properly formatted prompts with Question structure
- Maintains compatibility with LaTeX and \\boxed{} notation
"""

from typing import Dict, List
import datasets
import re

# Import the original helper functions
from lm_eval.tasks.hendrycks_math.utils import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
    strip_string
)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process documents by extracting problem, solution, and boxed answer.
    Same as original to maintain compatibility.
    """
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


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

    # Check if answer matches target
    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


def doc_to_target(doc):
    """
    Extract the target answer from the solution.
    Same as original processing.
    """
    return doc["answer"]
