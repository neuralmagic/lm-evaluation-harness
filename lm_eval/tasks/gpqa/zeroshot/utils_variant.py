"""
Standardized variant formatters for gpqa_diamond_zeroshot_variant.

This variant follows the complete standardization requirements:
- Uses "the final answer is (X)" pattern consistently
- Converts from multiple_choice to generate_until
- Properly formatted prompts with Question/Options structure
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


def preprocess(text):
    """Clean and normalize text content."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process documents by shuffling answer choices and tracking correct answer.

    This maintains the same shuffling logic as the original to ensure fairness.
    """
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "Question": doc["Question"],
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": chr(65 + correct_answer_index),  # Store as letter: A, B, C, or D
        }
        return out_doc

    return dataset.map(_process_doc)


def doc_to_text(doc):
    """
    Format the question with options for the user message.

    Returns a prompt that includes:
    - Question text
    - Options labeled A-D
    - Explicit instruction to think step by step and reply with "the final answer is (X)"
    """
    question = doc["Question"]

    prompt = f"Question:\n{question}\n"
    prompt += "Options:\n"
    prompt += f"(A) {doc['choice1']}\n"
    prompt += f"(B) {doc['choice2']}\n"
    prompt += f"(C) {doc['choice3']}\n"
    prompt += f"(D) {doc['choice4']}\n"

    # Use the standardized instruction
    prompt += '\n\nThink step by step and then finish your answer with "the final answer is (X)" where X is the correct letter choice.'

    return prompt


def doc_to_target(doc):
    """
    Extract the correct answer letter.

    Returns the letter (A, B, C, or D) of the correct answer.
    """
    return doc["answer"]
