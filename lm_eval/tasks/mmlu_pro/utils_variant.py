"""
Standardized variant formatters for mmlu_pro_variant.

This variant follows the complete standardization requirements:
- Uses "the final answer is (X)" pattern consistently
- Properly formatted prompts with Question/Options structure
- Separated few-shot examples with chain-of-thought
"""

from functools import partial

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def doc_to_text(example):
    """
    Format the question with options for the user message.

    Returns a prompt that includes:
    - Question text
    - Options labeled A-J
    - Explicit instruction to think step by step and reply with "the final answer is (X)"
    """
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        if i >= len(choices):
            break
        prompt += f"{choices[i]}. {opt.strip()}\n"

    # Use the standardized instruction with "the final answer is"
    return (
        prompt
        + '\n\nThink step by step and then finish your answer with "the final answer is (X)" where X is the correct letter choice.'
    )


def fewshot_doc_to_target(example):
    """
    Format the assistant response for few-shot examples.

    Extracts chain-of-thought content and cleans up formatting.
    Returns the CoT reasoning + final answer in standard format.
    """
    # Get the chain-of-thought content
    cot_content = example["cot_content"]

    # Clean up common prefixes
    cot_content = cot_content.replace("A: Let's think step by step.", "Let's think step by step.")
    cot_content = cot_content.replace("Answer: ", "")
    cot_content = cot_content.replace("A: ", "")

    return cot_content


def process_docs_generic(dataset, subject):
    """
    Filter dataset by subject category.

    Args:
        dataset: The full MMLU-Pro dataset
        subject: Subject name to filter by (e.g., "biology", "computer science")

    Returns:
        Filtered dataset containing only questions from the specified subject
    """
    return dataset.filter(lambda x: x["category"] == subject)


# Dynamically create process_docs functions for all subjects
# This avoids hardcoding each subject individually
SUBJECTS = {
    "biology": "biology",
    "business": "business",
    "chemistry": "chemistry",
    "computer_science": "computer science",  # Note: category uses space
    "economics": "economics",
    "engineering": "engineering",
    "health": "health",
    "history": "history",
    "law": "law",
    "math": "math",
    "other": "other",
    "philosophy": "philosophy",
    "physics": "physics",
    "psychology": "psychology"
}

# Generate process_* functions dynamically
for func_suffix, category_name in SUBJECTS.items():
    func_name = f"process_{func_suffix}"
    globals()[func_name] = partial(process_docs_generic, subject=category_name)
