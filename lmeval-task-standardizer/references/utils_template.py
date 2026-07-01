"""
Template for utils_chat.py file in standardized lmeval tasks.

This template shows the standard structure for formatting tasks with generate_until output type.
"""

# Standard choice letters for multiple choice questions
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def doc_to_text(example):
    """
    Format the question for the user message.

    This function should:
    1. Extract the question from the example
    2. Add options if they exist
    3. Add explicit format instructions

    Returns a string that will be the user message content.
    """
    prompt = "Question:\n"
    prompt += example["question"] + "\n"

    # Add options if present in the dataset
    if "options" in example and example["options"]:
        prompt += "Options:\n"
        for i, opt in enumerate(example["options"]):
            if i >= len(choices):
                break
            prompt += f"{choices[i]}. {opt.strip()}\n"

        # For tasks with options, specify letter choice
        prompt += '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'
    else:
        # For free-form tasks without options
        prompt += '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the answer.'

    return prompt


def fewshot_doc_to_target(example):
    """
    Format the assistant response for few-shot examples.

    This function should:
    1. Extract the chain-of-thought content if available
    2. Remove any "A:" or "Answer:" prefixes
    3. Ensure it ends with "the answer is (X)" format

    Returns a string that will be the assistant message content.
    """
    # Try to get CoT content from common field names
    cot_content = example.get("cot_content", "")

    if not cot_content:
        # If no CoT, try other common field names
        cot_content = example.get("explanation", "")

    if not cot_content:
        # If still no CoT, try to get the reasoning field
        cot_content = example.get("reasoning", "")

    # Clean up common prefixes
    cot_content = cot_content.replace("A: Let's think step by step.", "Let's think step by step.")
    cot_content = cot_content.replace("Answer: ", "")
    cot_content = cot_content.replace("A: ", "")

    # If we still have no content, create a minimal response
    if not cot_content:
        answer = example.get("answer", example.get("target", ""))
        cot_content = f"The answer is ({answer})."

    return cot_content


# Alternative template for tasks without options (free-form answers)
def doc_to_text_no_options(example):
    """Template for tasks without multiple choice options."""
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is your answer.'
    return prompt


# Alternative template for numeric answers
def doc_to_text_numeric(example):
    """Template for tasks with numeric answers."""
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the numeric answer.'
    return prompt
