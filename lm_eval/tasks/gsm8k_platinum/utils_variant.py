"""
Standardized variant formatters for gsm8k_platinum_cot_llama_variant.

This variant follows the complete standardization requirements:
- Uses "the final answer is (X)" pattern consistently
- Properly formatted prompts with Question structure
- Separated few-shot examples with chain-of-thought
"""


def doc_to_text(doc):
    """
    Format the math question for the user message.

    Returns a prompt that includes:
    - Question text
    - Explicit instruction to think step by step and reply with "the final answer is (X)"
    """
    question = doc["question"]

    # Use the standardized instruction with "the final answer is"
    prompt = f"Question:\n{question}\n\n"
    prompt += 'Think step by step and then finish your answer with "the final answer is (X)" where X is the numeric answer.'

    return prompt


def fewshot_doc_to_text(doc):
    """
    Format the question for few-shot examples (user message).
    Same as doc_to_text but used in fewshot_config.
    """
    return doc_to_text(doc)


def fewshot_doc_to_target(doc):
    """
    Format the assistant response for few-shot examples.

    Extracts or formats the chain-of-thought content with the final answer.
    Returns the CoT reasoning + final answer in standard format.
    """
    # Get the target answer which includes CoT reasoning
    target = doc.get("target", "")

    if not target:
        # If no target, try to construct from answer
        answer = doc.get("answer", "")
        if "####" in answer:
            numeric_answer = answer.split("####")[-1].strip()
            return f"The final answer is {numeric_answer}"
        return f"The final answer is {answer}"

    # Target already contains CoT + "The final answer is X" format from the original task
    # Just ensure it uses our standard phrasing
    if "The final answer is" in target:
        # Already in correct format
        return target
    else:
        # Add the standard ending if missing
        return target + f"\nThe final answer is {doc.get('answer', '').split('####')[-1].strip()}"


def doc_to_target(doc):
    """
    Extract the numeric answer from the document.

    Handles both formats:
    - answer field with "#### 42" format
    - target field with numeric value
    """
    answer = doc.get("answer", "")
    if "####" in answer:
        return answer.split("####")[-1].strip()

    target = doc.get("target", "")
    if target:
        return str(target).strip()

    return answer.strip()
