"""
Chat-template-friendly formatters for ifeval_chat variant.

This variant adds standardized prompt formatting to IFEval while preserving
the instruction-following evaluation logic.
"""

def doc_to_text(doc):
    """
    Format the IFEval prompt with standard instruction to think step by step.

    Args:
        doc: Document containing the 'prompt' field with the instruction-following task

    Returns:
        Formatted prompt string with thinking instruction appended
    """
    # Get the original prompt from the document
    original_prompt = doc["prompt"]

    # Add the standard thinking instruction
    # Note: We don't add "Reply with 'the final answer is (X)'" because
    # IFEval doesn't extract answers - it evaluates if instructions were followed
    formatted_prompt = original_prompt + "\n\nThink step by step."

    return formatted_prompt
