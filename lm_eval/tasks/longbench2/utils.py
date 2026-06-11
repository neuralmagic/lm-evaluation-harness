import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_a}\n"
    "(B) {choice_b}\n"
    "(C) {choice_c}\n"
    "(D) {choice_d}\n\n"
    "Let's think step by step, then end your response with: "
    '"The correct answer is (X)" where X is one of A, B, C, or D.'
)


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for LongBench2 truncation.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def _format_prompt(doc: dict) -> str:
    return PROMPT_TEMPLATE.format(
        context=doc["context"],
        question=doc["question"].strip(),
        choice_a=doc["choices"][0],
        choice_b=doc["choices"][1],
        choice_c=doc["choices"][2],
        choice_d=doc["choices"][3],
    )


def _resolve_max_len() -> int | None:
    max_len = os.getenv("LONGBENCH2_MAX_LEN")
    if not max_len:
        return None
    try:
        parsed = int(max_len)
    except ValueError:
        eval_logger.warning(
            f"Invalid LONGBENCH2_MAX_LEN={max_len!r}. Skipping task-level truncation."
        )
        return None
    if parsed <= 0:
        eval_logger.warning(
            f"LONGBENCH2_MAX_LEN must be positive, got {parsed}. Skipping task-level truncation."
        )
        return None
    return parsed


def _resolve_max_model_len(tokenizer_name: str) -> int | None:
    max_model_len = os.getenv("LONGBENCH2_MAX_MODEL_LEN")
    if max_model_len:
        try:
            parsed = int(max_model_len)
            if parsed > 0:
                return parsed
        except ValueError:
            eval_logger.warning(
                f"Invalid LONGBENCH2_MAX_MODEL_LEN={max_model_len!r}. Ignoring."
            )

    inferred = get_tokenizer(pretrained=tokenizer_name).model_max_length
    # Hugging Face uses very large sentinel values for "no fixed max length".
    if inferred and inferred < 10_000_000:
        return int(inferred)
    return None


def _resolve_effective_max_len(tokenizer_name: str) -> int | None:
    explicit_max_len = _resolve_max_len()
    if explicit_max_len is not None:
        return explicit_max_len

    max_model_len = _resolve_max_model_len(tokenizer_name)
    if max_model_len is None:
        return None

    reserve_gen_toks_raw = os.getenv("LONGBENCH2_RESERVE_GEN_TOKS", "32000")
    safety_margin_raw = os.getenv("LONGBENCH2_CONTEXT_SAFETY_MARGIN", "4096")
    try:
        reserve_gen_toks = int(reserve_gen_toks_raw)
        safety_margin = int(safety_margin_raw)
    except ValueError:
        eval_logger.warning(
            "Invalid truncation budgeting env vars. "
            "Expected integers for LONGBENCH2_RESERVE_GEN_TOKS and LONGBENCH2_CONTEXT_SAFETY_MARGIN."
        )
        return max_model_len

    effective = max_model_len - reserve_gen_toks - safety_margin
    if effective <= 0:
        eval_logger.warning(
            f"Computed non-positive effective max length ({effective}); "
            f"falling back to max_model_len={max_model_len}."
        )
        return max_model_len

    eval_logger.info(
        "LongBench2 truncation budget: "
        f"max_model_len={max_model_len}, reserve_gen_toks={reserve_gen_toks}, "
        f"safety_margin={safety_margin}, effective_max_len={effective}"
    )
    return effective


def _truncate_prompt(prompt: str, max_len: int, tokenizer_name: str) -> str:
    tokenizer = get_tokenizer(pretrained=tokenizer_name)
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        left_len = max_len // 2
        right_len = max_len - left_len
        input_ids = input_ids[:left_len] + input_ids[-right_len:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt


def doc_to_text(doc: dict) -> str:
    prompt = _format_prompt(doc)

    tokenizer_name = (
        os.getenv("LONGBENCH2_TOKENIZER")
        or os.getenv("LONGBENCH2_PRETRAINED")
        or os.getenv("TOKENIZER")
        or os.getenv("PRETRAINED")
    )
    max_len = _resolve_effective_max_len(tokenizer_name) if tokenizer_name else None

    if max_len is None:
        return prompt

    if not tokenizer_name:
        eval_logger.warning(
            "No tokenizer is configured for LongBench2 task-level truncation. "
            "Set LONGBENCH2_TOKENIZER (or LONGBENCH2_PRETRAINED), or TOKENIZER/PRETRAINED."
        )
        return prompt

    return _truncate_prompt(prompt, max_len=max_len, tokenizer_name=tokenizer_name)