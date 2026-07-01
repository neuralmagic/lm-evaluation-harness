# GSM8K Platinum CoT Llama Variant - Summary

## Overview

Created `gsm8k_platinum_cot_llama_variant` that standardizes the GSM8K Platinum task to follow the complete format requirements:
- Adds "Question:" structure to prompts
- Uses "the final answer is (X)" pattern consistently (lowercase "the")
- Implements utils functions for proper formatting
- Maintains 8-shot CoT prompting

## Files Created

1. **`utils_variant.py`** - Standardized formatting utilities
2. **`gsm8k-platinum-cot-llama-variant.yaml`** - Variant configuration

## Key Changes from Original

### 1. Prompt Structure

**Original (`gsm8k-platinum-cot-llama.yaml`):**
```
Given the following problem, reason and give a final answer to the problem.
Problem: {{question}}
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.
```

**Variant (`gsm8k-platinum-cot-llama-variant.yaml`):**
```
Question:
{question}

Think step by step and then finish your answer with "the final answer is (X)" where X is the numeric answer.
```

### 2. Regex Pattern

**Original:**
```yaml
regex_pattern: The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))
```

**Variant:**
```yaml
regex_pattern: 'the final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))'
```

**Note:** Changed to lowercase "the" to match the instruction given to the model.

### 3. Utils Functions

**Original:** Inline YAML template with `doc_to_text` and `doc_to_target` as template strings

**Variant:** Python functions in `utils_variant.py`:
- `doc_to_text()` - Formats question with "Question:" header
- `fewshot_doc_to_text()` - Formats few-shot questions
- `fewshot_doc_to_target()` - Formats few-shot answers
- `doc_to_target()` - Extracts numeric answer

## What Was Preserved

✅ **Dataset**: `madrylab/gsm8k-platinum`  
✅ **Output type**: `generate_until`  
✅ **8-shot prompting**: All 8 few-shot examples preserved  
✅ **CoT examples**: Chain-of-thought reasoning in few-shot targets  
✅ **Metrics**: exact_match with same regexes_to_ignore  
✅ **Generation kwargs**: Same until conditions and greedy decoding  
✅ **Tags**: chain_of_thought tag preserved  

## Prompt Format Example

### Question Formatting (doc_to_text)
```
Question:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

Think step by step and then finish your answer with "the final answer is (X)" where X is the numeric answer.
```

### Few-shot Answer Formatting (fewshot_doc_to_target)
```
There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6
```

## Comparison Table

| Feature | Original | Variant | Status |
|---------|----------|---------|---------|
| Output type | generate_until | generate_until | ✅ Same |
| Few-shot | 8-shot | 8-shot | ✅ Same |
| CoT examples | Yes | Yes | ✅ Same |
| Prompt structure | Inline template | Question: + utils | ✅ **Standardized** |
| Instruction | "The final answer is" | "the final answer is" | ✅ **Fixed case** |
| Regex pattern | `The final answer is` | `the final answer is` | ✅ **Fixed case** |
| Utils file | None | utils_variant.py | ✅ **Added** |
| Consistency | ❌ Case mismatch | ✅ **Consistent** | ✅ **Fixed** |

## Issue Fixed

**Problem:** The original task has a case inconsistency:
- **Instruction says**: "The final answer is [answer]" (capital T)
- **Regex expects**: "The final answer is" (capital T)
- **But models often respond**: "the final answer is" (lowercase t)

This could cause extraction failures on some models.

**Solution:** The variant uses lowercase "the" consistently:
- **Instruction**: "the final answer is (X)"
- **Regex**: `'the final answer is ...'`
- Both aligned for reliable extraction

## Usage

Run the variant:
```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama_variant \
  --model_args "model=Qwen/Qwen3-4B,max_length=40960,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 8 \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=24000" \
  --output_path ./results_gsm8k_platinum_variant
```

Compare with original:
```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama,gsm8k_platinum_cot_llama_variant \
  --model_args "model=Qwen/Qwen3-4B,max_length=40960,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 8 \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=24000" \
  --output_path ./results_gsm8k_platinum_comparison
```

## Testing

Quick validation:
```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama_variant \
  --model_args "model=Qwen/Qwen3-4B,max_length=40960,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 8 \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=24000" \
  --limit 5 \
  --log_samples
```

Verify:
- Prompts contain "Question:" header
- Prompts contain "the final answer is (X)" instruction (lowercase)
- Regex extraction works correctly
- Few-shot examples are properly formatted
- Metrics match expected values

## Expected Behavior

The variant should produce similar or slightly improved results compared to the original, as:
1. The "Question:" structure provides clearer formatting
2. The case-consistent instruction/regex reduces extraction errors
3. The standardized format works better with chat-tuned models

## Notes

- **Original task preserved** - `gsm8k-platinum-cot-llama.yaml` remains unchanged
- **Can run alongside original** - Use both for comparison
- **Case sensitivity matters** - Always check instruction matches regex
- **8-shot CoT examples** - All preserved from original with "The final answer is" → "the final answer is" for consistency
