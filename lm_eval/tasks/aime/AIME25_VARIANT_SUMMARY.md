# AIME 2025 Variant - Summary

## Overview

Created `aime25_variant` that standardizes the AIME 2025 task to follow the complete format requirements:
- Adds "Question:" structure to prompts
- Uses "the final answer is (X)" pattern consistently
- Maintains compatibility with LaTeX and \\boxed{} notation
- Preserves advanced answer extraction logic for mathematical expressions

## Files Created

1. **`utils_variant.py`** - Standardized formatting utilities with custom answer extraction
2. **`aime25_variant.yaml`** - Variant configuration

## Key Changes from Original

### 1. Prompt Structure

**Original (`aime25.yaml`):**
```
Question: {{problem}}
Answer:
```

**Variant (`aime25_variant.yaml`):**
```
Question:
{problem}

Think step by step and then finish your answer with "the final answer is (X)" where X is your answer. You may use LaTeX formatting and \boxed{} notation if needed.
```

### 2. Answer Extraction

**Original:** Uses `utils.process_results` which extracts from:
1. `$...$` format
2. `\boxed{}` notation
3. Raw response

**Variant:** Uses `utils_variant.process_results` which:
1. **First** tries to extract from "the final answer is ..." pattern
2. Handles `\boxed{}` within the final answer
3. **Falls back** to original extraction logic if "the final answer is" not found
4. Maintains all mathematical answer normalization

### 3. Utils Functions

**Original:** Inline YAML template with `doc_to_text` as template string

**Variant:** Python functions in `utils_variant.py`:
- `doc_to_text()` - Formats question with "Question:" header and standard instruction
- `process_results()` - Enhanced extraction supporting both standard format and LaTeX/boxed notation

## What Was Preserved

✅ **Dataset**: `math-ai/aime25`  
✅ **Output type**: `generate_until`  
✅ **Zero-shot**: `num_fewshot: 0` (AIME is designed for zero-shot)  
✅ **Metrics**: `exact_match` with mathematical equivalence checking  
✅ **Generation kwargs**: Same until conditions, greedy decoding, max 32K tokens  
✅ **Tags**: `math_word_problems` tag preserved  
✅ **Answer normalization**: All LaTeX/mathematical normalization from original utils.py  

## Prompt Format Example

### Question Formatting (doc_to_text)
```
Question:
How many ordered pairs of integers $(a, b)$ satisfy all of the following inequalities?
  \begin{align*}
    a^2 + b^2 &< 16 \\
    a^2 + b^2 &< 8a \\
    a^2 + b^2 &< 8b
  \end{align*}

Think step by step and then finish your answer with "the final answer is (X)" where X is your answer. You may use LaTeX formatting and \boxed{} notation if needed.
```

### Expected Model Response Format
```
Let me work through this step by step.

[... mathematical reasoning ...]

Therefore, there are 6 ordered pairs that satisfy all three inequalities.

the final answer is 6
```

**Or with LaTeX/boxed notation:**
```
[... mathematical reasoning ...]

the final answer is \boxed{6}
```

## Answer Extraction Logic

The variant's `process_results` function handles multiple formats:

1. **Standard format**: `the final answer is 6` → extracts `6`
2. **With boxed**: `the final answer is \boxed{6}` → extracts `6`
3. **With LaTeX**: `the final answer is \frac{1}{2}` → extracts `\frac{1}{2}`
4. **Fallback**: If "the final answer is" not found, uses original extraction (looks for `\boxed{}`, `$...$`, etc.)

This ensures backward compatibility with models that don't follow the new format while encouraging the standardized output.

## Comparison Table

| Feature | Original | Variant | Status |
|---------|----------|---------|---------|
| Output type | generate_until | generate_until | ✅ Same |
| Few-shot | 0-shot | 0-shot | ✅ Same |
| Prompt structure | Template | Question: + utils | ✅ **Standardized** |
| Instruction | "Answer:" | "the final answer is (X)" | ✅ **Standardized** |
| LaTeX support | Yes | Yes | ✅ **Preserved** |
| \\boxed{} support | Yes | Yes | ✅ **Preserved** |
| Answer extraction | boxed/$...$ only | "final answer is" + fallback | ✅ **Enhanced** |
| Math normalization | Yes | Yes | ✅ **Preserved** |
| Utils file | utils.py | utils_variant.py | ✅ **Added** |

## Why This Matters for AIME

AIME (American Invitational Mathematics Examination) is a prestigious competition with:
- **Complex problems**: Requires deep mathematical reasoning
- **Numerical answers**: All answers are integers from 0 to 999
- **LaTeX formatting**: Problems and solutions often use mathematical notation
- **\\boxed{} convention**: Standard way to highlight final answers in math

The variant maintains all these conventions while adding standardized prompting.

## Usage

Run the variant:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks aime25_variant \
    --device cuda:0 \
    --batch_size auto
```

Compare with original:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks aime25,aime25_variant \
    --device cuda:0 \
    --batch_size auto
```

## Testing

Quick validation:
```bash
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks aime25_variant \
    --limit 2 \
    --log_samples
```

Verify:
- Prompts contain "Question:" header
- Prompts contain "the final answer is (X)" instruction
- LaTeX and \\boxed{} notation still work
- Answer extraction handles both standard and LaTeX formats
- Mathematical equivalence checking works

## Expected Behavior

The variant should produce similar results to the original, as:
1. The "Question:" structure provides clearer formatting
2. The "the final answer is (X)" instruction guides model output
3. The extraction logic gracefully handles both formats
4. All mathematical normalization is preserved

Models may perform slightly better with explicit "think step by step" instruction.

## Notes

- **Original task preserved** - `aime25.yaml` remains unchanged
- **Can run alongside original** - Use both for comparison
- **LaTeX compatible** - Supports full mathematical notation
- **Backward compatible** - Falls back to original extraction if needed
- **Zero-shot only** - AIME is designed for zero-shot evaluation (no few-shot examples)
- **Answer range**: AIME answers are integers 0-999 (but extraction handles any format)
