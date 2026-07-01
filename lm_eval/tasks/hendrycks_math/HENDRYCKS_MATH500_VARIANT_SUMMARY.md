# Hendrycks MATH-500 Variant - Summary

## Overview

Created `hendrycks_math500_variant` that standardizes the MATH-500 benchmark task to follow the complete format requirements:
- Adds "Question:" structure to prompts
- Uses "the final answer is (X)" pattern consistently
- Maintains compatibility with LaTeX and \\boxed{} notation
- Preserves advanced answer extraction logic for mathematical expressions

## Files Created

1. **`utils_variant.py`** - Standardized formatting utilities with custom answer extraction
2. **`hendrycks_math_algebra_variant.yaml`** - Base variant configuration
3. **`hendrycks_math500_variant.yaml`** - MATH-500 specific variant

## Key Changes from Original

### 1. Prompt Structure

**Original (`hendrycks_math_algebra.yaml`):**
```
Problem: {{problem}}
Answer:
```

**Variant (`hendrycks_math_algebra_variant.yaml`):**
```
Question:
{problem}

Think step by step and then finish your answer with "the final answer is (X)" where X is your answer. You may use LaTeX formatting and \boxed{} notation if needed.
```

### 2. Answer Extraction

**Original:** Uses `utils.process_results` which extracts from:
1. `$...$` format
2. Falls back to raw response

**Variant:** Uses `utils_variant.process_results` which:
1. **First** tries to extract from "the final answer is ..." pattern
2. Handles `\boxed{}` within the final answer
3. **Falls back** to original extraction logic if "the final answer is" not found
4. Maintains all mathematical answer normalization

### 3. Generation Parameters

**Original:**
```yaml
generation_kwargs:
  until:
    - "Problem:"
  do_sample: false
  temperature: 0
```

**Variant:**
```yaml
generation_kwargs:
  until:
    - "Question:"
    - "Problem:"
  do_sample: false
  temperature: 0
  max_gen_toks: 4096
```

### 4. Utils Functions

**Original:** Inline YAML template with `doc_to_text` as template string

**Variant:** Python functions in `utils_variant.py`:
- `process_docs()` - Extracts problem, solution, and boxed answer
- `doc_to_text()` - Formats question with "Question:" header and standard instruction
- `process_results()` - Enhanced extraction supporting both standard format and LaTeX/boxed notation
- `doc_to_target()` - Returns the extracted answer

## What Was Preserved

✅ **Dataset**: `HuggingFaceH4/MATH-500`  
✅ **Output type**: `generate_until`  
✅ **Metrics**: `exact_match` with mathematical equivalence checking  
✅ **Answer normalization**: All LaTeX/mathematical normalization from original utils.py  
✅ **Greedy decoding**: `do_sample: false, temperature: 0`  
✅ **Tags**: `math_word_problems` tag preserved  
✅ **Split configuration**: `training_split: null, test_split: test`  

## About MATH-500

MATH-500 is a subset of the Hendrycks MATH benchmark:
- **500 problems**: Curated from the full MATH dataset
- **Difficulty levels**: Covers Level 1-5 problems
- **Topics**: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus
- **LaTeX heavy**: Problems and solutions use extensive mathematical notation
- **\\boxed{} answers**: Standard convention for final answers

## Prompt Format Example

### Question Formatting (doc_to_text)
```
Question:
If $f(x) = x^2 - 2x + 5$ and $g(x) = x + 3$, what is $f(g(2))$?

Think step by step and then finish your answer with "the final answer is (X)" where X is your answer. You may use LaTeX formatting and \boxed{} notation if needed.
```

### Expected Model Response Format
```
Let me work through this step by step.

First, I need to find g(2):
g(2) = 2 + 3 = 5

Now I need to find f(5):
f(5) = 5^2 - 2(5) + 5
f(5) = 25 - 10 + 5
f(5) = 20

the final answer is 20
```

**Or with LaTeX/boxed notation:**
```
[... mathematical reasoning ...]

the final answer is \boxed{20}
```

## Answer Extraction Logic

The variant's `process_results` function handles multiple formats:

1. **Standard format**: `the final answer is 20` → extracts `20`
2. **With boxed**: `the final answer is \boxed{20}` → extracts `20`
3. **With LaTeX**: `the final answer is \frac{1}{2}` → extracts `\frac{1}{2}`
4. **Fallback**: If "the final answer is" not found, uses original extraction (looks for `$...$`, etc.)

This ensures backward compatibility with models that don't follow the new format while encouraging the standardized output.

## Comparison Table

| Feature | Original | Variant | Status |
|---------|----------|---------|---------|
| Dataset | MATH-500 | MATH-500 | ✅ Same |
| Output type | generate_until | generate_until | ✅ Same |
| Prompt structure | Template | Question: + utils | ✅ **Standardized** |
| Instruction | "Answer:" | "the final answer is (X)" | ✅ **Standardized** |
| LaTeX support | Yes | Yes | ✅ **Preserved** |
| \\boxed{} support | Yes | Yes | ✅ **Preserved** |
| Answer extraction | $...$ only | "final answer is" + fallback | ✅ **Enhanced** |
| Math normalization | Yes | Yes | ✅ **Preserved** |
| Max tokens | (default) | 4096 | ✅ **Added** |
| Utils file | utils.py | utils_variant.py | ✅ **Added** |

## Relationship to Other MATH Tasks

The variant uses `hendrycks_math_algebra_variant.yaml` as its base, which can also be used to create variants for other MATH subjects:

**Potential variants:**
- `hendrycks_math_geometry_variant.yaml`
- `hendrycks_math_counting_and_prob_variant.yaml`
- `hendrycks_math_intermediate_algebra_variant.yaml`
- `hendrycks_math_num_theory_variant.yaml`
- `hendrycks_math_prealgebra_variant.yaml`
- `hendrycks_math_precalc_variant.yaml`

Each would include `hendrycks_math_algebra_variant.yaml` and override `dataset_name` and `task`.

## Usage

Run the variant:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks hendrycks_math500_variant \
    --device cuda:0 \
    --batch_size auto
```

Compare with original:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks hendrycks_math500,hendrycks_math500_variant \
    --device cuda:0 \
    --batch_size auto
```

## Testing

Quick validation:
```bash
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks hendrycks_math500_variant \
    --limit 3 \
    --log_samples
```

Verify:
- Prompts contain "Question:" header
- Prompts contain "the final answer is (X)" instruction
- LaTeX and \\boxed{} notation still work
- Answer extraction handles both standard and LaTeX formats
- Mathematical equivalence checking works (e.g., "0.5" = "\frac{1}{2}")

## Expected Behavior

The variant should produce similar results to the original, as:
1. The "Question:" structure provides clearer formatting
2. The "the final answer is (X)" instruction guides model output
3. The extraction logic gracefully handles both formats
4. All mathematical normalization is preserved

Models may perform slightly better with explicit "think step by step" instruction.

## Notes

- **Original task preserved** - `hendrycks_math500.yaml` remains unchanged
- **Can run alongside original** - Use both for comparison
- **LaTeX compatible** - Supports full mathematical notation
- **Backward compatible** - Falls back to original extraction if needed
- **Base template provided** - `hendrycks_math_algebra_variant.yaml` can be reused for other MATH subjects
- **No few-shot**: Original is zero-shot, variant maintains this
- **4096 tokens**: Sufficient for complex mathematical reasoning with chain-of-thought
