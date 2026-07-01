# GPQA Diamond Zeroshot Variant - Summary

## Overview

Created `gpqa_diamond_zeroshot_variant` that standardizes the GPQA Diamond task to follow the complete format requirements:
- **Converts from `multiple_choice` to `generate_until`**
- Adds "Question:" and "Options:" structure to prompts
- Uses "the final answer is (X)" pattern consistently
- Maintains answer shuffling for fairness

## Files Created

1. **`utils_variant.py`** - Standardized formatting utilities
2. **`_gpqa_zeroshot_variant_template.yaml`** - Template for all GPQA variant tasks
3. **`gpqa_diamond_zeroshot_variant.yaml`** - Diamond variant configuration

## Key Changes from Original

### 1. Output Type Change

**Original (`_gpqa_zeroshot_yaml`):**
```yaml
output_type: multiple_choice
doc_to_choice: ["(A)", "(B)", "(C)", "(D)"]
```

**Variant (`_gpqa_zeroshot_variant_template.yaml`):**
```yaml
output_type: generate_until
filter_list:
  - name: "custom-extract"
    filter:
      - function: "regex"
        regex_pattern: 'the final answer is \*{0,2}\(?([ABCD])\)?\*{0,2}'
```

### 2. Prompt Structure

**Original:**
```
What is the correct answer to this question:{{Question}}
Choices:
(A) {{choice1}}
(B) {{choice2}}
(C) {{choice3}}
(D) {{choice4}}
Answer:
```

**Variant:**
```
Question:
{Question}
Options:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}

Think step by step and then finish your answer with "the final answer is (X)" where X is the correct letter choice.
```

### 3. Metrics

**Original:**
```yaml
metric_list:
  - metric: acc
  - metric: acc_norm
```

**Variant:**
```yaml
metric_list:
  - metric: exact_match
    ignore_case: true
    ignore_punctuation: true
```

### 4. Utils Functions

**Original:** `process_docs` only, with inline template strings

**Variant:** Enhanced `utils_variant.py` with:
- `process_docs()` - Shuffles choices and preserves Question field
- `doc_to_text()` - Formats question with "Question:" and "Options:" structure
- `doc_to_target()` - Returns letter (A-D) without parentheses

## What Was Preserved

✅ **Dataset**: `Idavidrein/gpqa` with `gpqa_diamond` subset  
✅ **Answer shuffling**: Same randomization logic for fairness  
✅ **Zero-shot**: `num_fewshot: 0`  
✅ **Preprocessing**: Same text cleaning (title removal, whitespace)  
✅ **4 choices**: A, B, C, D format  
✅ **Greedy decoding**: `do_sample: false, temperature: 0.0`  

## About GPQA

GPQA (Graduate-Level Google-Proof Q&A) is a challenging multiple-choice benchmark with:
- **Domain**: Graduate-level science questions (Biology, Physics, Chemistry)
- **Difficulty**: Designed to be challenging even for experts
- **Diamond subset**: Highest quality questions, validated by multiple domain experts
- **Google-proof**: Questions designed to be difficult to answer via search

## Prompt Format Example

### Question Formatting (doc_to_text)
```
Question:
In a certain chemical reaction, the activation energy is 50 kJ/mol and the rate constant at 298K is 1.0 × 10^-3 s^-1. What is the rate constant at 350K? (Assume the pre-exponential factor remains constant)
Options:
(A) 2.5 × 10^-3 s^-1
(B) 5.2 × 10^-3 s^-1
(C) 8.7 × 10^-3 s^-1
(D) 1.2 × 10^-2 s^-1

Think step by step and then finish your answer with "the final answer is (X)" where X is the correct letter choice.
```

### Expected Model Response
```
Let me work through this using the Arrhenius equation.

[... reasoning about activation energy and temperature ...]

Using k2/k1 = exp[Ea/R × (1/T1 - 1/T2)], I can calculate:
k2 = 1.0 × 10^-3 × exp[50000/8.314 × (1/298 - 1/350)]
k2 ≈ 8.7 × 10^-3 s^-1

the final answer is (C)
```

## Comparison Table

| Feature | Original | Variant | Status |
|---------|----------|---------|---------|
| Output type | multiple_choice | generate_until | ✅ **Converted** |
| Prompt structure | Template | Question:/Options: | ✅ **Standardized** |
| Instruction | "Answer:" | "the final answer is (X)" | ✅ **Standardized** |
| Answer format | (A) with parens | A without parens | ✅ **Simplified** |
| Metrics | acc, acc_norm | exact_match | ✅ **Changed** |
| Answer shuffling | Yes | Yes | ✅ **Preserved** |
| Zero-shot | Yes | Yes | ✅ **Preserved** |
| Utils file | utils.py | utils_variant.py | ✅ **Added** |

## Why Convert from Multiple Choice?

**Benefits of generate_until:**
1. ✅ **Consistent with other tasks**: All standardized tasks use generate_until
2. ✅ **Chain-of-thought friendly**: Allows model to reason before answering
3. ✅ **More realistic**: Matches how humans solve problems (explain then answer)
4. ✅ **Better for instruction-tuned models**: Chat models work better with explicit instructions

**Original multiple_choice approach:**
- Compares log-probabilities of each choice
- Works well for base models
- Less natural for chat-tuned models

## Usage

Run the variant:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks gpqa_diamond_zeroshot_variant \
    --device cuda:0 \
    --batch_size auto
```

Compare with original:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks gpqa_diamond_zeroshot,gpqa_diamond_zeroshot_variant \
    --device cuda:0 \
    --batch_size auto
```

## Creating Other GPQA Variants

The template `_gpqa_zeroshot_variant_template.yaml` can be used for other GPQA subsets:

```yaml
# gpqa_main_zeroshot_variant.yaml
dataset_name: gpqa_main
include: _gpqa_zeroshot_variant_template.yaml
task: gpqa_main_zeroshot_variant
```

```yaml
# gpqa_extended_zeroshot_variant.yaml
dataset_name: gpqa_extended
include: _gpqa_zeroshot_variant_template.yaml
task: gpqa_extended_zeroshot_variant
```

## Testing

Quick validation:
```bash
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks gpqa_diamond_zeroshot_variant \
    --limit 3 \
    --log_samples
```

Verify:
- Prompts contain "Question:" header
- Prompts contain "Options:" with A-D choices
- Prompts contain "the final answer is (X)" instruction
- Regex extraction works correctly
- Answers are shuffled (different order each run due to random seed)
- Metrics are computed correctly

## Expected Behavior

Results may differ from the original because:
1. **Different evaluation method**: generate_until vs multiple_choice
2. **Different scoring**: exact_match on generated text vs log-probability comparison
3. **Model-dependent**: Chat models may perform better with explicit instructions

For **base models** (non-chat): Original multiple_choice may perform better  
For **instruction-tuned models**: Variant should perform better or comparably

## Notes

- **Original task preserved** - `gpqa_diamond_zeroshot.yaml` remains unchanged
- **Can run alongside original** - Compare both evaluation methods
- **Template provided** - Easy to create variants for other GPQA subsets
- **Answer shuffling** - Maintains fairness (same random shuffling as original)
- **Zero-shot only** - GPQA is designed for zero-shot evaluation
- **Graduate-level difficulty** - Expect lower accuracy than typical benchmarks
