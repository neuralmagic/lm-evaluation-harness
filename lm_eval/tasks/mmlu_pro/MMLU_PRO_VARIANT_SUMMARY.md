# MMLU-Pro Variant - Summary

## Overview

Created `mmlu_pro_variant` that **fixes the inconsistency** in the existing `_mmlu_pro_chat_yaml` reference and implements the **complete standardized format** with "the final answer is (X)" pattern.

## Problem Identified

The existing `_mmlu_pro_chat_yaml` has a **mismatch**:
- **utils_chat.py** (line 17): `"the answer is (X)"`
- **Regex pattern** (line 15): `'answer is \*{0,2}\(?([ABCDEFGHIJ])\)?\*{0,2}'`

This is missing both "the" and "final" from the regex pattern, which could cause extraction failures.

## Files Created

### Core Template Files
1. **`utils_variant.py`** - Standardized formatting utilities with correct "the final answer is" instruction
2. **`_mmlu_pro_variant_template.yaml`** - Template with corrected regex pattern
3. **`_mmlu_pro_variant.yaml`** - Group configuration for all subjects

### Subject-Specific Files (14 total)
- `mmlu_pro_variant_biology.yaml`
- `mmlu_pro_variant_business.yaml`
- `mmlu_pro_variant_chemistry.yaml`
- `mmlu_pro_variant_computer_science.yaml`
- `mmlu_pro_variant_economics.yaml`
- `mmlu_pro_variant_engineering.yaml`
- `mmlu_pro_variant_health.yaml`
- `mmlu_pro_variant_history.yaml`
- `mmlu_pro_variant_law.yaml`
- `mmlu_pro_variant_math.yaml`
- `mmlu_pro_variant_other.yaml`
- `mmlu_pro_variant_philosophy.yaml`
- `mmlu_pro_variant_physics.yaml`
- `mmlu_pro_variant_psychology.yaml`

## Key Changes from Reference (_mmlu_pro_chat_yaml)

### ✅ FIXED: Regex Pattern
**Before:**
```yaml
regex_pattern: 'answer is \*{0,2}\(?([ABCDEFGHIJ])\)?\*{0,2}'
```

**After:**
```yaml
regex_pattern: 'the final answer is \*{0,2}\(?([ABCDEFGHIJ])\)?\*{0,2}'
```

### ✅ UPDATED: Prompt Instruction
**Before (utils_chat.py):**
```python
+ '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'
```

**After (utils_variant.py):**
```python
+ '\n\nThink step by step and then finish your answer with "the final answer is (X)" where X is the correct letter choice.'
```

### ✅ Enhanced: Documentation
- Added metadata description
- Improved code comments
- Created comprehensive summary

## What Was Preserved

✅ **Dataset**: `TIGER-Lab/MMLU-Pro`  
✅ **Output type**: `generate_until`  
✅ **Few-shot**: 5-shot with CoT examples  
✅ **Fewshot config**: Proper separation with `doc_to_text` and `fewshot_doc_to_target`  
✅ **Options**: A-J letter choices  
✅ **Metrics**: exact_match with case/punctuation insensitivity  
✅ **Generation kwargs**: max_gen_toks=4096, do_sample=true  

## Prompt Format Example

### Question Formatting (doc_to_text)
```
Question:
What is the primary function of mitochondria in eukaryotic cells?
Options:
A. Protein synthesis
B. Energy production through ATP synthesis
C. DNA replication
D. Lipid storage
E. Cell division

Think step by step and then finish your answer with "the final answer is (X)" where X is the correct letter choice.
```

### Few-shot Answer Formatting (fewshot_doc_to_target)
```
Let's think step by step. Mitochondria are known as the powerhouse of the cell because they generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy. While they do contain their own DNA and can replicate, their primary function is energy production through cellular respiration. The final answer is (B).
```

## Comparison Table

| Feature | _mmlu_pro_chat_yaml | mmlu_pro_variant | Status |
|---------|---------------------|------------------|---------|
| Output type | generate_until | generate_until | ✅ Same |
| Few-shot | 5-shot | 5-shot | ✅ Same |
| Options | A-J | A-J | ✅ Same |
| Prompt instruction | "the answer is (X)" | "the final answer is (X)" | ✅ **Fixed** |
| Regex pattern | `answer is` | `the final answer is` | ✅ **Fixed** |
| Utils file | utils_chat.py | utils_variant.py | ✅ New |
| Consistency | ❌ Mismatch | ✅ **Consistent** | ✅ **Fixed** |

## Usage

Run a single subject:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu_pro_variant_biology \
    --device cuda:0 \
    --batch_size auto
```

Run all subjects:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu_pro_variant \
    --device cuda:0 \
    --batch_size auto
```

Compare with original:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu_pro,mmlu_pro_variant \
    --device cuda:0 \
    --batch_size auto
```

## Testing

Quick validation test:
```bash
# Test with a small model on limited examples
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu_pro_variant_biology \
    --limit 5 \
    --log_samples
```

Verify:
- Prompts contain "the final answer is (X)" instruction
- Regex extraction works correctly
- Few-shot examples are properly formatted
- Metrics are computed correctly

## Why This Matters

**Consistency is critical for evaluation:**
1. **Instruction-pattern mismatch** can cause models to respond with one format while extraction expects another
2. **Regex failures** lead to incorrect 0% scores even when models answer correctly
3. **Standardization** ensures fair comparison across different tasks and models

The variant fixes the mismatch and ensures:
- ✅ Models are instructed to use "the final answer is (X)"
- ✅ Regex extracts "the final answer is (X)"
- ✅ Both instruction and extraction are aligned

## Notes

- **Original mmlu_pro tasks remain unchanged** - this is a new variant
- **Can run alongside** existing mmlu_pro_chat tasks for comparison
- **Follows complete standard** - ready for integration with other standardized tasks
- **All 14 subjects included** - biology, business, chemistry, computer science, economics, engineering, health, history, law, math, other, philosophy, physics, psychology
