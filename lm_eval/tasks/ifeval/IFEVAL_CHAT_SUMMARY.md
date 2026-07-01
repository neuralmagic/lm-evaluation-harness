# IFEval Chat Variant - Summary

## Overview

Created `ifeval_chat` variant that adds standardized prompt formatting to the IFEval instruction-following evaluation task.

## Files Created

1. **`utils_chat.py`** - Chat formatting utilities
2. **`ifeval_chat.yaml`** - Chat variant configuration

## Key Changes

### From Original (ifeval.yaml)
```yaml
task: ifeval
doc_to_text: prompt  # Raw prompt field
metadata:
  version: 4.0
```

### To Chat Variant (ifeval_chat.yaml)
```yaml
task: ifeval_chat
doc_to_text: !function utils_chat.doc_to_text  # Formatted with thinking instruction
metadata:
  version: 1.0
  description: "Chat variant of IFEval with standardized 'Think step by step' instruction"
```

## What Was Preserved

✅ **Dataset configuration**: `google/IFEval`  
✅ **Output type**: `generate_until` (already correct)  
✅ **Zero-shot evaluation**: `num_fewshot: 0`  
✅ **Custom evaluation logic**: `process_results: !function utils.process_results`  
✅ **All metrics**: prompt_level_strict_acc, inst_level_strict_acc, prompt_level_loose_acc, inst_level_loose_acc  
✅ **Generation kwargs**: temperature=0.0, max_gen_toks=1280, greedy decoding  

## What Was Added

➕ **Standardized prompt formatting**: `utils_chat.doc_to_text()` appends "Think step by step." to each prompt  
➕ **Chat-friendly structure**: Uses function reference for doc_to_text instead of direct field access  

## What Was NOT Added (Intentionally)

❌ **No filter_list**: IFEval doesn't extract answers - it evaluates if instructions were followed  
❌ **No "final answer is (X)" instruction**: Not applicable to instruction-following tasks  
❌ **No multiple choice options**: IFEval is free-form instruction following  
❌ **No few-shot configuration**: Task is designed for zero-shot evaluation  

## Prompt Format Comparison

### Original IFEval
```
Write a 300+ word essay about the importance of time management.
Mention the word "productivity" at least 3 times.
Your entire response should be in English, and in all lowercase letters.
```

### IFEval Chat Variant
```
Write a 300+ word essay about the importance of time management.
Mention the word "productivity" at least 3 times.
Your entire response should be in English, and in all lowercase letters.

Think step by step.
```

## Why This Approach?

IFEval is unique because:
1. **No "correct answer"** - Success is measured by instruction adherence, not answer correctness
2. **Complex evaluation** - Uses custom logic to verify constraints (word count, keyword mentions, formatting rules)
3. **Diverse instructions** - 25 types of verifiable instructions, from length constraints to formatting requirements

The chat variant adds minimal formatting ("Think step by step.") to encourage reasoning without interfering with the instruction-following evaluation.

## Usage

Run the chat variant:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks ifeval_chat \
    --device cuda:0 \
    --batch_size auto
```

Compare with original:
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks ifeval,ifeval_chat \
    --device cuda:0 \
    --batch_size auto
```

## Expected Behavior

The chat variant should produce similar or slightly improved results compared to the original, as "Think step by step." may encourage more careful instruction following without changing the evaluation criteria.

## Testing

To verify the variant works correctly:

```bash
# Quick test with 10 examples
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks ifeval_chat \
    --limit 10 \
    --log_samples
```

Check that:
- All prompts have "Think step by step." appended
- The custom metrics are computed correctly
- Results are in the expected range (0-1 for accuracy metrics)

## Notes

- **Version reset to 1.0**: This is a new variant, not an update to the original task
- **Backward compatible**: Original `ifeval` task remains unchanged
- **Minimal intervention**: Only adds thinking instruction, preserves all evaluation logic
- **No answer extraction**: Unlike standard Q&A tasks, ifeval_chat doesn't need filter_list
