---
name: lmeval-task-standardizer
description: Standardize lmeval harness task formats to use generate_until output type with consistent prompting patterns. Use when the user mentions "evaluation refactor", "standardize task format", "refactor lmeval task", "convert to generate_until", or wants to update existing evaluation tasks to follow the standard format with "final answer is (X)" pattern.
---

# LM Eval Task Standardizer

This skill helps you refactor existing lmeval harness tasks to follow a standardized format that uses:
- `generate_until` as the only output type (no multiple choice)
- Consistent prompt formatting with questions and options in the user message
- Standardized answer extraction using the "final answer is (X)" regex pattern
- Proper separation of few-shot examples into individual user/assistant pairs

## When to Use This Skill

Use this skill when you need to:
- Refactor an existing lmeval task to the standardized format
- Convert multiple choice tasks to generate_until format
- Standardize prompting patterns across zero-shot and few-shot tasks
- Ensure consistent answer format with "the final answer is (X)"

## Standard Format Requirements

### Output Type
All tasks must use `output_type: generate_until`. Never use multiple_choice or other output types.

### Prompt Structure

**Zero-shot with options:**
```
Question: [question text]
Options: (A) [option1] (B) [option2] (C) [option3] (D) [option4]
Think step by step. Reply with 'the final answer is (X)'.
```

**Zero-shot without options:**
```
Question: [question text]
Think step by step. Reply with 'the final answer is (X)'.
```

**Few-shot prompting:**
Each example must be separated into distinct user/assistant message pairs. Do NOT concatenate multiple examples into a single request.

Example structure for few-shot with options:
```python
[
  {
    "role": "user", 
    "content": "Question: What is the capital of France?\nOptions: (A) London (B) Paris (C) Prague (D) Lisbon\nThink step by step. Reply with 'the final answer is (X)'."
  },
  {
    "role": "assistant", 
    "content": "The capital of France is Paris, which is known as the City of Light... the final answer is (B)."
  },
  # More examples follow the same pattern
]
```

### Answer Extraction

The regex pattern must expect "the final answer is (X)" format:
```yaml
filter_list:
  - name: "custom-extract"
    filter:
      - function: "regex"
        regex_pattern: 'the final answer is \*{0,2}\(?([A-Z])\)?\*{0,2}'
        group_select: -1
      - function: "take_first"
```

This pattern handles variations like:
- "the final answer is (B)"
- "the final answer is **B**"
- "the final answer is B"

## Workflow

### Step 1: Analyze the Existing Task

Read the task directory to understand:
1. Current task configuration (YAML files)
2. Current output type and prompt format
3. Whether the task has options or free-form answers
4. Dataset structure and splits
5. Existing utility functions

Look for:
- Task YAML files (usually in the task directory)
- Utils files (Python files with doc_to_text, doc_to_target functions)
- Dataset configuration

### Step 2: Determine Task Characteristics

Identify:
- **Has options?** Does the dataset include multiple choice options?
- **Answer format:** What does the target answer look like? (letter, text, number)
- **Few-shot support:** Does the dataset have a validation/few-shot split?
- **CoT examples:** Are chain-of-thought examples available in the dataset?

### Step 3: Create the Python Utils File

Generate `utils_chat.py` with two key functions:

**`doc_to_text(example)`** - Formats the question for the user message:
```python
def doc_to_text(example):
    """Format question with or without options."""
    prompt = "Question:\n" + example["question"] + "\n"
    
    # If task has options, add them
    if "options" in example:
        prompt += "Options:\n"
        for i, opt in enumerate(example["options"]):
            prompt += f"{choices[i]}. {opt.strip()}\n"
    
    prompt += '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'
    return prompt
```

**`fewshot_doc_to_target(example)`** - Formats the assistant's response:
```python
def fewshot_doc_to_target(example):
    """Format the assistant response for few-shot examples."""
    # Extract or format the CoT content
    cot_content = example.get("cot_content", "")
    # Remove any "A:" or "Answer:" prefixes
    cot_content = cot_content.replace("A: Let's think step by step.", "Let's think step by step.")
    return cot_content
```

**Why this matters:** The utils file ensures that questions and format instructions are provided directly in the user/assistant messages, not concatenated or hidden in system prompts. This makes the prompting explicit and testable.

### Step 4: Create the YAML Configuration

Generate a YAML config that follows this structure:

```yaml
dataset_path: [dataset_name]
test_split: test
fewshot_split: validation  # or train, if applicable
fewshot_config:
  sampler: first_n
  doc_to_text: !function utils_chat.doc_to_text
  doc_to_target: !function utils_chat.fewshot_doc_to_target
output_type: generate_until
doc_to_text: !function utils_chat.doc_to_text
doc_to_target: [answer_field]  # e.g., "answer", "target", etc.
filter_list:
  - name: "custom-extract"
    filter:
      - function: "regex"
        regex_pattern: 'answer is \*{0,2}\(?([A-Z])\)?\*{0,2}'
        group_select: -1
      - function: "take_first"
generation_kwargs:
  until: []
  max_gen_toks: 4096
  do_sample: true
num_fewshot: 5  # or 0 for zero-shot
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
metadata:
  version: 1.0
```

**Key points:**
- `output_type: generate_until` is mandatory
- `fewshot_config` references the utils functions for few-shot formatting
- `doc_to_text` references the utils function for the main task prompting
- The regex pattern is standardized across all tasks
- `max_gen_toks: 4096` allows for chain-of-thought reasoning

### Step 5: Handle Edge Cases

**Free-form answers (no options):**
- Adjust the regex pattern if answers are not letters (e.g., numbers, words)
- Example: `regex_pattern: 'answer is \*{0,2}([0-9]+)\*{0,2}'` for numeric answers
- Example: `regex_pattern: 'answer is \*{0,2}([A-Za-z\s]+)\*{0,2}'` for text answers

**Multiple answer formats in dataset:**
- Check if the dataset has both letter and text answers
- Normalize them in the utils functions if needed

**Missing CoT examples:**
- If no CoT content is available, `fewshot_doc_to_target` should just return the answer
- Consider using a simple format: "The answer is (X)."

### Step 6: Preserve Important Metadata

When refactoring, preserve:
- Dataset path and splits
- Metric configurations
- Any custom generation kwargs
- Task-specific metadata
- Version information

### Step 7: Create Backup and Output Files

1. Create a backup of the original files (add `.bak` extension)
2. Write the new `utils_chat.py` file
3. Write the new YAML configuration file (with `_chat` suffix if creating a variant)
4. Provide a summary of changes made

## Example Refactoring

**Before (multiple choice format):**
```yaml
output_type: multiple_choice
doc_to_choice: !function utils.doc_to_choice
doc_to_target: !function utils.doc_to_target
```

**After (generate_until format):**
```yaml
output_type: generate_until
doc_to_text: !function utils_chat.doc_to_text
doc_to_target: answer
filter_list:
  - name: "custom-extract"
    filter:
      - function: "regex"
        regex_pattern: 'the final answer is \*{0,2}\(?([A-Z])\)?\*{0,2}'
```

## Quality Checklist

Before finalizing the refactored task, verify:

- [ ] `output_type` is set to `generate_until`
- [ ] Question and format instructions are in the user message content
- [ ] The prompt explicitly instructs: "Think step by step. Reply with 'the final answer is (X)'"
- [ ] Few-shot examples are separated into individual user/assistant pairs (not concatenated)
- [ ] The regex pattern matches "the final answer is (X)" format
- [ ] `utils_chat.py` has both `doc_to_text` and `fewshot_doc_to_target` functions (if few-shot is supported)
- [ ] Original metadata and dataset configuration are preserved
- [ ] Backup files are created before overwriting

## Output Format

After refactoring, provide:

1. **Summary of changes:**
   - What output type was changed from/to
   - Whether options were added/removed from prompts
   - Changes to answer extraction logic

2. **File locations:**
   - Path to new/updated YAML config
   - Path to new/updated utils_chat.py
   - Path to backup files

3. **Next steps:**
   - How to test the refactored task
   - Any manual verification needed
   - Suggested validation commands

## Common Pitfalls to Avoid

1. **Don't concatenate few-shot examples** - Each example needs separate user/assistant messages
2. **Don't use multiple_choice output type** - Always use generate_until
3. **Don't hide format instructions** - They must be explicit in the prompt text
4. **Don't forget the regex pattern** - Must match "the final answer is (X)"
5. **Don't lose few-shot support** - If the original task had it, preserve it
6. **Don't change the answer format** - If answers are letters, keep them as letters in the regex
