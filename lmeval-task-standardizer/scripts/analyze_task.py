#!/usr/bin/env python3
"""
Analyze an existing lmeval task to understand its structure.

This script examines a task directory and reports:
- Current output type
- Dataset configuration
- Whether options are present
- Current prompt format
- Answer extraction method
"""

import os
import sys
import yaml
import json
from pathlib import Path


def load_yaml_file(yaml_path):
    """Load a YAML file, handling custom tags."""
    # Simple loader that ignores custom tags
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        pass

    def unknown_constructor(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return f"!{tag_suffix} {loader.construct_scalar(node)}"
        return None

    SafeLoaderIgnoreUnknown.add_multi_constructor('!', unknown_constructor)

    with open(yaml_path, 'r') as f:
        return yaml.load(f, Loader=SafeLoaderIgnoreUnknown)


def analyze_task_directory(task_dir):
    """Analyze a task directory and return key information."""
    task_path = Path(task_dir)

    if not task_path.exists():
        return {"error": f"Task directory does not exist: {task_dir}"}

    analysis = {
        "task_directory": str(task_path),
        "yaml_files": [],
        "python_files": [],
        "output_types": set(),
        "has_options": None,
        "has_fewshot": None,
        "current_format": "unknown"
    }

    # Find YAML files
    yaml_files = list(task_path.glob("*.yaml")) + list(task_path.glob("*.yml"))
    analysis["yaml_files"] = [str(f) for f in yaml_files]

    # Find Python files
    python_files = list(task_path.glob("*.py"))
    analysis["python_files"] = [str(f) for f in python_files]

    # Analyze YAML files
    for yaml_file in yaml_files:
        try:
            config = load_yaml_file(yaml_file)
            if config:
                # Check output type
                if "output_type" in config:
                    analysis["output_types"].add(config["output_type"])

                # Check for few-shot config
                if "fewshot_config" in config or config.get("num_fewshot", 0) > 0:
                    analysis["has_fewshot"] = True

                # Check for doc_to_choice (indicates multiple choice)
                if "doc_to_choice" in str(config):
                    analysis["current_format"] = "multiple_choice"

                # Check for doc_to_text
                if "doc_to_text" in str(config):
                    if "generate_until" in analysis["output_types"]:
                        analysis["current_format"] = "generate_until"

        except Exception as e:
            analysis.setdefault("warnings", []).append(f"Error reading {yaml_file}: {e}")

    # Convert set to list for JSON serialization
    analysis["output_types"] = list(analysis["output_types"])

    return analysis


def print_analysis(analysis):
    """Print analysis in a readable format."""
    if "error" in analysis:
        print(f"ERROR: {analysis['error']}")
        return

    print("=" * 60)
    print("TASK ANALYSIS")
    print("=" * 60)
    print(f"\nTask Directory: {analysis['task_directory']}")
    print(f"\nYAML Files Found: {len(analysis['yaml_files'])}")
    for f in analysis['yaml_files']:
        print(f"  - {f}")

    print(f"\nPython Files Found: {len(analysis['python_files'])}")
    for f in analysis['python_files']:
        print(f"  - {f}")

    print(f"\nOutput Types: {', '.join(analysis['output_types']) if analysis['output_types'] else 'None found'}")
    print(f"Current Format: {analysis['current_format']}")
    print(f"Has Few-shot: {analysis['has_fewshot']}")

    if analysis.get("warnings"):
        print("\nWARNINGS:")
        for w in analysis["warnings"]:
            print(f"  - {w}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if "multiple_choice" in analysis["output_types"]:
        print("✗ Task uses multiple_choice output type")
        print("  → Needs refactoring to generate_until")

    if "generate_until" in analysis["output_types"]:
        print("✓ Task already uses generate_until")
        print("  → Check if prompt format matches standard")

    if not analysis["has_fewshot"]:
        print("ℹ Task does not appear to have few-shot configuration")
        print("  → May need to add fewshot_config if dataset supports it")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_task.py <task_directory>")
        print("\nExample:")
        print("  python analyze_task.py /path/to/lm_eval/tasks/mmlu")
        sys.exit(1)

    task_dir = sys.argv[1]
    analysis = analyze_task_directory(task_dir)

    # Print readable output
    print_analysis(analysis)

    # Optionally save JSON
    if len(sys.argv) > 2 and sys.argv[2] == "--json":
        output_file = Path(task_dir) / "task_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    main()
