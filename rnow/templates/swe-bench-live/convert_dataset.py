#!/usr/bin/env python3
"""
Convert SWE-bench-Live HuggingFace dataset to ReinforceNow train.jsonl format.

Usage:
    pip install datasets
    python convert_dataset.py --split lite --output train.jsonl
    python convert_dataset.py --split full --output train_full.jsonl --limit 100
"""

import argparse
import json

SYSTEM_PROMPT = """You are an expert software engineer. Your task is to fix the GitHub issue described below.

The repository is available at /testbed. Use the provided tools to:
1. Explore the codebase to understand the issue
2. Locate the relevant files
3. Make the necessary code changes
4. Verify your fix by running the tests

Be methodical: read the issue carefully, search for relevant code, understand the context before making changes."""


def instance_id_to_docker_image(instance_id):
    # type: (str) -> str
    """Convert SWE-bench instance_id to Docker image name.

    The convention is:
    - Replace '__' with '_1776_'
    - Lowercase everything
    - Prefix with 'starryzhang/sweb.eval.x86_64.'
    """
    name = instance_id.replace("__", "_1776_").lower()
    return f"starryzhang/sweb.eval.x86_64.{name}"


def convert_row(row, max_pass_to_pass=5, max_problem_chars=12000):
    # type: (dict, int, int) -> dict
    """Convert a single HuggingFace dataset row to train.jsonl format.

    Args:
        row: Dataset row from HuggingFace
        max_pass_to_pass: Max number of pass_to_pass tests to include (default 5).
                         Set to 0 to omit entirely, -1 for all (can be huge).
        max_problem_chars: Max characters for problem statement (default 12000, ~3K tokens).
                          Some GitHub issues have huge log dumps that exceed context windows.
    """
    instance_id = row["instance_id"]
    problem_statement = row["problem_statement"]

    # Truncate long problem statements (some have massive log dumps)
    if len(problem_statement) > max_problem_chars:
        problem_statement = (
            problem_statement[:max_problem_chars] + "\n\n[... truncated due to length ...]"
        )
    test_cmds = row.get("test_cmds", [])
    fail_to_pass = row.get("FAIL_TO_PASS", [])
    pass_to_pass = row.get("PASS_TO_PASS", [])

    # Parse JSON strings if needed (HF sometimes stores lists as JSON strings)
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    if isinstance(pass_to_pass, str):
        pass_to_pass = json.loads(pass_to_pass)
    if isinstance(test_cmds, str):
        test_cmds = json.loads(test_cmds)

    # Limit pass_to_pass to save space
    if max_pass_to_pass == 0:
        pass_to_pass = []
    elif max_pass_to_pass > 0:
        pass_to_pass = pass_to_pass[:max_pass_to_pass]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Issue: {problem_statement}"},
        ],
        "rewards": ["has_changes", "resolved"],
        "tools": ["bash", "read_file", "write_file", "edit_file", "search_files", "list_files"],
        "docker": instance_id_to_docker_image(instance_id),
        "metadata": {
            "instance_id": instance_id,
            "test_cmds": test_cmds,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert SWE-bench-Live to train.jsonl")
    parser.add_argument(
        "--split",
        default="lite",
        help="Dataset split: lite (300), test (1000), full (1888), verified (500)",
    )
    parser.add_argument(
        "--output", default="train.jsonl", help="Output file path (default: train.jsonl)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of instances (default: all)"
    )
    args = parser.parse_args()

    # Import here so the script can show help without datasets installed
    from datasets import load_dataset

    print(f"Loading SWE-bench-Live dataset (split: {args.split})...")
    ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split=args.split)

    count = 0
    with open(args.output, "w") as f:
        for row in ds:
            if args.limit and count >= args.limit:
                break

            try:
                entry = convert_row(row)
                f.write(json.dumps(entry) + "\n")
                count += 1
            except Exception as e:
                print(f"Warning: Skipping {row.get('instance_id', 'unknown')}: {e}")

    print(f"Wrote {count} entries to {args.output}")
    print("\nNext steps:")
    print(f"  1. Review the generated file: head -1 {args.output} | python -m json.tool")
    print("  2. Run training: rnow run")


if __name__ == "__main__":
    main()
