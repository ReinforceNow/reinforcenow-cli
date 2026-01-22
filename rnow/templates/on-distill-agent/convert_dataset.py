#!/usr/bin/env python3
"""
Convert MedBrowseComp dataset from HuggingFace to train.jsonl format.

Usage:
    python convert_dataset.py                              # Full 605 samples
    python convert_dataset.py --split MedBrowseComp_50     # Small test (50)
    python convert_dataset.py --limit 100                  # First 100 samples
"""

import argparse
import json

from datasets import load_dataset

SYSTEM_PROMPT = """You are a medical research assistant. Your task is to find specific information about clinical trials, drugs, FDA approvals, patents, and pharmaceutical companies.

You have access to the browse tool to fetch web pages. Use it to access:
- Clinical trials: https://clinicaltrials.gov/study/NCT...
- FDA drug database: https://www.accessdata.fda.gov/scripts/cder/daf/
- Drug patents: https://www.accessdata.fda.gov/scripts/cder/ob/
- Stock prices: https://finance.yahoo.com/quote/SYMBOL

Return answers in the exact format requested in the prompt."""


def convert_entry(entry: dict) -> dict:
    """Convert a single MedBrowseComp entry to train.jsonl format.

    Note: On-policy distillation uses teacher KL penalty for supervision,
    NOT reward functions. The 'rewards' field is omitted.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry["prompt"]},
        ],
        "tools": ["browse"],
        # No 'rewards' - distillation uses teacher KL penalty, not reward functions
        "metadata": {
            "expected_answer": entry["gold"],
            "task_name": entry["task_name"],
            "split": entry["split"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MedBrowseComp to train.jsonl")
    parser.add_argument(
        "--split",
        type=str,
        default="MedBrowseComp_605",
        choices=["MedBrowseComp_50", "MedBrowseComp_605", "MedBrowseComp_CUA"],
        help="Dataset split (default: MedBrowseComp_605 = 605 training samples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries (for testing)",
    )
    args = parser.parse_args()

    print(f"Loading MedBrowseComp dataset (split: {args.split})...")
    dataset = load_dataset("AIM-Harvard/MedBrowseComp", split=args.split)

    print(f"Converting {len(dataset)} entries...")
    entries = []
    for i, entry in enumerate(dataset):
        if args.limit and i >= args.limit:
            break
        entries.append(convert_entry(entry))

    print(f"Writing to {args.output}...")
    with open(args.output, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Done! Converted {len(entries)} entries.")
    print("\nTask distribution:")
    task_counts = {}
    for entry in entries:
        task = entry["metadata"]["task_name"]
        task_counts[task] = task_counts.get(task, 0) + 1
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
