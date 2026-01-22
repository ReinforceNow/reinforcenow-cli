#!/usr/bin/env python3
"""
Convert MedBrowseComp HuggingFace dataset to prompts.jsonl.

Downloads the full training set (605 samples) by default.

Usage:
    python convert_dataset.py                     # Full 605 samples
    python convert_dataset.py --split MedBrowseComp_50  # Small test (50)
    python convert_dataset.py --limit 100         # First 100 samples
"""

import argparse
import json

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser(description="Convert MedBrowseComp to prompts.jsonl")
    p.add_argument("-o", "--output", default="prompts.jsonl", help="Output file")
    p.add_argument(
        "--split",
        default="MedBrowseComp_605",
        choices=["MedBrowseComp_50", "MedBrowseComp_605", "MedBrowseComp_CUA"],
        help="Dataset split (default: MedBrowseComp_605 = 605 training samples)",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = p.parse_args()

    print(f"Loading MedBrowseComp ({args.split})...")
    ds = load_dataset("AIM-Harvard/MedBrowseComp", split=args.split)

    num = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Converting {num} of {len(ds)} samples...")

    with open(args.output, "w") as f:
        for i in range(num):
            entry = ds[i]
            f.write(
                json.dumps(
                    {
                        "prompt": entry["prompt"],
                        "metadata": {
                            "expected_answer": entry["gold"],
                            "task_name": entry.get("task_name", ""),
                        },
                    }
                )
                + "\n"
            )

    print(f"Saved {num} prompts to {args.output}")

    # Show task distribution
    from collections import Counter

    tasks = Counter(ds[i].get("task_name", "unknown") for i in range(num))
    print("\nTask distribution:")
    for task, count in tasks.most_common():
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
