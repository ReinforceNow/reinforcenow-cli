#!/usr/bin/env python3
"""
Convert rollouts from rnow test to train.jsonl format for SFT training.

Usage:
    python convert_rollouts.py                    # Process all rollouts/*.json
    python convert_rollouts.py -o train.jsonl    # Custom output file
    python convert_rollouts.py --filter-success  # Only include successful rollouts
"""

import argparse
import json
from pathlib import Path


def convert_rollouts(
    rollouts_dir: Path = Path("rollouts"),
    output_file: Path = Path("train.jsonl"),
    filter_success: bool = True,
) -> int:
    """Convert rollout JSON files to train.jsonl format.

    Returns number of converted rollouts.
    """
    if not rollouts_dir.exists():
        print(f"Error: {rollouts_dir} not found. Run 'rnow test' first.")
        return 0

    rollout_files = sorted(rollouts_dir.glob("*.json"))
    if not rollout_files:
        print(f"No rollout files found in {rollouts_dir}/")
        return 0

    converted = 0
    skipped = 0

    with open(output_file, "w") as f:
        for rollout_path in rollout_files:
            try:
                data = json.loads(rollout_path.read_text())
            except json.JSONDecodeError:
                print(f"  Skipping {rollout_path.name}: invalid JSON")
                skipped += 1
                continue

            # Check completion status
            completed = data.get("completed")
            if filter_success and completed is not True:
                skipped += 1
                continue

            # Get conversation
            conversation = data.get("conversation", [])
            if not conversation:
                skipped += 1
                continue

            # Build train.jsonl entry
            entry = {"messages": conversation}

            # Include metadata if present
            if data.get("metadata"):
                entry["metadata"] = data["metadata"]

            f.write(json.dumps(entry) + "\n")
            converted += 1

    print(f"Converted {converted} rollouts to {output_file}")
    if skipped:
        print(f"Skipped {skipped} (incomplete or failed)")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert rollouts to train.jsonl")
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path("rollouts"),
        help="Rollouts directory (default: rollouts/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("train.jsonl"),
        help="Output file (default: train.jsonl)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed/incomplete rollouts",
    )
    args = parser.parse_args()

    convert_rollouts(
        rollouts_dir=args.dir,
        output_file=args.output,
        filter_success=not args.include_failed,
    )


if __name__ == "__main__":
    main()
