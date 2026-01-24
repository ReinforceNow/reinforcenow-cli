#!/usr/bin/env python3
"""
Convert HuggingFace zenml/cuad-deepseek dataset to train.jsonl for SFT.

This dataset has DeepSeek-R1 reasoning traces for legal contract classification.
Perfect for distilling reasoning capabilities into smaller models.

Usage:
    python convert_hf_dataset.py
    python convert_hf_dataset.py --max-samples 100
    python convert_hf_dataset.py --split train --max-samples 1000
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert CUAD-DeepSeek to train.jsonl")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to convert")
    parser.add_argument("--split", default="train", help="Dataset split (train/validation/test)")
    parser.add_argument("-o", "--output", default="train.jsonl", help="Output file")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess

        subprocess.run(["pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    print(f"Loading zenml/cuad-deepseek ({args.split} split)...")
    dataset = load_dataset("zenml/cuad-deepseek", split=args.split)

    print(f"Dataset has {len(dataset)} samples")

    # System prompt for legal classification
    system_prompt = """You are a legal expert specializing in contract analysis. Your task is to classify contract clauses into one of the legal categories.

When analyzing a clause:
1. Read the clause carefully within its context
2. Think through the legal implications step by step
3. Consider which category best fits the clause
4. Provide your reasoning, then give the final classification

Format your response as:
<reasoning>
[Your detailed analysis]
</reasoning>

Classification: [category]"""

    entries = []
    for i, row in enumerate(dataset):
        if i >= args.max_samples:
            break

        # Build the user message with clause and context
        clause = row.get("clause", "")
        context = row.get("clause_with_context", "")
        contract_type = row.get("contract_type", "Unknown")

        user_content = f"""Contract Type: {contract_type}

Clause with Context:
{context}

Specific Clause to Classify:
{clause}

What is the legal classification of this clause?"""

        # Build assistant response from reasoning trace and label
        reasoning = row.get("reasoning_trace", "")
        label = row.get("label", "NONE")
        rationale = row.get("rationale", "")

        # Format assistant response
        assistant_content = f"""<reasoning>
{reasoning}

Summary: {rationale}
</reasoning>

Classification: {label}"""

        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "label": label,
                "contract_type": contract_type,
                "contract_name": row.get("contract_name", ""),
            },
        }
        entries.append(entry)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples...")

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nConverted {len(entries)} samples to {output_path}")
    print("\nTo train:")
    print("  rnow run")


if __name__ == "__main__":
    main()
