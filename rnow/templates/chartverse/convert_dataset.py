"""
Convert ChartVerse-RL-40K from HuggingFace to rnow format.

Usage:
    uv run --with datasets --with pillow python convert_dataset.py --limit 100
    uv run --with datasets --with pillow python convert_dataset.py  # Full dataset
"""

import argparse
import base64
import io
import json
from pathlib import Path


def image_to_base64(img) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def convert_chartverse(limit: int | None = None):
    """Download and convert ChartVerse-RL-40K to rnow format."""
    from datasets import load_dataset

    print("Loading ChartVerse-RL-40K from HuggingFace...")
    ds = load_dataset("opendatalab/ChartVerse-RL-40K", split="train")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"Processing first {len(ds)} samples...")
    else:
        print(f"Processing all {len(ds)} samples...")

    # Convert to rnow format
    output_file = Path("train.jsonl")
    converted = 0
    errors = 0

    with open(output_file, "w") as f:
        for i, row in enumerate(ds):
            try:
                # Convert image to base64
                img = row["images"][0]  # PIL Image
                image_b64 = image_to_base64(img)

                # Build messages from prompt
                messages = []
                for turn in row["prompt"]:
                    content = turn["content"]
                    role = turn["role"]

                    # Add boxed instruction to system prompt
                    if role == "system":
                        content = content.strip() + "\n\nProvide your final answer inside \\boxed{}"

                    if "<image>" in content:
                        # Multi-part content with image (base64 inline)
                        text = content.replace("<image>", "").strip()
                        messages.append(
                            {
                                "role": role,
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image_b64,
                                    },
                                    {"type": "text", "text": text},
                                ],
                            }
                        )
                    else:
                        messages.append({"role": role, "content": content})

                # Extract ground truth
                ground_truth = row["reward_model"]["ground_truth"]

                entry = {
                    "messages": messages,
                    "rewards": ["accuracy"],
                    "metadata": {
                        "answer": ground_truth,
                        "category": row.get("category", "chart"),
                        "ability": row.get("ability", "chart"),
                        "source": row.get("source", "chartverse"),
                    },
                }

                f.write(json.dumps(entry) + "\n")
                converted += 1

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1} samples...")

            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                errors += 1

    print(f"\nDone! Converted {converted} samples, {errors} errors")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ChartVerse-RL-40K to rnow format")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )
    args = parser.parse_args()

    convert_chartverse(limit=args.limit)
