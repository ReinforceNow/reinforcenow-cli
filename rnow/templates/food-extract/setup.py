#!/usr/bin/env python3
"""Download FoodExtract-1k dataset: python setup.py --samples 100"""

import argparse
import base64
import io
import json

PROMPT = 'Extract food and drink items from this image. Respond with JSON: {"is_food": 0|1, "food_items": [...], "drink_items": [...]}'


def image_to_base64(img) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.samples} samples from FoodExtract-1k...")
    ds = load_dataset("mrdbourke/FoodExtract-1k-Vision", split=f"train[:{args.samples}]")

    data = []

    for i, s in enumerate(ds):
        print(f"{i+1}/{args.samples}")

        # Convert image to base64 inline
        img = s["image"]
        image_b64 = image_to_base64(img)

        # Get labels
        label = s["output_label_json"]
        food_items = label.get("food_items", [])
        drink_items = label.get("drink_items", [])
        is_food = label.get("is_food", 0)

        data.append(
            {
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_b64},
                            {"type": "text", "text": "What food and drinks are in this image?"},
                        ],
                    },
                ],
                "rewards": ["extraction_accuracy"],
                "metadata": {
                    "is_food": is_food,
                    "food_items": food_items,
                    "drink_items": drink_items,
                },
            }
        )

    with open("train.jsonl", "w") as f:
        for e in data:
            f.write(json.dumps(e) + "\n")

    print(f"Done: {len(data)} samples")


if __name__ == "__main__":
    main()
