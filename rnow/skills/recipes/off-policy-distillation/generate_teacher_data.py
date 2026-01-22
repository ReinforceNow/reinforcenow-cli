#!/usr/bin/env python3
"""
Generate teacher completions for off-policy distillation using OpenRouter.

This script takes a file of prompts and generates completions from a teacher model,
outputting a train.jsonl file ready for SFT training.

Requirements:
    pip install httpx python-dotenv tqdm

Setup:
    1. Get an API key from https://openrouter.ai
    2. Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-your-key-here

Usage:
    python generate_teacher_data.py --input prompts.txt --output train.jsonl
    python generate_teacher_data.py --input prompts.jsonl --model openai/gpt-5.2 --concurrency 10
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def load_prompts(input_path: str) -> list[dict]:
    """
    Load prompts from input file.

    Supports:
    - Plain text files (one prompt per line)
    - JSONL files with {"prompt": "...", "system": "...", "metadata": {...}}
    """
    prompts = []
    path = Path(input_path)

    if not path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON first
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    if "prompt" not in data:
                        print(f"Warning: Line {i+1} missing 'prompt' field, skipping")
                        continue
                    prompts.append(data)
                else:
                    # JSON but not a dict, treat as plain text
                    prompts.append({"prompt": line})
            except json.JSONDecodeError:
                # Plain text line
                prompts.append({"prompt": line})

    return prompts


def load_existing_outputs(output_path: str) -> set[str]:
    """Load already-generated prompts to support resuming."""
    existing = set()
    path = Path(output_path)

    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extract the user message to identify the prompt
                    for msg in data.get("messages", []):
                        if msg.get("role") == "user":
                            existing.add(msg.get("content", ""))
                            break
                except json.JSONDecodeError:
                    continue

    return existing


async def generate_completion(
    client: httpx.AsyncClient,
    prompt_data: dict,
    model: str,
    default_system: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single completion from the teacher model."""
    async with semaphore:
        # Build messages
        messages = []

        # System prompt (per-entry overrides default)
        system = prompt_data.get("system", default_system)
        if system:
            messages.append({"role": "system", "content": system})

        # User prompt
        messages.append({"role": "user", "content": prompt_data["prompt"]})

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://reinforcenow.ai",
            "X-Title": "ReinforceNow Distillation",
        }

        try:
            response = await client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            result = response.json()
            assistant_content = result["choices"][0]["message"]["content"]

            # Build output entry
            output_messages = messages.copy()
            output_messages.append({"role": "assistant", "content": assistant_content})

            entry = {"messages": output_messages}

            # Preserve metadata and add teacher info
            metadata = prompt_data.get("metadata", {}).copy()
            metadata["teacher"] = model
            if metadata:
                entry["metadata"] = metadata

            return entry

        except httpx.TimeoutException:
            print(f"\nTimeout for prompt: {prompt_data['prompt'][:50]}...")
            return None
        except httpx.HTTPStatusError as e:
            print(f"\nHTTP error {e.response.status_code}: {e.response.text[:100]}")
            return None
        except Exception as e:
            print(f"\nError: {e}")
            return None


async def generate_all(
    prompts: list[dict],
    output_path: str,
    model: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    timeout: float,
    samples: int,
) -> None:
    """Generate completions for all prompts concurrently."""
    # Load existing to support resume
    existing = load_existing_outputs(output_path)

    # Filter out already-generated prompts
    remaining = [p for p in prompts if p["prompt"] not in existing]

    if len(remaining) < len(prompts):
        print(
            f"Resuming: {len(prompts) - len(remaining)} already generated, {len(remaining)} remaining"
        )

    if not remaining:
        print("All prompts already generated!")
        return

    # Expand for multiple samples
    if samples > 1:
        expanded = []
        for p in remaining:
            for _ in range(samples):
                expanded.append(p.copy())
        remaining = expanded
        print(f"Generating {samples} samples per prompt ({len(remaining)} total)")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        # Create tasks
        tasks = [
            generate_completion(
                client, p, model, system, max_tokens, temperature, timeout, semaphore
            )
            for p in remaining
        ]

        # Process with progress bar
        completed = 0
        failed = 0

        # Open output file in append mode
        with open(output_path, "a") as f:
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Generating",
                unit="prompt",
            ):
                result = await coro
                if result:
                    f.write(json.dumps(result) + "\n")
                    f.flush()  # Ensure progress is saved
                    completed += 1
                else:
                    failed += 1

    print(f"\nCompleted: {completed}, Failed: {failed}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher completions for off-policy distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input prompts.txt --output train.jsonl
  %(prog)s --input prompts.jsonl --model openai/gpt-5.2 --concurrency 10
  %(prog)s --input data.jsonl --system "You are a helpful assistant." --temperature 0.9
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file with prompts (txt or jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="train.jsonl",
        help="Output file path (default: train.jsonl)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="anthropic/claude-sonnet-4",
        help="Teacher model in OpenRouter format (default: anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--system",
        "-s",
        default=None,
        help="System prompt for all completions (can be overridden per-entry in jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per completion (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Number of parallel requests (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of completions to generate per prompt (default: 1)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_env()

    # Check for API key
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY not found")
        print("\nSetup:")
        print("  1. Get an API key from https://openrouter.ai")
        print("  2. Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-your-key-here")
        sys.exit(1)

    # Load prompts
    print(f"Loading prompts from: {args.input}")
    prompts = load_prompts(args.input)
    print(f"Loaded {len(prompts)} prompts")

    if not prompts:
        print("No prompts found!")
        sys.exit(1)

    print(f"Teacher model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Run generation
    asyncio.run(
        generate_all(
            prompts=prompts,
            output_path=args.output,
            model=args.model,
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            timeout=args.timeout,
            samples=args.samples,
        )
    )


if __name__ == "__main__":
    main()
