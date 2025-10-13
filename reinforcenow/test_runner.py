"""
Test runner for generation and reward functions using OpenAI
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm import tqdm
import importlib.util
import sys

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class Sample:
    """Simplified Sample class matching slime's interface"""
    def __init__(self, prompt: str, label: Optional[str] = None, metadata: Optional[Dict] = None):
        self.prompt = prompt
        self.response = ""
        self.label = label
        self.metadata = metadata or {}
        self.reward = None
        self.status = "PENDING"
        self.tokens = []
        self.response_length = 0


async def generate_with_openai(
    client: AsyncOpenAI,
    sample: Sample,
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_tokens: int = 1024
) -> Sample:
    """Generate response using OpenAI API"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": sample.prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        sample.response = response.choices[0].message.content or ""
        sample.status = "COMPLETED"
        sample.response_length = response.usage.completion_tokens

    except Exception as e:
        sample.status = "ABORTED"
        sample.response = f"Error: {e}"

    return sample


def load_reward_function(project_dir: Path):
    """Dynamically load reward function from project directory"""
    reward_file = project_dir / "reward_function.py"

    if not reward_file.exists():
        raise FileNotFoundError(f"reward_function.py not found in {project_dir}")

    # Load module
    spec = importlib.util.spec_from_file_location("reward_function", reward_file)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load reward_function.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["reward_function"] = module
    spec.loader.exec_module(module)

    # Get reward function
    if not hasattr(module, "reward"):
        raise AttributeError("reward_function.py must define a 'reward' function")

    return module.reward


def load_generation_function(project_dir: Path):
    """Dynamically load custom generation function if it exists"""
    gen_file = project_dir / "generation_function.py"

    if not gen_file.exists():
        return None

    # Load module
    spec = importlib.util.spec_from_file_location("generation_function", gen_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["generation_function"] = module
    spec.loader.exec_module(module)

    # Get generate function (decorator style)
    if hasattr(module, "generate"):
        return module.generate

    return None


async def test_sample(
    client: AsyncOpenAI,
    sample: Sample,
    reward_fn: Any,
    custom_generate_fn: Optional[Any] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_tokens: int = 1024
) -> Sample:
    """Test a single sample with generation and reward"""

    # Generate response
    if custom_generate_fn:
        # Use custom generator (decorator style)
        sample = await custom_generate_fn(
            lambda s: generate_with_openai(client, s, model, temperature, max_tokens),
            sample
        )
    else:
        # Use default OpenAI generation
        sample = await generate_with_openai(client, sample, model, temperature, max_tokens)

    # Compute reward
    if sample.status == "COMPLETED":
        try:
            sample.reward = reward_fn(sample)
        except Exception as e:
            sample.reward = 0.0
            sample.status = "ABORTED"
            print(f"Error computing reward: {e}")

    return sample


async def run_test(
    dataset_file: Path,
    project_dir: Path,
    api_key: str,
    n_samples: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """Run test on dataset samples"""

    if AsyncOpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")

    # Load functions
    reward_fn = load_reward_function(project_dir)
    custom_generate_fn = load_generation_function(project_dir)

    if custom_generate_fn:
        print(f"✓ Using custom generation function")
    else:
        print(f"✓ Using default OpenAI generation")

    # Load dataset
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    samples = []
    with open(dataset_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data = json.loads(line)
            sample = Sample(
                prompt=data.get("prompt", ""),
                label=data.get("label"),
                metadata=data.get("metadata", {})
            )
            samples.append(sample)

    if not samples:
        raise ValueError("No samples loaded from dataset")

    print(f"✓ Loaded {len(samples)} samples from dataset\n")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Run tests
    print(f"Running tests with model: {model}")
    results = []

    pbar = tqdm(total=len(samples), desc="Testing samples")
    for sample in samples:
        result = await test_sample(
            client,
            sample,
            reward_fn,
            custom_generate_fn,
            model,
            temperature,
            max_tokens
        )
        results.append(result)
        pbar.update(1)
    pbar.close()

    # Compute statistics
    completed = [s for s in results if s.status == "COMPLETED"]
    rewards = [s.reward for s in completed if s.reward is not None]

    stats = {
        "total_samples": len(results),
        "completed": len(completed),
        "aborted": len([s for s in results if s.status == "ABORTED"]),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "avg_response_length": sum(s.response_length for s in completed) / len(completed) if completed else 0,
    }

    return {
        "stats": stats,
        "results": results
    }
