"""
Test runner for generation and reward functions using OpenAI
Matches slime's interface exactly so users can test their functions locally
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm import tqdm
import importlib.util
import sys
from argparse import Namespace

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class Sample:
    """Simplified Sample class matching slime's interface"""

    class Status:
        PENDING = "PENDING"
        COMPLETED = "COMPLETED"
        ABORTED = "ABORTED"
        TRUNCATED = "TRUNCATED"

    def __init__(self, prompt: str, label: Optional[str] = None, metadata: Optional[Dict] = None):
        self.prompt = prompt
        self.response = ""
        self.label = label
        self.metadata = metadata or {}
        self.reward = None
        self.status = Sample.Status.PENDING
        self.tokens = []
        self.response_length = 0
        self.rollout_log_probs = None
        self.prompt_tokens = []
        self.loss_mask = None
        self.weight_versions = []


async def generate_with_openai(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any]
) -> Sample:
    """
    Generate response using OpenAI API
    Signature matches slime's generate(args, sample, sampling_params)
    """
    try:
        client = args._openai_client

        response = await client.chat.completions.create(
            model=sampling_params.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": sample.prompt}],
            temperature=sampling_params.get("temperature", 0.8),
            max_tokens=sampling_params.get("max_new_tokens", 1024)
        )

        sample.response = response.choices[0].message.content or ""
        sample.status = Sample.Status.COMPLETED
        sample.response_length = response.usage.completion_tokens

    except Exception as e:
        sample.status = Sample.Status.ABORTED
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
    """
    Dynamically load custom generation function if it exists
    Looks for 'generate' function with signature: generate(args, sample, sampling_params)
    """
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

    # Get generate function (matching slime's signature)
    if hasattr(module, "generate"):
        return module.generate

    return None


async def test_sample(
    args: Namespace,
    sample: Sample,
    reward_fn: Any,
    sampling_params: dict[str, Any],
    custom_generate_fn: Optional[Any] = None
) -> Sample:
    """
    Test a single sample with generation and reward
    Matches slime's workflow exactly
    """

    # Generate response (matching slime's pattern)
    if custom_generate_fn:
        # Use custom generator with slime's exact signature
        sample = await custom_generate_fn(args, sample, sampling_params)
    else:
        # Use default OpenAI generation
        sample = await generate_with_openai(args, sample, sampling_params)

    # Compute reward (only if generation completed)
    if sample.status == Sample.Status.COMPLETED:
        try:
            # Check if reward function is async
            import inspect
            if inspect.iscoroutinefunction(reward_fn):
                # Async reward: await reward(args, sample)
                sample.reward = await reward_fn(args, sample)
            else:
                # Sync reward: reward(sample)
                sample.reward = reward_fn(sample)
        except Exception as e:
            sample.reward = 0.0
            sample.status = Sample.Status.ABORTED
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
    """Run test on dataset samples matching slime's interface"""

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

    # Initialize OpenAI client and create mock args object
    client = AsyncOpenAI(api_key=api_key)

    # Create args namespace matching slime's structure
    args = Namespace()
    args._openai_client = client  # Store client for generate function

    # Create sampling_params matching slime's structure
    sampling_params = {
        "model": model,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "top_p": 1.0,
        "top_k": -1,
    }

    # Run tests
    print(f"Running tests with model: {model}")
    results = []

    pbar = tqdm(total=len(samples), desc="Testing samples")
    for sample in samples:
        result = await test_sample(
            args,
            sample,
            reward_fn,
            sampling_params,
            custom_generate_fn
        )
        results.append(result)
        pbar.update(1)
    pbar.close()

    # Compute statistics
    completed = [s for s in results if s.status == Sample.Status.COMPLETED]
    rewards = [s.reward for s in completed if s.reward is not None]

    stats = {
        "total_samples": len(results),
        "completed": len(completed),
        "aborted": len([s for s in results if s.status == Sample.Status.ABORTED]),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "avg_response_length": sum(s.response_length for s in completed) / len(completed) if completed else 0,
    }

    return {
        "stats": stats,
        "results": results
    }
