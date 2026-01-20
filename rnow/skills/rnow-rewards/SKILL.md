---
name: rnow-rewards
description: Write reward functions for ReinforceNow RL training. Use when creating @reward decorated functions, writing rewards.py, using precondition rewards, sandbox rewards, or llm_judge. Triggers on "reward function", "@reward", "RewardArgs", "precondition", "llm_judge".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Writing Reward Functions for ReinforceNow

Reward functions compute the training signal for reinforcement learning. They evaluate model responses and return a score between 0.0 and 1.0.

## Basic Structure

Every reward function must:
1. Be decorated with `@reward`
2. Accept `(args: RewardArgs, messages: list)` as parameters
3. Return a `float` between 0.0 and 1.0

```python
from rnow.core import reward, RewardArgs

@reward
async def my_reward(args: RewardArgs, messages: list) -> float:
    """Evaluate the model's response."""
    response = messages[-1]["content"]
    # Your evaluation logic here
    return 1.0 if condition else 0.0
```

## RewardArgs Object

`args` provides access to data from train.jsonl:

| Field | Description | Example |
|-------|-------------|---------|
| `args.metadata` | Dict from `metadata` field | `args.metadata["answer"]` |
| `args.variables` | Dict from `variables` field | `args.variables["topic"]` |
| `args.secrets` | User secrets from .env | `args.secrets["OPENAI_API_KEY"]` |

## Messages Format

`messages` is a list of conversation turns:

```python
[
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4"}
]
```

Get the last assistant response:
```python
response = messages[-1]["content"]
```

## Reward Patterns

### 1. Exact Match

```python
@reward
async def exact_match(args: RewardArgs, messages: list) -> float:
    """Check if response exactly matches expected answer."""
    response = messages[-1]["content"].strip().lower()
    expected = args.metadata["answer"].strip().lower()
    return 1.0 if response == expected else 0.0
```

### 2. Contains Answer

```python
@reward
async def contains_answer(args: RewardArgs, messages: list) -> float:
    """Check if response contains the expected answer."""
    response = messages[-1]["content"]
    expected = args.metadata["answer"]
    return 1.0 if expected in response else 0.0
```

### 3. Numerical Comparison (with tolerance)

```python
import re

@reward
async def numerical_accuracy(args: RewardArgs, messages: list) -> float:
    """Check if extracted number is within 1% of expected."""
    response = messages[-1]["content"]
    expected = float(args.metadata["answer"])

    # Extract numbers from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if not numbers:
        return 0.0

    predicted = float(numbers[-1])  # Take last number
    tolerance = abs(expected) * 0.01  # 1% tolerance
    return 1.0 if abs(predicted - expected) <= tolerance else 0.0
```

### 4. Math Verification (LaTeX)

**IMPORTANT: Must use `async def`** - math-verify/antlr4 is NOT thread-safe. Sync functions run in a thread pool which causes sympy parsing to silently fail.

```python
from math_verify import LatexExtractionConfig, parse, verify

@reward
async def math_accuracy(args: RewardArgs, messages: list) -> float:
    """Verify mathematical equivalence using math-verify.

    NOTE: Must be async - math-verify is NOT thread-safe.
    """
    gold = parse(args.metadata["answer"])
    pred = parse(
        messages[-1]["content"],
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )
    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
```

Add to requirements.txt:
```
math-verify==0.5.0
```

### 5. JSON Structure Validation

```python
import json

@reward
async def valid_json(args: RewardArgs, messages: list) -> float:
    """Check if response is valid JSON with required fields."""
    response = messages[-1]["content"]
    required_fields = args.metadata.get("required_fields", [])

    try:
        data = json.loads(response)
        for field in required_fields:
            if field not in data:
                return 0.0
        return 1.0
    except json.JSONDecodeError:
        return 0.0
```

### 6. Length-Based Reward

```python
@reward
async def appropriate_length(args: RewardArgs, messages: list) -> float:
    """Reward responses within target length range."""
    response = messages[-1]["content"]
    min_len = args.metadata.get("min_length", 50)
    max_len = args.metadata.get("max_length", 500)

    length = len(response)
    if length < min_len or length > max_len:
        return 0.0
    return 1.0
```

### 7. Regex Pattern Match

```python
import re

@reward
async def pattern_match(args: RewardArgs, messages: list) -> float:
    """Check if response matches required pattern."""
    response = messages[-1]["content"]
    pattern = args.metadata["pattern"]
    return 1.0 if re.search(pattern, response) else 0.0
```

## Precondition Rewards

Preconditions act as gates. If ANY precondition returns 0, the total reward is 0.

```python
@reward(precondition=True)
async def has_answer_tag(args: RewardArgs, messages: list) -> float:
    """GATE: Response must contain Answer: tag."""
    response = messages[-1]["content"]
    return 1.0 if "Answer:" in response else 0.0

@reward(precondition=True)
async def no_refusal(args: RewardArgs, messages: list) -> float:
    """GATE: Response must not be a refusal."""
    response = messages[-1]["content"].lower()
    refusals = ["i cannot", "i can't", "i'm unable", "i am unable"]
    return 0.0 if any(r in response for r in refusals) else 1.0

@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Main accuracy reward (only applied if preconditions pass)."""
    # This only runs if has_answer_tag AND no_refusal both return 1.0
    response = messages[-1]["content"]
    expected = args.metadata["answer"]
    return 1.0 if expected in response else 0.0
```

## Sandbox Rewards

Use `sandbox=True` when rewards need to:
- Execute code
- Check files created by tools
- Access isolated environment state

**IMPORTANT**: Entries using sandbox rewards MUST have `docker` field in train.jsonl.

```python
@reward(sandbox=True, timeout=120)
async def code_runs(args: RewardArgs, messages: list) -> float:
    """Check if the generated code executes without errors."""
    import subprocess
    result = subprocess.run(
        ["python", "solution.py"],
        capture_output=True,
        timeout=60
    )
    return 1.0 if result.returncode == 0 else 0.0

@reward(sandbox=True)
async def file_created(args: RewardArgs, messages: list) -> float:
    """Check if expected file was created by tools."""
    import os
    expected_file = args.metadata["expected_file"]
    return 1.0 if os.path.exists(expected_file) else 0.0

@reward(sandbox=True)
async def test_passes(args: RewardArgs, messages: list) -> float:
    """Run pytest and check if tests pass."""
    import subprocess
    result = subprocess.run(
        ["pytest", "-q", "test_solution.py"],
        capture_output=True
    )
    return 1.0 if result.returncode == 0 else 0.0
```

train.jsonl entry:
```json
{"messages": [...], "rewards": ["code_runs"], "docker": "python:3.11-slim"}
```

## LLM Judge

Use another LLM to evaluate responses.

**IMPORTANT: Requires OPENAI_API_KEY**

Create a `.env` file in your project directory:
```
OPENAI_API_KEY=sk-your-api-key-here
```

Your secrets are encrypted and securely stored in the ReinforceNow platform. They are never logged or exposed.

```python
from rnow.core import llm_judge

@reward
async def quality_score(args: RewardArgs, messages: list) -> float:
    """Use GPT to evaluate response quality."""
    response = messages[-1]["content"]
    question = args.metadata["question"]

    prompt = f"""Rate this response on a scale of 0-1.

Question: {question}
Response: {response}

Return only a number between 0 and 1."""

    return llm_judge(prompt, secrets=args.secrets)
```

### LLM Judge with Custom Schema

```python
@reward
async def detailed_evaluation(args: RewardArgs, messages: list) -> float:
    """Detailed evaluation with custom schema."""
    response = messages[-1]["content"]

    custom_schema = {
        "type": "object",
        "properties": {
            "accuracy": {"type": "integer", "minimum": 0, "maximum": 10},
            "clarity": {"type": "integer", "minimum": 0, "maximum": 10},
            "completeness": {"type": "integer", "minimum": 0, "maximum": 10}
        },
        "required": ["accuracy", "clarity", "completeness"]
    }

    prompt = f"""Evaluate this response:
{response}

Rate accuracy, clarity, and completeness from 0-10."""

    # Returns average of scores normalized to 0-1
    result = llm_judge(
        prompt,
        secrets=args.secrets,
        schema=custom_schema,
        model="gpt-5.2-nano"
    )
    return result / 10.0  # Normalize to 0-1
```

### LLM Judge Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | Evaluation prompt |
| `secrets` | None | Dict with API keys |
| `model` | "gpt-5.2-nano" | Model to use |
| `schema` | binary 0/1 | Custom JSON schema |
| `score_key` | "score" | Field to extract |
| `temperature` | 0.0 | Sampling temperature |
| `max_tokens` | 256 | Max response tokens |
| `timeout` | 30 | Request timeout |

## Combining Multiple Rewards

In train.jsonl, list all rewards to apply:

```json
{
  "messages": [{"role": "user", "content": "Solve: 2+2"}],
  "rewards": ["has_format", "accuracy", "clarity"],
  "metadata": {"answer": "4"}
}
```

The total reward is calculated based on preconditions:
- If any `precondition=True` reward returns 0 → total = 0
- Otherwise → weighted average of all rewards

## Async vs Sync

Both work, but some libraries require async:

```python
# Async (recommended - runs on main event loop)
@reward
async def my_async_reward(args: RewardArgs, messages: list) -> float:
    result = await some_async_operation()
    return result

# Sync (runs in thread pool - simpler for pure computation)
@reward
def my_sync_reward(args: RewardArgs, messages: list) -> float:
    return 1.0 if condition else 0.0
```

**IMPORTANT**: Some libraries are NOT thread-safe and MUST use `async def`:
- `math-verify` / `latex2sympy2` / `antlr4` - sympy parsing fails silently in threads

## Common Mistakes

### Wrong: Return value outside 0-1
```python
@reward
async def bad(args: RewardArgs, messages: list) -> float:
    return 10  # ERROR: Must be 0.0-1.0
```

### Wrong: Missing type hints
```python
@reward
async def bad(args, messages):  # ERROR: Missing types
    return 1.0
```

### Wrong: Using sandbox=True without docker field
```python
@reward(sandbox=True)
async def check_file(args: RewardArgs, messages: list) -> float:
    # ERROR if train.jsonl entry lacks "docker" field
    return 1.0 if os.path.exists("output.txt") else 0.0
```

### Right: Clamp values to valid range
```python
@reward
async def safe_score(args: RewardArgs, messages: list) -> float:
    score = calculate_score()  # Might return any float
    return max(0.0, min(1.0, score))  # Clamp to 0-1
```

## Testing Rewards Locally

```bash
rnow test -n 3 --verbose
```

This runs rollouts and shows reward breakdowns for debugging.
