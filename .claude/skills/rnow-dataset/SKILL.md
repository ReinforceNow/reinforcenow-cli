---
name: rnow-dataset
description: Convert HuggingFace datasets to ReinforceNow format. Use when creating train.jsonl from HuggingFace datasets, formatting data for SFT or RL training, or writing reward functions for math datasets. Triggers on "HuggingFace", "dataset", "train.jsonl", "convert dataset", "math reward", "latex".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Converting HuggingFace Datasets to ReinforceNow Format

ReinforceNow uses `train.jsonl` format for training data. This guide shows how to convert HuggingFace datasets.

## SFT (Supervised Fine-Tuning) Format

For SFT, each line needs `messages` - no rewards required.

### Basic SFT Structure

```json
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
{"messages": [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
```

### Converting HuggingFace Dataset to SFT

```python
from datasets import load_dataset
import json

# Load dataset from HuggingFace
dataset = load_dataset("your-dataset-name", split="train")

# Convert to train.jsonl
with open("train.jsonl", "w") as f:
    for row in dataset:
        # Adapt these field names to match your dataset
        entry = {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }
        f.write(json.dumps(entry) + "\n")
```

### Example: Converting Alpaca-style Dataset

```python
from datasets import load_dataset
import json

dataset = load_dataset("tatsu-lab/alpaca", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        messages = []

        # Add instruction as user message
        if row.get("input"):
            user_content = f"{row['instruction']}\n\nInput: {row['input']}"
        else:
            user_content = row["instruction"]

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": row["output"]})

        f.write(json.dumps({"messages": messages}) + "\n")
```

### Update config.yml for SFT

```yaml
dataset_type: sft
data:
  train_file: train.jsonl
  batch_size: 8
```

---

## RL (Reinforcement Learning) Format

For RL, each line needs `messages`, `rewards`, and optionally `metadata`.

### Basic RL Structure

```json
{"messages": [{"role": "user", "content": "Question"}], "rewards": ["accuracy"], "metadata": {"answer": "expected"}}
```

**Important**: For RL, messages should only contain the prompt (user message). The model generates the assistant response during training.

### Converting HuggingFace Dataset to RL

```python
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("your-math-dataset", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        entry = {
            "messages": [
                {"role": "user", "content": row["question"]}
            ],
            "rewards": ["accuracy"],  # References function in rewards.py
            "metadata": {
                "expected_answer": row["answer"]  # Ground truth for reward
            }
        }
        f.write(json.dumps(entry) + "\n")
```

### Example: Converting GSM8K Math Dataset

```python
from datasets import load_dataset
import json
import re

dataset = load_dataset("gsm8k", "main", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        # Extract final answer from GSM8K format (#### followed by number)
        answer_match = re.search(r"####\s*(.+)$", row["answer"])
        final_answer = answer_match.group(1).strip() if answer_match else row["answer"]

        entry = {
            "messages": [
                {"role": "user", "content": row["question"]}
            ],
            "rewards": ["accuracy"],
            "metadata": {
                "expected_answer": final_answer
            }
        }
        f.write(json.dumps(entry) + "\n")
```

### Example: Converting MATH Dataset

```python
from datasets import load_dataset
import json

dataset = load_dataset("hendrycks/competition_math", split="train")

with open("train.jsonl", "w") as f:
    for row in dataset:
        entry = {
            "messages": [
                {"role": "user", "content": row["problem"]}
            ],
            "rewards": ["accuracy"],
            "metadata": {
                "expected_answer": row["solution"]  # Contains LaTeX
            }
        }
        f.write(json.dumps(entry) + "\n")
```

---

## Math Reward Functions

### Option 1: Using math-verify (for LaTeX/numerical answers)

Best for datasets with LaTeX expressions like `\boxed{42}` or `\frac{1}{2}`.

**requirements.txt:**
```
math-verify==0.5.0
```

**rewards.py:**
```python
from math_verify import LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, get_response, reward


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Verify mathematical equivalence using math-verify."""
    gold = parse(args.metadata["expected_answer"])
    pred = parse(
        get_response(messages),
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )
    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
```

### Option 2: Using llm_judge (for complex/text answers)

Best for answers that need semantic understanding or aren't purely numerical.

**IMPORTANT: Requires OPENAI_API_KEY**

Create a `.env` file in your project directory:
```
OPENAI_API_KEY=sk-your-api-key-here
```

Your secrets are encrypted and securely stored in the ReinforceNow platform.

**rewards.py:**
```python
from rnow.core import RewardArgs, get_response, llm_judge, reward


@reward(timeout=120)
def accuracy(args: RewardArgs, messages: list) -> float:
    """Judge if model's answer matches expected using LLM."""
    expected = args.metadata["expected_answer"]
    model_answer = get_response(messages)

    prompt = (
        f"Expected: {expected}\n"
        f"Model: {model_answer}\n\n"
        "Is the model's final answer mathematically equal to expected? "
        "Ignore formatting (\\boxed, LaTeX). Equivalent forms count (1/2=0.5=50%). "
        "Answer only: Yes or No"
    )

    return llm_judge(prompt)
```

### Option 3: Combining Both (Recommended for Math)

Use math-verify first, fall back to llm_judge for edge cases.

**requirements.txt:**
```
math-verify==0.5.0
```

**rewards.py:**
```python
from math_verify import LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, get_response, llm_judge, reward


@reward(timeout=120)
def accuracy(args: RewardArgs, messages: list) -> float:
    """Verify math with math-verify, fallback to LLM judge."""
    expected = args.metadata["expected_answer"]
    response = get_response(messages)

    # Try math-verify first (faster, more reliable for pure math)
    gold = parse(expected)
    pred = parse(
        response,
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )

    if gold and pred:
        return 1.0 if verify(gold, pred) else 0.0

    # Fallback to LLM judge for complex cases
    prompt = (
        f"Expected: {expected}\n"
        f"Model: {response}\n\n"
        "Is the model's final answer mathematically equal to expected? "
        "Ignore formatting. Answer only: Yes or No"
    )
    return llm_judge(prompt)
```

---

## Common Dataset Patterns

### Q&A Dataset

```python
# Input format: {"question": "...", "answer": "..."}
entry = {
    "messages": [{"role": "user", "content": row["question"]}],
    "rewards": ["accuracy"],
    "metadata": {"expected_answer": row["answer"]}
}
```

### Instruction-Following Dataset

```python
# Input format: {"instruction": "...", "input": "...", "output": "..."}
content = row["instruction"]
if row.get("input"):
    content += f"\n\nInput: {row['input']}"

entry = {
    "messages": [{"role": "user", "content": content}],
    "rewards": ["accuracy"],
    "metadata": {"expected_answer": row["output"]}
}
```

### Multi-turn Conversation (SFT only)

```python
# Input format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
messages = []
for turn in row["conversations"]:
    role = "user" if turn["from"] == "human" else "assistant"
    messages.append({"role": role, "content": turn["value"]})

entry = {"messages": messages}
```

---

## Testing Your Dataset

After creating train.jsonl:

```bash
# Validate the format
rnow run --dry-run

# Test with a few samples
rnow test -n 3 --verbose
```

---

## Secrets and Environment Variables

For `llm_judge` or any reward function needing API keys:

1. Create `.env` in your project:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

2. Access in rewards.py via `args.secrets`:
   ```python
   api_key = args.secrets["OPENAI_API_KEY"]
   ```

3. Secrets are encrypted and stored securely on the ReinforceNow platform. They are never logged or exposed.
