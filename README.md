<div align="center">
  <img
    alt="ReinforceNow CLI"
    src="./assets/header.png"
    width="100%"
  >
  <br><br>

[![PyPI version](https://img.shields.io/pypi/v/rnow?color=blue)](https://pypi.org/project/rnow/)
[![Docs](https://img.shields.io/badge/docs-reinforcenow.ai-blue)](https://reinforcenow.ai/docs)
[![Follow on X](https://img.shields.io/badge/Follow_on_X-@reinforcenow-black?labelColor=white)](https://x.com/reinforcenowai)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

</div>

# Documentation

See the [documentation](https://www.reinforcenow.ai/docs/getting-started/quickstart) for a technical overview of the platform and [train your first agent](https://www.reinforcenow.ai/docs/getting-started/first-agent)

# Quick Start

### 1. Install

```bash
pip install rnow
```

### 2. Authenticate

```bash
rnow login
```

### 3. Create & Run Your First Project

```bash
rnow init --template rl-single
rnow run
```

That's it! Your training run will start on ReinforceNow's infrastructure. Monitor progress in the [dashboard](https://reinforcenow.ai/home).

![ReinforceNow Graph](./assets/reinforcenow-graph.png)

# Core Concepts

Go from raw data to a reliable AI agent in production. ReinforceNow gives you the flexibility to define:

### 1. Reward Functions

Define how your model should be evaluated using the `@reward` decorator:

```python
from rnow.core import reward, RewardArgs

@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the model's answer matches ground truth."""
    response = messages[-1]["content"]
    expected = args.metadata["answer"]
    return 1.0 if expected in response else 0.0
```

→ [Write your first reward function](https://www.reinforcenow.ai/docs/getting-started/first-reward)

### 2. Tools (for Agents)

Give your model the ability to call functions during training:

```python
from rnow.core import tool

@tool
def search(query: str, max_results: int = 5) -> dict:
    """Search the web for information."""
    # Your implementation here
    return {"results": [...]}
```

→ [Train an agent with custom tools](https://www.reinforcenow.ai/docs/getting-started/first-agent)

### 3. Training Data

Create a `train.jsonl` file with your prompts and reward assignments:

```json
{"messages": [{"role": "user", "content": "Balance the equation: Fe + O2 → Fe2O3"}], "rewards": ["accuracy"], "metadata": {"answer": "4Fe + 3O2 → 2Fe2O3"}}
{"messages": [{"role": "user", "content": "Balance the equation: H2 + O2 → H2O"}], "rewards": ["accuracy"], "metadata": {"answer": "2H2 + O2 → 2H2O"}}
{"messages": [{"role": "user", "content": "Balance the equation: N2 + H2 → NH3"}], "rewards": ["accuracy"], "metadata": {"answer": "N2 + 3H2 → 2NH3"}}
```

→ [Learn about training data format](https://www.reinforcenow.ai/docs/cli-reference/train-data)

# Contributing

We welcome contributions! ❤️ Please open an issue to discuss your ideas before submitting a PR.

<br>
<div align="center">
  <img
    alt="ReinforceNow"
    src="./assets/footer.png"
    width="100%"
  >
</div>
