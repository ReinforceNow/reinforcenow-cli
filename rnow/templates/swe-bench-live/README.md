# SWE-bench Live Template

Train agents to fix real GitHub issues using the [SWE-bench-Live](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live) benchmark.

## Quick Start

```bash
# Initialize the template
rnow init -t swe-bench-live

# Run training (includes 10-sample dataset)
rnow run
```

## Generate More Data

The template includes 10 sample instances. To train on more data:

```bash
# Install dependencies
uv venv .venv --python 3.11 --seed
uv pip install --python .venv/bin/python datasets

# Generate lite split (300 instances)
.venv/bin/python convert_dataset.py --split lite --output train.jsonl

# Or full dataset (1888 instances)
.venv/bin/python convert_dataset.py --split full --output train.jsonl

# Or a custom subset
.venv/bin/python convert_dataset.py --split lite --limit 50 --output train.jsonl

# Clean up
rm -rf .venv
```

## How It Works

Each training instance:
1. Spins up a **per-instance Docker container** with the exact repository state at issue time
2. Agent uses tools (`bash`, `read_file`, `edit_file`, etc.) to explore and fix the issue
3. Reward runs the test suite and checks if FAIL_TO_PASS tests now pass

## Files

- `config.yml` - Training configuration (multi-turn RL with 30 max turns)
- `env.py` - Tools with `sandbox=True` for execution in the Docker container
- `rewards.py` - Test-based reward checking FAIL_TO_PASS/PASS_TO_PASS transitions
- `convert_dataset.py` - Script to convert HuggingFace dataset to train.jsonl
- `train.jsonl` - Sample instances (use convert_dataset.py for full dataset)

## Dataset Format

Each entry in train.jsonl includes:

```json
{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Issue: ..."}],
  "rewards": ["has_changes", "resolved"],
  "tools": ["bash", "read_file", "write_file", "edit_file", "search_files", "list_files"],
  "docker": "starryzhang/sweb.eval.x86_64.repo_1776_name-123",
  "metadata": {
    "instance_id": "repo__name-123",
    "test_cmds": ["pytest -rA tests/"],
    "fail_to_pass": ["tests/test_foo.py::test_bar"],
    "pass_to_pass": ["tests/test_foo.py::test_existing"]
  }
}
```

## Docker Images

SWE-bench-Live provides pre-built Docker images on DockerHub. The naming convention:
- Instance ID: `repo__name-123`
- Docker image: `starryzhang/sweb.eval.x86_64.repo_1776_name-123`

The repository is mounted at `/testbed` inside the container.

## Rewards

- **has_changes** (precondition): Agent must attempt to modify files
- **resolved**: Runs tests and checks:
  - `1.0` if all FAIL_TO_PASS tests pass AND no regressions
  - `0.5` if FAIL_TO_PASS tests pass but introduced regressions
  - `0.0` if FAIL_TO_PASS tests still fail

## Customization

### Using a subset of instances
```bash
python convert_dataset.py --split lite --limit 50 --output train.jsonl
```

### Using the full dataset
```bash
python convert_dataset.py --split test --output train_full.jsonl
```

### Adjusting training parameters
Edit `config.yml`:
- `rollout.max_turns`: Number of tool calls per episode (default: 30)
- `data.batch_size`: Instances per batch (default: 4)
- `model.path`: Base model to finetune
