# On-Policy Distillation Template

This template demonstrates **on-policy distillation**, where a smaller "student" model learns to match a larger "teacher" model's behavior through KL divergence supervision.

## How It Works

1. **Student generates** - The student model (Qwen3-8B) generates responses to prompts
2. **Teacher grades** - The teacher model (Qwen3-32B) computes log probabilities for each token
3. **KL penalty** - Reverse KL divergence `(log p_student - log p_teacher)` becomes the advantage signal
4. **Student learns** - Tokens where student diverges from teacher get negative advantage, guiding the student to match teacher behavior

This follows the approach described in [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation).

## Why On-Policy Distillation?

Compared to **off-policy distillation** (SFT on teacher-generated data):

- **On-policy** trains on the student's own distribution, avoiding exposure bias
- Student learns from its own mistakes, not just teacher's examples
- More sample efficient for capability transfer

## Files

- `config.yml` - Training configuration with teacher model specified
- `train.jsonl` - Prompts for training (no rewards needed - teacher provides supervision)

## Configuration

```yaml
dataset_type: distill  # Enables distillation mode

teacher:
  path: Qwen/Qwen3-32B  # Larger teacher model
```

The KL penalty coefficient (1.0) and discount factor (0.0) are set automatically following best practices.

## Usage

```bash
# Initialize project
rnow init -t distill-reasoning -n "My Distilled Model"

# Run training
rnow run
```

## Tips

1. **Teacher quality matters** - Use the best teacher you can afford. Larger models generally produce better students.

2. **Prompt diversity** - Include diverse prompts that cover the capabilities you want to transfer.

3. **No rewards needed** - Unlike RL, distillation doesn't require reward functions. The teacher's behavior is the reward signal.

4. **Scaling** - Start with a smaller dataset to verify the setup works, then scale up.
