import re

from rnow.core import RewardArgs, reward


@reward(precondition=True)
async def format(args: RewardArgs, messages: list) -> float:
    """Check for <think>...</think><answer>...</answer> format."""
    response = messages[-1]["content"]
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.search(pattern, response, re.DOTALL) else 0.0


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if equation equals target."""
    response = messages[-1]["content"]
    target = args.metadata["target"]

    # Extract equation from <answer> tags
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if not match:
        return 0.0

    equation = match.group(1).strip()

    try:
        result = eval(equation)
        return 1.0 if abs(result - target) < 0.0001 else 0.0
    except:
        return 0.0
