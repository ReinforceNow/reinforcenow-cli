import re

from rnow.core import RewardArgs, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    response = messages[-1]["content"]
    ground_truth = args.metadata["ground_truth"]

    # Extract content from \boxed{...}
    match = re.search(r"\\boxed\{(.+?)\}", response, re.DOTALL)

    if not match:
        return 0.0  # No boxed answer = no reward

    answer = match.group(1).strip()
    return 1.0 if ground_truth in answer else 0.0
