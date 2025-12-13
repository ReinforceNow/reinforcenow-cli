import re

from rnow.core import RewardArgs, reward


@reward(precondition=True)
def format(args: RewardArgs, messages: list) -> float:
    """Check for \\boxed{} format."""
    response = messages[-1]["content"]
    return 1.0 if re.search(r"\\boxed\{", response) else 0.0


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if equation equals target and uses all numbers exactly once."""
    response = messages[-1]["content"]
    target = args.metadata["target"]
    numbers = args.metadata["numbers"]

    # Extract equation from \boxed{} (take the last one)
    matches = re.findall(r"\\boxed\{([^}]*)\}", response)
    if not matches:
        return 0.0

    equation = matches[-1].strip()

    # Check all numbers used exactly once
    used = [int(n) for n in re.findall(r"\d+", equation)]
    if sorted(used) != sorted(numbers):
        return 0.0

    try:
        result = eval(equation)
        return 1.0 if abs(result - target) < 0.0001 else 0.0
    except:
        return 0.0
