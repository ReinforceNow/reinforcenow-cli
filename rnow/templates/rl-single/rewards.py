import re

from math_verify import parse, verify

from rnow.core import RewardArgs, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the boxed answer matches the expected answer using math-verify."""
    response = messages[-1]["content"]
    expected = args.metadata["expected_answer"]

    # Extract last boxed answer
    matches = re.findall(r"\\+boxed\{([^}]*)\}", response)
    if not matches:
        return 0.0

    answer = matches[-1]

    # Wrap in \boxed{} to give math_verify proper LaTeX context
    # (required for tuples, complex expressions, etc.)
    gold = parse(rf"\boxed{{{expected}}}", parsing_timeout=None)
    pred = parse(rf"\boxed{{{answer}}}", parsing_timeout=None)
    return 1.0 if verify(gold, pred, timeout_seconds=None) else 0.0
