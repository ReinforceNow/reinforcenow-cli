from rnow.core import RewardArgs, reward


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    """Check if the boxed answer matches the expected answer.

    Hints:
    - Get response: messages[-1]["content"]
    - Get expected: args.metadata["expected_answer"]
    - Handle "A or B" format in expected answers
    - Extract ALL \\boxed{} answers from response
    - Return 1.0 if any match, 0.0 otherwise
    """
    pass
