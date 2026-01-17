from rnow.core import RewardArgs, get_response, reward


@reward
def answer_correctness(args: RewardArgs, messages: list) -> float:
    """Check if the expected answer appears in the model's response."""
    answer = str(args.metadata["answer"])
    return 1.0 if answer in get_response(messages) else 0.0
