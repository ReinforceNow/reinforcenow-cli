from rnow.core import RewardArgs, get_response, reward


@reward
def answer_correctness(args: RewardArgs, messages: list) -> float:
    answer = str(args.metadata["answer"])
    return 1.0 if answer in get_response(messages) else 0.0
