from rnow.core import RewardArgs, get_response, reward


@reward
def answer_correctness(args: RewardArgs, messages: list) -> float:
    val = args.metadata["answer"]
    answer = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
    return 1.0 if answer in get_response(messages) else 0.0
