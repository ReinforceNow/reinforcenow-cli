from math_verify import LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, get_response, reward


@reward
def answer_correctness(args: RewardArgs, messages: list) -> float:
    gold = parse(args.metadata["answer"])
    pred = parse(
        get_response(messages), extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )
    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
