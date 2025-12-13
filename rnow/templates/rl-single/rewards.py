from math_verify import LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, reward


@reward
def accuracy(args: RewardArgs, messages: list) -> float:
    gold = parse(args.metadata["expected_answer"])
    pred = parse(
        messages[-1]["content"], extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )
    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
