"""
FinQA Reward - Answer Correctness

Uses math_verify for robust numerical comparison (same as rl-single template).
"""

from math_verify import LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, reward


@reward
def answer_correctness(args: RewardArgs, messages: list) -> float:
    """
    Check if model's answer matches ground truth using math_verify.

    Handles:
    - Different numerical formats (0.5, 1/2, 50%)
    - LaTeX expressions
    - Symbolic equivalence
    """
    expected = args.metadata.get("answer")
    if expected is None:
        return 0.0

    gold = parse(str(expected))
    pred = parse(
        messages[-1]["content"], extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )

    if not pred:
        return 0.0

    return 1.0 if verify(gold, pred) else 0.0
