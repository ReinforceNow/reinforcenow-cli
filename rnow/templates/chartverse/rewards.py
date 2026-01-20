from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, get_response, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Verify numerical answer using math_verify.

    NOTE: Must be async - math-verify/antlr4 is NOT thread-safe.
    """
    gold = parse(args.metadata["answer"])
    response = get_response(messages)

    # Try LaTeX extraction first (for \boxed{} answers)
    pred = parse(response, extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])

    # Fallback to plain expression extraction
    if not pred:
        pred = parse(response, extraction_config=[ExprExtractionConfig()])

    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
