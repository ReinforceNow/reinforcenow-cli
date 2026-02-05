from math_verify import LatexExtractionConfig, parse, verify

from rnow.core import RewardArgs, get_response, reward


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Verify mathematical equivalence using math-verify.

    NOTE: Must be async to run on main event loop. math-verify/antlr4 is NOT
    thread-safe - running in asyncio.to_thread causes sympy parsing to fail.
    """
    gold = parse(args.metadata["expected_answer"])
    pred = parse(
        get_response(messages), extraction_config=[LatexExtractionConfig(boxed_match_priority=0)]
    )
    if not pred:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
