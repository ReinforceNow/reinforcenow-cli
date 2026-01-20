from rnow.core import RewardArgs, get_response, llm_judge, reward


@reward(timeout=120)
async def accuracy(args: RewardArgs, messages: list) -> float:
    """Judge if model's numerical answer matches expected."""
    expected = args.metadata["answer"]
    model_answer = get_response(messages)

    prompt = (
        f"Expected: {expected}\n"
        f"Model: {model_answer}\n\n"
        "Match? (15.4%=15.4, -13.3% → 13.3 drop; no approximations)\n"
        "Answer 1 or 0."
    )

    return llm_judge(prompt)
