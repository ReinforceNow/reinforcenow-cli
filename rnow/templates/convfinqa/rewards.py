from rnow.core import RewardArgs, get_response, llm_judge, reward


@reward(timeout=120)
def accuracy(args: RewardArgs, messages: list) -> float:
    """Judge if model's numerical answer matches expected."""
    expected = args.metadata["answer"]
    model_answer = get_response(messages)

    prompt = (
        f"Expected: {expected}\n"
        f"Model: {model_answer}\n\n"
        "Is the model's final answer mathematically equal to expected? "
        "Ignore formatting (\\boxed, LaTeX). Equivalent forms count (1/2=0.5=50%). "
        "Answer only: Yes or No"
    )

    return llm_judge(prompt)
