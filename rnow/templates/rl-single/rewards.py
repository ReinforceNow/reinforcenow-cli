from rnow.core import reward, RewardArgs


@reward(parse_reasoning=True)
async def accuracy(args: RewardArgs, messages: list) -> float:

    response = messages[-1]["content"]
    
    ground_truth = args.metadata["ground_truth"]

    if ground_truth in response:
        return 1.0
    else:
        return 0.0
