# Multi-turn RL reward functions
from rnow.core import reward

@reward(parse_reasoning=True)
async def step_accuracy(args, sample, **kwargs):
    """
    Reward for each step in a multi-turn conversation.
    """
    messages = sample.get("messages", [])
    num_turns = len([m for m in messages if m.get("role") == "assistant"])
    expected_steps = sample.get("metadata", {}).get("steps", 1)

    # Reward based on having the right number of turns
    if num_turns == expected_steps:
        return 1.0
    elif num_turns > 0:
        # Partial credit based on how close we are
        return max(0.3, 1.0 - abs(num_turns - expected_steps) * 0.2)
    else:
        return 0.0

@reward(parse_reasoning=True)
async def final_answer(args, sample, **kwargs):
    """
    Reward for the final answer quality.
    """
    messages = sample.get("messages", [])
    if not messages:
        return 0.0

    # Check if the conversation reached completion
    last_message = messages[-1]
    if last_message.get("role") == "assistant":
        content = last_message.get("content", "").lower()
        expected = sample.get("metadata", {}).get("answer", "")

        # For math problems, check if answer is in the response
        if expected and expected != "complete":
            if expected in content:
                return 1.0
            else:
                return 0.2
        # For explanatory tasks, check for completion
        elif expected == "complete":
            # Simple heuristic: longer responses are better
            if len(content) > 100:
                return 1.0
            elif len(content) > 50:
                return 0.6
            else:
                return 0.3

    return 0.0