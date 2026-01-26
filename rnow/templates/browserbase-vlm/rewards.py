"""
Rewards for VLM Browser Agent

Rewards:
1. used_browser (precondition): Must use at least one browser tool
2. accuracy (LLM judge): Evaluate if the final answer matches expected
"""

import os
import re
from rnow.core import reward, RewardArgs, llm_judge


@reward(precondition=True)
async def used_browser(args: RewardArgs, messages: list) -> float:
    """
    Gate reward: Agent must SUCCESSFULLY use browser tools to get any reward.

    Returns 1.0 if any browser tool was used and succeeded, 0.0 otherwise.
    This ensures the agent actually browses instead of answering from memory.
    """
    browser_tools = {"click", "type_text", "press_key", "scroll", "wait", "navigate", "screenshot"}

    # Track which tool calls succeeded by checking tool responses
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                if tool_name in browser_tools:
                    # Check if the next message is a successful tool response
                    if i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        if next_msg.get("role") == "tool":
                            content = next_msg.get("content", "")
                            # Tool failed if response contains error
                            if "<tool_error>" not in content:
                                return 1.0

    return 0.0


@reward
async def accuracy(args: RewardArgs, messages: list) -> float:
    """
    Evaluate if the model's answer matches the expected answer.

    Uses an LLM judge for semantic evaluation - handles paraphrasing,
    different formats, and equivalent expressions.
    """
    expected = args.metadata.get("expected_answer", "")
    if not expected:
        return 0.0

    # Extract the model's final answer from the last assistant message
    model_answer = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                model_answer = content
                break
            elif isinstance(content, list):
                # Handle content blocks (text, tool_calls, etc.)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        model_answer = block.get("text", "")
                        break
                if model_answer:
                    break

    if not model_answer:
        return 0.0

    # Use LLM judge for semantic evaluation
    score = await llm_judge(
        prompt=f"""You are evaluating whether a model's response correctly answers a question.

Expected Answer: {expected}

Model's Response:
{model_answer}

Does the model's response contain the correct answer? The answer may be phrased differently
but should be semantically equivalent. For example:
- "Michio Sugeno" matches "Sugeno, Michio"
- "2010" matches "in 2010" or "the year 2010"
- "June 24, 1957" matches "24 June 1957"

Return score=1 if correct, score=0 if incorrect.""",
        model="gpt-4o-mini",
        max_tokens=100,
    )

    return score
