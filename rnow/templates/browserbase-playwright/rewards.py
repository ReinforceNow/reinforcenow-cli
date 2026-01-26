from rnow.core import RewardArgs, get_response, llm_judge, reward


@reward(precondition=True)
async def used_browser(args: RewardArgs, messages: list) -> float:
    """Gate: must use browser tools to get any reward."""
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                # Playwright MCP tools start with "browser_"
                if name.startswith("browser_"):
                    return 1.0
    return 0.0


@reward(timeout=120)
async def accuracy(args: RewardArgs, messages: list) -> float:
    """LLM judge: check if the response contains the correct answer.

    Uses binary scoring with chain-of-thought reasoning.
    Based on LLM-as-a-Judge best practices.
    """
    response = get_response(messages)
    expected = args.metadata.get("expected_answer", "").strip()

    if not expected:
        return 0.0

    # Simple, effective prompt based on LLM-as-a-Judge best practices:
    # 1. Binary scoring (0 or 1)
    # 2. Clear criteria definition
    # 3. Asks for reasoning (chain of thought)
    # 4. Handles semantic equivalence, not just exact match
    prompt = f"""Compare the model's response to the expected answer.

EXPECTED ANSWER: {expected}

MODEL RESPONSE: {response}

TASK: Determine if the model's response contains an answer that is semantically equivalent to the expected answer.

CRITERIA FOR CORRECT (score=1):
- The response contains the same factual information as expected
- Minor wording differences are OK (e.g., "Michio Sugeno" vs "Sugeno, Michio")
- Equivalent forms count (e.g., "2010" = "in 2010", "$120,000" = "120000 euros")
- The answer can appear anywhere in the response

CRITERIA FOR INCORRECT (score=0):
- The response contradicts the expected answer
- The response gives a different answer entirely
- The response says "I don't know" or fails to answer
- The expected information is missing from the response

Think step by step, then give your verdict."""

    return await llm_judge(prompt, secrets=args.secrets)
