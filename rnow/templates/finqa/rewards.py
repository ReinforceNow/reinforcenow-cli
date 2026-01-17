import os

import requests

from rnow.core import RewardArgs, get_response, reward

API_URL = "https://api.openai.com/v1/responses"


@reward(timeout=300)
def llm_judge(args: RewardArgs, messages: list) -> float:
    """Judge if model's numerical answer matches expected."""
    expected = args.metadata["answer"]
    model_answer = get_response(messages)

    prompt = (
        f"Expected: {expected}\n"
        f"Model: {model_answer}\n\n"
        "Match? (15.4%=15.4, -13.3% → 13.3 drop; no approximations)\n"
        "Answer 1 or 0."
    )

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "gpt-5-nano", "input": prompt, "reasoning": {"effort": "low"}},
    )
    resp.raise_for_status()

    output = resp.json()["output"]
    text = next(x["content"][0]["text"] for x in output if x.get("type") == "message")
    return 1.0 if text.strip() == "1" else 0.0
