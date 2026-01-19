"""
LLM Judge - Use an LLM to evaluate responses.

Provides a simple LLM judge that takes a prompt and returns a score.
Supports any OpenAI-compatible API endpoint.
"""

import os
import re
from typing import Any

import requests


def llm_judge(
    prompt: str,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    secrets: dict[str, Any] | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 10,
    timeout: int = 60,
    **kwargs: Any,
) -> float:
    """
    Send a prompt to an LLM and parse the response as a score.

    Args:
        prompt: The prompt to send to the LLM
        api_url: OpenAI-compatible API endpoint (default: OpenAI)
        api_key: API key (takes priority over secrets/env)
        secrets: Dict of secrets (e.g., args.secrets) - checks for OPENAI_API_KEY
        model: Model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Max tokens in response (default: 10)
        timeout: Request timeout in seconds (default: 60)
        **kwargs: Additional parameters to pass to the API

    Returns:
        Float between 0.0 and 1.0

    Example:
        @reward(timeout=120)
        def accuracy(args: RewardArgs, messages: list) -> float:
            prompt = f"Expected: {args.metadata['answer']}\\nModel: {get_response(messages)}\\nMatch? Answer 1 or 0."
            return llm_judge(prompt, secrets=args.secrets)
    """
    url = api_url or os.environ.get(
        "LLM_JUDGE_API_URL", "https://api.openai.com/v1/chat/completions"
    )

    # Resolve API key: explicit > secrets dict > env vars
    key = api_key
    if not key and secrets:
        key = secrets.get("OPENAI_API_KEY") or secrets.get("LLM_JUDGE_API_KEY")
    if not key:
        key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not key:
        # Debug: print available env vars that might contain API keys
        env_keys = [
            k
            for k in os.environ
            if "API" in k.upper() or "KEY" in k.upper() or "OPENAI" in k.upper()
        ]
        print(f"[llm_judge DEBUG] No API key found. Relevant env vars: {env_keys}")
        print(f"[llm_judge DEBUG] OPENAI_API_KEY in env: {'OPENAI_API_KEY' in os.environ}")
        raise ValueError(
            "No API key provided. Pass api_key, secrets=args.secrets, or set OPENAI_API_KEY env var."
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    data = resp.json()

    # Standard OpenAI chat completion format
    if "choices" in data:
        text = data["choices"][0]["message"]["content"].strip()
    # OpenAI responses API format
    elif "output" in data:
        output = data["output"]
        text = next((x["content"][0]["text"] for x in output if x.get("type") == "message"), "0")
    else:
        raise ValueError(f"Unexpected API response format: {data}")

    return _parse_score(text)


def _parse_score(text: str) -> float:
    """Parse a numeric score from LLM output."""
    text = text.strip()

    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        pass

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        if score > 1:
            score = score / 100
        return max(0.0, min(1.0, score))

    if text.lower() in ("1", "yes", "true", "correct"):
        return 1.0
    if text.lower() in ("0", "no", "false", "incorrect"):
        return 0.0

    return 0.0
