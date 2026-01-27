"""
LLM Judge - Use an LLM to evaluate responses with structured outputs.

Provides an LLM judge that uses OpenAI's Responses API with structured outputs
to reliably return scores. Always uses structured outputs for guaranteed
valid responses.
"""

import os
from typing import Any

import aiohttp

# Default schema: binary 0/1 score
DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the response meets the criteria, 0 otherwise",
        }
    },
    "required": ["score"],
    "additionalProperties": False,
}


async def llm_judge(
    prompt: str,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    secrets: dict[str, Any] | None = None,
    model: str = "gpt-5-nano",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 520,
    schema: dict | None = None,
    score_key: str = "score",
    **kwargs: Any,
) -> float:
    """
    Send a prompt to an LLM and get a score using structured outputs.

    Uses OpenAI's Responses API with structured outputs to guarantee a valid JSON response.
    By default returns 0 or 1, but you can provide a custom schema.

    Args:
        prompt: The prompt to send to the LLM (should ask for a score)
        api_url: OpenAI Responses API endpoint (default: OpenAI)
        api_key: API key (takes priority over secrets/env)
        secrets: Dict of secrets (e.g., args.secrets) - checks for OPENAI_API_KEY
        model: Model to use (default: gpt-5-nano)
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Max tokens in response (default: 1024)
        timeout: Request timeout in seconds (default: 60)
        schema: Custom JSON schema for structured output (default: binary 0/1)
        score_key: Key to extract score from response (default: "score")
        **kwargs: Additional parameters to pass to the API

    Returns:
        Float between 0.0 and 1.0

    Example (default 0/1 scoring):
        @reward
        async def quality(args: RewardArgs, messages: list) -> float:
            response = get_response(messages)
            prompt = f"Is this response helpful? Answer 1 or 0.\\n\\n{response}"
            return await llm_judge(prompt, secrets=args.secrets)

    Example (custom schema with 0-10 scale):
        @reward
        async def detailed_quality(args: RewardArgs, messages: list) -> float:
            custom_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 10}
                },
                "required": ["reasoning", "score"],
                "additionalProperties": False
            }
            score = await llm_judge(
                prompt,
                secrets=args.secrets,
                schema=custom_schema,
            )
            return score / 10.0  # Normalize to 0-1
    """
    url = api_url or os.environ.get("LLM_JUDGE_API_URL", "https://api.openai.com/v1/responses")

    # Resolve API key: explicit > secrets dict > env vars
    key = api_key
    if not key and secrets:
        key = secrets.get("OPENAI_API_KEY") or secrets.get("LLM_JUDGE_API_KEY")
    if not key:
        key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not key:
        raise ValueError(
            "No API key provided. Pass api_key, secrets=args.secrets, or set OPENAI_API_KEY env var."
        )

    # Use custom schema or default binary schema
    output_schema = schema or DEFAULT_SCHEMA

    # Build the request payload for Responses API with structured outputs
    payload = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": max_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "judge_score",
                "strict": True,
                "schema": output_schema,
            }
        },
        **kwargs,
    }

    # Only add temperature for models that support it (gpt-5 models don't)
    if not model.startswith("gpt-5"):
        payload["temperature"] = temperature

    async with (
        aiohttp.ClientSession() as session,
        session.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp,
    ):
        resp.raise_for_status()
        data = await resp.json()

    return _parse_structured_response(data, score_key)


def _parse_structured_response(data: dict, score_key: str = "score") -> float:
    """Parse structured output response (JSON with score field)."""
    import json

    # Standard OpenAI chat completion format
    if "choices" in data:
        text = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(text)
            score = parsed.get(score_key, 0)
            return float(score)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return 0.0

    # OpenAI responses API format
    if "output" in data:
        for item in data["output"]:
            if item.get("type") == "message":
                for content in item.get("content", []):
                    # Handle both "text" and "output_text" content types
                    if content.get("type") in ("text", "output_text"):
                        try:
                            parsed = json.loads(content["text"])
                            score = parsed.get(score_key, 0)
                            return float(score)
                        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                            pass
        return 0.0

    raise ValueError(f"Unexpected API response format: {data}")
