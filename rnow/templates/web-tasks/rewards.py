"""Web task completion reward using LLM-as-judge."""

import json
import os
import urllib.request

from rnow.core import RewardArgs, reward


@reward
def task_complete(args: RewardArgs, messages: list) -> float:
    """Evaluate if the web agent completed the task."""
    task = args.metadata.get("task", "")
    if not task:
        return 0.0

    api_key = args.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return 0.0

    # Build conversation summary
    conv = "\n".join(f"{m['role']}: {str(m.get('content', ''))[:500]}" for m in messages[-10:])

    prompt = f"""Task: {task}

Conversation:
{conv}

Did the agent complete the task? Answer 0 (no) or 1 (yes)."""

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(
            {
                "model": "gpt-5-nano",
                "instructions": "You evaluate if a web agent completed a task. Answer only 0 or 1.",
                "input": prompt,
                "max_output_tokens": 16,  # Minimum required by OpenAI Responses API
            }
        ).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    for item in result.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                text = c.get("text", "").strip()
                if "1" in text:
                    return 1.0
    return 0.0
