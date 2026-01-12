"""InSTA-150k criteria evaluation reward using GPT-5-nano."""

import json
import urllib.request

from rnow.core import RewardArgs, reward


@reward
def criteria_met(args: RewardArgs, messages: list) -> float:
    """Evaluate if agent satisfied the criteria using LLM-as-judge."""
    criteria = args.metadata.get("criteria", [])
    if not criteria:
        return 0.0

    api_key = args.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return 0.0

    conversation = "\n".join(f"{m['role'].upper()}: {m.get('content', '')}" for m in messages)

    instructions = """You evaluate if a web agent completed a task based on criteria.
For each criterion, respond with 1 if met, 0 if not.
Respond with ONLY a comma-separated list like: 1,0,1"""

    prompt = f"""Task: {args.metadata.get('instruction', '')}

Criteria:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(criteria))}

Agent conversation:
{conversation[:4000]}"""

    try:
        data = json.dumps(
            {
                "model": "gpt-5-nano",
                "instructions": instructions,
                "input": prompt,
                "max_output_tokens": 64,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        for item in result.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        scores = [
                            int(x) for x in c.get("text", "").split(",") if x.strip() in ("0", "1")
                        ]
                        return sum(scores) / len(criteria) if scores else 0.0
        return 0.0
    except Exception:
        return 0.0
