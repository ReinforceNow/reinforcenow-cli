import json
import re

from rnow.core import RewardArgs, get_response, reward


@reward
def extraction_accuracy(args: RewardArgs, messages: list) -> float:
    """Check food/drink extraction accuracy."""
    response = get_response(messages)

    # Try to parse JSON from response
    try:
        # Find JSON in response
        match = re.search(r"\{[^{}]*\}", response)
        if match:
            pred = json.loads(match.group())
        else:
            return 0.0
    except:
        return 0.0

    score = 0.0
    gt_food = set(args.metadata.get("food_items", []))
    gt_drink = set(args.metadata.get("drink_items", []))
    gt_is_food = args.metadata.get("is_food", 0)

    # is_food correct (20%)
    if pred.get("is_food") == gt_is_food:
        score += 0.2

    # Food items overlap (40%)
    pred_food = set(pred.get("food_items", []))
    if gt_food:
        overlap = (
            len(gt_food & pred_food) / len(gt_food | pred_food) if (gt_food | pred_food) else 1.0
        )
        score += 0.4 * overlap
    elif not pred_food:
        score += 0.4

    # Drink items overlap (40%)
    pred_drink = set(pred.get("drink_items", []))
    if gt_drink:
        overlap = (
            len(gt_drink & pred_drink) / len(gt_drink | pred_drink)
            if (gt_drink | pred_drink)
            else 1.0
        )
        score += 0.4 * overlap
    elif not pred_drink:
        score += 0.4

    return score
