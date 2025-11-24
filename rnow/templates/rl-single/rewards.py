from rnow.core import reward


@reward(parse_reasoning=True)
async def accuracy(args, sample, **kwargs):
    """
    Simple accuracy reward for sentiment classification.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    import re

    # Get the response from messages
    messages = sample.get("messages", [])
    response = messages[-1].get("content", "").strip()

    # Strip all markdown formatting, punctuation, and extra text
    # Remove markdown bold/italic
    response = re.sub(r'\*+([^*]+)\*+', r'\1', response)  # **text** or *text*
    response = re.sub(r'_+([^_]+)_+', r'\1', response)    # __text__ or _text_

    # Remove any sentences/phrases, just extract the sentiment word
    # Look for "positive" or "negative" as standalone words
    pattern = r'\b(positive|negative)\b'
    matches = re.findall(pattern, response.lower())

    # If we found exactly one sentiment word, use it
    if len(matches) == 1:
        response = matches[0]
    else:
        # Fallback: just take the last word after stripping punctuation
        # This handles cases like "positive." or "negative!"
        response = re.sub(r'[^\w\s]', '', response).strip().split()[-1].lower() if response else ""

    ground_truth = sample.get("metadata", {}).get("ground_truth", "").lower()

    # Exact match after cleaning
    if response == ground_truth:
        return 1.0
    else:
        return 0.0