# generation.py
# Quickstart Tutorial: Sentiment Analysis Task

def generate(prompt: str, **kwargs) -> str:
    """
    Generate a sentiment analysis response.

    This is a quickstart example that analyzes the sentiment of text.
    In a real scenario, this would call an LLM API or use a local model.

    Args:
        prompt: Input text containing a review or statement to analyze
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        Sentiment classification: "positive", "negative", or "neutral"
    """
    # Dummy implementation for quickstart tutorial
    # Replace with actual LLM API call or model inference

    prompt_lower = prompt.lower()

    # Simple keyword-based sentiment (for demonstration only)
    positive_words = ["good", "great", "excellent", "amazing", "love", "best"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "poor"]

    positive_count = sum(word in prompt_lower for word in positive_words)
    negative_count = sum(word in prompt_lower for word in negative_words)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"
