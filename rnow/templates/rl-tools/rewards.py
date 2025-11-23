# RL with tools - demonstrates @reward and @tool decorators with automatic schema inference
from rnow.core import reward



# Define reward functions
@reward
async def tool_usage(args, sample, **kwargs):
    """
    Reward for using the correct tool.
    """
    messages = sample.get("messages", [])
    expected_tool = sample.get("metadata", {}).get("expected_tool", "")

    if len(messages) < 2:
        return 0.0

    response = messages[-1].get("content", "")

    # Check if response contains a tool call
    # This is a simplified check - in real implementation,
    # you'd parse the actual tool call format
    tool_indicators = {
        "calculator": ["calculate", "calculator", "add", "multiply", "divide"],
        "weather": ["weather", "temperature", "conditions"],
        "currency": ["convert", "currency", "USD", "EUR", "GBP"]
    }

    # Check if the expected tool was mentioned
    if expected_tool in tool_indicators:
        for indicator in tool_indicators[expected_tool]:
            if indicator.lower() in response.lower():
                return 1.0

    return 0.2  # Partial credit for any response


@reward
async def correct_answer(args, sample, **kwargs):
    """
    Reward for providing the correct answer.
    """
    messages = sample.get("messages", [])
    expected = sample.get("metadata", {}).get("expected_answer", "")

    if len(messages) < 2:
        return 0.0

    response = messages[-1].get("content", "").lower()

    # For numeric answers, check if the number is in the response
    if expected.isdigit():
        if expected in response:
            return 1.0
        else:
            return 0.0

    # For other answers, check if key information is present
    elif expected == "weather_info":
        if any(word in response for word in ["temperature", "weather", "cloudy", "sunny", "°C"]):
            return 1.0
        else:
            return 0.3

    elif expected == "conversion":
        if any(word in response for word in ["converted", "EUR", "rate", "currency"]):
            return 1.0
        else:
            return 0.3

    return 0.0