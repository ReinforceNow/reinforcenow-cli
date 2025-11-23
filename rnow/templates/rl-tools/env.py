
from rnow.core import tool

@tool
async def calculator(operation: str, a: float, b: float) -> dict:
    """
    Simple calculator tool for basic arithmetic operations.
    Supports add, subtract, multiply, and divide operations.
    """
    if operation == "add":
        return {"result": a + b}
    elif operation == "subtract":
        return {"result": a - b}
    elif operation == "multiply":
        return {"result": a * b}
    elif operation == "divide":
        if b != 0:
            return {"result": a / b}
        else:
            return {"error": "Division by zero"}
    else:
        return {"error": "Unknown operation"}


@tool
async def weather(city: str) -> dict:
    """
    Get weather information for a city.
    Returns temperature, conditions, and humidity.
    """
    # In real implementation, this would call a weather API
    return {
        "city": city,
        "temperature": "22°C",
        "conditions": "Partly cloudy",
        "humidity": "65%"
    }


@tool
async def currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert currency from one type to another.
    Supports USD, EUR, and GBP conversions.
    """
    # Mock conversion rates
    rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27
    }

    rate = rates.get((from_currency, to_currency), 1.0)
    return {
        "original": amount,
        "converted": amount * rate,
        "from": from_currency,
        "to": to_currency,
        "rate": rate
    }
