"""
Tool entry point for ReinforceNow.
"""
import inspect
from typing import Callable, Dict, Optional

# Global registry for tool functions
TOOL_REGISTRY: Dict[str, Callable] = {}


def tool(fn: Callable = None, *, schema: dict = None, description: str = None) -> Callable:
    """
    Decorator to register tool functions using the function name.

    Usage:
        @tool
        async def multiply(arguments: dict):
            return arguments["a"] * arguments["b"]

        @tool(schema={"a": "number", "b": "number"})
        async def divide(arguments: dict):
            return arguments["a"] / arguments["b"]
    """
    def decorator(func):
        # Register with function name as key
        TOOL_REGISTRY[func.__name__] = func

        # Add metadata
        func._is_tool = True
        func._schema = schema
        func._description = description or f"Tool: {func.__name__}"

        return func

    # Support both @tool and @tool(schema={...})
    if fn is None:
        return decorator
    return decorator(fn)