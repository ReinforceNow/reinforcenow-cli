"""
Tool entry point for ReinforceNow with validation.
"""
import inspect
from typing import Callable, Dict, Any, get_type_hints

# Global registry for tool functions
TOOL_REGISTRY: Dict[str, Callable] = {}


def _infer_schema(func: Callable) -> dict:
    """
    Infer JSON schema from function signature.
    Raises errors if parameters are missing type hints.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'cls']:
            continue

        # Require type hint for every parameter
        if param_name not in hints:
            raise TypeError(
                f"Missing type hint for parameter '{param_name}' in tool '{func.__name__}'. "
                f"All parameters must have type annotations."
            )

        # Map Python types to JSON schema types
        param_type = hints[param_name]
        json_type = "string"  # default
        if param_type == int:
            json_type = "integer"
        elif param_type == float:
            json_type = "number"
        elif param_type == bool:
            json_type = "boolean"
        elif param_type == list:
            json_type = "array"
        elif param_type == dict:
            json_type = "object"

        properties[param_name] = {"type": json_type}

        # Mark as required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def tool(fn: Callable = None, *, description: str = None) -> Callable:
    """
    Decorator to register tool functions with validation.

    Enforces:
      - Every parameter must have a type hint
      - Function must have a docstring or provided description

    Usage:
        @tool
        def multiply(a: int, b: int) -> int:
            '''Multiply two numbers.'''
            return a * b

        @tool
        async def weather(location: str) -> dict:
            '''Get weather for a location.'''
            return {"temp": 72, "conditions": "sunny"}
    """
    def decorator(func: Callable) -> Callable:
        # Validate documentation
        if not (func.__doc__ or description):
            raise ValueError(
                f"Tool '{func.__name__}' must have a docstring or description. "
                f"Add a docstring to the function or use @tool(description='...')."
            )

        # Infer and validate schema (this checks type hints)
        try:
            schema = _infer_schema(func)
        except TypeError as e:
            # Re-raise with more context
            raise TypeError(f"Tool registration failed: {e}") from e

        # Registration
        func._is_tool = True
        func._tool_name = func.__name__
        func._schema = schema
        func._description = (description or func.__doc__).strip()

        # Register for introspection
        TOOL_REGISTRY[func._tool_name] = func

        return func

    # Support both @tool and @tool(description="...")
    return decorator(fn) if fn else decorator