"""
Tool entry point for ReinforceNow with robust validation.

Validates at decorator-time:
- Function has docstring or description
- No *args/**kwargs in signature
- All parameters have type hints
- Return type is declared and JSON-serializable
- Supports Optional[T], list[T], dict[str, T], Literal[...], Union types

Validates at runtime:
- Arguments match schema (required keys, no extra keys, type coercion)
"""

import inspect
from collections.abc import Callable
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# Global registry for tool functions
TOOL_REGISTRY: dict[str, Callable] = {}

# Types that are JSON-serializable
JSON_SERIALIZABLE_TYPES = (str, int, float, bool, list, dict, type(None))


def clear_tool_registry() -> None:
    """Clear the tool registry (useful for testing multiple projects)."""
    TOOL_REGISTRY.clear()


def _map_type_to_json_schema(py_type: Any) -> dict[str, Any]:
    """
    Map a Python type annotation to a JSON Schema fragment.

    Supports:
    - Primitives: str, int, float, bool
    - Collections: list, List[T], dict, Dict[str, T]
    - Optional: Optional[T], T | None, Union[T, None]
    - Literal: Literal["foo", "bar"]
    - Any: defaults to string
    """
    origin = get_origin(py_type) or py_type
    args = get_args(py_type)

    # Simple primitives
    if origin is str:
        return {"type": "string"}
    if origin is int:
        return {"type": "integer"}
    if origin is float:
        return {"type": "number"}
    if origin is bool:
        return {"type": "boolean"}
    if origin is type(None):
        return {"type": "null"}

    # List[T] / list[T]
    if origin in (list, list):
        if args:
            return {
                "type": "array",
                "items": _map_type_to_json_schema(args[0]),
            }
        return {"type": "array"}

    # Dict[K, V] / dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Literal["foo", "bar"] -> enum
    try:
        from typing import Literal

        if origin is Literal:
            return {"type": "string", "enum": list(args)}
    except ImportError:
        pass

    # Optional[T] / Union[T, None] / T | None
    if origin is Union:
        # Filter out None types
        non_none_args = [a for a in args if a is not type(None)]
        has_none = len(non_none_args) < len(args)

        if len(non_none_args) == 1:
            # Optional[T] case
            schema = _map_type_to_json_schema(non_none_args[0])
            if has_none and "type" in schema:
                # Allow null
                current_type = schema["type"]
                if isinstance(current_type, list):
                    schema["type"] = current_type + ["null"]
                else:
                    schema["type"] = [current_type, "null"]
            return schema
        else:
            # Union[A, B, C] case - use anyOf
            schemas = [_map_type_to_json_schema(a) for a in non_none_args]
            if has_none:
                schemas.append({"type": "null"})
            return {"anyOf": schemas}

    # Any or unknown type - default to string
    return {"type": "string"}


def _is_json_serializable_type(py_type: Any) -> bool:
    """Check if a type annotation represents a JSON-serializable type."""
    origin = get_origin(py_type) or py_type
    args = get_args(py_type)

    # Direct JSON types
    if origin in JSON_SERIALIZABLE_TYPES:
        return True

    # List/dict with JSON-serializable contents
    if origin in (list, list):
        if args:
            return _is_json_serializable_type(args[0])
        return True

    if origin is dict:
        if args and len(args) >= 2:
            return _is_json_serializable_type(args[1])
        return True

    # Optional/Union
    if origin is Union:
        return all(_is_json_serializable_type(a) for a in args)

    # Literal
    try:
        from typing import Literal

        if origin is Literal:
            return True
    except ImportError:
        pass

    # Any
    return py_type is Any


def _infer_schema(func: Callable) -> dict[str, Any]:
    """
    Infer JSON schema from function signature with full type support.

    Raises:
        TypeError: If parameters are missing type hints or use *args/**kwargs
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # Disallow *args/**kwargs - tools must have explicit parameters
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"Tool '{func.__name__}' cannot use *args. "
                "Define explicit, typed parameters instead."
            )
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"Tool '{func.__name__}' cannot use **kwargs. "
                "Define explicit, typed parameters instead."
            )

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Require type hint for every parameter
        if param_name not in hints:
            raise TypeError(
                f"Missing type hint for parameter '{param_name}' in tool '{func.__name__}'. "
                "All parameters must have type annotations."
            )

        param_type = hints[param_name]
        properties[param_name] = _map_type_to_json_schema(param_type)

        # Mark as required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _validate_return_type(func: Callable) -> None:
    """
    Validate that the function has a return type annotation that is JSON-serializable.

    Raises:
        TypeError: If return type is missing or not JSON-serializable
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # Check return annotation exists
    if sig.return_annotation is inspect.Signature.empty and "return" not in hints:
        raise TypeError(
            f"Tool '{func.__name__}' must declare a return type annotation. "
            "Add -> ReturnType to the function signature."
        )

    return_type = hints.get("return", sig.return_annotation)

    # Check return type is JSON-serializable
    if not _is_json_serializable_type(return_type):
        raise TypeError(
            f"Tool '{func.__name__}' return type '{return_type}' is not JSON-serializable. "
            "Use dict, list, str, int, float, bool, or None."
        )


def validate_tool_args(
    tool_name: str, schema: dict[str, Any], args: dict[str, Any]
) -> tuple[bool, str | None]:
    """
    Validate tool arguments against the schema.

    Args:
        tool_name: Name of the tool (for error messages)
        schema: The tool's JSON schema
        args: Arguments to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required = schema.get("required", [])
    props = schema.get("properties", {})
    allow_additional = schema.get("additionalProperties", True)

    # Check for missing required arguments
    for key in required:
        if key not in args:
            return False, f"Tool '{tool_name}': missing required argument '{key}'."

    # Check for unexpected arguments
    if not allow_additional:
        for key in args:
            if key not in props:
                return False, f"Tool '{tool_name}': unexpected argument '{key}'."

    # Type validation with coercion attempts
    for key, value in args.items():
        if key not in props:
            continue

        prop_schema = props[key]
        expected_type = prop_schema.get("type")

        if expected_type is None:
            continue

        # Handle array types (can be ["string", "null"])
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]

        # Check if value matches any expected type
        valid = False
        for exp_type in expected_types:
            if (
                exp_type == "string"
                and isinstance(value, str)
                or exp_type == "integer"
                and isinstance(value, int)
                and not isinstance(value, bool)
                or exp_type == "number"
                and isinstance(value, int | float)
                and not isinstance(value, bool)
                or exp_type == "boolean"
                and isinstance(value, bool)
                or exp_type == "array"
                and isinstance(value, list)
                or exp_type == "object"
                and isinstance(value, dict)
                or exp_type == "null"
                and value is None
            ):
                valid = True

        if not valid:
            # Attempt type coercion for common cases
            coerced, coerced_value = _try_coerce(value, expected_types)
            if coerced:
                args[key] = coerced_value
            else:
                return False, (
                    f"Tool '{tool_name}': argument '{key}' expected type "
                    f"{expected_types}, got {type(value).__name__}."
                )

    return True, None


def _try_coerce(value: Any, expected_types: list[str]) -> tuple[bool, Any]:
    """
    Try to coerce a value to one of the expected types.

    Returns:
        Tuple of (success, coerced_value)
    """
    for exp_type in expected_types:
        try:
            if exp_type == "integer" and isinstance(value, str):
                return True, int(value)
            if exp_type == "number" and isinstance(value, str):
                return True, float(value)
            if exp_type == "boolean" and isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True, True
                if value.lower() in ("false", "0", "no"):
                    return True, False
            if exp_type == "string" and not isinstance(value, str):
                return True, str(value)
        except (ValueError, TypeError):
            continue

    return False, value


def tool(fn: Callable = None) -> Callable:
    """
    Decorator to register tool functions with robust validation.

    Validates at decorator-time:
    - Function has a non-empty docstring
    - No *args/**kwargs in signature
    - All parameters have type hints
    - Return type is declared and JSON-serializable

    Both sync and async functions are supported. Execution strategy
    is determined automatically at runtime.

    Usage:
        @tool
        def web_search(query: str) -> dict:
            '''Search the web.'''
            return requests.get(...).json()

        @tool
        def calculator(expr: str) -> float:
            '''Evaluate math expression.'''
            return eval(expr)

    Supported parameter types:
        - Primitives: str, int, float, bool
        - Collections: list, List[T], dict, Dict[str, T]
        - Optional: Optional[T], T | None
        - Literal: Literal["option1", "option2"]
        - Union: Union[str, int]
    """

    def decorator(func: Callable) -> Callable:
        # 1. Validate docstring (must be non-empty)
        doc = (func.__doc__ or "").strip()
        if not doc:
            raise ValueError(
                f"Tool '{func.__name__}' must have a non-empty docstring. "
                "Add a docstring to the function."
            )

        # 2. Validate return type
        try:
            _validate_return_type(func)
        except TypeError as e:
            raise TypeError(f"Tool registration failed: {e}") from e

        # 3. Infer and validate schema (checks type hints, no *args/**kwargs)
        try:
            schema = _infer_schema(func)
        except TypeError as e:
            raise TypeError(f"Tool registration failed: {e}") from e

        # 4. Warn if overwriting existing tool
        if func.__name__ in TOOL_REGISTRY:
            import warnings

            warnings.warn(
                f"Tool '{func.__name__}' is being overwritten in the registry.",
                UserWarning,
                stacklevel=2,
            )

        # 5. Attach metadata and register
        func._is_tool = True
        func._tool_name = func.__name__
        func._schema = schema
        func._description = doc  # Already validated and stripped above

        TOOL_REGISTRY[func._tool_name] = func

        return func

    # Support both @tool and @tool()
    return decorator(fn) if fn else decorator


def validate_tools_file(filepath) -> list:
    """
    Validate an env.py file without executing it.

    Parses the AST to find @tool decorated functions and checks:
    - Function has a non-empty docstring
    - No *args/**kwargs
    - Has type annotations for all parameters
    - Has a return type annotation

    Both sync and async functions are supported.

    Returns a list of error messages (empty if valid).
    """
    import ast
    from pathlib import Path

    errors = []
    filepath = Path(filepath)

    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return [f"Syntax error in {filepath.name}: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Check if function has @tool decorator
            is_tool = False
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "tool"
                    or (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "tool"
                    )
                ):
                    is_tool = True

            if is_tool:
                # Both async and sync functions are allowed

                # Check for docstring using ast.get_docstring (canonical way)
                doc = ast.get_docstring(node)
                if not (doc or "").strip():
                    errors.append(
                        f"Tool '{node.name}' must have a non-empty docstring. "
                        "Add a docstring to describe what the tool does and its arguments."
                    )

                # Check for *args/**kwargs (not allowed in tools)
                if node.args.vararg:
                    errors.append(
                        f"Tool '{node.name}' cannot use *{node.args.vararg.arg}. "
                        "Define explicit, typed parameters instead."
                    )
                if node.args.kwarg:
                    errors.append(
                        f"Tool '{node.name}' cannot use **{node.args.kwarg.arg}. "
                        "Define explicit, typed parameters instead."
                    )

                # Check return type annotation
                if node.returns is None:
                    errors.append(
                        f"Tool '{node.name}' must have a return type annotation. "
                        "Add '-> ReturnType' to the function signature."
                    )

                # Check parameter type annotations (args + kwonlyargs, skip 'self' and 'cls')
                all_args = list(node.args.args) + list(node.args.kwonlyargs)
                for arg in all_args:
                    if arg.arg in ("self", "cls"):
                        continue
                    if arg.annotation is None:
                        errors.append(
                            f"Tool '{node.name}': parameter '{arg.arg}' must have a type annotation."
                        )

    return errors
