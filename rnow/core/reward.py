"""
Reward entry point for ReinforceNow with validation.

Validates at decorator-time:
- Function has correct signature: (args: RewardArgs, messages: list) -> float
- Function has docstring or description

Both sync and async functions are supported. Execution strategy is
determined automatically at runtime.
"""

import inspect
from collections.abc import Callable
from typing import get_type_hints

from rnow.models import RewardArgs

# Global registry for reward functions
REWARD_REGISTRY: dict[str, Callable] = {}


def clear_reward_registry() -> None:
    """Clear the reward registry (useful for testing multiple projects)."""
    REWARD_REGISTRY.clear()


def is_precondition(name: str) -> bool:
    """Check if a reward function is marked as a precondition."""
    fn = REWARD_REGISTRY.get(name)
    if fn is None:
        return False
    return getattr(fn, "_is_precondition", False)


def compute_total_reward(reward_results: dict[str, float]) -> float:
    """
    Compute total reward with precondition logic.

    Uses the _is_precondition attribute on registered reward functions.
    If any precondition reward is 0, total reward is 0.
    Otherwise, total reward is sum of all rewards.

    Args:
        reward_results: Dict mapping reward name to its value

    Returns:
        Total reward value
    """
    precondition_sum = 0.0
    other_sum = 0.0

    for name, value in reward_results.items():
        if is_precondition(name):
            # If any precondition fails (returns 0), total reward is 0
            if value == 0.0:
                return 0.0
            precondition_sum += value
        else:
            other_sum += value

    return precondition_sum + other_sum


def _validate_reward_signature(func: Callable) -> None:
    """
    Validate that a reward function has the correct signature.

    Expected signature:
        def reward_fn(args: RewardArgs, messages: list) -> float

    Both sync and async functions are supported.

    Raises:
        TypeError: If signature doesn't match expected format
    """

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Check parameter count (should be exactly 2: args and messages)
    if len(params) != 2:
        raise TypeError(
            f"Reward '{func.__name__}' must have exactly 2 parameters: "
            "(args: RewardArgs, messages: list). Got {len(params)} parameters."
        )

    # Get type hints
    hints = get_type_hints(func)

    # Check first parameter (args: RewardArgs)
    first_param = params[0]
    if first_param.name not in hints:
        raise TypeError(
            f"Reward '{func.__name__}': parameter '{first_param.name}' must have "
            "type hint 'RewardArgs'."
        )
    first_type = hints[first_param.name]
    if first_type is not RewardArgs:
        raise TypeError(
            f"Reward '{func.__name__}': first parameter must be typed as 'RewardArgs', "
            f"got '{first_type}'."
        )

    # Check second parameter (messages: list)
    second_param = params[1]
    if second_param.name not in hints:
        raise TypeError(
            f"Reward '{func.__name__}': parameter '{second_param.name}' must have "
            "type hint 'list'."
        )
    second_type = hints[second_param.name]
    # Allow list or List (from typing)
    from typing import get_origin

    second_origin = get_origin(second_type) or second_type
    if second_origin not in (list, list):
        raise TypeError(
            f"Reward '{func.__name__}': second parameter must be typed as 'list', "
            f"got '{second_type}'."
        )

    # Check return type
    if "return" not in hints:
        raise TypeError(f"Reward '{func.__name__}' must declare return type '-> float'.")
    return_type = hints["return"]
    if return_type is not float:
        raise TypeError(f"Reward '{func.__name__}' must return 'float', got '{return_type}'.")


def reward(fn: Callable = None, *, precondition: bool = False) -> Callable:
    """
    Decorator to register reward functions with validation.

    Validates at decorator-time:
    - Signature is (args: RewardArgs, messages: list) -> float

    Both sync and async functions are supported. Execution strategy
    is determined automatically at runtime.

    Usage:
        @reward
        def accuracy(args: RewardArgs, messages: list) -> float:
            ground_truth = args.metadata.get("ground_truth")
            response = messages[-1].get("content", "")
            return 1.0 if ground_truth in response else 0.0

        @reward(precondition=True)  # Mark as precondition reward
        def format_check(args: RewardArgs, messages: list) -> float:
            # If this returns 0, total reward is 0
            # If this returns 1, total reward is 1 + sum(other rewards)
            return 1.0 if valid_format else 0.0

    Args:
        precondition: If True, this reward acts as a gate:
            - If precondition reward is 0, total reward is 0
            - If precondition reward is 1, total reward is 1 + sum(other rewards)
    """

    def decorator(func):
        # Validate signature (async, correct params, return type)
        try:
            _validate_reward_signature(func)
        except TypeError as e:
            raise TypeError(f"Reward registration failed: {e}") from e

        # Warn if overwriting existing reward
        if func.__name__ in REWARD_REGISTRY:
            import warnings

            warnings.warn(
                f"Reward '{func.__name__}' is being overwritten in the registry.",
                UserWarning,
                stacklevel=2,
            )

        # Store metadata
        func._is_reward = True
        func._reward_name = func.__name__
        func._is_precondition = precondition

        # Register the function
        REWARD_REGISTRY[func.__name__] = func
        return func

    # Support both @reward and @reward(precondition=False)
    return decorator(fn) if fn else decorator


def validate_rewards_file(filepath) -> list:
    """
    Validate a rewards.py file without executing it.

    Parses the AST to find @reward decorated functions and checks:
    - Has correct number of parameters
    - Return values are between 0.0 and 1.0

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

    def get_numeric_value(node) -> float | None:
        """Extract numeric value from AST node if it's a constant."""
        # Python 3.8+: ast.Constant
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        # Python 3.7: ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return float(node.n)
        # Handle negative numbers: ast.UnaryOp with ast.USub
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = get_numeric_value(node.operand)
            if inner is not None:
                return -inner
        return None

    def check_return_values(func_node, func_name: str):
        """Check all return statements in a function for valid reward values."""
        for child in ast.walk(func_node):
            if isinstance(child, ast.Return) and child.value is not None:
                # Check direct numeric returns
                value = get_numeric_value(child.value)
                if value is not None and (value < 0.0 or value > 1.0):
                    errors.append(
                        f"Reward '{func_name}' returns {value}, must be between 0.0 and 1.0 "
                        f"(line {child.lineno})"
                    )

                # Check ternary expressions: x if cond else y
                if isinstance(child.value, ast.IfExp):
                    for branch in [child.value.body, child.value.orelse]:
                        value = get_numeric_value(branch)
                        if value is not None and (value < 0.0 or value > 1.0):
                            errors.append(
                                f"Reward '{func_name}' returns {value}, must be between 0.0 and 1.0 "
                                f"(line {child.lineno})"
                            )

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Check if function has @reward decorator
            is_reward = False
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "reward"
                    or (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "reward"
                    )
                ):
                    is_reward = True

            if is_reward:
                # Both async and sync functions are allowed

                # Check parameter count
                params = node.args.args
                if len(params) != 2:
                    errors.append(
                        f"Reward '{node.name}' must have exactly 2 parameters: "
                        f"(args: RewardArgs, messages: list). Got {len(params)} parameters."
                    )

                # Check return type annotation
                if node.returns is None:
                    errors.append(
                        f"Reward '{node.name}' must have a return type annotation '-> float'."
                    )

                # Check parameter type annotations
                for i, arg in enumerate(params):
                    if arg.annotation is None:
                        param_hint = "RewardArgs" if i == 0 else "list"
                        errors.append(
                            f"Reward '{node.name}': parameter '{arg.arg}' must have type hint '{param_hint}'."
                        )

                # Check return values are between 0.0 and 1.0
                check_return_values(node, node.name)

    return errors


def get_reward_names_from_file(filepath) -> set[str]:
    """
    Extract reward function names from a rewards.py file without executing it.

    Parses the AST to find @reward decorated functions and returns their names.

    Returns:
        Set of reward function names defined in the file.
    """
    import ast
    from pathlib import Path

    names = set()
    filepath = Path(filepath)

    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return names

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Check if function has @reward decorator
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "reward"
                    or (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "reward"
                    )
                ):
                    names.add(node.name)

    return names
