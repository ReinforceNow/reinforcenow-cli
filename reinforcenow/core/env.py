"""
Environment entry point for ReinforceNow.
"""
import inspect
from typing import Dict

# Global registry for environment classes
ENV_REGISTRY: Dict[str, type] = {}


def env(cls: type = None, *, name: str = None, max_turns: int = 1) -> type:
    """
    Decorator to register environment classes.

    Usage:
        @env
        class CustomEnv(Env):
            pass
    """
    def decorator(env_class):
        if not inspect.isclass(env_class):
            raise TypeError(f"@env can only decorate classes, not {type(env_class)}")

        # Register with class name as key (or custom name)
        registry_name = name or env_class.__name__
        ENV_REGISTRY[registry_name] = env_class

        # Add metadata
        env_class._is_env = True
        env_class._max_turns = max_turns

        return env_class

    # Support both @env and @env(name="...", max_turns=5)
    if cls is None:
        return decorator
    return decorator(cls)