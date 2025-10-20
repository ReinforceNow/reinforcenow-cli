import inspect
import json
import re
from string import Template
from typing import Dict, Any, Callable, Optional

from reinforcenow.models import Env, StopCondition, Action, StepResult, Observation


# Global registry for environment classes
ENV_REGISTRY: Dict[str, type] = {}


class ReinforceNowEnv(Env):
    """Environment for both single-turn and multi-turn RL training with reward and tool registry support."""

    def __init__(
        self,
        data: dict,
        renderer: Any,
        reward_registry: Dict[str, Callable],
        tool_registry: Optional[Dict[str, Callable]] = None,
        max_turns: int = 1,
        max_tokens: int = 2048,
    ):
        self.messages_templates = data["messages"]
        self.reward_names = data["rewards"]
        self.variables = data.get("variables", {})
        self.metadata = data["metadata"]
        self.tool_registry = tool_registry or {}

        # Collect reward functions (validated)
        self.reward_fns = []
        for name in self.reward_names:
            if name not in reward_registry:
                raise ValueError(f"Reward function '{name}' not found in registry")
            self.reward_fns.append(reward_registry[name])

        self.renderer = renderer
        self.max_turns = max_turns
        self.max_tokens = max_tokens

        # Substitute context variables into message templates
        ctx = {**self.metadata, **self.variables}
        self.messages = [
            {"role": msg["role"], "content": Template(msg["content"]).safe_substitute(ctx)}
            for msg in self.messages_templates
        ]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return initial observation and stop condition."""
        self.turn_count = 0
        self.prompt_tokens = sum(
            len(self.renderer.tokenizer.encode(m["content"])) for m in self.messages
        )
        self.conversation = self.messages.copy()

        observation = self.renderer.build_generation_prompt(messages=self.conversation)
        stop = self.renderer.get_stop_sequences()
        return observation, stop

    async def step(self, action: Action) -> StepResult:
        """Execute one environment step, optionally invoking tools and computing rewards."""
        self.turn_count += 1
        response = self.renderer.tokenizer.decode(action)
        self.conversation.append({"role": "assistant", "content": response})

        # Tool Calling
        tool_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if tool_match and self.tool_registry:
            try:
                tool_data = json.loads(tool_match.group(1))
                tool_name = tool_data.get("name")
                args = tool_data.get("arguments", {})

                if tool_name not in self.tool_registry:
                    raise ValueError(f"Tool '{tool_name}' not found in registry")

                tool_fn = self.tool_registry[tool_name]
                tool_result = (
                    await tool_fn(args) if inspect.iscoroutinefunction(tool_fn) else tool_fn(args)
                )

                self.conversation.append({
                    "role": "tool",
                    "content": f"<tool_result>{json.dumps(tool_result)}</tool_result>"
                })
            except Exception as e:
                self.conversation.append({
                    "role": "tool",
                    "content": f"<tool_error>{str(e)}</tool_error>"
                })

        # --- REWARD COMPUTATION ---
        total_reward = 0.0
        metrics = {"turn": self.turn_count}
        done = self.turn_count >= self.max_turns

        if done:
            sample = {
                "messages": self.conversation,
                "rewards": {},
                "variables": self.variables,
                "metadata": self.metadata,
            }

            for fn, name in zip(self.reward_fns, self.reward_names):
                value = await fn(None, sample)
                sample["rewards"][name] = value

            total_reward = sum(sample["rewards"].values()) / len(sample["rewards"])
            # Only keep total_reward in metrics for averaging
            metrics["total_reward"] = float(total_reward)

            # Store everything else in rollout_data for logging
            self.rollout_data = {
                "reward_breakdown": sample["rewards"],
                "reward_details": {f"reward/{name}": float(value) for name, value in sample["rewards"].items()},
                "prompt_id": self.metadata.get("prompt_index", 0),
                "step": self.turn_count,
                "rollout_id": self.turn_count,
                "rollout_data": {
                    "totalTokens": len(action),
                    "completion": len(action),
                    "prompt": self.prompt_tokens,
                    "truncated": False,
                    "metadata": self.metadata,
                },
                "messages": self.conversation,
            }

        observation = self.renderer.build_generation_prompt(messages=self.conversation)
        stop = self.renderer.get_stop_sequences()

        return StepResult(
            reward=total_reward,
            episode_done=done,
            next_observation=observation,
            next_stop_condition=stop,
            metrics=metrics,
        )


class TelemetryWrapper:
    """Adds telemetry validation and tracing to custom environments."""

    def __init__(self, user_env: Env, renderer: Any):
        self.user_env = user_env
        self.renderer = renderer
        self.turn_count = 0

        if not hasattr(user_env, "metadata"):
            raise AttributeError("Environment must define `self.metadata`.")
        if not getattr(user_env, "messages", None):
            raise AttributeError("Environment must define a non-empty `self.messages` list.")

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        tokenizer = getattr(self.renderer, "tokenizer", None)
        self.user_env.prompt_tokens = (
            sum(len(tokenizer.encode(m["content"])) for m in self.user_env.messages if "content" in m)
            if tokenizer else 0
        )
        return await self.user_env.initial_observation()

    async def step(self, action: Action) -> StepResult:
        self.turn_count += 1
        result = await self.user_env.step(action)
        metadata = getattr(self.user_env, "metadata", {})

        telemetry = {
            "reward_breakdown": getattr(result, "metrics", {}).get("reward_breakdown", {}),
            "prompt_id": metadata.get("prompt_index"),
            "step": self.turn_count,
            "rollout_id": self.turn_count,
            "rollout_data": {
                "totalTokens": len(action),
                "completion": len(action),
                "prompt": getattr(self.user_env, "prompt_tokens", 0),
                "truncated": False,
                "metadata": metadata,
            },
        }

        result.metrics = telemetry
        return result


# --- ENV REGISTRATION ---
def env(cls: type = None, *, name: str = None, max_turns: int = 1, use_telemetry: bool = True):
    """Decorator to register environments with telemetry support."""
    def decorator(env_class):
        if not inspect.isclass(env_class):
            raise TypeError("@env can only decorate classes")

        if use_telemetry:
            env_class._use_telemetry = True
        ENV_REGISTRY[name or env_class.__name__] = env_class
        env_class._is_env = True
        env_class._max_turns = max_turns
        return env_class

    return decorator if cls is None else decorator(cls)


def create_env(env_class_or_name: str | type, *args, **kwargs) -> Env:
    """Factory for creating registered environments with telemetry wrapping."""
    env_class = (
        ENV_REGISTRY[env_class_or_name]
        if isinstance(env_class_or_name, str)
        else env_class_or_name
    )
    env_instance = env_class(*args, **kwargs)
    if getattr(env_class, "_use_telemetry", False):
        renderer = kwargs.get("renderer") or (args[1] if len(args) > 1 else None)
        if renderer:
            env_instance = TelemetryWrapper(env_instance, renderer)
    return env_instance




__all__ = ["ReinforceNowEnv", "TelemetryWrapper", "env", "ENV_REGISTRY", "create_env"]
