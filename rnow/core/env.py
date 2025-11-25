from __future__ import annotations

import inspect
import json
import re
from string import Template
from typing import Any, Callable

from rnow.models import Env, StopCondition, Action, StepResult, Observation


# Global registry for environment classes
ENV_REGISTRY: dict[str, type] = {}

# Global trace logger callback
TRACE_LOGGER: Callable[[dict], None] | None = None

# Regex for parsing tool calls - captures everything between tags (handles nested JSON)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def set_trace_logger(logger: Callable[[dict], None] | None) -> None:
    """Set the global trace logger callback."""
    global TRACE_LOGGER
    TRACE_LOGGER = logger


def _build_tools_block(tool_registry: dict[str, Callable]) -> str:
    """Build the <tools> XML block from registered tool functions."""
    if not tool_registry:
        return ""

    tools_json = []
    for name, fn in tool_registry.items():
        schema = getattr(fn, "_schema", {"type": "object", "properties": {}})
        description = getattr(fn, "_description", "No description available.")
        tools_json.append({
            "name": name,
            "description": description,
            "parameters": schema,
        })

    tools_block = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_list}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<value>"}}}}
</tool_call>
""".format(tools_list=json.dumps(tools_json, indent=2))

    return tools_block


class ReinforceNowEnv(Env):
    """Environment for both single-turn and multi-turn RL training with reward and tool registry support."""

    def __init__(
        self,
        data: dict,
        renderer: Any,
        reward_registry: dict[str, Callable],
        tool_registry: dict[str, Callable] | None = None,
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

        # Inject tools block into system prompt if tools are registered
        if self.tool_registry:
            tools_block = _build_tools_block(self.tool_registry)
            self._inject_tools_into_system_prompt(tools_block)

    def _inject_tools_into_system_prompt(self, tools_block: str) -> None:
        """Inject tools block into the system message, or create one if missing."""
        for msg in self.messages:
            if msg["role"] == "system":
                # Prepend tools block to existing system message
                msg["content"] = tools_block + "\n\n" + msg["content"]
                return

        # No system message found, insert one at the beginning
        self.messages.insert(0, {"role": "system", "content": tools_block})

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
        # Use renderer.parse_response to properly handle stop tokens like <|im_end|>
        message, _ = self.renderer.parse_response(action)
        response = message["content"]
        self.conversation.append({"role": "assistant", "content": response})

        # Tool Calling - handle multiple tool calls
        tool_matches = TOOL_CALL_RE.findall(response)
        tools_called = len(tool_matches) > 0

        for raw_call in tool_matches:
            if not self.tool_registry:
                break
            try:
                tool_data = json.loads(raw_call)
                tool_name = tool_data.get("name")
                args = tool_data.get("arguments", {})

                if tool_name not in self.tool_registry:
                    self.conversation.append({
                        "role": "tool",
                        "content": f"<tool_error>Tool '{tool_name}' not found in registry</tool_error>"
                    })
                    continue

                tool_fn = self.tool_registry[tool_name]
                # Call tool with unpacked arguments (tools use typed params, not args dict)
                tool_result = (
                    await tool_fn(**args) if inspect.iscoroutinefunction(tool_fn) else tool_fn(**args)
                )

                self.conversation.append({
                    "role": "tool",
                    "content": f"<tool_result>{json.dumps(tool_result)}</tool_result>"
                })
            except json.JSONDecodeError as e:
                self.conversation.append({
                    "role": "tool",
                    "content": f"<tool_error>Invalid JSON in tool call: {str(e)}</tool_error>"
                })
            except Exception as e:
                self.conversation.append({
                    "role": "tool",
                    "content": f"<tool_error>{str(e)}</tool_error>"
                })

        # --- REWARD COMPUTATION ---
        total_reward = 0.0
        metrics = {"turn": self.turn_count, "tools_called": tools_called}
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

            # Add individual reward metrics (numeric only) for averaging
            for name, value in sample["rewards"].items():
                metrics[f"reward/{name}"] = float(value)

            # Store trace data on the environment instance for external access
            completion_tokens = len(action)
            total_tokens = self.prompt_tokens + completion_tokens

            # Create metadata without iteration and batch (redundant with step)
            clean_metadata = {k: v for k, v in self.metadata.items() if k not in ["iteration", "batch"]}

            self.rollout_data = {
                "reward": total_reward,
                "reward_breakdown": sample["rewards"],
                "prompt_id": self.metadata.get("prompt_index", 0),
                "turn": self.turn_count,  # Turn within the episode
                "rollout_id": self.metadata.get("rollout_id", self.metadata.get("env_id", self.turn_count)),
                "total_tokens": total_tokens,
                "completion": completion_tokens,
                "promptTokens": self.prompt_tokens,
                "messages": self.conversation,
                "rollout_data": {
                    "totalTokens": total_tokens,
                    "completion": completion_tokens,
                    "promptTokens": self.prompt_tokens,
                    "truncated": False,
                    "metadata": clean_metadata,
                },
            }

            # Fire-and-forget trace logging
            if TRACE_LOGGER is not None:
                try:
                    TRACE_LOGGER(self.rollout_data)
                except Exception as e:
                    # Don't kill training if trace logging fails
                    import sys
                    print(f"[ReinforceNowEnv] Failed to log trace: {e}", file=sys.stderr)

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
        # Pass through the result as-is, no telemetry modification needed
        # Trace logging is now handled via the global callback in ReinforceNowEnv
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




__all__ = ["ReinforceNowEnv", "TelemetryWrapper", "env", "ENV_REGISTRY", "create_env", "set_trace_logger"]
