"""
Token counting utilities for accurate context window validation.

Supports:
- gpt-oss: Uses openai-harmony render_conversation_for_completion (canonical API)
- Qwen/other HF models: Uses HuggingFace tokenizers with chat template approximation
- Fallback: Conservative char-based estimate
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

# Tokenizer cache to avoid reloading
_tokenizer_cache: dict[str, Any] = {}


def get_tokenizer_for_model(model_path: str) -> tuple[str, Any] | None:
    """
    Get the appropriate tokenizer for a model.

    Returns:
        Tuple of (tokenizer_type, tokenizer) or None if unavailable.
        tokenizer_type is "harmony" for gpt-oss, "hf" for HuggingFace models.
    """
    if model_path in _tokenizer_cache:
        return _tokenizer_cache[model_path]

    # gpt-oss models use openai-harmony
    if "gpt-oss" in model_path.lower():
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding

            tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            _tokenizer_cache[model_path] = ("harmony", tokenizer)
            return _tokenizer_cache[model_path]
        except ImportError:
            pass
    else:
        # Try HuggingFace tokenizers (requires tokenizer.json in repo)
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_pretrained(model_path)
            _tokenizer_cache[model_path] = ("hf", tokenizer)
            return _tokenizer_cache[model_path]
        except Exception:
            # tokenizer.json not available or repo requires auth
            pass

    _tokenizer_cache[model_path] = None
    return None


def count_tokens_for_gpt_oss(
    messages: list[dict], tools: list[dict], instructions: str = ""
) -> int:
    """
    Count tokens for gpt-oss using the canonical Harmony API.

    This uses render_conversation_for_completion which is the correct way
    to count tokens exactly as gpt-oss runtime sees them.

    System messages from the dataset are merged into DeveloperContent.instructions,
    since Harmony's SystemContent is for model metadata, not user system prompts.
    """
    try:
        from openai_harmony import (
            Conversation,
            DeveloperContent,
            HarmonyEncodingName,
            Message,
            Role,
            SystemContent,
            ToolDescription,
            load_harmony_encoding,
        )
    except ImportError:
        # Conservative fallback: count JSON chars as tokens
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) + sum(
            len(json.dumps(t, ensure_ascii=False)) for t in tools
        )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Extract system messages from dataset - these become developer instructions
    # (Harmony's SystemContent is model metadata, not user system prompts)
    sys_texts = [
        m.get("content", "")
        for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    sys_text = "\n\n".join(t.strip() for t in sys_texts if t.strip())

    # Combine with any additional instructions
    combined_instructions = "\n\n".join(s for s in [instructions.strip(), sys_text] if s).strip()

    # Build tool descriptions using canonical API
    tool_descs = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        try:
            td = ToolDescription.new(
                t.get("name", "unknown"),
                t.get("description", "") or "",
                parameters=t.get("schema") or {},
            )
            tool_descs.append(td)
        except Exception:
            # Skip malformed tools
            pass

    # Build DeveloperContent with instructions and tools
    dev = DeveloperContent.new()
    if combined_instructions:
        dev = dev.with_instructions(combined_instructions)
    if tool_descs:
        dev = dev.with_function_tools(tool_descs)

    # Build conversation using canonical from_role_and_content
    convo_msgs = [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(Role.DEVELOPER, dev),
    ]

    # Add non-system messages
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system":
            continue  # Already handled in DeveloperContent

        content = m.get("content", "") or ""

        if role == "user":
            convo_msgs.append(Message.from_role_and_content(Role.USER, content))
        elif role == "assistant":
            convo_msgs.append(Message.from_role_and_content(Role.ASSISTANT, content))

    convo = Conversation.from_messages(convo_msgs)
    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    return len(tokens)


def count_tokens_for_hf_model(messages: list[dict], tools: list[dict], model_path: str) -> int:
    """
    Count tokens for HuggingFace models using tokenizers library.

    Uses ChatML-style template as approximation since we don't have
    the exact chat template. This may slightly over-count which is safer
    than under-counting.
    """
    tokenizer_info = get_tokenizer_for_model(model_path)

    # Build prompt with ChatML-style template (common for Qwen, etc.)
    # This is an approximation but errs on the side of over-counting
    prompt_parts = []

    # Add tools as system content
    if tools:
        tool_lines = ["You have access to the following tools:\n"]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            schema = tool.get("schema", {})
            tool_lines.append(f"### {name}")
            if desc:
                tool_lines.append(f"{desc}")
            if schema:
                tool_lines.append(f"Parameters: {json.dumps(schema)}")
            tool_lines.append("")
        prompt_parts.append(f"<|im_start|>system\n{''.join(tool_lines)}<|im_end|>")

    # Add messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    full_prompt = "\n".join(prompt_parts)

    if tokenizer_info is None:
        # Conservative estimate: 1 char = 1 token (over-estimates)
        return len(full_prompt)

    tokenizer_type, tokenizer = tokenizer_info
    try:
        if tokenizer_type == "hf":
            encoded = tokenizer.encode(full_prompt)
            return len(encoded.ids)
        else:
            # Shouldn't happen, but handle gracefully
            return len(full_prompt)
    except Exception:
        return len(full_prompt)


def count_tokens_for_prompt(messages: list[dict], tools: list[dict], model_path: str) -> int:
    """
    Count tokens for a prompt with the correct format for the model.

    For gpt-oss: Uses Harmony render_conversation_for_completion (exact)
    For HF models: Uses tokenizer with chat template approximation (conservative)
    Fallback: Char-based estimate (very conservative)
    """
    is_gpt_oss = "gpt-oss" in model_path.lower()

    if is_gpt_oss:
        return count_tokens_for_gpt_oss(messages, tools)

    return count_tokens_for_hf_model(messages, tools, model_path)


def get_max_prompt_tokens(
    path: Path,
    tools: list[dict],
    model_path: str = "",
    sample_size: int = 0,
    top_k: int = 20,
) -> int:
    """
    Scan train.jsonl and return the maximum prompt token count.

    Uses top-K strategy: keeps top K candidates by char count, then
    token-counts all of them to find the true maximum. This handles
    cases where char count doesn't correlate with token count
    (e.g., CJK text, code, JSON).

    Args:
        path: Path to train.jsonl
        tools: List of tool definitions to include in token count
        model_path: Model path for tokenizer selection
        sample_size: Max lines to scan (0 = all)
        top_k: Number of candidates to keep for token counting
    """
    # Min-heap of (char_count, index, messages) - keeps smallest at top
    # Index is needed to make tuples comparable when char_counts are equal
    heap: list[tuple[int, int, list[dict]]] = []

    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msgs = rec.get("messages")
                if not isinstance(msgs, list):
                    continue

                # Calculate char count for this example
                char_count = sum(
                    len(m.get("content", ""))
                    for m in msgs
                    if isinstance(m, dict) and isinstance(m.get("content", ""), str)
                )

                # Keep top_k largest by char count
                if len(heap) < top_k:
                    heapq.heappush(heap, (char_count, i, msgs))
                elif char_count > heap[0][0]:
                    heapq.heappushpop(heap, (char_count, i, msgs))

    except Exception:
        return 0

    if not heap:
        return 0

    # Token-count all candidates and return the maximum
    max_tokens = 0
    for _chars, _idx, msgs in heap:
        tokens = count_tokens_for_prompt(msgs, tools, model_path)
        max_tokens = max(max_tokens, tokens)

    return max_tokens


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Conservative token estimate from character count.
    Uses 2 chars per token which typically over-estimates.
    """
    return char_count // 2
