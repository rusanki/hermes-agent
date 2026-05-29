"""Pure helpers for the `claude-cli` model provider.

This module provides the no-subprocess building blocks for a Hermes provider
that drives the `claude` CLI (`claude -p`) as an inference backend. It is
modeled on ``agent/copilot_acp_client.py`` (which drives `copilot --acp` the
same way): Hermes owns the tool-execution loop, Claude is a pure inference
engine, and Hermes parses ``<tool_call>{...}</tool_call>`` markup out of
Claude's text output and executes the tools itself.

Only the pure helpers live here:

* :func:`_split_system_message` splits a leading ``system`` message out of the
  conversation so it can be supplied to the CLI separately (the CLI takes a
  dedicated system prompt), keeping it out of the rendered transcript.
* :func:`_format_messages_as_prompt` renders the remaining conversation into a
  single prompt string. Unlike the ACP version it does NOT embed the
  how-to-emit-tool-calls instruction prose (that lives in the system prompt in
  a later task); it keeps only the available-tool *schema*.
* :func:`_render_message_content` normalises message content to a string.
* :func:`_render_assistant_tool_calls` renders an assistant message's
  ``tool_calls`` into ``<tool_call>`` blocks.
* :func:`_render_tool_response` renders a ``role == "tool"`` message into a
  ``<tool_response>`` block.
* :func:`_extract_tool_calls_from_text` pulls OpenAI-shaped tool calls out of
  Claude's text output (XML blocks first, bare JSON as a fallback).

The subprocess client class itself is added in a later task.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _split_system_message(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split a leading ``system`` message out of the conversation.

    If ``messages`` is non-empty and its first entry has ``role == "system"``,
    return ``(rendered_system_text, messages[1:])`` where the system text is the
    message's content rendered to a string via the same logic as
    :func:`_render_message_content`. Otherwise return ``("", messages)``.
    """

    if not messages:
        return "", messages
    first = messages[0]
    if isinstance(first, dict) and str(first.get("role") or "").strip().lower() == "system":
        return _render_message_content(first.get("content")), messages[1:]
    return "", messages


def _render_assistant_tool_calls(tool_calls: Any) -> list[str]:
    """Render an assistant message's ``tool_calls`` into ``<tool_call>`` blocks.

    Mirrors ``agent/agent_runtime_helpers.py`` (~lines 120-136): if a call's
    ``function.arguments`` is a JSON string, parse it to an object first so the
    transcript carries structured arguments; fall back to the raw string on a
    parse failure.
    """

    blocks: list[str] = []
    if not isinstance(tool_calls, list):
        return blocks
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                arguments: Any = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                arguments = raw_args
        else:
            arguments = raw_args
        payload = {"name": name.strip(), "arguments": arguments}
        blocks.append(
            "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
        )
    return blocks


def _render_tool_response(message: dict[str, Any]) -> str:
    """Render a ``role == "tool"`` message into a ``<tool_response>`` block.

    Mirrors ``agent/agent_runtime_helpers.py`` (~lines 154-175): the inner JSON
    carries ``tool_call_id``/``name``/``content``, and string content that looks
    like JSON is parsed back into an object so it round-trips structurally.
    """

    tool_content: Any = message.get("content")
    if isinstance(tool_content, str):
        try:
            if tool_content.strip().startswith(("{", "[")):
                tool_content = json.loads(tool_content)
        except (json.JSONDecodeError, AttributeError):
            pass  # Keep as string if not valid JSON

    call_id = message.get("tool_call_id")
    name = message.get("name")
    payload = {
        "tool_call_id": call_id if isinstance(call_id, str) else "",
        "name": name if isinstance(name, str) else "",
        "content": tool_content,
    }
    return (
        "<tool_response>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_response>"
    )


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are the inference backend for Hermes.",
        "Continue the conversation from the latest user request.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema):\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        rendered = _render_message_content(message.get("content"))

        body_parts: list[str] = []
        if rendered:
            body_parts.append(rendered)

        if role == "assistant":
            # Render tool_calls so tool-call-only turns are not dropped. (The
            # ACP version's ``if not rendered: continue`` would wrongly skip an
            # assistant turn whose content is empty but which carries tool
            # calls.)
            body_parts.extend(_render_assistant_tool_calls(message.get("tool_calls")))
        elif role == "tool":
            if message.get("name") or message.get("tool_call_id"):
                body_parts.append(_render_tool_response(message))

        if not body_parts:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n" + "\n".join(body_parts))

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        # Accept both the OpenAI ``function``-wrapped shape and the flat
        # ``{"name", "arguments"}`` shape. The flat shape is what
        # ``_format_messages_as_prompt`` renders into ``<tool_call>`` blocks,
        # so it is the shape Claude is steered to echo; the wrapped shape keeps
        # the bare-JSON OpenAI fallback (``_TOOL_CALL_JSON_RE``) working.
        fn = obj.get("function")
        if not isinstance(fn, dict):
            fn = obj
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


class ClaudeCliError(RuntimeError):
    """Base error for failures surfaced by the ``claude`` CLI ``result`` event."""


class ClaudeCliQuotaError(ClaudeCliError):
    """The CLI reported a usage/quota exhaustion error."""


class ClaudeCliAuthError(ClaudeCliError):
    """The CLI reported an authentication/authorization error."""


# Substrings (matched case-insensitively) that classify a CLI error message.
# NOTE: these markers are heuristic — they pattern-match free-text error
# wording, so they are deliberately conservative to avoid false positives.
# In particular the bare token "401" and the bare word "login" are NOT used,
# because they match unrelated text ("processed 401k tokens", "weblogin
# failed"); the 401 forms are matched via a delimited regex (so "401k" does
# not trip it) and "login" is replaced with more specific phrasings.
_QUOTA_MARKERS = ("out of extra usage", "extra usage", "/settings/usage")
_AUTH_MARKERS = (
    "unauthorized",
    "authenticate",
    "invalid api key",
    "log in",
    "/login",
    "please login",
)
# A 401 status that appears as a delimited token after a space / "status" /
# "http" / "error" — but not as part of a larger word/number like "401k".
_AUTH_401_RE = re.compile(r"(?:\bstatus|\bhttp|\berror|\s)\s*401(?!\w)", re.IGNORECASE)


def _classify_cli_error(message: str) -> ClaudeCliError:
    """Map an error ``result`` message to the most specific error class.

    Quota wording wins over auth wording; anything unmatched is the base
    :class:`ClaudeCliError`.
    """

    lowered = (message or "").lower()
    if any(marker in lowered for marker in _QUOTA_MARKERS):
        return ClaudeCliQuotaError(message)
    if any(marker in lowered for marker in _AUTH_MARKERS) or _AUTH_401_RE.search(lowered):
        return ClaudeCliAuthError(message)
    return ClaudeCliError(message)


def _parse_stream_json_lines(lines: Any) -> SimpleNamespace:
    """Parse the line-delimited JSON event stream from ``claude -p``.

    ``claude -p --output-format stream-json --verbose`` emits one JSON object
    per stdout line. This consumes ``lines`` (an iterable of strings) and folds
    them into a :class:`types.SimpleNamespace` with attributes ``text`` (str),
    ``stop_reason`` (str | None), ``usage`` (dict), ``cost_usd`` (float) and
    ``raw_result`` (the terminal ``result`` event dict, or ``None``).

    Malformed lines are skipped (never crash). ``assistant`` events accumulate
    text and refresh the latest usage. A ``result`` event is terminal: on error
    it raises a classified :class:`ClaudeCliError`; on success it supplies the
    authoritative final text/usage/cost. If the stream ends without a ``result``
    event (e.g. truncated output), the accumulated state is returned as-is
    rather than raising.
    """

    buffer: list[str] = []
    last_usage: dict[str, Any] = {}

    for line in lines:
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue

        event_type = obj.get("type")

        if event_type == "assistant":
            message = obj.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text")
                            if isinstance(text, str):
                                buffer.append(text)
                usage = message.get("usage")
                if isinstance(usage, dict):
                    last_usage = usage
            continue

        if event_type == "result":
            if obj.get("is_error") or obj.get("api_error_status"):
                message_text = obj.get("result")
                if not isinstance(message_text, str) or not message_text:
                    message_text = "claude CLI reported an error"
                raise _classify_cli_error(message_text)

            usage = obj.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            try:
                cost_usd = float(obj.get("total_cost_usd") or 0.0)
            except (TypeError, ValueError):
                cost_usd = 0.0
            result_text = obj.get("result")
            text = result_text if isinstance(result_text, str) and result_text else "".join(buffer)
            return SimpleNamespace(
                text=text,
                stop_reason=obj.get("stop_reason"),
                usage=usage,
                cost_usd=cost_usd,
                raw_result=obj,
            )

        if event_type == "rate_limit_event":
            logger.debug("claude-cli rate_limit_event: %s", obj)
            continue

        # Unknown / "system" events are ignored.

    # No terminal result event seen — return what we accumulated.
    return SimpleNamespace(
        text="".join(buffer),
        stop_reason=None,
        usage=last_usage or {},
        cost_usd=0.0,
        raw_result=None,
    )
