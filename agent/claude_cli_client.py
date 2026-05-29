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
import os
import re
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_DEFAULT_MODEL = "claude-opus-4-8"

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


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ``claude`` CLI processes."""

    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    """Build the environment for the child ``claude`` CLI process.

    SECURITY: ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` and
    ``ANTHROPIC_BASE_URL`` are all scrubbed so the CLI can NEVER fall back to a
    metered billing path. The provider exists to drive subscription OAuth
    (Claude Code print mode). ``ANTHROPIC_API_KEY`` is the obvious API-key path;
    ``ANTHROPIC_AUTH_TOKEN``/``ANTHROPIC_BASE_URL`` are the documented
    *alternate*/custom-gateway auth path — leaving any of them in the
    environment would silently route inference through a metered or custom
    endpoint, undercutting the subscription-OAuth-only billing guarantee. This
    scrub is the core billing safeguard for the provider — do not remove it.
    (Note: ``ANTHROPIC_TOKEN`` is Hermes-managed OAuth, unrelated to the child
    CLI's own auth, and is intentionally left intact.)
    """

    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    env.pop("ANTHROPIC_BASE_URL", None)
    return env


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip()
        or os.getenv("CLAUDE_CLI_PATH", "").strip()
        or "claude"
    )


def _normalize_model(m: str | None) -> str:
    """Strip a provider prefix from a model id, defaulting when empty.

    Accepts ``claude-cli/<id>`` and ``anthropic/<id>`` forms (the prefixes
    Hermes may attach when routing to this provider) and returns the bare CLI
    model id. ``None``/empty falls back to :data:`_DEFAULT_MODEL`.
    """

    if not m or not str(m).strip():
        return _DEFAULT_MODEL
    name = str(m).strip()
    for prefix in ("claude-cli/", "anthropic/"):
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
            break
    return name or _DEFAULT_MODEL


TOOL_MARKUP_INSTRUCTION = (
    "You are being used as a pure inference engine. You have NO tools of your "
    "own and cannot run commands, read files, or take actions directly.\n"
    "When an action IS needed, you MUST request it by emitting a tool call as a "
    "single line of markup and nothing else for that call:\n"
    '<tool_call>{"name": "<tool_name>", "arguments": "<json-string-of-args>"}</tool_call>\n'
    "Rules:\n"
    "- Emit exactly one JSON object per <tool_call> block; \"arguments\" must be "
    "a JSON string (a string whose contents are themselves valid JSON).\n"
    "- Emit one <tool_call> block per tool call; do not wrap multiple calls in "
    "one block.\n"
    "- Do NOT apologize for lacking tools and do NOT claim you cannot perform an "
    "action — instead emit the appropriate <tool_call> and let Hermes execute "
    "it.\n"
    "- If no tool is needed, just answer the user normally with plain text."
)


class _ChatCompletions:
    def __init__(self, client: "ClaudeCliClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "ClaudeCliClient"):
        self.completions = _ChatCompletions(client)


class ClaudeCliClient:
    """Minimal OpenAI-client-compatible facade driving the ``claude`` CLI.

    Each ``chat.completions.create`` call renders the conversation into a single
    prompt, spawns ``claude -p --output-format stream-json`` as a short-lived
    subprocess, parses the line-delimited JSON it streams back, and converts the
    result into the minimal shape Hermes expects from an OpenAI client. Hermes
    owns the tool-execution loop; the CLI is a pure inference engine.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        model: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args) if args else []
        self._default_model = model
        self.chat = _ChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    def _build_system_prompt(self, system_text: str) -> str:
        if system_text and system_text.strip():
            return system_text + "\n\n" + TOOL_MARKUP_INSTRUCTION
        return TOOL_MARKUP_INSTRUCTION

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        system_text, rest = _split_system_message(messages or [])
        prompt = _format_messages_as_prompt(rest, model, tools, tool_choice)
        sys_prompt = self._build_system_prompt(system_text)
        eff_model = _normalize_model(model or self._default_model)

        lines = self._run_claude(prompt, sys_prompt, eff_model, timeout)
        parsed = _parse_stream_json_lines(lines)
        tool_calls, cleaned = _extract_tool_calls_from_text(parsed.text)

        prompt_tokens = parsed.usage.get("input_tokens", 0)
        completion_tokens = parsed.usage.get("output_tokens", 0)
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=parsed.usage.get("cache_read_input_tokens", 0)
            ),
        )
        assistant_message = SimpleNamespace(
            content=cleaned,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage, model=eff_model)

    def _run_claude(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        timeout: float | None,
    ) -> list[str]:
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        cmd = [
            self._command,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "text",
            "--tools",
            "",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--model",
            model,
            "--system-prompt",
            system_prompt,
        ] + self._args

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise ClaudeCliAuthError(
                "claude CLI not found; install it or set "
                "HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH"
            ) from exc

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        try:
            try:
                stdout, stderr = proc.communicate(
                    input=prompt, timeout=effective_timeout
                )
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.communicate()
                raise ClaudeCliError(
                    f"claude CLI timed out after {effective_timeout}s"
                ) from exc

            stdout = stdout or ""
            if proc.returncode != 0 and not stdout.strip():
                stderr_snippet = (stderr or "").strip()[:2000]
                raise ClaudeCliError(
                    f"claude CLI exited with code {proc.returncode}"
                    + (f": {stderr_snippet}" if stderr_snippet else "")
                )
            return stdout.splitlines()
        finally:
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None
