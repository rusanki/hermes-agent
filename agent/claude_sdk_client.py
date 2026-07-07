"""claude-sdk provider — drives the Claude Agent SDK (Python) as the inference
+ native-tool-use engine behind Hermes' OpenAI-client facade.

Confirmed SDK API surface (Task 1, claude-agent-sdk==0.2.110):

* Version pinned as optional extra ``claude-sdk``; ``provider.claude_sdk`` in
  ``tools/lazy_deps.py``.
* ``ClaudeAgentOptions`` HAS a ``model`` field (str|None) and ``permission_mode``
  (Literal includes 'default','acceptEdits','plan','bypassPermissions','dontAsk',
  'auto'). All needed option fields exist: ``system_prompt``, ``mcp_servers``,
  ``allowed_tools``, ``disallowed_tools``, ``can_use_tool`` (MUST be async),
  ``resume``, ``max_turns``, ``cli_path``, ``env``.
* Blocks: ``TextBlock.text``; ``ThinkingBlock.thinking`` / ``.signature``;
  ``ToolUseBlock.id`` / ``.name`` / ``.input``; ``AssistantMessage.content``
  (list of blocks) + ``.usage`` (dict|None); ``ResultMessage.usage``
  (dict[str,Any]|None) + ``.result`` / ``.is_error`` / ``.total_cost_usd``.
* CRITICAL: ``usage`` is an UNTYPED dict; the token keys ``input_tokens`` /
  ``output_tokens`` / ``cache_read_input_tokens`` are a runtime CLI contract, NOT
  a typed schema — later tasks MUST read them defensively via ``.get(key, 0)``.
  (Not needed in THIS task, but recorded here for Tasks 4-8.)
* ``tool(name, description, input_schema, annotations=None)`` -> decorator over
  ``async def handler(args: dict) -> dict``.
* ``create_sdk_mcp_server(name, version='1.0.0', tools=None)`` ->
  ``McpSdkServerConfig``.

Sibling of ``agent/claude_cli_client.py``; that file is untouched (rollback lane).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import threading
from collections import OrderedDict
from types import SimpleNamespace  # noqa: F401  (re-exported for test/consumer parity)
from typing import Any

logger = logging.getLogger(__name__)

CLAUDE_SDK_MARKER_BASE_URL = "claude-sdk://local"
_DEFAULT_MODEL = "claude-haiku-4-5"  # matches claude-cli default; override via config


def _normalize_model(m: str | None) -> str:
    """Strip a provider prefix from a model id, defaulting when empty.

    Mirrors :func:`agent.claude_cli_client._normalize_model` (accepting the
    ``claude-sdk/`` prefix this lane may carry, plus ``anthropic/``). ``None``/
    empty falls back to :data:`_DEFAULT_MODEL`. IMPORTANT: this default is a
    LAST-RESORT only — the real model must arrive via the ``model`` kwarg so we
    never silently downgrade a caller's opus/sonnet request to haiku.
    """

    if not m or not str(m).strip():
        return _DEFAULT_MODEL
    name = str(m).strip()
    for prefix in ("claude-sdk/", "anthropic/"):
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
            break
    return name or _DEFAULT_MODEL


def _resolve_home_dir() -> str:
    """Return a stable HOME for the SDK-spawned ``claude`` CLI process.

    Billing-critical: reuse ``claude_cli_client``'s implementation so both lanes
    resolve HOME identically (subscription-OAuth credential lookup depends on it).
    """

    from agent.claude_cli_client import _resolve_home_dir as _home

    return _home()


def _safe_content(content: Any) -> str:
    """Render OpenAI message content to a string, tolerating multimodal blocks.

    Reuses claude-cli's renderer so a list/dict content (vision parts) joins its
    text pieces instead of crashing a naive ``str``-join. Never raises.
    """

    try:
        from agent.claude_cli_client import _render_message_content
        return _render_message_content(content)
    except Exception:  # pragma: no cover - defensive last resort
        return "" if content is None else str(content)


def _tool_names(tools: list | None) -> tuple[str, ...]:
    """Stable ordered tuple of tool names for seed/turn comparison."""

    names: list[str] = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str):
            names.append(name)
    return tuple(names)


def _build_sdk_env() -> dict[str, str]:
    """Environment for the SDK-spawned ``claude`` subprocess.

    SECURITY (billing invariant): scrub every var that could flip inference off
    the subscription OAuth lane, exactly as ``claude_cli_client._build_subprocess_env``
    does (which scrubs the 3 core ANTHROPIC_* vars). This adds the Bedrock/Vertex/
    model-override vars as defense-in-depth for the SDK path. Do NOT remove this
    scrub. (``ANTHROPIC_TOKEN`` is Hermes-managed OAuth for the child CLI and is
    intentionally left intact, matching ``claude_cli_client``.)
    """

    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "ANTHROPIC_MODEL",
    ):
        env.pop(k, None)
    return env


class _AwaitableResponse:
    """A resolved chat-completion response usable awaited OR directly.

    Dual-use wrapper: attribute access (primary loop, un-awaited) AND awaitable
    (async aux/vision path). Same shape as ``claude_cli_client._AwaitableResponse``
    — a bare ``SimpleNamespace`` crashes the aux ``await`` path with
    ``TypeError: object types.SimpleNamespace can't be used in 'await' expression``.

    * The primary conversation loop reads attributes off the result directly
      (no ``await``).
    * The auxiliary/async path (``auxiliary_client.async_call_llm`` /
      ``vision_analyze``) does ``response = await client.chat.completions.create(...)``.

    Uses ``__slots__`` + ``object.__setattr__`` (matching the sibling) so the
    ``__getattr__`` proxy never shadows internal state.
    """

    __slots__ = ("_resolved",)

    def __init__(self, resolved: Any):
        object.__setattr__(self, "_resolved", resolved)

    def __await__(self):
        # Already-resolved value; hand it back without yielding to the loop.
        async def _identity() -> Any:
            return self._resolved
        return _identity().__await__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolved, name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic aid only
        return f"_AwaitableResponse({self._resolved!r})"


class _ChatCompletions:
    def __init__(self, client: "ClaudeSdkClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return _AwaitableResponse(self._client._create_chat_completion(**kwargs))


class _ChatNamespace:
    def __init__(self, client: "ClaudeSdkClient"):
        self.completions = _ChatCompletions(client)


class ClaudeSdkClient:
    """Minimal OpenAI-client-compatible facade driving the Claude Agent SDK.

    Sibling of :class:`agent.claude_cli_client.ClaudeCliClient`. Accepts the SAME
    constructor kwargs so the dispatch site (Task 9) can construct either lane
    identically. This skeleton wires the facade (``chat.completions.create`` ->
    ``_AwaitableResponse``) and the env/HOME safeguards; the real SDK turn is
    filled in by Tasks 4-8 (``_create_chat_completion`` raises until then, or
    delegates to a test-only ``_run_turn_stub`` hook).
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
        self.api_key = api_key or "claude-sdk"
        self.base_url = base_url or CLAUDE_SDK_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        # SDK's ``cli_path`` option: an explicit path to the ``claude`` binary,
        # or None to let the SDK resolve it. Task 9 passes ``command`` here.
        self._cli_path = command or None
        self._args = list(args) if args else []
        self._default_model = model
        self.chat = _ChatNamespace(self)
        self.is_closed = False
        # Test-only injection point for the SDK turn (Tasks 4-8 replace the real
        # path); when None, ``_create_chat_completion`` raises NotImplementedError.
        self._run_turn_stub = None

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        if self._run_turn_stub is not None:
            return self._run_turn_stub(**kwargs)

        # --- FRESH context snapshot per create() (Task 5 RBAC requirement). ---
        ctx = _capture_ctx()
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools") or []
        model = kwargs.get("model") or self._default_model
        eff_model = _normalize_model(model)

        # Extract the system text (applied via ClaudeAgentOptions, NOT as a turn)
        # and the non-system conversation. Guard against multimodal (list/dict)
        # content — _render_message_content joins the text parts, never crashes.
        system_text = "\n\n".join(
            _safe_content(m.get("content"))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "system"
        ).strip()

        turn_ids = {
            k: kwargs.get(k, "")
            for k in ("task_id", "tool_call_id", "session_id", "turn_id", "api_request_id")
        }

        key = _session_key_for(kwargs)
        session = _get_or_make_session(key)
        decision = session.reconcile(messages)

        if decision.reseed:
            logger.info("claude-sdk: session reseed (%s)", decision.reseed_reason)
            if session.sdk_client is not None:
                _disconnect_client_on_bridge(session.sdk_client)
            session.sdk_client = None
            session.session_id = ""
            session.shadow = []
            new_prompt = _render_history_for_reseed(
                _strip_system(messages), eff_model, tools)
            sent_user_msgs = _strip_system(messages)
        else:
            new_prompt = "\n\n".join(
                _safe_content(m.get("content"))
                for m in decision.new_messages
                if isinstance(m, dict)
            ).strip()
            sent_user_msgs = decision.new_messages

        text, usage = _get_bridge().run(
            self._run_sdk_turn(
                session, new_prompt, eff_model, tools, ctx, turn_ids, system_text),
            timeout=getattr(self, "_turn_timeout", None),
        )

        # Keep the shadow in lock-step with what the live SDK session now knows:
        # the user message(s) we sent + the assistant final text (Task 6 strategy).
        session.shadow.extend(sent_user_msgs)
        session.shadow.append({"role": "assistant", "content": text})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        usage_ns = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        )
        assistant_message = SimpleNamespace(
            content=text,
            tool_calls=None,  # claude-sdk: SDK ran the full inner loop -> always None
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        # finish_reason is ALWAYS "stop": the SDK produced a single-shot final
        # answer, so Hermes' outer loop does not iterate (Approach B).
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage_ns, model=eff_model)

    async def _run_sdk_turn(
        self, session, new_prompt, eff_model, tools, ctx, turn_ids, system_text,
    ):
        """Run one full agentic SDK turn on the bridge loop; return (text, usage).

        Builds the persistent ``ClaudeSDKClient`` once per (re)seed with the
        current tools wired through the Hermes RBAC MCP proxy, then queries and
        drains ``receive_response()`` (which terminates after the ResultMessage).
        Accumulates TextBlock output and the ResultMessage usage dict.
        """

        if session.sdk_client is None:
            opts = ClaudeAgentOptions(
                system_prompt=system_text or None,
                mcp_servers={
                    HERMES_MCP_SERVER_NAME: _build_mcp_server(
                        tools, capture_ctx=ctx, turn_ids=turn_ids),
                },
                allowed_tools=["mcp__hermes__*"],
                model=eff_model or None,
                cli_path=self._cli_path,
                env=_build_sdk_env(),
                max_turns=getattr(self, "_max_turns", None),
            )
            client = ClaudeSDKClient(options=opts)
            await client.connect()
            session.sdk_client = client
            session._seed_tool_names = _tool_names(tools)
        else:
            # Options (system_prompt/tools) are fixed at construction; a tool-set
            # change mid-session is an edge case v1 ignores (log it if detected).
            current = _tool_names(tools)
            if current != getattr(session, "_seed_tool_names", current):
                logger.warning(
                    "claude-sdk: tool-set changed within a live session; "
                    "reusing the seed tools (v1 limitation)")

        await session.sdk_client.query(new_prompt)

        final_text_parts: list[str] = []
        usage_dict: dict = {}
        saw_result = False
        async for msg in session.sdk_client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        final_text_parts.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        logger.debug("claude-sdk thinking: %.120s", block.thinking)
                session.session_id = getattr(msg, "session_id", session.session_id)
            elif isinstance(msg, ResultMessage):
                usage_dict = msg.usage or {}
                session.session_id = getattr(msg, "session_id", session.session_id)
                saw_result = True
                break  # receive_response() ends after ResultMessage anyway

        if not saw_result:
            raise _SdkNoResultError("stream ended without ResultMessage")

        prompt_tokens = usage_dict.get("input_tokens", 0)
        completion_tokens = usage_dict.get("output_tokens", 0)
        cached_tokens = usage_dict.get("cache_read_input_tokens", 0)
        return (
            "".join(final_text_parts),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cached_tokens": cached_tokens,
            },
        )

    def close(self) -> None:
        self.is_closed = True


class _BridgeLoop:
    """One daemon-thread asyncio event loop shared by all SDK clients.

    Mirrors the daemon-loop pattern in tools/mcp_tool.py. Sync callers submit
    coroutines via run(); the loop lives for the process lifetime.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_forever, name="claude-sdk-bridge", daemon=True)
        self._thread.start()
        self._closed = False

    def _run_forever(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: float | None = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


_BRIDGE: _BridgeLoop | None = None
_BRIDGE_LOCK = threading.Lock()


def _get_bridge() -> _BridgeLoop:
    """Process-wide singleton bridge loop."""
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = _BridgeLoop()
        return _BRIDGE


class _SdkNoResultError(RuntimeError):
    """The SDK stream ended without a ``ResultMessage``.

    Defined here so :meth:`ClaudeSdkClient._run_sdk_turn` can raise it; Task 8
    extends error handling (retry/interrupt/surfacing) around this signal.
    """


# ---------------------------------------------------------------------------
# Task 6: history reconciliation (pure logic — no SDK calls, threads, or RBAC).
#
# The SDK owns its own conversation server-side, so on each Hermes ``create()``
# we must NOT re-send the whole transcript (that would double history and blow
# the context window). Instead we send only the new user message(s). But if
# Hermes rewrote/compressed/restarted history (a real event), the live SDK
# session is out of sync and we must reseed a FRESH session from the rendered
# history. ``reconcile`` is the state machine that distinguishes those cases:
#   * false "reseed" on a normal turn  -> throws away the live session (costly)
#   * false "no reseed" on a rewrite   -> corrupts the conversation
# Both guards matter; both have dedicated tests.
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class _ReconcileDecision:
    reseed: bool
    new_messages: list = field(default_factory=list)
    reseed_reason: str = ""


def _strip_system(messages: list) -> list:
    return [m for m in (messages or []) if m.get("role") != "system"]


@dataclass
class _SdkSession:
    session_id: str
    shadow: list = field(default_factory=list)   # non-system OpenAI messages fed so far
    sdk_client: object | None = None             # live ClaudeSDKClient (Task 7)

    # SHADOW-UPDATE STRATEGY (for Task 7 — do NOT "fix" reconcile to compensate):
    # The shadow list mirrors BOTH roles the live SDK session already knows about
    # (system prompts are stripped — they are applied via ``ClaudeAgentOptions``,
    # not as conversation turns). After each successful turn, Task 7's caller MUST
    # append to ``shadow`` the sent user message(s) AND the assistant final text as
    # ``{"role": "assistant", "content": <text>}``. That is why the normal-turn
    # test pre-seeds shadow with ``[u1, a1]``: the next incoming's ``[u1, a1]``
    # prefix then matches, and only the trailing new user message is detected as
    # ``new_messages``. Keeping the shadow in lock-step with what the SDK has seen
    # is what makes the cheap prefix check correct — the reconcile logic stays
    # pure and must not be patched to paper over a shadow-update omission.

    def reconcile(self, incoming: list) -> _ReconcileDecision:
        """Compare incoming (system stripped) against the shadow list.

        Match  = shadow is a prefix of incoming_nonsys AND the only new trailing
                 entries are appendable (the previous assistant text the SDK
                 already produced + the new user message). We send only the
                 messages after the shadow prefix that are NEW user turns.
        Mismatch = shadow is NOT a prefix (compression/rewrite/restart/switch) =>
                 reseed from rendered history.
        """
        incoming_nonsys = _strip_system(incoming)

        # Empty-shadow guard (fresh process / new session object). A naive prefix
        # check would treat ``shadow == incoming_nonsys[:0]`` as ``[] == []`` -> a
        # spurious match. But an empty shadow paired with pre-existing history
        # (any assistant turn, or more than one message) means the incoming carries
        # a transcript this fresh SDK session has NEVER seen -> we MUST reseed. Only
        # a brand-new conversation (exactly one user message, no assistant history)
        # is a legitimate first turn that appends without reseeding.
        if not self.shadow:
            has_prior_history = (
                len(incoming_nonsys) > 1
                or any(m.get("role") == "assistant" for m in incoming_nonsys)
            )
            if has_prior_history:
                return _ReconcileDecision(reseed=True, reseed_reason="restart-no-shadow")
            # Fresh first turn: fall through to the append path below, which
            # returns the single new user message as new_messages.

        n = len(self.shadow)
        if self.shadow != incoming_nonsys[:n]:
            reason = ("restart-no-shadow" if not self.shadow
                      else "history-rewritten")
            return _ReconcileDecision(reseed=True, reseed_reason=reason)
        # Prefix matches. The tail beyond the shadow is what the SDK hasn't seen.
        tail = incoming_nonsys[n:]
        # The SDK session already contains the assistant reply it produced, so only
        # NEW user messages in the tail need sending.
        new_user = [m for m in tail if m.get("role") == "user"]
        return _ReconcileDecision(reseed=False, new_messages=new_user)


def _render_history_for_reseed(messages: list, model: str, tools: list) -> str:
    """Render prior history as one synthetic first user message on a fresh SDK
    session. Reuse claude-cli's renderer so there's no new rendering code."""
    from agent.claude_cli_client import _format_messages_as_prompt
    return _format_messages_as_prompt(messages, model, tools, None)


# ---------------------------------------------------------------------------
# Task 7: per-session store (LRU) of live SDK sessions.
#
# A Hermes ``create()`` maps to a session key; each key owns one persistent
# ``_SdkSession`` (holding the live ``ClaudeSDKClient`` + shadow history). We
# cap the number of live sessions and evict the oldest LRU-style, disconnecting
# its subprocess best-effort. An evicted key that returns later simply rebuilds
# a fresh session (empty shadow) and reconcile reseeds from rendered history —
# correct-but-cold; we deliberately do NOT keep resume state for v1.
# ---------------------------------------------------------------------------
_SESSION_CAP = int(os.getenv("HERMES_CLAUDE_SDK_SESSION_CAP", "8") or "8")
_SESSIONS: "OrderedDict[str, _SdkSession]" = OrderedDict()
_SESSIONS_LOCK = threading.Lock()


def _session_key_for(kwargs: dict) -> str:
    """Derive a stable session key from create() kwargs.

    Prefer an explicit ``session_key``; fall back to ``user`` (OpenAI's stable
    per-user field, which Hermes populates with the verified user id), else a
    single ``"default"`` bucket. reconcile() keeps history correct regardless of
    how coarse the key is, so a single stable value is sufficient for v1.
    """

    return kwargs.get("session_key") or kwargs.get("user") or "default"


def _disconnect_client_on_bridge(client: object) -> None:
    """Best-effort ``client.disconnect()`` scheduled on the bridge loop.

    Called from the LRU-eviction path (and monkeypatched by tests to spy). Never
    raises: eviction must not fail because a dead/half-open subprocess won't
    disconnect, and the bridge may not even be running yet.
    """

    if client is None:
        return
    try:
        _get_bridge().run(client.disconnect(), timeout=5)
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.debug("claude-sdk: disconnect on evict failed", exc_info=True)


def _get_or_make_session(key: str) -> "_SdkSession":
    """LRU get-or-create for the session keyed by ``key``.

    Moves an existing key to the most-recent end. On insertion beyond
    :data:`_SESSION_CAP`, evicts the oldest entry and disconnects its live SDK
    client best-effort. The evicted ``_SdkSession`` (its session_id + shadow) is
    discarded; a later request with that key rebuilds cold via reseed.
    """

    with _SESSIONS_LOCK:
        existing = _SESSIONS.get(key)
        if existing is not None:
            _SESSIONS.move_to_end(key)
            return existing
        session = _SdkSession(session_id="")
        _SESSIONS[key] = session
        _SESSIONS.move_to_end(key)
        while len(_SESSIONS) > _SESSION_CAP:
            _evicted_key, evicted = _SESSIONS.popitem(last=False)
            if getattr(evicted, "sdk_client", None) is not None:
                _disconnect_client_on_bridge(evicted.sdk_client)
                evicted.sdk_client = None
        return session


# ---------------------------------------------------------------------------
# Task 5 — SECURITY-CRITICAL: in-process MCP tool proxy.
#
# Every native tool the model invokes on the SDK lane is routed through Hermes'
# single-fire tool pipeline (``handle_function_call``), which runs RBAC, audit,
# and redaction. We expose each OpenAI-shaped tool as an in-process MCP tool
# named ``mcp__hermes__<original_name>``; the SDK CLI calls it, the proxy
# handler dispatches into ``handle_function_call`` under a captured contextvars
# snapshot so the RBAC hook sees the verified user_id on the bridge thread.
#
# ``handle_function_call`` is imported at module scope (not by closure) so
# tests can ``monkeypatch.setattr(module, "handle_function_call", ...)`` — the
# proxy handler calls the bare module-global name, which is resolved through
# this module's namespace at call time.
# ---------------------------------------------------------------------------
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    create_sdk_mcp_server,
)
from claude_agent_sdk import tool as sdk_tool  # noqa: E402

from model_tools import handle_function_call  # noqa: E402  the single-fire hook entry

HERMES_MCP_SERVER_NAME = "hermes"


def _capture_ctx() -> contextvars.Context:
    """Snapshot the current contextvars (incl. gateway ``_SESSION_USER_ID``) at
    create() entry, so proxy handlers running on the bridge thread see the
    verified user_id via ``_get_user_id_for_hooks()``."""
    return contextvars.copy_context()


def _extract_text_and_error(raw: str) -> tuple[str, bool]:
    """``handle_function_call`` returns a JSON string. A block/error is
    ``{"error": <msg>}`` (a denial). Anything else is a normal result;
    return it verbatim as text, ``is_error=False``."""
    if not isinstance(raw, str):
        return str(raw), False
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return raw, False  # non-JSON string result -> pass through
    if isinstance(obj, dict) and "error" in obj:
        return str(obj["error"]), True
    return raw, False


def _to_mcp_result(raw: str) -> dict:
    """Translate ``handle_function_call``'s JSON-STRING result into the SDK MCP
    result shape. A hook block ``{"error": msg}`` -> ``is_error=True``."""
    text, is_error = _extract_text_and_error(raw)
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def _make_proxy_handler(original_name, *, capture_ctx, turn_ids=None):
    """Build an async MCP tool handler that dispatches into Hermes' executor
    pipeline under the captured context. RBAC/audit/redaction all run inside
    ``handle_function_call`` (``skip_pre_tool_call_hook=False`` => hook fires
    once)."""
    turn_ids = turn_ids or {}

    async def _handler(args: dict) -> dict:
        def _dispatch():
            # Module-global lookup so monkeypatch.setattr can override it.
            return handle_function_call(
                original_name, args,
                skip_pre_tool_call_hook=False,  # RBAC hook fires here
                task_id=turn_ids.get("task_id", ""),
                tool_call_id=turn_ids.get("tool_call_id", ""),
                session_id=turn_ids.get("session_id", ""),
                turn_id=turn_ids.get("turn_id", ""),
                api_request_id=turn_ids.get("api_request_id", ""),
            )

        # Run the (sync) Hermes dispatch under the captured contextvars so the
        # RBAC hook sees the verified user_id on the bridge thread.
        raw = capture_ctx.run(_dispatch)
        return _to_mcp_result(raw)

    return _handler


def _build_mcp_server(openai_tools, *, capture_ctx, turn_ids):
    """Wrap each OpenAI function tool as an in-process MCP tool named
    ``mcp__hermes__<original_name>``."""
    sdk_tools = []
    for t in (openai_tools or []):
        fn = t.get("function", t)
        name = fn["name"]
        desc = fn.get("description", "")
        schema = fn.get("parameters", {"type": "object", "properties": {}})
        handler = _make_proxy_handler(name, capture_ctx=capture_ctx, turn_ids=turn_ids)
        sdk_tools.append(sdk_tool(name, desc, schema)(handler))
    return create_sdk_mcp_server(name=HERMES_MCP_SERVER_NAME, tools=sdk_tools)
