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
import logging
import os
import threading
from types import SimpleNamespace  # noqa: F401  (re-exported for test/consumer parity)
from typing import Any

logger = logging.getLogger(__name__)

CLAUDE_SDK_MARKER_BASE_URL = "claude-sdk://local"
_DEFAULT_MODEL = "claude-haiku-4-5"  # matches claude-cli default; override via config


def _resolve_home_dir() -> str:
    """Return a stable HOME for the SDK-spawned ``claude`` CLI process.

    Billing-critical: reuse ``claude_cli_client``'s implementation so both lanes
    resolve HOME identically (subscription-OAuth credential lookup depends on it).
    """

    from agent.claude_cli_client import _resolve_home_dir as _home

    return _home()


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
        raise NotImplementedError("claude-sdk turn not yet implemented")

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
