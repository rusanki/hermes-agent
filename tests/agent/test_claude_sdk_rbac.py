"""Task 5 — security-critical: every claude-sdk model tool call routes through
Hermes' RBAC/audit/redaction pipeline (``handle_function_call``) via an
in-process MCP tool proxy.

Load-bearing invariants asserted here:
  (a) the ORIGINAL tool name (not ``mcp__hermes__<name>``) reaches
      ``handle_function_call``;
  (b) ``skip_pre_tool_call_hook=False`` — the RBAC hook fires inside;
  (c) a hook BLOCK (a ``{"error": <msg>}`` JSON string) surfaces as an
      ``is_error`` MCP tool result the model sees — NOT an exception, NOT a
      silent success;
  (d) the verified user_id captured at create() entry is visible inside the
      handler (cron-Phase-1 identity invariant).
"""
import asyncio
import json


def test_proxy_denied_tool_returns_block_as_is_error(monkeypatch):
    """A tool the RBAC hook denies comes back as an is_error tool result the
    model sees — NOT a raised exception, NOT a silent success."""
    from agent import claude_sdk_client as m

    calls = {}

    def fake_handle_function_call(name, args, **kw):
        calls["name"] = name
        calls["skip"] = kw.get("skip_pre_tool_call_hook")
        # Simulate a hook BLOCK — the REAL shape: a JSON string {"error": <msg>}.
        return json.dumps({"error": "role 'member' is not permitted to use terminal"})

    monkeypatch.setattr(m, "handle_function_call", fake_handle_function_call, raising=False)

    handler = m._make_proxy_handler("terminal", capture_ctx=m._capture_ctx(), turn_ids={})
    result = asyncio.run(handler({"command": "rm -rf /"}))

    assert calls["name"] == "terminal"           # original name, not mcp__hermes__terminal
    assert calls["skip"] is False                # hook MUST fire inside
    assert result["is_error"] is True            # {"error": ...} -> is_error True
    assert "not permitted" in result["content"][0]["text"]


def test_proxy_success_result_passes_through(monkeypatch):
    """A normal (non-error) JSON-string result passes through as is_error=False."""
    from agent import claude_sdk_client as m

    def fake_handle_function_call(name, args, **kw):
        return json.dumps({"stdout": "hello", "exit_code": 0})

    monkeypatch.setattr(m, "handle_function_call", fake_handle_function_call, raising=False)
    handler = m._make_proxy_handler("terminal", capture_ctx=m._capture_ctx(), turn_ids={})
    result = asyncio.run(handler({"command": "echo hello"}))
    assert result["is_error"] is False
    assert "hello" in result["content"][0]["text"]


def test_proxy_forwards_verified_user_id(monkeypatch):
    """The contextvar captured at create() entry must be visible on the bridge
    thread so _get_user_id_for_hooks() sees the real user_id."""
    from agent import claude_sdk_client as m
    from gateway import session_context

    seen = {}

    def fake_handle_function_call(name, args, **kw):
        # handle_function_call internally reads the contextvar via the hook;
        # here we assert the contextvar itself is visible under capture_ctx.run.
        seen["uid"] = session_context.get_session_user_id()
        return "{}"

    monkeypatch.setattr(m, "handle_function_call", fake_handle_function_call, raising=False)

    # Set the user_id in THIS thread's context, capture it, then run the handler
    # (the handler runs the dispatch under the captured context).
    tokens = session_context.set_session_vars(user_id="U_TEST_123")
    try:
        ctx = m._capture_ctx()
    finally:
        session_context.clear_session_vars(tokens)
    handler = m._make_proxy_handler("read_file", capture_ctx=ctx, turn_ids={})
    asyncio.run(handler({"path": "/x"}))
    assert seen["uid"] == "U_TEST_123"


def test_tools_convert_to_mcp_server(monkeypatch):
    from agent.claude_sdk_client import _build_mcp_server, _capture_ctx
    tools = [{"type": "function", "function": {
        "name": "terminal", "description": "run",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    server = _build_mcp_server(tools, capture_ctx=_capture_ctx(), turn_ids={})
    assert server is not None
