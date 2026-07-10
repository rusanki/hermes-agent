import asyncio
import json
from pathlib import Path


def _read_lines(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def test_disabled_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_REQUEST_TRACE", "0")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    ctx = rt.trace_turn_start(session_id="s", user_id="u", platform="slack",
                              model="m", provider="p", inbound="hi")
    assert ctx is None
    # All hooks must no-op on a None ctx without raising.
    rt.trace_tool_call(ctx, name="x", args={}, result="r", duration=0.1, is_error=False)
    rt.trace_turn_end(ctx, response="done", finish_reason="stop", usage={})


def test_disabled_is_case_insensitive(monkeypatch, tmp_path):
    # Kill-switch parsing must not be case-sensitive: "False"/"NO"/"FALSE" must
    # all disable tracing just like the lowercase literals do.
    monkeypatch.setenv("HERMES_REQUEST_TRACE", "False")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    ctx = rt.trace_turn_start(session_id="s", user_id="u", platform="slack",
                              model="m", provider="p", inbound="hi")
    assert ctx is None


def test_full_turn_writes_one_record_with_tools(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_REQUEST_TRACE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    ctx = rt.trace_turn_start(session_id="s1", user_id="U1", platform="slack",
                              model="claude-sdk/claude-sonnet-5", provider="claude-sdk",
                              inbound="what crons exist?")
    assert ctx is not None
    rt.trace_tool_call(ctx, name="cronjob", args={"action": "list"},
                       result='{"count": 2}', duration=0.01, is_error=False)
    rt.trace_tool_call(ctx, name="read_file", args={"path": "/x"},
                       result="contents", duration=0.05, is_error=False)
    rt.trace_turn_end(ctx, response="You have 2 crons.", finish_reason="stop",
                      usage={"prompt_tokens": 10, "completion_tokens": 5})
    recs = _read_lines(rt._trace_path())
    assert len(recs) == 1
    r = recs[0]
    assert r["session_id"] == "s1" and r["user_id"] == "U1"
    assert r["provider"] == "claude-sdk" and r["inbound"] == "what crons exist?"
    assert r["response"] == "You have 2 crons."
    assert len(r["tools"]) == 2
    assert r["tools"][0]["name"] == "cronjob"
    assert "list" in r["tools"][0]["args"]        # args serialized to a string
    assert r["tools"][0]["result"] == '{"count": 2}'
    assert r["usage"]["prompt_tokens"] == 10


def test_secrets_are_redacted(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_REQUEST_TRACE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    secret = "sk-ant-api03-CANARYSECRETVALUE1234567890abcdef"
    ctx = rt.trace_turn_start(session_id="s", user_id="u", platform="slack",
                              model="m", provider="p", inbound=f"my key is {secret}")
    rt.trace_tool_call(ctx, name="terminal", args={"cmd": f"echo {secret}"},
                       result=f"output {secret}", duration=0.1, is_error=False)
    rt.trace_turn_end(ctx, response=f"used {secret}", finish_reason="stop", usage={})
    blob = Path(rt._trace_path()).read_text()
    assert secret not in blob, "raw secret must be redacted from the trace"


def test_rotation_moves_to_dot1(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_REQUEST_TRACE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REQUEST_TRACE_MAX_MB", "0.0001")  # ~100 bytes
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    for i in range(5):
        ctx = rt.trace_turn_start(session_id=f"s{i}", user_id="u", platform="p",
                                  model="m", provider="p", inbound="x" * 200)
        rt.trace_turn_end(ctx, response="y" * 200, finish_reason="stop", usage={})
    from pathlib import Path
    assert Path(rt._trace_path() + ".1").exists(), "rotation must create a .1 backup"
    # current file exists and is smaller than the total written
    assert Path(rt._trace_path()).exists()


def test_max_bytes_falls_back_on_non_finite_or_nonpositive(monkeypatch, tmp_path):
    # nan/inf/-inf all parse via float() without a ValueError, and negative/zero
    # values parse fine too -- _max_bytes() must still fall back to the 50MB
    # default for all of them, and must never raise.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    default_bytes = int(50 * 1024 * 1024)
    for bad_value in ("nan", "inf", "-inf", "-1", "0", "-0.0"):
        monkeypatch.setenv("HERMES_REQUEST_TRACE_MAX_MB", bad_value)
        assert rt._max_bytes() == default_bytes, f"input {bad_value!r} must fall back to default"


def test_max_bytes_never_raises_on_garbage_input(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    for bad_value in ("nan", "inf", "-inf", "-1", "0", "not-a-number", "", "  "):
        monkeypatch.setenv("HERMES_REQUEST_TRACE_MAX_MB", bad_value)
        rt._max_bytes()  # must not raise


def test_max_bytes_accepts_valid_positive_value(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REQUEST_TRACE_MAX_MB", "10")
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    assert rt._max_bytes() == int(10 * 1024 * 1024)


def test_write_failure_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_REQUEST_TRACE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    # Force the write to blow up; trace_turn_end must swallow it.
    monkeypatch.setattr(rt, "_write_record",
                        lambda rec: (_ for _ in ()).throw(OSError("disk full")))
    ctx = rt.trace_turn_start(session_id="s", user_id="u", platform="p",
                              model="m", provider="p", inbound="hi")
    rt.trace_turn_end(ctx, response="ok", finish_reason="stop", usage={})  # must NOT raise


def test_sdk_proxy_records_tool_under_captured_ctx(monkeypatch, tmp_path):
    """The proxy handler runs on the bridge thread under a captured context; a
    tool call there MUST record into the turn's trace ctx (the whole reason this
    feature exists). We simulate by setting the ContextVar, capturing the
    context, then running the handler under it — mirroring the RBAC user_id path
    (see test_proxy_forwards_verified_user_id in test_claude_sdk_rbac.py).
    """
    monkeypatch.delenv("HERMES_REQUEST_TRACE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)
    from agent import claude_sdk_client as m

    # Begin a turn -> sets the ContextVar (mirrors turn_context.py: trace_turn_start
    # MUST run before the provider's _capture_ctx() snapshot).
    ctx = rt.trace_turn_start(session_id="s", user_id="u", platform="slack",
                              model="claude-sdk/x", provider="claude-sdk", inbound="q")

    def fake_handle_function_call(name, args, **kw):
        return '{"ok": true}'
    monkeypatch.setattr(m, "handle_function_call", fake_handle_function_call, raising=False)

    # Capture the context AFTER trace_turn_start, same as _create_chat_completion
    # does in production (ctx = _capture_ctx() runs after turn_context.py's
    # trace_turn_start call). We do NOT wrap the outer asyncio.run() call itself
    # in captured.run(...): a contextvars.Context cannot be entered twice
    # concurrently, and _handler's own internal `await
    # asyncio.to_thread(capture_ctx.run, _dispatch)` already enters it once from
    # the worker thread — wrapping the top-level call too raises "cannot enter
    # context: ... is already entered" (confirmed empirically). Instead, mirror
    # the RBAC test pattern: the ContextVar was set on THIS thread before
    # _capture_ctx(), and asyncio.run() itself snapshots the calling thread's
    # current context to run the coroutine, so a bare asyncio.run(handler(...))
    # already sees the ContextVar at the handler's top level.
    captured = m._capture_ctx()
    handler = m._make_proxy_handler("cronjob", capture_ctx=captured, turn_ids={})
    asyncio.run(handler({"action": "list"}))

    # The proxy must have appended a tool event to THE SAME turn ctx.
    assert any(t["name"] == "cronjob" for t in ctx["tools"]), \
        "SDK-proxy tool call must be recorded into the turn trace ctx"
