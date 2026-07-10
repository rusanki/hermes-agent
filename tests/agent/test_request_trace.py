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


def test_finalize_turn_flushes_and_resets_trace_ctx(monkeypatch, tmp_path):
    """Wiring check for Task 6: finalize_turn must flush agent._request_trace_ctx
    via trace_turn_end (writing one record) and reset it to None afterward, so a
    stale ctx can't leak into the next turn on the same agent object.

    Uses a minimal agent stub (mirrors the one in
    test_turn_finalizer_cleanup_guard.py) rather than the real Agent -- just
    enough surface for finalize_turn to run its full body without mocking.
    """
    monkeypatch.delenv("HERMES_REQUEST_TRACE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    from agent import request_trace as rt
    importlib.reload(rt)

    from agent.turn_finalizer import finalize_turn

    class _StubBudget:
        used = 1
        max_total = 10
        remaining = 9

    class _StubCompressor:
        last_prompt_tokens = 0

    class _StubAgent:
        def __init__(self):
            self.max_iterations = 10
            self.iteration_budget = _StubBudget()
            self.context_compressor = _StubCompressor()
            self.model = "stub/model"
            self.provider = "stub"
            self.base_url = "http://stub"
            self.session_id = "sess-trace"
            self.quiet_mode = True
            self.platform = "cli"
            self._interrupt_requested = False
            self._interrupt_message = None
            self._tool_guardrail_halt_decision = None
            self._response_was_previewed = False
            self._skill_nudge_interval = 0
            self._iters_since_skill = 0
            for attr in (
                "session_input_tokens", "session_output_tokens",
                "session_cache_read_tokens", "session_cache_write_tokens",
                "session_reasoning_tokens", "session_prompt_tokens",
                "session_completion_tokens", "session_total_tokens",
                "session_estimated_cost_usd",
            ):
                setattr(self, attr, 0)
            self.session_cost_status = "ok"
            self.session_cost_source = "stub"
            # Populated by trace_turn_start before finalize_turn runs, exactly
            # like turn_context.py does at turn-start in production.
            self._request_trace_ctx = rt.trace_turn_start(
                session_id=self.session_id, user_id="U1", platform="cli",
                model=self.model, provider=self.provider, inbound="do a thing",
            )

        def _save_trajectory(self, *a, **k):
            pass

        def _cleanup_task_resources(self, *a, **k):
            pass

        def _drop_trailing_empty_response_scaffolding(self, *a, **k):
            pass

        def _persist_session(self, *a, **k):
            pass

        def _emit_status(self, *a, **k):
            pass

        def _safe_print(self, *a, **k):
            pass

        def _handle_max_iterations(self, messages, n):
            return "unused"

        def _file_mutation_verifier_enabled(self):
            return False

        def _turn_completion_explainer_enabled(self):
            return False

        def _drain_pending_steer(self):
            return None

        def clear_interrupt(self):
            pass

        def _sync_external_memory_for_turn(self, **k):
            pass

    agent = _StubAgent()
    assert agent._request_trace_ctx is not None

    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        {"role": "assistant", "content": "all done"},
    ]

    result = finalize_turn(
        agent,
        final_response="all done",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == "all done"

    # The trace record was flushed to disk with the response captured.
    recs = _read_lines(rt._trace_path())
    assert len(recs) == 1
    assert recs[0]["session_id"] == "sess-trace"
    assert recs[0]["response"] == "all done"

    # Ctx reset so a subsequent turn on the same agent object starts clean.
    assert agent._request_trace_ctx is None


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
