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
