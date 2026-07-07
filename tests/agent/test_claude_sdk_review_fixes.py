"""Regression tests for the PR-review findings on the claude-sdk provider.

Each test pins one verified bug fix (F1..F10) so it cannot silently return.
"""
import asyncio
import json

import pytest


# ── F5: _SESSION_CAP must not crash the import on a non-numeric env ──────────
def test_env_int_tolerates_garbage(monkeypatch):
    from agent.claude_sdk_client import _env_int
    monkeypatch.setenv("HERMES_CLAUDE_SDK_SESSION_CAP", "eight")
    assert _env_int("HERMES_CLAUDE_SDK_SESSION_CAP", 8) == 8
    monkeypatch.setenv("HERMES_CLAUDE_SDK_SESSION_CAP", "8 x")
    assert _env_int("HERMES_CLAUDE_SDK_SESSION_CAP", 8) == 8


# ── F3: a success payload carrying a falsy "error" key is NOT is_error ───────
def test_success_with_falsy_error_key_not_flagged():
    from agent.claude_sdk_client import _extract_text_and_error
    # null / empty / zero error values are NOT denials.
    for payload in ('{"output": "ok", "error": null}',
                    '{"output": "ok", "error": ""}',
                    '{"result": "no error here"}'):
        text, is_error = _extract_text_and_error(payload)
        assert is_error is False, payload
    # A truthy top-level error IS a denial.
    text, is_error = _extract_text_and_error('{"error": "role member denied terminal"}')
    assert is_error is True
    assert "denied" in text


# ── F2: no session_key/session_id/user must NOT collapse users to one bucket ─
def test_session_key_falls_back_per_instance_not_global():
    from agent.claude_sdk_client import _session_key_for
    k1 = _session_key_for({}, fallback="inst-A")
    k2 = _session_key_for({}, fallback="inst-B")
    assert k1 == "inst-A" and k2 == "inst-B"
    assert k1 != k2  # distinct clients never share the "default" bucket
    # An explicit session_id wins and is stable.
    assert _session_key_for({"session_id": "S1"}, fallback="inst-A") == "S1"
    assert _session_key_for({"session_key": "K1", "session_id": "S1"},
                            fallback="inst-A") == "K1"


def test_two_clients_get_distinct_instance_keys():
    from agent.claude_sdk_client import ClaudeSdkClient
    a, b = ClaudeSdkClient(), ClaudeSdkClient()
    assert a._instance_key != b._instance_key


# ── F4: cosmetic assistant-text transforms must NOT force a reseed ──────────
def test_think_block_transform_does_not_reseed():
    from agent.claude_sdk_client import _SdkSession
    sess = _SdkSession(session_id="s")
    # Shadow holds the RAW SDK assistant text (with a <think> block).
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "<think>reasoning</think>answer"}]
    # Incoming replays the TRANSFORMED assistant text (Hermes stripped <think>).
    incoming = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "u2"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is False, "cosmetic think-strip must not reseed"
    assert decision.new_messages == [{"role": "user", "content": "u2"}]


def test_real_rewrite_still_reseeds():
    from agent.claude_sdk_client import _SdkSession
    sess = _SdkSession(session_id="s")
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "a1"}]
    incoming = [{"role": "user", "content": "[compressed]"},
                {"role": "user", "content": "u2"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is True  # normalization must not mask a genuine rewrite


# ── F1/F10: an errored/timed-out turn must clear shadow so next turn reseeds ─
def test_error_turn_resets_shadow_forcing_reseed(monkeypatch):
    from agent import claude_sdk_client as m

    async def _boom(self, *a, **k):
        raise m._SdkSubprocessError("subprocess died")

    monkeypatch.setattr(m.ClaudeSdkClient, "_run_sdk_turn", _boom, raising=False)
    m._SESSIONS.clear()
    client = m.ClaudeSdkClient()
    # Pre-seed a session with a populated shadow (as if a prior turn succeeded).
    key = client._instance_key
    sess = m._get_or_make_session(key)
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "a1"}]
    with pytest.raises(Exception):
        client.chat.completions.create(
            model="x", messages=[{"role": "user", "content": "u1"},
                                  {"role": "assistant", "content": "a1"},
                                  {"role": "user", "content": "u2"}], tools=[])
    # After the error the shadow MUST be cleared (so the next turn reseeds the
    # full rendered history instead of matching a stale prefix and dropping it).
    assert sess.shadow == []
    assert sess.sdk_client is None


# ── F9/G1: each session owns a distinct per-session SYNC turn lock ───────────
def test_session_has_distinct_sync_lock():
    import threading
    from agent.claude_sdk_client import _SdkSession
    s1 = _SdkSession(session_id="s1")
    s2 = _SdkSession(session_id="s2")
    # A real, acquirable threading lock, and NOT shared between sessions.
    assert isinstance(s1.lock, type(threading.Lock()))
    assert s1.lock is not s2.lock
    with s1.lock:
        assert s1.lock.locked()
        assert not s2.lock.locked()  # locking one must not lock the other


# ── G2: a timed-out session is REPLACED in the store (fresh lock, no poison) ─
def test_replace_session_swaps_in_fresh_object():
    from agent import claude_sdk_client as m
    m._SESSIONS.clear()
    old = m._get_or_make_session("k1")
    old.shadow = [{"role": "user", "content": "stale"}]
    m._replace_session("k1", old)
    new = m._SESSIONS["k1"]
    assert new is not old
    assert new.shadow == []
    assert new.lock is not old.lock  # fresh lock -> no poisoned-lock reuse


# ── G6: no new user message short-circuits (no empty SDK query) ──────────────
def test_empty_query_short_circuits_to_cached_text(monkeypatch):
    from agent import claude_sdk_client as m

    called = {"ran": False}

    async def _should_not_run(self, *a, **k):
        called["ran"] = True
        return ("SHOULD NOT HAPPEN", {})

    monkeypatch.setattr(m.ClaudeSdkClient, "_run_sdk_turn", _should_not_run, raising=False)
    m._SESSIONS.clear()
    client = m.ClaudeSdkClient()
    key = client._instance_key
    sess = m._get_or_make_session(key)
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "prior answer"}]
    # Incoming == shadow (no new user turn): must NOT run a turn, must return the
    # cached last assistant text.
    resp = client.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "u1"},
                  {"role": "assistant", "content": "prior answer"}], tools=[])
    assert called["ran"] is False
    assert resp.choices[0].message.content == "prior answer"


# ── G3: turn_ids must carry a non-empty task_id so tool sessions don't collide ─
def test_turn_ids_task_id_defaults_to_session_key(monkeypatch):
    from agent import claude_sdk_client as m

    seen = {}

    async def _capture(self, session, new_prompt, eff_model, tools, ctx, turn_ids, system_text):
        seen["turn_ids"] = dict(turn_ids)
        return ("ok", {})

    monkeypatch.setattr(m.ClaudeSdkClient, "_run_sdk_turn", _capture, raising=False)
    m._SESSIONS.clear()
    client = m.ClaudeSdkClient()
    client.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "hi"}], tools=[],
        session_id="CONV_42")
    # task_id must be non-empty and tie to the conversation, not the shared ""
    # "default" bucket that would bleed terminal/browser state across convos.
    assert seen["turn_ids"]["task_id"] == "CONV_42"
    assert seen["turn_ids"]["session_id"] == "CONV_42"


# ── G(think): normalization strips ALL reasoning tag variants, not just <think> ─
def test_normalize_strips_all_reasoning_variants():
    from agent.claude_sdk_client import _normalize_content
    for tag in ("think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought"):
        raw = f"<{tag}>hidden</{tag}>visible"
        assert _normalize_content(raw) == "visible", tag


def test_reasoning_variant_transform_does_not_reseed():
    from agent.claude_sdk_client import _SdkSession
    sess = _SdkSession(session_id="s")
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "<reasoning>x</reasoning>answer"}]
    incoming = [{"role": "user", "content": "u1"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "u2"}]
    assert sess.reconcile(incoming).reseed is False


# ── F7: LRU eviction disconnects the evicted client (outside the lock) ───────
def test_lru_evicts_and_disconnects_outside_lock(monkeypatch):
    from agent import claude_sdk_client as m
    monkeypatch.setattr(m, "_SESSION_CAP", 2, raising=False)
    m._SESSIONS.clear()
    disconnected = []

    class _FakeClient:
        def __init__(self, sid):
            self.sid = sid

    monkeypatch.setattr(
        m, "_disconnect_client_on_bridge",
        lambda c: disconnected.append(getattr(c, "sid", None)), raising=False)

    s1 = m._get_or_make_session("k1")
    s1.sdk_client = _FakeClient("s1")
    s2 = m._get_or_make_session("k2")
    s2.sdk_client = _FakeClient("s2")
    m._get_or_make_session("k3")  # evicts k1
    assert "k1" not in m._SESSIONS
    assert "k2" in m._SESSIONS and "k3" in m._SESSIONS
    assert disconnected == ["s1"]
