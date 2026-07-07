import asyncio


def test_bridge_runs_coroutine_and_returns_result():
    from agent.claude_sdk_client import _BridgeLoop
    bridge = _BridgeLoop()
    try:
        async def _work():
            await asyncio.sleep(0)
            return 42
        assert bridge.run(_work()) == 42
    finally:
        bridge.shutdown()


def test_bridge_shutdown_is_idempotent():
    from agent.claude_sdk_client import _BridgeLoop
    bridge = _BridgeLoop()
    bridge.shutdown()
    bridge.shutdown()  # must not raise


def test_lru_evicts_oldest_and_disconnects(monkeypatch):
    from agent import claude_sdk_client as m
    # Force a small cap.
    monkeypatch.setattr(m, "_SESSION_CAP", 2, raising=False)
    # Reset the store.
    m._SESSIONS.clear()
    disconnected = []

    class _FakeClient:
        def __init__(self, sid):
            self.sid = sid
    # Create 3 sessions past the cap of 2 via the internal helper.
    s1 = m._get_or_make_session("k1")
    s1.sdk_client = _FakeClient("s1")
    s2 = m._get_or_make_session("k2")
    s2.sdk_client = _FakeClient("s2")
    # Spy on the disconnect-on-evict path: monkeypatch the module's disconnect helper.
    monkeypatch.setattr(m, "_disconnect_client_on_bridge",
                        lambda c: disconnected.append(getattr(c, "sid", None)), raising=False)
    s3 = m._get_or_make_session("k3")   # should evict k1
    assert "k1" not in m._SESSIONS
    assert "k2" in m._SESSIONS and "k3" in m._SESSIONS
    assert disconnected == ["s1"]
