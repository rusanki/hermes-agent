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
