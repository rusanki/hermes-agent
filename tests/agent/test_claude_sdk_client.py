from types import SimpleNamespace
import asyncio


def test_create_returns_awaitable_response_dual_use():
    from agent.claude_sdk_client import ClaudeSdkClient, _AwaitableResponse
    client = ClaudeSdkClient()
    client._run_turn_stub = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None,
                                    reasoning=None, reasoning_content=None,
                                    reasoning_details=None),
            finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
        model="claude-sdk-stub")
    resp = client.chat.completions.create(
        model="claude-sdk", messages=[{"role": "user", "content": "hello"}], tools=[])
    assert isinstance(resp, _AwaitableResponse)
    assert resp.choices[0].message.content == "hi"
    assert resp.choices[0].finish_reason == "stop"
    awaited = asyncio.run(_await(resp))
    assert awaited.choices[0].message.content == "hi"


async def _await(x):
    return await x


def test_env_scrub_removes_billing_vars(monkeypatch):
    from agent.claude_sdk_client import _build_sdk_env
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.setenv(k, "SENTINEL")
    env = _build_sdk_env()
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        assert k not in env, f"{k} must be scrubbed to protect subscription billing"
