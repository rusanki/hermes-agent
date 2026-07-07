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


def test_turn_maps_final_text_and_usage(monkeypatch):
    from agent import claude_sdk_client as m

    async def _fake_run_turn(self, session, new_prompt, eff_model, tools, ctx, turn_ids, system_text):
        return ("final answer",
                {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15, "cached_tokens": 3})

    monkeypatch.setattr(m.ClaudeSdkClient, "_run_sdk_turn", _fake_run_turn, raising=False)

    client = m.ClaudeSdkClient()
    resp = client.chat.completions.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.choices[0].message.content == "final answer"
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.usage.prompt_tokens_details.cached_tokens == 3


def test_normalize_model_strips_prefix_and_defaults():
    from agent.claude_sdk_client import _normalize_model, _DEFAULT_MODEL
    assert _normalize_model("claude-sdk/claude-opus-4-8") == "claude-opus-4-8"
    assert _normalize_model("anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert _normalize_model("") == _DEFAULT_MODEL
    assert _normalize_model(None) == _DEFAULT_MODEL
    assert _normalize_model("claude-opus-4-8") == "claude-opus-4-8"
