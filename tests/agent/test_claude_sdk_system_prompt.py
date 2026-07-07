"""Tool-use enforcement gating for the ``claude-sdk`` provider (Task 9).

``claude-cli`` drives Claude through the ``<tool_call>`` TEXT-MARKUP protocol,
so under ``auto`` it MUST receive TOOL_USE_ENFORCEMENT_GUIDANCE (the same
weak-tool-calling nudge given to gpt/gemini).  ``claude-sdk`` is the opposite:
the SDK uses NATIVE ``tool_use`` blocks, so it must NOT get that markup
guidance — otherwise the model would be told to emit ``<tool_call>`` text that
the SDK path never parses.

The gate lives in ``agent/system_prompt.py`` (``_is_claude_cli``) and already
excludes claude-sdk.  These tests LOCK that behaviour by exercising the real
``build_system_prompt_parts`` code path.  We assert claude-sdk does NOT get the
guidance while claude-cli DOES — proving the test discriminates.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


# "Tool-use enforcement" is the leading text of TOOL_USE_ENFORCEMENT_GUIDANCE.
_MARKER = "Tool-use enforcement"


class TestToolUseEnforcementClaudeSdk:
    def test_absent_for_claude_sdk_provider_under_auto(self):
        # claude-sdk uses NATIVE tool_use — must NOT get the <tool_call> markup
        # guidance even under "auto".
        agent = _make_agent(
            valid_tool_names=["read_file"],
            provider="claude-sdk",
            model="claude-sdk/claude-opus-4-8",
            _tool_use_enforcement="auto",
        )
        assert _MARKER not in _stable_prompt(agent)

    def test_absent_for_claude_sdk_base_url_under_auto(self):
        # Base-url marker path must also NOT trip the claude-cli-only gate.
        agent = _make_agent(
            valid_tool_names=["read_file"],
            provider="",
            base_url="claude-sdk://local",
            _tool_use_enforcement="auto",
        )
        assert _MARKER not in _stable_prompt(agent)

    def test_present_for_claude_cli_under_auto_proves_discrimination(self):
        # DISCRIMINATOR: the identical setup with the claude-cli provider MUST
        # get the guidance — proves the claude-sdk absence above is meaningful
        # and not a trivially-empty prompt.
        agent = _make_agent(
            valid_tool_names=["read_file"],
            provider="claude-cli",
            model="claude-cli/claude-opus-4-8",
            _tool_use_enforcement="auto",
        )
        assert _MARKER in _stable_prompt(agent)
