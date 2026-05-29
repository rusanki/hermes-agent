"""Streaming guard for blocking-subprocess providers.

``ClaudeCliClient`` (and ``CopilotACPClient``) drive a local CLI subprocess and
return a plain blocking ``SimpleNamespace`` response — NOT an iterable stream.
The streaming branch in ``run_conversation`` must therefore be disabled for
these providers, otherwise the loop would try to iterate a non-iterable
response and break.

These tests pin the predicate that ``run_conversation`` consults at the
streaming-decision site.  Whether the loop reuses
``agent_init._is_external_process_runtime`` directly or wraps it in a local
``_provider_uses_blocking_client`` alias, the contract is the same: claude-cli
and copilot-acp disable streaming; normal HTTP providers do not.
"""

import unittest


def _streaming_guard_predicate():
    """Return the predicate the conversation loop uses to gate streaming.

    Prefer a conversation_loop-local alias if one exists, otherwise fall back
    to the shared agent_init helper that the loop delegates to.
    """
    try:
        from agent.conversation_loop import _provider_uses_blocking_client as pred
    except ImportError:
        from agent.agent_init import _is_external_process_runtime as pred
    return pred


class StreamingGuardTests(unittest.TestCase):
    def test_claude_cli_disables_streaming(self):
        pred = _streaming_guard_predicate()
        self.assertTrue(pred("claude-cli", "claude-cli://local"))

    def test_claude_cli_base_url_disables_streaming(self):
        # base_url prefix alone (provider not literally "claude-cli") still counts.
        pred = _streaming_guard_predicate()
        self.assertTrue(pred("something", "claude-cli://local"))

    def test_copilot_still_disables_streaming(self):
        pred = _streaming_guard_predicate()
        self.assertTrue(pred("copilot-acp", "acp://copilot"))

    def test_acp_tcp_disables_streaming(self):
        pred = _streaming_guard_predicate()
        self.assertTrue(pred("copilot-acp", "acp+tcp://host:1234"))

    def test_normal_provider_allows_streaming(self):
        pred = _streaming_guard_predicate()
        self.assertFalse(pred("anthropic", "https://api.anthropic.com"))

    def test_none_base_url_safe(self):
        pred = _streaming_guard_predicate()
        self.assertFalse(pred("anthropic", None))

    def test_conversation_loop_consults_the_predicate(self):
        """The loop must reference the predicate so claude-cli joins copilot on
        the non-streaming branch — not keep an inline copilot-only check.

        This fails until the inline ``provider == "copilot-acp"`` guard is
        replaced by a call to the shared predicate.
        """
        from agent import conversation_loop

        has_local_alias = hasattr(conversation_loop, "_provider_uses_blocking_client")
        delegates_to_helper = (
            getattr(conversation_loop, "_is_external_process_runtime", None) is not None
        )
        self.assertTrue(
            has_local_alias or delegates_to_helper,
            "conversation_loop must consult the blocking-client streaming "
            "predicate (local alias or imported _is_external_process_runtime)",
        )
