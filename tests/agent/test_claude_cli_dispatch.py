import unittest
from types import SimpleNamespace
from agent import agent_runtime_helpers as arh
from agent.claude_cli_client import ClaudeCliClient


class ClaudeCliDispatchTests(unittest.TestCase):
    def _agent(self, provider="claude-cli"):
        return SimpleNamespace(provider=provider, _client_kwargs={}, _client_log_context=lambda: "")

    def test_dispatch_by_provider_returns_claude_cli_client(self):
        client = arh.create_openai_client(
            self._agent(), {"base_url": "claude-cli://local"}, reason="test", shared=False)
        self.assertIsInstance(client, ClaudeCliClient)

    def test_dispatch_by_base_url_returns_claude_cli_client(self):
        # provider name not claude-cli but base_url marker present
        client = arh.create_openai_client(
            self._agent(provider="something"), {"base_url": "claude-cli://local"}, reason="test", shared=False)
        self.assertIsInstance(client, ClaudeCliClient)

    def test_copilot_still_dispatches_to_copilot_client(self):
        from agent.copilot_acp_client import CopilotACPClient
        client = arh.create_openai_client(
            self._agent(provider="copilot-acp"), {"base_url": "acp://copilot"}, reason="test", shared=False)
        self.assertIsInstance(client, CopilotACPClient)
