import unittest
from types import SimpleNamespace
from agent import agent_runtime_helpers as arh
from agent.claude_sdk_client import ClaudeSdkClient
from agent.claude_cli_client import ClaudeCliClient


class ClaudeSdkDispatchTests(unittest.TestCase):
    def _agent(self, provider="claude-sdk"):
        return SimpleNamespace(provider=provider, _client_kwargs={}, _client_log_context=lambda: "")

    def test_dispatch_by_provider(self):
        client = arh.create_openai_client(
            self._agent(), {"base_url": "claude-sdk://local"}, reason="test", shared=False)
        self.assertIsInstance(client, ClaudeSdkClient)

    def test_dispatch_by_base_url(self):
        client = arh.create_openai_client(
            self._agent(provider="something"), {"base_url": "claude-sdk://local"}, reason="test", shared=False)
        self.assertIsInstance(client, ClaudeSdkClient)

    def test_claude_cli_still_dispatches_to_cli(self):
        client = arh.create_openai_client(
            self._agent(provider="claude-cli"), {"base_url": "claude-cli://local"}, reason="test", shared=False)
        self.assertIsInstance(client, ClaudeCliClient)


if __name__ == "__main__":
    unittest.main()
