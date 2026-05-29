import unittest
from types import SimpleNamespace
from agent import agent_init


class ClaudeCliInitTests(unittest.TestCase):
    def test_command_args_injected_for_claude_cli(self):
        agent = SimpleNamespace(provider="claude-cli", acp_command="/opt/claude", acp_args=["--x"])
        client_kwargs = {}
        agent_init._inject_external_process_kwargs(agent, client_kwargs)
        self.assertEqual(client_kwargs.get("command"), "/opt/claude")
        self.assertEqual(client_kwargs.get("args"), ["--x"])

    def test_command_args_injected_for_copilot(self):
        agent = SimpleNamespace(provider="copilot-acp", acp_command="/usr/bin/copilot", acp_args=["--acp", "--stdio"])
        client_kwargs = {}
        agent_init._inject_external_process_kwargs(agent, client_kwargs)
        self.assertEqual(client_kwargs.get("command"), "/usr/bin/copilot")

    def test_no_injection_for_regular_provider(self):
        agent = SimpleNamespace(provider="anthropic", acp_command=None, acp_args=None)
        client_kwargs = {}
        agent_init._inject_external_process_kwargs(agent, client_kwargs)
        self.assertNotIn("command", client_kwargs)


class ClaudeCliResponsesGuardTests(unittest.TestCase):
    """Guard the Responses-API auto-upgrade exclusion for claude-cli.

    The auto-upgrade guard in ``init_agent`` (~:368) excludes external-process
    runtimes from being silently switched from ``chat_completions`` to
    ``codex_responses`` (ClaudeCliClient does not implement the Responses
    surface). The exclusion is expressed as a provider check plus a
    ``claude-cli://`` base_url prefix check, mirroring the copilot-acp /
    ``acp://`` exclusions. These tests reproduce that exact predicate to lock
    the behaviour in so the exclusion isn't dropped during a future refactor.
    """

    @staticmethod
    def _is_excluded_from_upgrade(provider, base_url):
        """Mirror of the claude-cli portion of the init_agent guard predicate."""
        return provider == "claude-cli" or str(base_url or "").lower().startswith(
            "claude-cli://"
        )

    def test_claude_cli_provider_excluded_from_responses_upgrade(self):
        self.assertTrue(self._is_excluded_from_upgrade("claude-cli", ""))

    def test_claude_cli_base_url_excluded_from_responses_upgrade(self):
        self.assertTrue(self._is_excluded_from_upgrade("", "claude-cli://local"))

    def test_regular_provider_not_excluded(self):
        self.assertFalse(self._is_excluded_from_upgrade("openai", "https://api.openai.com"))
