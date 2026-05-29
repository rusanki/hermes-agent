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


class ExternalProcessRuntimeTests(unittest.TestCase):
    """Guard the Responses-API auto-upgrade exclusion for external-process runtimes.

    The auto-upgrade guard in ``init_agent`` (~:368) excludes external-process
    runtimes from being silently switched from ``chat_completions`` to
    ``codex_responses`` (CopilotACPClient / ClaudeCliClient do not implement the
    Responses surface). The exclusion is delegated to the pure
    ``_is_external_process_runtime`` predicate; these tests assert that REAL
    helper so the lock-in tracks the source of truth rather than a copy.
    """

    def test_claude_cli_provider_is_external_process(self):
        self.assertTrue(agent_init._is_external_process_runtime("claude-cli", "claude-cli://local"))

    def test_claude_cli_base_url_is_external_process(self):
        self.assertTrue(agent_init._is_external_process_runtime("something", "claude-cli://local"))

    def test_copilot_is_external_process(self):
        self.assertTrue(agent_init._is_external_process_runtime("copilot-acp", "acp://copilot"))

    def test_acp_tcp_base_url_is_external_process(self):
        self.assertTrue(agent_init._is_external_process_runtime("x", "acp+tcp://host:1234"))

    def test_regular_provider_is_not_external_process(self):
        self.assertFalse(agent_init._is_external_process_runtime("anthropic", "https://api.anthropic.com"))

    def test_none_base_url_safe(self):
        self.assertFalse(agent_init._is_external_process_runtime("anthropic", None))
