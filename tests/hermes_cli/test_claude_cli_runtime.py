import unittest
from unittest.mock import patch

from hermes_cli import runtime_provider


class ClaudeCliRuntimeTests(unittest.TestCase):
    def test_runtime_for_claude_cli(self):
        # resolve_runtime_provider is the real runtime-resolution entrypoint
        # (mirrors how the copilot-acp branch is exercised). The auth resolver
        # uses hermes_cli.auth.shutil.which to locate the `claude` binary, so we
        # mock that to resolve without a real install. We also clear the env so
        # the command resolves to the default "claude".
        env = {
            "HERMES_CLAUDE_CLI_COMMAND": "",
            "CLAUDE_CLI_PATH": "",
            "HERMES_CLAUDE_CLI_ARGS": "",
            "CLAUDE_CLI_BASE_URL": "",
        }
        with patch.dict("os.environ", env, clear=False), patch(
            "hermes_cli.auth.shutil.which", return_value="/usr/bin/claude"
        ):
            rt = runtime_provider.resolve_runtime_provider(requested="claude-cli")

        # Mirror the copilot-acp runtime dict shape exactly.
        self.assertEqual(rt["provider"], "claude-cli")
        self.assertEqual(rt["api_mode"], "chat_completions")
        self.assertTrue(str(rt["base_url"]).startswith("claude-cli://"))
        self.assertEqual(rt["api_key"], "claude-cli")
        self.assertEqual(rt["command"], "/usr/bin/claude")
        self.assertEqual(rt["args"], [])
        self.assertEqual(rt["source"], "process")
        self.assertEqual(rt["requested_provider"], "claude-cli")


if __name__ == "__main__":
    unittest.main()
