"""Auxiliary-client routing for the ``claude-cli`` provider (Task 10).

The auxiliary client (context compression, session-title generation, vision
side-tasks) must route claude-cli traffic through the ``claude -p`` SUBPROCESS
(``ClaudeCliClient``), NOT through the direct-HTTP Anthropic OAuth path
(``build_anthropic_client``).  Routing aux to the HTTP path would silently bill
against "extra usage" — the exact problem the claude-cli provider exists to
avoid.  These tests are the billing safeguard.
"""

import unittest
from unittest.mock import patch

from agent import auxiliary_client
from agent.claude_cli_client import ClaudeCliClient


class ClaudeCliAuxTests(unittest.TestCase):
    def test_aux_routes_through_claude_cli_not_http(self):
        # ``resolve_external_process_provider_credentials`` resolves the
        # ``claude`` binary via ``shutil.which`` and raises if absent.  Mock it
        # so the test does not depend on a real claude install being on PATH.
        # ``_read_main_model`` is forced empty so the assertion below proves the
        # cheap aux DEFAULT model is what avoids the empty-model (None,None)
        # guard — not some ambient main-model config.
        with (
            patch("hermes_cli.auth.shutil.which", return_value="/usr/bin/claude"),
            patch("agent.auxiliary_client._read_main_model", return_value=""),
        ):
            client, model = auxiliary_client.resolve_provider_client("claude-cli")

        # Must be a subprocess client, never None and never an Anthropic HTTP
        # client.
        self.assertIsInstance(client, ClaudeCliClient)
        # Non-empty model proves it did NOT hit the empty-model (None,None)
        # guard, and that the cheap aux default kicked in.
        self.assertTrue(model)

    def test_aux_claude_cli_async_mode_returns_async_wrapped_client(self):
        # Mirror copilot's async return shape: async_mode=True must still hand
        # back a usable (non-None) client + non-empty model.
        with (
            patch("hermes_cli.auth.shutil.which", return_value="/usr/bin/claude"),
            patch("agent.auxiliary_client._read_main_model", return_value=""),
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "claude-cli", async_mode=True
            )

        self.assertIsNotNone(client)
        self.assertTrue(model)

    def test_claude_cli_not_aliased_to_anthropic(self):
        # Guard against the billing trap: claude-cli must NOT normalize to the
        # anthropic (HTTP) provider.
        self.assertNotEqual(
            auxiliary_client._normalize_aux_provider("claude-cli"), "anthropic"
        )
        self.assertEqual(
            auxiliary_client._normalize_aux_provider("claude-cli"), "claude-cli"
        )


if __name__ == "__main__":
    unittest.main()
