"""Auxiliary-client routing for the ``claude-sdk`` provider (Task 9).

The auxiliary client (context compression, session-title generation, vision
side-tasks) for a ``claude-sdk`` PRIMARY must route through the CHEAP
``claude -p`` SUBPROCESS (``ClaudeCliClient``) — NOT a second, expensive
``ClaudeSdkClient`` (which runs the full SDK inner loop) and NOT the
direct-HTTP Anthropic OAuth path (``build_anthropic_client``), which would
silently bill against "extra usage".  These tests are the billing safeguard
mirroring ``test_claude_cli_aux.py``.
"""

import unittest
from unittest.mock import patch

from agent import auxiliary_client
from agent.claude_cli_client import ClaudeCliClient
from agent.claude_sdk_client import ClaudeSdkClient


class ClaudeSdkAuxTests(unittest.TestCase):
    def test_aux_for_claude_sdk_primary_yields_claude_cli_client(self):
        # LOAD-BEARING: aux resolution for a claude-sdk primary must hand back
        # the CHEAP subprocess ClaudeCliClient, never a ClaudeSdkClient (which
        # would run the full SDK inner loop for a trivial side-task) and never
        # an Anthropic HTTP client.  The claude-sdk command may be empty (SDK
        # bundles its CLI), so the aux path resolves claude-cli creds; mock
        # ``shutil.which`` so the fallback ``claude`` binary resolves.
        with (
            patch("hermes_cli.auth.shutil.which", return_value="/usr/bin/claude"),
            patch("agent.auxiliary_client._read_main_model", return_value=""),
        ):
            client, model = auxiliary_client.resolve_provider_client("claude-sdk")

        self.assertIsInstance(client, ClaudeCliClient)
        self.assertNotIsInstance(client, ClaudeSdkClient)
        # Non-empty model proves it did NOT hit the empty-model (None,None)
        # guard, and that the cheap aux default kicked in.
        self.assertTrue(model)

    def test_aux_claude_sdk_async_mode_preserves_claude_cli_client(self):
        # In async_mode the claude-sdk aux client must STILL be a ClaudeCliClient
        # (subprocess), NOT be re-wrapped into an AsyncOpenAI pointed at a
        # non-routable subprocess marker base_url.
        with (
            patch("hermes_cli.auth.shutil.which", return_value="/usr/bin/claude"),
            patch("agent.auxiliary_client._read_main_model", return_value=""),
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "claude-sdk", async_mode=True
            )

        self.assertIsInstance(client, ClaudeCliClient)  # NOT AsyncOpenAI, NOT ClaudeSdkClient
        self.assertNotIsInstance(client, ClaudeSdkClient)
        self.assertTrue(model)

    def test_aux_claude_sdk_aliases_route_through_claude_cli(self):
        # The claude-sdk aliases must normalise to claude-sdk (and thus aux to a
        # ClaudeCliClient), never leak out to a different provider.
        for alias in ("claude-agent-sdk", "claude_sdk"):
            self.assertEqual(
                auxiliary_client._normalize_aux_provider(alias), "claude-sdk"
            )

    def test_claude_sdk_not_aliased_to_anthropic(self):
        # Guard against the billing trap: claude-sdk must NOT normalise to the
        # anthropic (HTTP) provider.
        self.assertNotEqual(
            auxiliary_client._normalize_aux_provider("claude-sdk"), "anthropic"
        )
        self.assertEqual(
            auxiliary_client._normalize_aux_provider("claude-sdk"), "claude-sdk"
        )

    def test_requested_model_wins_over_default(self):
        # An explicitly requested model must take precedence over the cheap aux
        # DEFAULT.  ``_read_main_model`` is forced empty so the only non-default
        # source of a model is the requested ``model=`` argument.
        with (
            patch("hermes_cli.auth.shutil.which", return_value="/usr/bin/claude"),
            patch("agent.auxiliary_client._read_main_model", return_value=""),
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "claude-sdk", model="claude-opus-4-6"
            )

        self.assertIsInstance(client, ClaudeCliClient)
        self.assertIn("opus-4-6", model)
        self.assertNotIn("haiku", model)


if __name__ == "__main__":
    unittest.main()
