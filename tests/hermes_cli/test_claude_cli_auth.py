"""Tests for hermes_cli.auth — claude-cli external-process provider wiring.

Mirrors the copilot-acp precedent: PROVIDER_REGISTRY entry, credential
resolver (env-var precedence + default), status dispatch, and alias map.
"""

import os
import unittest
from unittest.mock import patch

from hermes_cli import auth


class ClaudeCliAuthTests(unittest.TestCase):
    def test_in_provider_registry_as_external_process(self):
        cfg = auth.PROVIDER_REGISTRY.get("claude-cli")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.auth_type, "external_process")

    def test_credential_resolver_default_command(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_CLAUDE_CLI_COMMAND", None)
            os.environ.pop("CLAUDE_CLI_PATH", None)
            # Make resolution deterministic even when `claude` is not installed:
            # the resolver must still report the requested command name.
            with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/claude"):
                creds = auth.resolve_external_process_provider_credentials("claude-cli")
            self.assertEqual(creds.get("command"), "/usr/local/bin/claude")

    def test_credential_resolver_env_override(self):
        with patch.dict(os.environ, {"HERMES_CLAUDE_CLI_COMMAND": "/opt/claude"}, clear=False):
            with patch("hermes_cli.auth.shutil.which", side_effect=lambda c: c):
                creds = auth.resolve_external_process_provider_credentials("claude-cli")
            self.assertEqual(creds.get("command"), "/opt/claude")

    def test_credential_resolver_path_fallback(self):
        with patch.dict(os.environ, {"CLAUDE_CLI_PATH": "/usr/bin/claude"}, clear=False):
            os.environ.pop("HERMES_CLAUDE_CLI_COMMAND", None)
            with patch("hermes_cli.auth.shutil.which", side_effect=lambda c: c):
                creds = auth.resolve_external_process_provider_credentials("claude-cli")
            self.assertEqual(creds.get("command"), "/usr/bin/claude")

    def test_credential_resolver_shape_matches_copilot(self):
        """Resolver returns the same dict keys copilot-acp returns."""
        with patch("hermes_cli.auth.shutil.which", side_effect=lambda c: c):
            creds = auth.resolve_external_process_provider_credentials("claude-cli")
        self.assertEqual(
            set(creds.keys()),
            {"provider", "api_key", "base_url", "command", "args", "source"},
        )
        self.assertEqual(creds["provider"], "claude-cli")
        self.assertEqual(creds["api_key"], "claude-cli")
        self.assertEqual(creds["args"], [])
        self.assertEqual(creds["source"], "process")

    def test_status_routes_through_external_process(self):
        status = auth.get_auth_status("claude-cli")
        # external-process status snapshots carry a provider key (unlike the
        # bare {"logged_in": False} returned for unrouted providers).
        self.assertEqual(status.get("provider"), "claude-cli")

    def test_status_probes_claude_binary(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_CLAUDE_CLI_COMMAND", None)
            os.environ.pop("CLAUDE_CLI_PATH", None)
            with patch("hermes_cli.auth.shutil.which", side_effect=lambda c: c) as which:
                status = auth.get_external_process_provider_status("claude-cli")
            which.assert_called_with("claude")
            self.assertEqual(status.get("command"), "claude")

    def test_aliases_map_to_claude_cli(self):
        # resolve_provider() is the canonicalization entrypoint that applies the
        # _PROVIDER_ALIASES map (same path copilot-acp aliases use).
        self.assertEqual(auth.resolve_provider("claude-code-cli"), "claude-cli")
        self.assertEqual(auth.resolve_provider("claude_subscription"), "claude-cli")
        # sanity: the canonical id resolves to itself
        self.assertEqual(auth.resolve_provider("claude-cli"), "claude-cli")


if __name__ == "__main__":
    unittest.main()
