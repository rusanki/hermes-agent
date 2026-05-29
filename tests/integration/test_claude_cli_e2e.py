"""End-to-end smoke test for the claude-cli provider (gated; spends subscription credit).

Set HERMES_CLAUDE_CLI_E2E=1 to run. Requires a working `claude` CLI login and
available Agent SDK credit / extra-usage balance. Skipped by default so CI and
ordinary test runs never spawn the real CLI or incur cost.
"""
from __future__ import annotations

import os
import unittest

from agent.claude_cli_client import ClaudeCliClient

_RUN = os.getenv("HERMES_CLAUDE_CLI_E2E") == "1"


@unittest.skipUnless(
    _RUN,
    "set HERMES_CLAUDE_CLI_E2E=1 to run (spends Claude subscription credit; needs `claude` login)",
)
class ClaudeCliE2ETests(unittest.TestCase):
    def test_text_turn(self):
        client = ClaudeCliClient(model="claude-opus-4-8")
        resp = client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[
                {"role": "system", "content": "You are a test harness probe."},
                {"role": "user", "content": "Reply with exactly this token and nothing else: E2E_OK"},
            ],
        )
        content = resp.choices[0].message.content or ""
        self.assertIn("E2E_OK", content)
        self.assertEqual(resp.choices[0].finish_reason, "stop")

    def test_tool_turn(self):
        client = ClaudeCliClient(model="claude-opus-4-8")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        resp = client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[
                {"role": "system", "content": "Use the provided tool when the user asks for the time."},
                {"role": "user", "content": "What time is it? Use the get_time tool."},
            ],
            tools=tools,
        )
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "get_time")


if __name__ == "__main__":
    unittest.main()
