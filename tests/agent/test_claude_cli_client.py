"""Unit tests for the claude-cli provider client (no real subprocess)."""
from __future__ import annotations
import json
import unittest
from agent.claude_cli_client import (
    _format_messages_as_prompt,
    _extract_tool_calls_from_text,
    _split_system_message,
)


class FormatterTests(unittest.TestCase):
    def test_split_system_message_extracts_leading_system(self):
        msgs = [{"role": "system", "content": "SYS PROMPT"}, {"role": "user", "content": "hello"}]
        system_text, rest = _split_system_message(msgs)
        self.assertEqual(system_text, "SYS PROMPT")
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0]["role"], "user")

    def test_split_system_message_no_system(self):
        msgs = [{"role": "user", "content": "hi"}]
        system_text, rest = _split_system_message(msgs)
        self.assertEqual(system_text, "")
        self.assertEqual(len(rest), 1)

    def test_formatter_does_not_duplicate_system_prompt(self):
        _, rest = _split_system_message([{"role": "system", "content": "SECRET_SYS"}, {"role": "user", "content": "hi"}])
        prompt = _format_messages_as_prompt(rest, model="claude-opus-4-8", tools=None)
        self.assertNotIn("SECRET_SYS", prompt)
        self.assertIn("hi", prompt)

    def test_formatter_renders_assistant_tool_calls_as_markup(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/x"}'}}]}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("/x", prompt)

    def test_formatter_renders_tool_result_as_tool_response(self):
        msgs = [{"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "hostX"}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        self.assertIn("<tool_response>", prompt)
        self.assertIn("hostX", prompt)
        self.assertIn("c1", prompt)

    def test_extract_single_tool_call(self):
        text = '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"/x\\"}"}</tool_call>'
        calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "read_file")
        self.assertEqual(cleaned, "")

    def test_extract_text_only_no_calls(self):
        calls, cleaned = _extract_tool_calls_from_text("just text")
        self.assertEqual(calls, [])
        self.assertEqual(cleaned, "just text")


if __name__ == "__main__":
    unittest.main()
