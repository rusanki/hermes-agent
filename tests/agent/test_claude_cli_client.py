"""Unit tests for the claude-cli provider client (no real subprocess)."""
from __future__ import annotations
import json
import unittest
from agent.claude_cli_client import (
    _format_messages_as_prompt,
    _extract_tool_calls_from_text,
    _split_system_message,
    _render_assistant_tool_calls,
    _render_tool_response,
    _parse_stream_json_lines,
    ClaudeCliQuotaError,
    ClaudeCliAuthError,
    ClaudeCliError,
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

    def test_render_extract_roundtrip(self):
        # An assistant tool_call (args as a JSON string) rendered into a prompt
        # and re-parsed must preserve the arguments. Extracted arguments is a
        # JSON STRING, so json.loads() of it must equal the original object.
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/x"}'}}]}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        calls, _ = _extract_tool_calls_from_text(prompt)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "read_file")
        self.assertEqual(json.loads(calls[0].function.arguments), {"path": "/x"})

    def test_multiple_tool_calls_in_one_message(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/a"}'}},
            {"id": "c2", "type": "function",
             "function": {"name": "write_file", "arguments": '{"path":"/b"}'}}]}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        self.assertIn("read_file", prompt)
        self.assertIn("write_file", prompt)
        self.assertEqual(prompt.count("<tool_call>"), 2)
        # The helper itself must emit exactly two blocks for two calls.
        blocks = _render_assistant_tool_calls(msgs[0]["tool_calls"])
        self.assertEqual(len(blocks), 2)

    def test_assistant_with_both_content_and_tool_calls(self):
        # Guards the divergence-from-ACP fix: an assistant turn carrying BOTH
        # text content and tool calls must render both, not drop either.
        msgs = [{"role": "assistant", "content": "thinking out loud",
                 "tool_calls": [
                     {"id": "c1", "type": "function",
                      "function": {"name": "read_file", "arguments": '{"path":"/x"}'}}]}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        self.assertIn("thinking out loud", prompt)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("read_file", prompt)

    def test_malformed_tool_call_json_is_dropped_but_consumed(self):
        # A malformed <tool_call> block yields zero calls, but the block is
        # still stripped from the cleaned text while surrounding text survives.
        text = "<tool_call>{not valid json}</tool_call> trailing"
        calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertIn("trailing", cleaned)
        self.assertNotIn("not valid json", cleaned)

    def test_tool_message_with_json_string_content(self):
        # A tool message whose content is a JSON string round-trips as a parsed
        # structure inside the <tool_response> block.
        msgs = [{"role": "tool", "tool_call_id": "c1", "name": "read_file",
                 "content": '{"result": "ok"}'}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        self.assertIn("<tool_response>", prompt)
        self.assertIn("result", prompt)
        self.assertIn("ok", prompt)
        # The helper itself must parse the JSON-string content into structure
        # (not leave it escaped), so the inner payload round-trips.
        block = _render_tool_response(msgs[0])
        inner = json.loads(block.split("<tool_response>\n", 1)[1].rsplit("\n</tool_response>", 1)[0])
        self.assertEqual(inner["content"], {"result": "ok"})


class StreamParserTests(unittest.TestCase):
    def _lines(self, *objs):
        return [json.dumps(o) for o in objs]

    def test_parses_assistant_text_and_result(self):
        lines = self._lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello "}], "usage": {"input_tokens": 10, "output_tokens": 2}}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}], "usage": {"input_tokens": 10, "output_tokens": 4}}},
            {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None, "stop_reason": "end_turn", "result": "hello world", "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 3}, "total_cost_usd": 0.01},
        )
        parsed = _parse_stream_json_lines(lines)
        self.assertEqual(parsed.text, "hello world")
        self.assertEqual(parsed.stop_reason, "end_turn")
        self.assertEqual(parsed.usage["output_tokens"], 4)
        self.assertEqual(parsed.usage["cache_read_input_tokens"], 3)
        self.assertAlmostEqual(parsed.cost_usd, 0.01)

    def test_quota_error_raises(self):
        lines = self._lines(
            {"type": "result", "subtype": "error", "is_error": True, "api_error_status": 400, "result": "You're out of extra usage. Add more at claude.ai/settings/usage and keep going."},
        )
        with self.assertRaises(ClaudeCliQuotaError):
            _parse_stream_json_lines(lines)

    def test_auth_error_raises(self):
        lines = self._lines(
            {"type": "result", "subtype": "error", "is_error": True, "api_error_status": 401, "result": "Unauthorized: please authenticate."},
        )
        with self.assertRaises(ClaudeCliAuthError):
            _parse_stream_json_lines(lines)

    def test_generic_error_raises_base(self):
        lines = self._lines(
            {"type": "result", "subtype": "error", "is_error": True, "api_error_status": 500, "result": "internal server error"},
        )
        with self.assertRaises(ClaudeCliError):
            _parse_stream_json_lines(lines)

    def test_malformed_line_is_skipped(self):
        lines = ["not json", json.dumps({"type": "result", "subtype": "success", "is_error": False, "api_error_status": None, "stop_reason": "end_turn", "result": "ok", "usage": {}, "total_cost_usd": 0.0})]
        parsed = _parse_stream_json_lines(lines)
        self.assertEqual(parsed.text, "ok")

    def test_assistant_buffer_used_when_no_result_text(self):
        lines = self._lines(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "partial"}], "usage": {}}},
        )
        parsed = _parse_stream_json_lines(lines)
        self.assertEqual(parsed.text, "partial")
        self.assertIsNone(parsed.stop_reason)


if __name__ == "__main__":
    unittest.main()
