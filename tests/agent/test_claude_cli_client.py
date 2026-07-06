"""Unit tests for the claude-cli provider client (no real subprocess)."""
from __future__ import annotations
import asyncio
import base64
import json
import os as _os
import subprocess
import unittest
from unittest.mock import MagicMock, patch
from agent.claude_cli_client import (
    ClaudeCliClient,
    _build_subprocess_env,
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
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "read_file")
        self.assertEqual(cleaned, "")
        self.assertEqual(malformed, [])

    def test_extract_text_only_no_calls(self):
        calls, cleaned, malformed = _extract_tool_calls_from_text("just text")
        self.assertEqual(calls, [])
        self.assertEqual(cleaned, "just text")
        self.assertEqual(malformed, [])

    def test_render_extract_roundtrip(self):
        # An assistant tool_call (args as a JSON string) rendered into a prompt
        # and re-parsed must preserve the arguments. Extracted arguments is a
        # JSON STRING, so json.loads() of it must equal the original object.
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/x"}'}}]}]
        prompt = _format_messages_as_prompt(msgs, model=None, tools=None)
        calls, _, _malformed = _extract_tool_calls_from_text(prompt)
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

    def test_malformed_tool_call_json_is_reported_and_left_visible(self):
        # A malformed <tool_call> block yields zero calls, is reported in
        # `malformed` with an error string, and is NEVER deleted from the
        # cleaned text — the raw block must stay visible (never silently
        # discarded), while surrounding text also survives.
        text = "<tool_call>{not valid json}</tool_call> trailing"
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertIn("trailing", cleaned)
        self.assertIn("not valid json", cleaned)
        self.assertEqual(len(malformed), 1)
        self.assertIn("not valid json", malformed[0]["raw"])
        self.assertTrue(malformed[0]["error"])

    def test_malformed_and_valid_blocks_in_same_text(self):
        # A malformed block alongside a valid one: the valid block is still
        # extracted and stripped from cleaned text, while the malformed block
        # is reported AND stays visible in cleaned text.
        text = (
            "Before.\n"
            "<tool_call>{not valid json}</tool_call>\n"
            'adapter <tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"/x\\"}"}</tool_call>\n'
            "After."
        )
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "read_file")
        self.assertEqual(len(malformed), 1)
        self.assertIn("not valid json", malformed[0]["raw"])
        self.assertIn("not valid json", cleaned)
        self.assertIn("Before.", cleaned)
        self.assertIn("After.", cleaned)
        # The valid block's markup must be gone; the malformed block's must
        # remain (it is not consumed).
        self.assertNotIn("read_file", cleaned)

    def test_malformed_json_error_string_is_descriptive(self):
        # The error string must describe the JSON failure, not be empty/generic.
        text = "<tool_call>{not valid json}</tool_call>"
        _, _, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(malformed), 1)
        self.assertIsInstance(malformed[0]["error"], str)
        self.assertGreater(len(malformed[0]["error"]), 0)

    def test_malformed_non_dict_json_result_reported(self):
        # Valid JSON that parses to a non-dict (e.g. a bare list) must be
        # reported as malformed, not silently ignored.
        text = "<tool_call>[1, 2, 3]</tool_call>"
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertEqual(len(malformed), 1)
        self.assertIn("[1, 2, 3]", cleaned)

    def test_malformed_missing_name_reported(self):
        # Valid JSON dict but missing/blank "name" must be reported as
        # malformed and left visible.
        text = '<tool_call>{"arguments": "{}"}</tool_call>'
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertEqual(len(malformed), 1)
        self.assertIn("arguments", cleaned)

    def test_malformed_blank_name_reported(self):
        text = '<tool_call>{"name": "  ", "arguments": "{}"}</tool_call>'
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertEqual(len(malformed), 1)

    def test_malformed_literal_control_char_in_json_string(self):
        # A literal newline (raw control char, not the two-char escape "\n")
        # embedded in a JSON string value: json.loads must reject this, and it
        # must be reported as malformed rather than silently dropped.
        text = '<tool_call>{"name": "write_file", "arguments": "{\\"content\\": \\"line1\nline2\\"}"}</tool_call>'
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertEqual(len(malformed), 1)
        self.assertIn("write_file", cleaned)

    def test_unclosed_tool_call_tag_detected(self):
        # Output truncated mid-block (max_tokens/timeout): an opening
        # <tool_call> with no matching closing tag must be reported as
        # malformed with a specific error, and left visible (not consumed).
        text = 'Sure, one sec.\n<tool_call>{"name": "cronjob", "argu'
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(calls, [])
        self.assertEqual(len(malformed), 1)
        self.assertEqual(malformed[0]["error"], "unclosed <tool_call> tag")
        self.assertIn("cronjob", malformed[0]["raw"])
        self.assertIn("Sure, one sec.", cleaned)
        self.assertIn("<tool_call>", cleaned)

    def test_unclosed_tag_raw_capped_at_500_chars(self):
        text = "<tool_call>" + ("x" * 1000)
        _, _, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(malformed), 1)
        self.assertLessEqual(len(malformed[0]["raw"]), 500)

    def test_malformed_block_raw_capped_at_500_chars(self):
        # A malformed (parse-failing) block's raw text is capped at 500 chars.
        text = "<tool_call>{not valid json " + ("x" * 1000) + "}</tool_call>"
        _, _, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(malformed), 1)
        self.assertLessEqual(len(malformed[0]["raw"]), 500)

    def test_no_unclosed_tag_false_positive_when_all_blocks_closed(self):
        # A normal closed block must NOT trigger unclosed-tag detection.
        text = '<tool_call>{"name": "read_file", "arguments": "{}"}</tool_call>'
        _, _, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(malformed, [])

    def test_extract_tool_call_with_nested_brace_arguments_object(self):
        # Regression guard: full nested-brace payloads must extract intact
        # under the widened `(.*?)` capture. The old `(\{.*?\})` also handled
        # these via backtracking; the real motivation for the widened capture
        # is that non-`{`-starting/garbled payloads now match and can be
        # routed to malformed handling (Task 2) instead of shipping as raw
        # unmatched text.
        text = (
            '<tool_call>{"name":"cronjob","arguments":{"action":"create",'
            '"job":{"schedule":"0 9 * * *","name":"digest"}}}</tool_call>'
        )
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "cronjob")
        # function.arguments is stringified in `_try_add_tool_call` since the
        # parsed "arguments" value here is an object, not already a string.
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"action": "create", "job": {"schedule": "0 9 * * *", "name": "digest"}},
        )
        self.assertEqual(cleaned, "")
        self.assertEqual(malformed, [])

    def test_extract_tool_call_with_escaped_braces_in_argument_string(self):
        # "arguments" as a JSON string whose contents themselves contain
        # braces/quotes (escaped) must not confuse the block extraction.
        text = (
            '<tool_call>{"name": "write_file", '
            '"arguments": "{\\"path\\": \\"/tmp/x\\", '
            '\\"content\\": \\"if (a) { b(); }\\"}"}</tool_call>'
        )
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "write_file")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"path": "/tmp/x", "content": "if (a) { b(); }"},
        )
        self.assertEqual(cleaned, "")
        self.assertEqual(malformed, [])

    def test_extract_pretty_printed_multiline_tool_call(self):
        # Pretty-printed JSON (newlines/indentation, nested object) inside the
        # block must still be captured in full.
        text = (
            "<tool_call>\n"
            "{\n"
            '  "name": "cronjob",\n'
            '  "arguments": {\n'
            '    "action": "create",\n'
            '    "job": {\n'
            '      "schedule": "0 9 * * *"\n'
            "    }\n"
            "  }\n"
            "}\n"
            "</tool_call>"
        )
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "cronjob")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"action": "create", "job": {"schedule": "0 9 * * *"}},
        )
        self.assertEqual(cleaned, "")
        self.assertEqual(malformed, [])

    def test_two_nested_brace_tool_calls_with_surrounding_prose(self):
        # Two nested-brace blocks in one message, with prose before/between/
        # after: both must be extracted in order, and the cleaned text must
        # retain the prose but no <tool_call> markup.
        text = (
            "Sure, I'll do both.\n"
            '<tool_call>{"name":"cronjob","arguments":{"action":"create",'
            '"job":{"schedule":"0 9 * * *"}}}</tool_call>\n'
            "Now the second one.\n"
            '<tool_call>{"name":"cronjob","arguments":{"action":"delete",'
            '"job":{"id":"abc"}}}</tool_call>\n'
            "Done."
        )
        calls, cleaned, malformed = _extract_tool_calls_from_text(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].function.name, "cronjob")
        self.assertEqual(calls[1].function.name, "cronjob")
        self.assertEqual(malformed, [])
        self.assertEqual(
            json.loads(calls[0].function.arguments)["action"], "create"
        )
        self.assertEqual(
            json.loads(calls[1].function.arguments)["action"], "delete"
        )
        self.assertIn("Sure, I'll do both.", cleaned)
        self.assertIn("Now the second one.", cleaned)
        self.assertIn("Done.", cleaned)
        self.assertNotIn("<tool_call>", cleaned)
        self.assertNotIn("</tool_call>", cleaned)

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

    def test_empty_lines(self):
        # No events at all: every field falls back to its empty default and
        # nothing is raised.
        parsed = _parse_stream_json_lines([])
        self.assertEqual(parsed.text, "")
        self.assertIsNone(parsed.stop_reason)
        self.assertEqual(parsed.usage, {})
        self.assertEqual(parsed.cost_usd, 0.0)
        self.assertIsNone(parsed.raw_result)

    def test_error_result_with_no_result_field(self):
        # An error result lacking a "result" message must still raise, using the
        # fallback message text.
        lines = self._lines(
            {"type": "result", "subtype": "error", "is_error": True, "api_error_status": 500},
        )
        with self.assertRaises(ClaudeCliError) as cm:
            _parse_stream_json_lines(lines)
        self.assertIn("error", str(cm.exception))

    def test_non_dict_json_line_skipped(self):
        # Bare JSON values (a number, a null) are valid JSON but not dict events;
        # they must be skipped without crashing.
        lines = [
            "123",
            "null",
            json.dumps({"type": "result", "subtype": "success", "is_error": False, "api_error_status": None, "stop_reason": "end_turn", "result": "ok", "usage": {}, "total_cost_usd": 0.0}),
        ]
        parsed = _parse_stream_json_lines(lines)
        self.assertEqual(parsed.text, "ok")

    def test_multiple_text_parts_in_one_assistant(self):
        # A single assistant event with multiple text parts and no result event:
        # the parts are concatenated in order.
        lines = self._lines(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}], "usage": {}}},
        )
        parsed = _parse_stream_json_lines(lines)
        self.assertEqual(parsed.text, "AB")

    def test_quota_markers_beat_auth_markers(self):
        # When an error message contains both quota and auth wording, quota wins.
        lines = self._lines(
            {"type": "result", "subtype": "error", "is_error": True, "api_error_status": 400, "result": "You're out of extra usage and also unauthorized."},
        )
        with self.assertRaises(ClaudeCliQuotaError):
            _parse_stream_json_lines(lines)

    def test_loose_401_not_misclassified(self):
        # A bare "401" embedded in an unrelated word ("401k") must NOT be treated
        # as an HTTP 401 / auth failure.
        lines = self._lines(
            {"type": "result", "subtype": "error", "is_error": True, "api_error_status": None, "result": "processed 401k tokens then failed"},
        )
        with self.assertRaises(ClaudeCliError) as cm:
            _parse_stream_json_lines(lines)
        self.assertNotIsInstance(cm.exception, ClaudeCliAuthError)


class ClientFacadeTests(unittest.TestCase):
    def _client_with_output(self, lines):
        client = ClaudeCliClient(model="claude-opus-4-8")
        client._run_claude = lambda prompt, system_prompt, model, timeout, image_dir=None: list(lines)
        return client

    def test_text_response_shape(self):
        lines = [json.dumps({"type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
                             "stop_reason": "end_turn", "result": "hi there", "usage": {"input_tokens": 5, "output_tokens": 2}, "total_cost_usd": 0.001})]
        client = self._client_with_output(lines)
        resp = client.chat.completions.create(model="claude-opus-4-8",
            messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}])
        self.assertEqual(resp.choices[0].message.content, "hi there")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertFalse(resp.choices[0].message.tool_calls)
        self.assertEqual(resp.usage.prompt_tokens, 5)
        self.assertEqual(resp.usage.completion_tokens, 2)

    def test_tool_call_response_shape(self):
        result_text = '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"/x\\"}"}</tool_call>'
        lines = [json.dumps({"type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
                             "stop_reason": "end_turn", "result": result_text, "usage": {}, "total_cost_usd": 0.0})]
        client = self._client_with_output(lines)
        resp = client.chat.completions.create(model="claude-opus-4-8", messages=[{"role": "user", "content": "read /x"}])
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "read_file")

    def test_quota_error_propagates(self):
        from agent.claude_cli_client import ClaudeCliQuotaError
        lines = [json.dumps({"type": "result", "is_error": True, "api_error_status": 400,
                             "result": "You're out of extra usage. Add more at claude.ai/settings/usage and keep going."})]
        client = self._client_with_output(lines)
        with self.assertRaises(ClaudeCliQuotaError):
            client.chat.completions.create(model="claude-opus-4-8", messages=[{"role": "user", "content": "hi"}])

    def test_accepts_unknown_kwargs(self):
        client = ClaudeCliClient(model="claude-opus-4-8", max_retries=0, default_headers={}, api_key=None, base_url=None)
        self.assertIsNotNone(client)

    def test_model_normalization(self):
        from agent.claude_cli_client import _normalize_model
        self.assertEqual(_normalize_model("anthropic/claude-opus-4-8"), "claude-opus-4-8")
        self.assertEqual(_normalize_model("claude-cli/claude-opus-4-8"), "claude-opus-4-8")
        self.assertEqual(_normalize_model("claude-opus-4-8"), "claude-opus-4-8")
        self.assertEqual(_normalize_model(None), "claude-opus-4-8")

    def test_subprocess_env_scrubs_anthropic_api_key(self):
        with patch.dict(_os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api-should-be-removed"}, clear=False):
            env = _build_subprocess_env()
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertIn("HOME", env)

    def test_subprocess_env_scrubs_alternate_anthropic_auth_vars(self):
        # The documented alternate / custom-gateway auth path must also be
        # scrubbed so inference can never be routed through a metered or custom
        # endpoint, undercutting the subscription-OAuth-only billing guarantee.
        with patch.dict(
            _os.environ,
            {
                "ANTHROPIC_API_KEY": "sk-ant-api-should-be-removed",
                "ANTHROPIC_AUTH_TOKEN": "token-should-be-removed",
                "ANTHROPIC_BASE_URL": "https://metered.example.com",
                "ANTHROPIC_TOKEN": "hermes-managed-oauth-keep-me",
            },
            clear=False,
        ):
            env = _build_subprocess_env()
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
            self.assertNotIn("ANTHROPIC_BASE_URL", env)
            # ANTHROPIC_TOKEN is Hermes-managed OAuth and must be left intact.
            self.assertEqual(env.get("ANTHROPIC_TOKEN"), "hermes-managed-oauth-keep-me")
            self.assertIn("HOME", env)


def _result_line(text, input_tokens=0, output_tokens=0, cache_read=0):
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "stop_reason": "end_turn",
        "result": text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
        },
        "total_cost_usd": 0.0,
    })


class RepairRetryTests(unittest.TestCase):
    """Malformed <tool_call> blocks trigger exactly one repair retry attempt.

    ``_run_claude`` is mocked with a side_effect list so each call in the
    sequence returns fixture stream-json stdout lines; invocation count is
    asserted directly on the mock.
    """

    def _client(self):
        return ClaudeCliClient(model="claude-opus-4-8")

    def test_malformed_only_then_valid_repair_uses_retry_result(self):
        # A block with a literal (raw) newline control char inside a JSON
        # string value: json.loads rejects this. No valid calls on the first
        # attempt -> exactly one repair retry -> retry succeeds.
        original_text = (
            'Sure, I will do that now.\n'
            '<tool_call>{"name": "write_file", "arguments": '
            '"{\\"content\\": \\"line1\nline2\\"}"}</tool_call>'
        )
        retry_text = '<tool_call>{"name": "write_file", "arguments": "{\\"content\\": \\"line1 line2\\"}"}</tool_call>'
        first_lines = [_result_line(original_text, input_tokens=10, output_tokens=20, cache_read=1)]
        second_lines = [_result_line(retry_text, input_tokens=5, output_tokens=7, cache_read=2)]

        client = self._client()
        mock_run = MagicMock(side_effect=[first_lines, second_lines])
        client._run_claude = mock_run

        with self.assertLogs("agent.claude_cli_client", level="INFO") as cm:
            resp = client.chat.completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "write something"}],
            )

        self.assertEqual(mock_run.call_count, 2)
        # The second call's prompt must contain the original assistant text
        # and the correction sentence.
        second_call_prompt = mock_run.call_args_list[1].args[0]
        self.assertIn("Sure, I will do that now.", second_call_prompt)
        self.assertIn("malformed <tool_call>", second_call_prompt)
        # Success is logged at INFO level.
        self.assertIn("repair retry succeeded", "\n".join(cm.output))

        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "write_file")
        # Usage tokens must be summed across both runs.
        self.assertEqual(resp.usage.prompt_tokens, 15)
        self.assertEqual(resp.usage.completion_tokens, 27)
        self.assertEqual(resp.usage.prompt_tokens_details.cached_tokens, 3)

    def test_malformed_only_retry_also_malformed_falls_back_to_original(self):
        original_text = "I've made the change.\n<tool_call>{not valid json}</tool_call>"
        retry_text = "Still broken.\n<tool_call>{also not valid}</tool_call>"
        first_lines = [_result_line(original_text, input_tokens=10, output_tokens=20, cache_read=1)]
        second_lines = [_result_line(retry_text, input_tokens=5, output_tokens=7, cache_read=2)]

        client = self._client()
        mock_run = MagicMock(side_effect=[first_lines, second_lines])
        client._run_claude = mock_run

        with self.assertLogs("agent.claude_cli_client", level="WARNING") as cm:
            resp = client.chat.completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "do something"}],
            )

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        # The RAW malformed block text from the ORIGINAL reply must be visible.
        self.assertIn("not valid json", resp.choices[0].message.content)
        self.assertFalse(resp.choices[0].message.tool_calls)
        # Both the malformed-block warning and the repair-failed warning fire.
        joined_logs = "\n".join(cm.output)
        self.assertIn("malformed", joined_logs.lower())
        self.assertIn("repair attempt failed", joined_logs.lower())
        # Retry tokens are real cost: usage is summed even on a failed repair.
        self.assertEqual(resp.usage.prompt_tokens, 15)
        self.assertEqual(resp.usage.completion_tokens, 27)
        self.assertEqual(resp.usage.prompt_tokens_details.cached_tokens, 3)

    def test_repair_run_exception_falls_back_to_original(self):
        # A repair run that raises (e.g. subprocess timeout) must never
        # destroy the original result: no exception propagates, the original
        # text (raw malformed block visible) is delivered, and the raised
        # warning is logged.
        original_text = "Done!\n<tool_call>{not valid json}</tool_call>"
        first_lines = [_result_line(original_text, input_tokens=10, output_tokens=20)]

        client = self._client()
        mock_run = MagicMock(side_effect=[first_lines, ClaudeCliError("timeout")])
        client._run_claude = mock_run

        with self.assertLogs("agent.claude_cli_client", level="WARNING") as cm:
            resp = client.chat.completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "do something"}],
            )

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertIn("not valid json", resp.choices[0].message.content)
        self.assertFalse(resp.choices[0].message.tool_calls)
        # Only the original run's usage is counted (the raising run returned
        # no parseable usage).
        self.assertEqual(resp.usage.prompt_tokens, 10)
        self.assertEqual(resp.usage.completion_tokens, 20)
        joined_logs = "\n".join(cm.output)
        self.assertIn("repair attempt raised", joined_logs)
        self.assertIn("timeout", joined_logs)

    def test_image_turn_malformed_skips_repair(self):
        # An image-bearing turn must NOT attempt repair: the image scratch dir
        # is already cleaned up and the repair run has tools disabled, so the
        # retry model would be told to Read a file it can't access. The
        # original fallback is delivered after exactly one subprocess run.
        original_text = "I see the image.\n<tool_call>{not valid json}</tool_call>"
        first_lines = [_result_line(original_text)]

        client = self._client()
        mock_run = MagicMock(side_effect=[first_lines])
        client._run_claude = mock_run

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe and act"},
                    {"type": "image_url", "image_url": {"url": _PNG_1PX_DATA_URL}},
                ],
            }
        ]
        with self.assertLogs("agent.claude_cli_client", level="WARNING") as cm:
            resp = client.chat.completions.create(
                model="claude-opus-4-8", messages=messages,
            )

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertIn("not valid json", resp.choices[0].message.content)
        joined_logs = "\n".join(cm.output)
        self.assertIn("skipping repair retry for image-bearing turn", joined_logs)

    def test_malformed_and_valid_mixed_no_repair_run(self):
        text = (
            "Partial success.\n"
            "<tool_call>{not valid json}</tool_call>\n"
            '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"/x\\"}"}</tool_call>'
        )
        lines = [_result_line(text)]
        client = self._client()
        mock_run = MagicMock(side_effect=[lines])
        client._run_claude = mock_run

        with self.assertLogs("agent.claude_cli_client", level="WARNING"):
            resp = client.chat.completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "do two things"}],
            )

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "read_file")
        self.assertIn("not valid json", resp.choices[0].message.content)

    def test_no_malformed_no_repair_run(self):
        lines = [_result_line("just a normal reply", input_tokens=3, output_tokens=4)]
        client = self._client()
        mock_run = MagicMock(side_effect=[lines])
        client._run_claude = mock_run

        resp = client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
        )

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.content, "just a normal reply")

    def test_unclosed_tag_triggers_repair(self):
        original_text = 'One moment.\n<tool_call>{"name": "cronjob", "argu'
        retry_text = '<tool_call>{"name": "cronjob", "arguments": "{\\"action\\": \\"list\\"}"}</tool_call>'
        first_lines = [_result_line(original_text)]
        second_lines = [_result_line(retry_text)]

        client = self._client()
        mock_run = MagicMock(side_effect=[first_lines, second_lines])
        client._run_claude = mock_run

        resp = client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "list my jobs"}],
        )

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "cronjob")


class ClientAwaitableCreateTests(unittest.TestCase):
    """``chat.completions.create`` must work BOTH awaited and un-awaited.

    The primary conversation loop consumes the result directly (no ``await``),
    but the auxiliary/async path — used by ``vision_analyze`` via
    ``auxiliary_client.async_call_llm`` / ``_retry_same_provider_async`` — does
    ``response = await client.chat.completions.create(**kwargs)``
    (see ``agent/auxiliary_client.py`` and ``tools/vision_tools.py``).  A bare
    ``SimpleNamespace`` return value crashes that path with
    ``TypeError: object types.SimpleNamespace can't be used in 'await'
    expression``, which is the image-reading failure on the claude-cli provider.
    """

    def _client_with_output(self, lines):
        client = ClaudeCliClient(model="claude-opus-4-8")
        client._run_claude = lambda prompt, system_prompt, model, timeout, image_dir=None: list(lines)
        return client

    _LINES = [
        json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "api_error_status": None, "stop_reason": "end_turn",
            "result": "a cat on a mat",
            "usage": {"input_tokens": 9, "output_tokens": 4},
            "total_cost_usd": 0.001,
        })
    ]

    def test_create_result_is_awaitable_for_async_aux_path(self):
        # Reproduces the vision-tool crash: the aux/async path awaits create().
        client = self._client_with_output(self._LINES)

        async def _call():
            return await client.chat.completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "describe this image"}],
            )

        resp = asyncio.run(_call())
        self.assertEqual(resp.choices[0].message.content, "a cat on a mat")
        self.assertEqual(resp.usage.prompt_tokens, 9)

    def test_create_result_still_usable_without_await(self):
        # Regression guard: the primary path consumes the result directly,
        # reading attributes off it without awaiting.  Must keep working.
        client = self._client_with_output(self._LINES)
        resp = client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(resp.choices[0].message.content, "a cat on a mat")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.usage.completion_tokens, 4)


_SUCCESS_RESULT_LINE = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "api_error_status": None,
    "stop_reason": "end_turn",
    "result": "ok",
    "usage": {},
    "total_cost_usd": 0.0,
})


class RunClaudeSubprocessTests(unittest.TestCase):
    """Exercise the real ``_run_claude`` subprocess body (no real subprocess).

    The ClientFacadeTests stub ``_run_claude`` wholesale, so the riskiest code —
    the subprocess spawn, timeout reap, and returncode handling — is otherwise
    untested. These patch ``subprocess.Popen`` at the module-qualified name and
    shape a ``MagicMock`` proc to match the exact call/attribute sequence
    ``_run_claude`` uses.
    """

    def _client(self):
        return ClaudeCliClient(model="claude-opus-4-8")

    def test_timeout_kills_and_reaps(self):
        # Highest-value test: guards the zombie-reap guarantee. On timeout,
        # _run_claude must kill the proc and call communicate() a SECOND time to
        # reap it, then raise a "timed out" ClaudeCliError.
        client = self._client()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="claude", timeout=900.0),
            ("", ""),
        ]
        with patch("agent.claude_cli_client.subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_proc
            with self.assertRaises(ClaudeCliError) as cm:
                client._run_claude("p", "sys", "claude-opus-4-8", 900.0)
        self.assertIn("timed out", str(cm.exception))
        mock_proc.kill.assert_called_once()
        self.assertEqual(mock_proc.communicate.call_count, 2)

    def test_file_not_found_raises_auth_error(self):
        client = self._client()
        with patch("agent.claude_cli_client.subprocess.Popen") as mock_popen:
            mock_popen.side_effect = FileNotFoundError()
            with self.assertRaises(ClaudeCliAuthError) as cm:
                client._run_claude("p", "sys", "claude-opus-4-8", 900.0)
        self.assertIn("not found", str(cm.exception))

    def test_nonzero_returncode_empty_stdout_raises_with_stderr(self):
        client = self._client()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "some stderr boom")
        mock_proc.returncode = 2
        with patch("agent.claude_cli_client.subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_proc
            with self.assertRaises(ClaudeCliError) as cm:
                client._run_claude("p", "sys", "claude-opus-4-8", 900.0)
        self.assertIn("boom", str(cm.exception))

    def test_nonzero_returncode_with_parseable_stdout_returns_lines(self):
        # rc != 0 but stdout carries a parseable success result line: _run_claude
        # must NOT raise and must defer to the parser by returning the lines.
        client = self._client()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (_SUCCESS_RESULT_LINE + "\n", "")
        mock_proc.returncode = 1
        with patch("agent.claude_cli_client.subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_proc
            lines = client._run_claude("p", "sys", "claude-opus-4-8", 900.0)
        self.assertEqual(lines, [_SUCCESS_RESULT_LINE])
        # Parser later yields "ok" from these lines.
        self.assertEqual(_parse_stream_json_lines(lines).text, "ok")

    def test_success_clears_active_process(self):
        client = self._client()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (_SUCCESS_RESULT_LINE + "\n", "")
        mock_proc.returncode = 0
        with patch("agent.claude_cli_client.subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_proc
            client._run_claude("p", "sys", "claude-opus-4-8", 900.0)
        self.assertIsNone(client._active_process)


# 1x1 transparent PNG, base64 — a minimal real image payload.
_PNG_1PX_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9"
    "awAAAABJRU5ErkJggg=="
)
_PNG_1PX_DATA_URL = f"data:image/png;base64,{_PNG_1PX_B64}"


class ImageExtractionTests(unittest.TestCase):
    """``_extract_image_data_urls`` pulls inline base64 images out of messages.

    The auxiliary vision path (``vision_analyze``) sends the image as an OpenAI
    ``image_url`` content part whose ``url`` is a ``data:<mime>;base64,...`` URL.
    The ``claude`` CLI cannot ingest inline base64 in ``--print`` mode, so the
    client must recover the raw bytes to materialise them as a file the CLI's
    Read tool can open.
    """

    def test_extracts_base64_image_part(self):
        from agent.claude_cli_client import _extract_image_data_urls
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": _PNG_1PX_DATA_URL}},
                ],
            }
        ]
        images = _extract_image_data_urls(messages)
        self.assertEqual(len(images), 1)
        mime, raw = images[0]
        self.assertEqual(mime, "image/png")
        self.assertEqual(raw, base64.b64decode(_PNG_1PX_B64))

    def test_no_images_returns_empty(self):
        from agent.claude_cli_client import _extract_image_data_urls
        messages = [{"role": "user", "content": "just text"}]
        self.assertEqual(_extract_image_data_urls(messages), [])


class VisionRoutingTests(unittest.TestCase):
    """When a turn carries an image, the client materialises it to a file and
    invokes the CLI with Read-tool access + the file path in the prompt.

    This is the path-based vision mechanism: ``claude -p`` reads the image via
    its own Read tool (``--allowedTools Read`` + ``--add-dir``) rather than
    receiving inline base64 (which it cannot accept)."""

    def _client(self):
        return ClaudeCliClient(model="claude-opus-4-8")

    def test_image_turn_invokes_run_claude_with_image_dir_and_path(self):
        client = self._client()
        captured = {}

        def _fake_run(prompt, system_prompt, model, timeout, image_dir=None):
            captured["prompt"] = prompt
            captured["image_dir"] = image_dir
            return [json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "api_error_status": None, "stop_reason": "end_turn",
                "result": "a tiny image", "usage": {},
            })]

        client._run_claude = _fake_run
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe the image"},
                    {"type": "image_url", "image_url": {"url": _PNG_1PX_DATA_URL}},
                ],
            }
        ]
        resp = client.chat.completions.create(model="claude-opus-4-8", messages=messages)
        self.assertEqual(resp.choices[0].message.content, "a tiny image")
        # _run_claude must have been told where the image lives...
        self.assertIsNotNone(captured.get("image_dir"))
        # ...and a concrete .png file must exist in that dir during the call,
        # with the path surfaced in the prompt so the model knows to Read it.
        self.assertIn(".png", captured["prompt"])
        self.assertIn(str(captured["image_dir"]), captured["prompt"])

    def test_text_only_turn_passes_no_image_dir(self):
        client = self._client()
        captured = {}

        def _fake_run(prompt, system_prompt, model, timeout, image_dir=None):
            captured["image_dir"] = image_dir
            return [json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "api_error_status": None, "stop_reason": "end_turn",
                "result": "hi", "usage": {},
            })]

        client._run_claude = _fake_run
        client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertIsNone(captured["image_dir"])


class RunClaudeImageArgsTests(unittest.TestCase):
    """``_run_claude`` grants scoped Read access only when an image dir is set."""

    def _client(self):
        client = ClaudeCliClient(model="claude-opus-4-8")
        return client

    def _run_and_capture_cmd(self, image_dir):
        client = self._client()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            json.dumps({"type": "result", "subtype": "success", "is_error": False,
                        "api_error_status": None, "stop_reason": "end_turn",
                        "result": "ok", "usage": {}}) + "\n",
            "",
        )
        mock_proc.returncode = 0
        with patch("agent.claude_cli_client.subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_proc
            client._run_claude("p", "sys", "claude-opus-4-8", 900.0, image_dir=image_dir)
            (args, _kwargs) = mock_popen.call_args
        return list(args[0])

    def test_image_dir_enables_read_tool_and_add_dir(self):
        cmd = self._run_and_capture_cmd("/tmp/some-img-dir")
        self.assertIn("--allowedTools", cmd)
        self.assertIn("Read", cmd)
        self.assertIn("--add-dir", cmd)
        self.assertIn("/tmp/some-img-dir", cmd)
        # The blanket tool-disable must NOT be present for image turns
        # (it would suppress Read).
        joined = " ".join(cmd)
        self.assertNotIn('--tools  ', f" {joined} ")  # no `--tools ""` pair

    def test_no_image_dir_keeps_tools_disabled(self):
        cmd = self._run_and_capture_cmd(None)
        self.assertIn("--tools", cmd)
        # Read tool must NOT be granted on a plain text turn.
        self.assertNotIn("--allowedTools", cmd)
        self.assertNotIn("--add-dir", cmd)


if __name__ == "__main__":
    unittest.main()
