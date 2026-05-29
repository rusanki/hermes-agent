"""Unit tests for the claude-cli provider client (no real subprocess)."""
from __future__ import annotations
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
        client._run_claude = lambda prompt, system_prompt, model, timeout: list(lines)
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


if __name__ == "__main__":
    unittest.main()
