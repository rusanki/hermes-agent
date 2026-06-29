import json
from agent import shell_hooks


def test_serialize_payload_carries_user_id_in_extra():
    out = shell_hooks._serialize_payload("pre_tool_call",
        {"tool_name":"terminal","args":{"cmd":"ls"},"session_id":"s1","user_id":"U123"})
    assert json.loads(out)["extra"].get("user_id") == "U123"


from unittest.mock import patch
from hermes_cli import plugins


def test_block_message_forwards_user_id_to_hooks():
    captured = {}
    def _fake_invoke(event, **kwargs):
        captured.update(kwargs); return []
    with patch.object(plugins, "invoke_hook", _fake_invoke):
        plugins.get_pre_tool_call_block_message("terminal", {"cmd":"ls"}, session_id="s1", user_id="U123")
    assert captured.get("user_id") == "U123"
