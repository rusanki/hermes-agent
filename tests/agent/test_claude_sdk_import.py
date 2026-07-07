def test_sdk_imports():
    import claude_agent_sdk as c

    for name in (
        "ClaudeSDKClient",
        "ClaudeAgentOptions",
        "tool",
        "create_sdk_mcp_server",
        "AssistantMessage",
        "TextBlock",
        "ToolUseBlock",
        "ResultMessage",
        "PermissionResultAllow",
        "PermissionResultDeny",
    ):
        assert hasattr(c, name), f"claude_agent_sdk missing {name}"
