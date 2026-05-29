"""claude-cli provider profile — drives `claude -p` as a subprocess (Claude subscription OAuth)."""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCliProfile(ProviderProfile):
    """Claude CLI — external subprocess, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled out-of-band by the CLI."""
        return None


claude_cli = ClaudeCliProfile(
    name="claude-cli",
    aliases=("claude-code-cli", "claude_subscription"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="claude-cli://local",
    auth_type="external_process",
)

register_provider(claude_cli)
