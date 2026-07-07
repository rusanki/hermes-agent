"""claude-sdk provider profile — drives the Claude Agent SDK (Python) in-process.

Sibling of claude-cli: native tool_use + SDK-managed sessions instead of the
<tool_call> text-markup protocol. Subscription OAuth, external-process runtime.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeSdkProfile(ProviderProfile):
    """Claude Agent SDK — external subprocess (via SDK), no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled out-of-band by the CLI/SDK."""
        return None


claude_sdk = ClaudeSdkProfile(
    name="claude-sdk",
    aliases=("claude-agent-sdk", "claude_sdk"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="claude-sdk://local",
    auth_type="external_process",
    default_aux_model="claude-haiku-4-5",
)

register_provider(claude_sdk)
