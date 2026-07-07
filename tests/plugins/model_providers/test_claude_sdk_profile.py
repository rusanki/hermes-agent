import unittest


class ClaudeSdkProfileTests(unittest.TestCase):
    def test_profile_registers_and_discoverable(self):
        import providers
        # get_provider_profile lazily triggers discovery, which scans the
        # bundled plugins/model-providers/ dir (handles dash-named dirs).
        prof = providers.get_provider_profile("claude-sdk")
        self.assertIsNotNone(prof)
        self.assertEqual(prof.name, "claude-sdk")
        self.assertEqual(prof.api_mode, "chat_completions")
        self.assertEqual(str(prof.base_url), "claude-sdk://local")
        self.assertEqual(prof.auth_type, "external_process")

    def test_aliases_resolve(self):
        import providers
        for alias in ("claude-agent-sdk", "claude_sdk"):
            prof = providers.get_provider_profile(alias)
            self.assertIsNotNone(prof, f"alias {alias} did not resolve")
            self.assertEqual(prof.name, "claude-sdk")
