import unittest


class ClaudeCliProfileTests(unittest.TestCase):
    def test_profile_registers_and_discoverable(self):
        import providers
        # get_provider_profile lazily triggers discovery, which scans the
        # bundled plugins/model-providers/ dir (handles dash-named dirs).
        prof = providers.get_provider_profile("claude-cli")
        self.assertIsNotNone(prof)
        self.assertEqual(prof.name, "claude-cli")
        self.assertEqual(prof.auth_type, "external_process")
        self.assertTrue(str(prof.base_url).startswith("claude-cli://"))

    def test_aliases_resolve(self):
        import providers
        for alias in ("claude-code-cli", "claude_subscription"):
            prof = providers.get_provider_profile(alias)
            self.assertIsNotNone(prof, f"alias {alias} did not resolve")
            self.assertEqual(prof.name, "claude-cli")
