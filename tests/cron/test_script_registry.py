import pytest


@pytest.fixture
def reg_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "scripts" / "staging").mkdir(parents=True)
    (hermes_home / "scripts" / "approved").mkdir(parents=True)
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import cron.script_registry as sr
    monkeypatch.setattr(sr, "_get_hermes_home", lambda: hermes_home, raising=False)
    return hermes_home


def test_sha256_of_content_matches_hashlib(reg_env):
    import hashlib
    import cron.script_registry as sr
    content = "print('hi')\n"
    assert sr.sha256_of(content) == hashlib.sha256(content.encode()).hexdigest()


def test_sha256_of_accepts_bytes(reg_env):
    import hashlib
    import cron.script_registry as sr
    assert sr.sha256_of(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_stage_rejects_path_escape_name(reg_env):
    import cron.script_registry as sr
    with pytest.raises(ValueError):
        sr.stage_script("../evil.py", "print(1)", requested_by="U1")
    with pytest.raises(ValueError):
        sr.stage_script("/abs/evil.py", "print(1)", requested_by="U1")


def test_stage_rejects_empty_name(reg_env):
    import cron.script_registry as sr
    with pytest.raises(ValueError):
        sr.stage_script("", "print(1)", requested_by="U1")
