import json

import pytest


@pytest.fixture
def script_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME with the registry directory layout, plus a stubbed
    session-user resolver on the TOOL module so tests can pick the caller uid."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "scripts" / "staging").mkdir(parents=True)
    (hermes_home / "scripts" / "approved").mkdir(parents=True)
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.script_registry as sr
    monkeypatch.setattr(sr, "_get_hermes_home", lambda: hermes_home, raising=False)

    import tools.cron_script_tools as cst

    # Mutable holder so a test can flip the "current caller" mid-test.
    state = {"uid": ""}
    monkeypatch.setattr(
        cst, "get_session_user_id", lambda: state["uid"], raising=True
    )
    return hermes_home, cst, state


def test_stage_list_show_roundtrip(script_env):
    hermes_home, cst, state = script_env
    state["uid"] = "U_ADMIN"
    content = "print('hello world')\n# a longer body\n" * 10

    staged = json.loads(cst.cron_script(action="stage", name="job.py", content=content))
    assert staged["staged"] is True
    assert staged["pending_approval"] is True
    assert staged["name"] == "job.py"
    sha = staged["sha256"]
    assert sha and len(sha) == 64

    listed = json.loads(cst.cron_script(action="list_pending"))
    names = [p["name"] for p in listed["pending"]]
    assert "job.py" in names

    shown = json.loads(cst.cron_script(action="show", name="job.py"))
    assert shown["name"] == "job.py"
    assert shown["content"] == content  # full content, not truncated
    assert shown["sha256"] == sha


def test_approve_records_verified_uid(script_env):
    hermes_home, cst, state = script_env

    state["uid"] = "U_ADMIN"
    staged = json.loads(cst.cron_script(action="stage", name="j.py", content="x=1\n"))
    sha = staged["sha256"]

    state["uid"] = "U_SUPER"
    approved = json.loads(
        cst.cron_script_approve(action="approve", name="j.py", sha256=sha)
    )
    assert approved["approved"] is True
    assert approved["name"] == "j.py"

    reg = json.loads(
        (hermes_home / "cron" / "approved_scripts.json").read_text(encoding="utf-8")
    )
    assert reg["j.py"]["approved_by"] == "U_SUPER"


def test_approve_stale_sha_returns_error(script_env):
    hermes_home, cst, state = script_env
    state["uid"] = "U_ADMIN"
    json.loads(cst.cron_script(action="stage", name="j.py", content="x=1\n"))

    state["uid"] = "U_SUPER"
    out = json.loads(
        cst.cron_script_approve(action="approve", name="j.py", sha256="deadbeef")
    )
    assert "error" in out
    # nothing got approved
    assert not (hermes_home / "scripts" / "approved" / "j.py").exists()


def test_revoke_roundtrip(script_env):
    hermes_home, cst, state = script_env
    state["uid"] = "U_ADMIN"
    staged = json.loads(cst.cron_script(action="stage", name="j.py", content="x=1\n"))

    state["uid"] = "U_SUPER"
    json.loads(cst.cron_script_approve(action="approve", name="j.py", sha256=staged["sha256"]))

    revoked = json.loads(cst.cron_script_approve(action="revoke", name="j.py"))
    assert revoked["revoked"] is True
    assert revoked["name"] == "j.py"
    assert not (hermes_home / "scripts" / "approved" / "j.py").exists()


def test_stage_without_name_returns_error(script_env):
    _, cst, state = script_env
    state["uid"] = "U_ADMIN"
    out = json.loads(cst.cron_script(action="stage", content="x=1\n"))
    assert "error" in out


def test_show_without_name_returns_error(script_env):
    _, cst, state = script_env
    out = json.loads(cst.cron_script(action="show"))
    assert "error" in out


def test_approve_without_name_returns_error(script_env):
    _, cst, state = script_env
    out = json.loads(cst.cron_script_approve(action="approve", sha256="abc"))
    assert "error" in out


def test_unknown_action_returns_error(script_env):
    _, cst, state = script_env
    out = json.loads(cst.cron_script(action="frobnicate"))
    assert "error" in out
    out2 = json.loads(cst.cron_script_approve(action="frobnicate"))
    assert "error" in out2


def test_valueerror_does_not_escape(script_env):
    """A registry-level ValueError (e.g. path escape) becomes an error dict, not a raise."""
    _, cst, state = script_env
    state["uid"] = "U_ADMIN"
    out = json.loads(cst.cron_script(action="stage", name="../evil.py", content="x=1\n"))
    assert "error" in out
