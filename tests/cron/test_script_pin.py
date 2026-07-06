"""Run-time enforcement of the cron script approval pin (Task 7).

These tests exercise the REAL ``cron.script_registry.is_approved`` gate that
``cron/scheduler.py``'s ``_run_job_script`` consults, so they opt OUT of the
autouse ``_approve_all_cron_scripts_by_default`` bypass fixture (see
``tests/cron/conftest.py``) via the ``real_script_pin`` marker.
"""
import pytest

import cron.scheduler as scheduler
import cron.script_registry as sr

pytestmark = pytest.mark.real_script_pin


@pytest.fixture
def sched_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "scripts" / "staging").mkdir(parents=True)
    (hermes_home / "scripts" / "approved").mkdir(parents=True)
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(scheduler, "_hermes_home", hermes_home, raising=False)
    monkeypatch.setattr(sr, "_get_hermes_home", lambda: hermes_home, raising=False)
    return hermes_home


def test_run_blocked_for_unapproved_script(sched_env):
    script = sched_env / "scripts" / "foo.py"
    script.write_text("print('hi')\n")
    ok, out = scheduler._run_job_script(str(script))
    assert ok is False
    assert "approv" in out.lower()


def test_run_allowed_for_approved_script(sched_env):
    content = "print('ok-marker')\n"
    staged = sr.stage_script("foo.py", content, requested_by="U1")
    sr.approve_script("foo.py", staged["sha256"], approver_uid="U_SUPER")
    approved = sched_env / "scripts" / "approved" / "foo.py"
    ok, out = scheduler._run_job_script(str(approved))
    assert ok is True
    assert "ok-marker" in out


def test_run_blocked_for_edited_approved_script(sched_env):
    content = "print('ok')\n"
    staged = sr.stage_script("foo.py", content, requested_by="U1")
    sr.approve_script("foo.py", staged["sha256"], approver_uid="U_SUPER")
    approved = sched_env / "scripts" / "approved" / "foo.py"
    approved.write_text("print('TAMPERED')\n")
    ok, out = scheduler._run_job_script(str(approved))
    assert ok is False


def test_run_blocked_for_missing_script(sched_env):
    ok, out = scheduler._run_job_script(str(sched_env / "scripts" / "approved" / "nope.py"))
    assert ok is False
