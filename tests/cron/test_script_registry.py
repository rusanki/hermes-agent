import pytest

# These are direct unit tests of cron.script_registry (including is_approved
# itself), so they must run against the REAL is_approved — opt out of the
# autouse approval-bypass fixture in conftest.py (Task 7).
pytestmark = pytest.mark.real_script_pin


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


def test_stage_writes_file_and_pending_record(reg_env):
    import cron.script_registry as sr
    res = sr.stage_script("fetch_x.py", "print('x')\n", requested_by="U_ADMIN")
    assert res["name"] == "fetch_x.py"
    assert res["sha256"] == sr.sha256_of("print('x')\n")
    staged = reg_env / "scripts" / "staging" / "fetch_x.py"
    assert staged.read_text() == "print('x')\n"
    pending = sr.list_pending()
    names = {p["name"] for p in pending}
    assert "fetch_x.py" in names
    rec = next(p for p in pending if p["name"] == "fetch_x.py")
    assert rec["sha256"] == res["sha256"]
    assert rec["requested_by"] == "U_ADMIN"
    assert "authored_at" in rec and "preview" in rec


def test_stage_rejects_oversized_content(reg_env):
    import cron.script_registry as sr
    big = "x" * (64 * 1024 + 1)
    with pytest.raises(ValueError):
        sr.stage_script("big.py", big, requested_by="U1")


def test_stage_rejects_when_pending_queue_full(reg_env):
    import cron.script_registry as sr
    for i in range(20):
        sr.stage_script(f"s{i}.py", "print(1)\n", requested_by="U1")
    with pytest.raises(ValueError):
        sr.stage_script("overflow.py", "print(1)\n", requested_by="U1")


def test_restage_same_name_updates_not_duplicates(reg_env):
    import cron.script_registry as sr
    sr.stage_script("a.py", "print(1)\n", requested_by="U1")
    sr.stage_script("a.py", "print(2)\n", requested_by="U1")
    pending = [p for p in sr.list_pending() if p["name"] == "a.py"]
    assert len(pending) == 1
    assert pending[0]["sha256"] == sr.sha256_of("print(2)\n")


def test_restage_when_queue_full_still_allowed(reg_env):
    # Re-staging an EXISTING name must not be rejected by the queue cap.
    import cron.script_registry as sr
    for i in range(20):
        sr.stage_script(f"s{i}.py", "print(1)\n", requested_by="U1")
    # s0.py already exists -> updating it is allowed even at capacity
    res = sr.stage_script("s0.py", "print('updated')\n", requested_by="U1")
    assert res["sha256"] == sr.sha256_of("print('updated')\n")


def test_approve_pins_content_and_copies_to_approved(reg_env):
    import cron.script_registry as sr
    staged = sr.stage_script("f.py", "print('f')\n", requested_by="U1")
    res = sr.approve_script("f.py", staged["sha256"], approver_uid="U_SUPER")
    assert res["approved"] is True
    approved_file = reg_env / "scripts" / "approved" / "f.py"
    assert approved_file.read_text() == "print('f')\n"
    reg = sr._read_json(sr._approved_registry_path())
    assert reg["f.py"]["sha256"] == staged["sha256"]
    assert reg["f.py"]["approved_by"] == "U_SUPER"
    assert "approved_at" in reg["f.py"]


def test_approve_clears_pending_record(reg_env):
    import cron.script_registry as sr
    staged = sr.stage_script("f.py", "print('f')\n", requested_by="U1")
    sr.approve_script("f.py", staged["sha256"], approver_uid="U_SUPER")
    assert all(p["name"] != "f.py" for p in sr.list_pending())


def test_approve_rejects_stale_sha(reg_env):
    import cron.script_registry as sr
    sr.stage_script("f.py", "print('f')\n", requested_by="U1")
    with pytest.raises(ValueError):
        sr.approve_script("f.py", "deadbeef", approver_uid="U_SUPER")


def test_approve_rehashes_staging_file_not_pending_record(reg_env):
    # TOCTOU: staging file changed between stage and approve -> reject even if
    # the caller passes the ORIGINAL sha (pin must re-hash the file on disk).
    import cron.script_registry as sr
    staged = sr.stage_script("f.py", "print('orig')\n", requested_by="U1")
    (reg_env / "scripts" / "staging" / "f.py").write_text("print('EVIL')\n")
    with pytest.raises(ValueError):
        sr.approve_script("f.py", staged["sha256"], approver_uid="U_SUPER")


def test_approve_missing_staged_raises(reg_env):
    import cron.script_registry as sr
    with pytest.raises(ValueError):
        sr.approve_script("nope.py", "abc", approver_uid="U_SUPER")


def test_is_approved_true_only_on_matching_hash(reg_env):
    import cron.script_registry as sr
    sr.stage_script("f.py", "print('f')\n", requested_by="U1")
    sr.approve_script("f.py", sr.sha256_of("print('f')\n"), approver_uid="U_SUPER")
    approved_file = reg_env / "scripts" / "approved" / "f.py"
    assert sr.is_approved(str(approved_file)) is True
    approved_file.write_text("print('tampered')\n")
    assert sr.is_approved(str(approved_file)) is False


def test_is_approved_false_for_unknown_or_missing(reg_env):
    import cron.script_registry as sr
    assert sr.is_approved(str(reg_env / "scripts" / "approved" / "nope.py")) is False


def test_is_approved_false_for_path_outside_approved_dir(reg_env):
    # A path that resolves outside approved/ must never be considered approved.
    import cron.script_registry as sr
    outside = reg_env / "scripts" / "staging" / "f.py"
    outside.write_text("print('x')\n")
    assert sr.is_approved(str(outside)) is False


def test_revoke_removes_pin(reg_env):
    import cron.script_registry as sr
    sr.stage_script("f.py", "print('f')\n", requested_by="U1")
    sr.approve_script("f.py", sr.sha256_of("print('f')\n"), approver_uid="U_SUPER")
    approved_file = reg_env / "scripts" / "approved" / "f.py"
    assert sr.is_approved(str(approved_file)) is True
    out = sr.revoke_script("f.py", revoked_by="U_SUPER")
    assert out["revoked"] is True
    assert sr.is_approved(str(approved_file)) is False


def test_revoke_unknown_reports_false(reg_env):
    import cron.script_registry as sr
    out = sr.revoke_script("never.py", revoked_by="U_SUPER")
    assert out["revoked"] is False


def test_show_staged_returns_full_content_and_sha(reg_env):
    import cron.script_registry as sr
    sr.stage_script("f.py", "print('full body here')\n", requested_by="U1")
    shown = sr.show_staged("f.py")
    assert shown["content"] == "print('full body here')\n"
    assert shown["sha256"] == sr.sha256_of("print('full body here')\n")


def test_show_staged_missing_raises(reg_env):
    import cron.script_registry as sr
    with pytest.raises(ValueError):
        sr.show_staged("nope.py")


def test_is_approved_fail_closed_on_bad_type(reg_env):
    # Any error (e.g. a non-path type) must fail closed to False, never raise.
    import cron.script_registry as sr
    assert sr.is_approved(None) is False
    assert sr.is_approved(12345) is False
