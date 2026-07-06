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
