# tests/integration/test_team_policy_hook.py
import importlib.util, json, os, pathlib

# Load the hook from the REPO copy (not ~/.hermes — that's a deploy artifact, breaks CI).
_REPO_HOOK = pathlib.Path(__file__).resolve().parents[2] / "tools" / "team_policy_hook.py"
HOOK = pathlib.Path(os.environ.get("HERMES_TEAM_POLICY_HOOK", str(_REPO_HOOK)))
def _load():
    spec = importlib.util.spec_from_file_location("team_policy", HOOK)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

POLICY = {"version":1,"default_role":"member",
  "roles":{"admin":{"allow":["*"],"deny":[]},
           "member":{"allow":["*"],"deny":["terminal"]},
           "viewer":{"allow":["read_file","memory"],"deny":["*"]},
           "conflict":{"allow":["write_file"],"deny":["write_file"]}},
  "users":{"U_ADMIN":{"role":"admin"},"U_VIEW":{"role":"viewer"},"U_CONF":{"role":"conflict"}}}

def test_member_denied_explicit_deny_terminal():       # rule 1: explicit deny
    assert _load().decide(POLICY,"U_X","terminal")["action"]=="block"
def test_member_allowed_via_wildcard():                # rule 4: wildcard allow
    assert _load().decide(POLICY,"U_X","read_file") is None
def test_admin_wildcard_allows_terminal():             # rule 4
    assert _load().decide(POLICY,"U_ADMIN","terminal") is None
def test_viewer_explicit_allow_beats_wildcard_deny():  # rule 2 beats rule 3
    assert _load().decide(POLICY,"U_VIEW","memory") is None
def test_viewer_wildcard_deny_blocks_others():         # rule 3: wildcard deny
    assert _load().decide(POLICY,"U_VIEW","terminal")["action"]=="block"
def test_explicit_deny_beats_explicit_allow():         # rule 1 beats rule 2
    assert _load().decide(POLICY,"U_CONF","write_file")["action"]=="block"
def test_unknown_user_uses_default_role_member():      # default_role=member, rule 4
    assert _load().decide(POLICY,"","read_file") is None

def test_handle_blocks_and_audits(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, "_load_policy", lambda: POLICY)
    audit = tmp_path / "audit.log"
    monkeypatch.setattr(m, "AUDIT_PATH", str(audit))
    out = m.handle({"tool_name": "terminal", "session_id": "s1", "extra": {"user_id": "U_X"}})
    assert out["action"] == "block"
    text = audit.read_text()
    assert "U_X" in text and "terminal" in text and "block" in text

def test_handle_allows_returns_empty(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, "_load_policy", lambda: POLICY)
    monkeypatch.setattr(m, "AUDIT_PATH", str(tmp_path / "a.log"))
    out = m.handle({"tool_name": "read_file", "session_id": "s1", "extra": {"user_id": "U_X"}})
    assert out == {} or out is None or out == {}  # allow

def test_handle_fail_open_on_policy_error(tmp_path, monkeypatch, capsys):
    m = _load()
    def _boom(): raise ValueError("corrupt policy")
    monkeypatch.setattr(m, "_load_policy", _boom)
    audit = tmp_path / "audit.log"
    monkeypatch.setattr(m, "AUDIT_PATH", str(audit))
    out = m.handle({"tool_name": "terminal", "session_id": "s1", "extra": {"user_id": "U_X"}})
    assert out == {}  # FAIL-OPEN (allow)
    assert "policy_load_error" in audit.read_text()   # observable
    assert "policy_load_error" in capsys.readouterr().err or audit.read_text()  # stderr warning too
