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

# --- empty/whitespace user_id -> 'system' role (cron/scheduled runs) ----------
# Cron/scheduled invocations carry user_id="" (no Slack user sent them). Before
# the 2026-07-01 fix this fell through to default_role=member, which denies
# `terminal` and silently broke a scheduled digest. This policy fixture defines
# an explicit `system` role that empty/whitespace user_ids map onto.
SYSTEM_POLICY = {"version":1,"default_role":"member",
  "roles":{"system":{"allow":["terminal","read_file","cronjob","send_message"],
                      "deny":["write_file","patch","execute_code"]},
           "member":{"allow":["*"],"deny":["terminal"]},
           "admin":{"allow":["*"],"deny":[]}},
  "users":{"U_ADMIN":{"role":"admin"},"U_MEM":{"role":"member"}}}

def test_empty_user_id_system_role_allows_terminal():
    assert _load().decide(SYSTEM_POLICY,"","terminal") is None
def test_empty_user_id_system_role_blocks_write_file():
    assert _load().decide(SYSTEM_POLICY,"","write_file")["action"]=="block"
def test_whitespace_user_id_treated_as_empty_system_role():
    assert _load().decide(SYSTEM_POLICY,"   ","terminal") is None
def test_known_member_user_still_denied_terminal_unchanged():
    assert _load().decide(SYSTEM_POLICY,"U_MEM","terminal")["action"]=="block"
def test_admin_user_unchanged_with_system_role_present():
    assert _load().decide(SYSTEM_POLICY,"U_ADMIN","terminal") is None
def test_unknown_nonempty_user_id_uses_default_role_unchanged():
    assert _load().decide(SYSTEM_POLICY,"U_UNKNOWN","terminal")["action"]=="block"  # default_role=member denies terminal
def test_empty_user_id_without_system_role_falls_to_default_role():
    # POLICY (module-level fixture above) defines no 'system' role: empty uid
    # must fall to default_role=member exactly as before this change.
    assert _load().decide(POLICY,"","terminal")["action"]=="block"

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
    assert "policy_load_error" in audit.read_text()   # observable in the audit log
    # stderr warning must ALSO be emitted (a broken policy must be visible, not silent).
    # NOTE: this pins the stderr behavior specifically — do NOT weaken to
    # `... or audit.read_text()`, which is always truthy and tests nothing.
    assert "failing open" in capsys.readouterr().err

# --- per-user, per-session mutating-action limit -------------------------------
LIMIT_POLICY = {"version":1,"default_role":"member",
  "roles":{"member":{"allow":["*"],"deny":[],"limits":{"max_mutating_per_session":2}},
           "admin":{"allow":["*"],"deny":[]}},
  "users":{"U_ADMIN":{"role":"admin"}}}

def _setup(m, tmp_path, monkeypatch, policy=LIMIT_POLICY):
    monkeypatch.setattr(m, "_load_policy", lambda: policy)
    monkeypatch.setattr(m, "AUDIT_PATH", str(tmp_path/"audit.log"))
    monkeypatch.setattr(m, "COUNTS_DIR", str(tmp_path/"counts"))

def test_mutating_under_cap_allowed_and_counts(tmp_path, monkeypatch):
    m=_load(); _setup(m,tmp_path,monkeypatch)
    p={"tool_name":"write_file","session_id":"s1","extra":{"user_id":"U1"}}
    assert m.handle(p) == {}            # 1st allowed
    assert m.handle(p) == {}            # 2nd allowed (cap=2)
    assert m.handle(p)["action"]=="block"  # 3rd over cap -> block

def test_non_mutating_never_limited(tmp_path, monkeypatch):
    m=_load(); _setup(m,tmp_path,monkeypatch)
    p={"tool_name":"read_file","session_id":"s1","extra":{"user_id":"U1"}}
    for _ in range(5):
        assert m.handle(p) == {}        # never limited

def test_limit_is_per_user(tmp_path, monkeypatch):
    m=_load(); _setup(m,tmp_path,monkeypatch)
    a={"tool_name":"write_file","session_id":"s1","extra":{"user_id":"U_A"}}
    b={"tool_name":"write_file","session_id":"s1","extra":{"user_id":"U_B"}}
    m.handle(a); m.handle(a)            # U_A hits cap
    assert m.handle(a)["action"]=="block"
    assert m.handle(b) == {}            # U_B independent, still allowed

def test_no_cap_means_unlimited(tmp_path, monkeypatch):
    m=_load(); _setup(m,tmp_path,monkeypatch)
    p={"tool_name":"write_file","session_id":"s1","extra":{"user_id":"U_ADMIN"}}  # admin: no limits
    for _ in range(5):
        assert m.handle(p) == {}

def test_explicit_deny_still_blocks_before_limit(tmp_path, monkeypatch):
    # a tool denied by role is blocked regardless of limits (and not counted)
    pol={"version":1,"default_role":"member",
         "roles":{"member":{"allow":["*"],"deny":["terminal"],"limits":{"max_mutating_per_session":2}}},"users":{}}
    m=_load(); _setup(m,tmp_path,monkeypatch,pol)
    p={"tool_name":"terminal","session_id":"s1","extra":{"user_id":"U1"}}
    assert m.handle(p)["action"]=="block"

# --- Action-aware creation gate (Task 6) -------------------------------------
GATE_POLICY = {"version": 1, "default_role": "member",
  "roles": {
    "superadmin": {"allow": ["*"], "deny": []},
    "admin": {"allow": ["*"], "deny": ["terminal", "execute_code", "write_file", "patch"]},
    "member": {"allow": ["*"], "deny": ["terminal", "execute_code", "write_file", "patch"]},
    "system": {"allow": ["read_file", "memory", "cronjob"], "deny": ["*"]},
  },
  "users": {
    "U_SUPER": {"role": "superadmin"},
    "U_ADMIN": {"role": "admin"},
    "U_MEMBER": {"role": "member"},
  }}

def _gate_handle(m, tool, action, uid, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_load_policy", lambda: GATE_POLICY)
    monkeypatch.setattr(m, "AUDIT_PATH", str(tmp_path / "audit.log"))
    payload = {"tool_name": tool, "session_id": "s", "extra": {"user_id": uid}}
    if action is not None:
        payload["tool_input"] = {"action": action}
    return m.handle(payload)

def test_gate_member_cronjob_create_blocked(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "create", "U_MEMBER", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_admin_cronjob_create_allowed(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "create", "U_ADMIN", tmp_path, monkeypatch)
    assert not out

def test_gate_member_cronjob_list_allowed(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "list", "U_MEMBER", tmp_path, monkeypatch)
    assert not out

def test_gate_member_cronjob_delete_blocked(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "delete", "U_MEMBER", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_member_cron_script_stage_blocked(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cron_script", "stage", "U_MEMBER", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_admin_cron_script_stage_allowed(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cron_script", "stage", "U_ADMIN", tmp_path, monkeypatch)
    assert not out

def test_gate_member_cron_script_show_allowed(tmp_path, monkeypatch):
    # show/list_pending are not gated (read-only review)
    out = _gate_handle(_load(), "cron_script", "show", "U_MEMBER", tmp_path, monkeypatch)
    assert not out

def test_gate_admin_cron_script_approve_blocked(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cron_script_approve", "approve", "U_ADMIN", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_superadmin_cron_script_approve_allowed(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cron_script_approve", "approve", "U_SUPER", tmp_path, monkeypatch)
    assert not out

def test_gate_approve_missing_action_still_superadmin_only(tmp_path, monkeypatch):
    # Even without tool_input.action, the approve tool requires superadmin (fail-safe).
    out = _gate_handle(_load(), "cron_script_approve", None, "U_ADMIN", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_decide_is_action_aware_directly(tmp_path, monkeypatch):
    m = _load()
    # decide() must accept the action arg and block member create
    assert m.decide(GATE_POLICY, "U_MEMBER", "cronjob", "create")["action"] == "block"
    assert m.decide(GATE_POLICY, "U_ADMIN", "cronjob", "create") is None
    # backward-compat: decide() still works WITHOUT the action arg (defaults None)
    assert m.decide(GATE_POLICY, "U_MEMBER", "read_file") is None

# --- MUST-FIX 1: gate the REAL cron mutating verbs (create/update/remove) -----
# The live cronjob tool's mutating actions are create/update/remove (verified in
# tools/cronjob_tools.py). `update` is the EDIT verb -- a member calling
# cronjob(action="update", ...) can rewrite an existing job's script, so it MUST
# be gated. The earlier set used placeholder edit/delete and MISSED update.
def test_gate_member_cronjob_update_blocked(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "update", "U_MEMBER", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_member_cronjob_remove_blocked(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "remove", "U_MEMBER", tmp_path, monkeypatch)
    assert out.get("action") == "block"

def test_gate_admin_cronjob_update_allowed(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", "update", "U_ADMIN", tmp_path, monkeypatch)
    assert not out

# --- MUST-FIX 2: a non-string action must NOT crash the hook (fail-OPEN=bypass) -
# A dict/list/int action would raise AttributeError in _required_role. Because
# the hook's caller treats a crashed hook as NO BLOCK, that fails OPEN and
# bypasses the gate. The hook must return a dict and never raise.
def test_gate_nonstring_action_dict_does_not_crash(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", {"nested": 1}, "U_MEMBER", tmp_path, monkeypatch)
    assert isinstance(out, dict)   # returned a decision, did not raise

def test_gate_nonstring_action_int_does_not_crash(tmp_path, monkeypatch):
    out = _gate_handle(_load(), "cronjob", 5, "U_MEMBER", tmp_path, monkeypatch)
    assert isinstance(out, dict)   # returned a decision, did not raise

def test_gate_approve_nonstring_action_still_blocked_for_admin(tmp_path, monkeypatch):
    # cron_script_approve is gated at the TOOL level (action ignored), so a
    # non-string action must NOT let a non-superadmin through.
    out = _gate_handle(_load(), "cron_script_approve", {"x": 1}, "U_ADMIN", tmp_path, monkeypatch)
    assert out.get("action") == "block"

# --- MINOR 1: gate blocks are audited with reason="rbac_gate" (greppable) ------
def test_gate_block_audited_with_rbac_gate_reason(tmp_path, monkeypatch):
    audit = tmp_path / "audit.log"
    _gate_handle(_load(), "cronjob", "create", "U_MEMBER", tmp_path, monkeypatch)
    assert "rbac_gate" in audit.read_text()
