# tools/team_policy_hook.py  (deployed to ~/.hermes/hooks/team_policy.py in a later task)
"""pre_tool_call policy hook for the shared team Hermes. Stdin JSON in, decision JSON out.

This module currently provides the pure decision logic (_role_for, decide).
I/O wrapper (handle/main), audit, and limits are added in later tasks.
"""
import json, sys  # sys/json imported now; used by later tasks' main()

def _role_for(policy, user_id):
    # Cron/scheduled runs invoke tools with user_id="" (no Slack user sent
    # them). Without this mapping they fall to default_role and inherit
    # member's denials -- this silently broke a scheduled digest on
    # 2026-07-01. Map empty/whitespace user_id to a dedicated 'system' role,
    # but only if the policy actually defines one: policies without a
    # 'system' role are unaffected (empty uid falls to default_role exactly
    # as before -- fail-safe, no accidental privilege grant from this alone).
    if not (user_id or "").strip() and "system" in (policy.get("roles") or {}):
        return "system"
    u = (policy.get("users") or {}).get(user_id or "")
    return (u or {}).get("role") or policy.get("default_role") or "member"

# The REAL mutating cronjob actions are create/update/remove (verified in
# tools/cronjob_tools.py): `update` is the EDIT verb (rewrites an existing
# job's script/prompt), `remove` is delete. The add/edit/delete/rm aliases are
# harmless extra coverage in case the serializer ever emits a synonym.
_CRONJOB_MUTATING_ACTIONS = {"create", "add", "update", "edit", "delete", "remove", "rm"}
# member and system are BOTH rank 1 (deliberate: both sit below admin). system
# is a distinct role name so cron/scheduled runs get their own allow/deny list,
# but it must NOT outrank admin for the creation gate.
_ROLE_RANK = {"viewer": 0, "member": 1, "system": 1, "admin": 2, "superadmin": 3}

def _required_role(tool_name, action):
    """Minimum role for an elevated (tool, action), or None if not gated here."""
    # A non-string action (dict/list/int) must not raise here: normalize it to
    # "" so it reads as "not a named mutating action" for cronjob/cron_script
    # (falls through to normal allow/deny). cron_script_approve is gated at the
    # tool level below, so its superadmin requirement is unaffected by action.
    a = (action if isinstance(action, str) else "").strip().lower()
    if tool_name == "cron_script_approve":
        return "superadmin"                 # tool-level: any/no action needs superadmin
    if tool_name == "cron_script" and a == "stage":
        return "admin"
    if tool_name == "cronjob" and a in _CRONJOB_MUTATING_ACTIONS:
        return "admin"
    return None

def _role_meets(role, required):
    return _ROLE_RANK.get(role, 0) >= _ROLE_RANK.get(required, 99)

def decide(policy, user_id, tool_name, action=None):
    """Return a block dict {"action":"block","message":...}, or None to allow.

    An action-aware creation gate runs FIRST: for elevated (tool, action) pairs
    (cron creation/edit, script staging/approval) the caller's role must meet a
    minimum rank, else BLOCK. The gate only ADDS constraints -- it can block or
    fall through, never grant. Then the existing allow/deny precedence applies:

    Precedence (first match wins):
      1. explicit deny  (tool_name in role.deny)   -> BLOCK
      2. explicit allow (tool_name in role.allow)  -> ALLOW  (overrides wildcard deny)
      3. wildcard deny  ("*" in role.deny)         -> BLOCK
      4. wildcard allow ("*" in role.allow)        -> ALLOW
      5. otherwise (allowlist miss)                -> BLOCK
    """
    role = _role_for(policy, user_id)
    required = _required_role(tool_name, action)
    if required is not None and not _role_meets(role, required):
        return {"action": "block",
                "message": (f"Your role '{role}' cannot perform '{tool_name}' "
                            f"action '{action}' (requires {required}).")}
    rules = (policy.get("roles") or {}).get(role) or {}
    deny, allow = rules.get("deny") or [], rules.get("allow") or []
    block = {"action": "block",
             "message": f"Your role '{role}' is not permitted to use '{tool_name}'."}
    if tool_name in deny:    # 1
        return block
    if tool_name in allow:   # 2
        return None
    if "*" in deny:          # 3
        return block
    if "*" in allow:         # 4
        return None
    return block             # 5

import os, time, fcntl

POLICY_PATH = os.path.expanduser("~/.hermes/team_policy.json")
AUDIT_PATH = os.path.expanduser("~/.hermes/logs/team_policy_audit.log")
COUNTS_DIR = os.path.expanduser("~/.hermes/cache/team_policy_counts")

# Tools that mutate state and therefore count against a role's per-session cap.
_MUTATING = {"terminal", "execute_code", "write_file", "patch"}

def _count_path(session_id):
    return os.path.join(COUNTS_DIR, f"{session_id or 'none'}.json")

def _cap_for(policy, role):
    """Return the per-session mutating cap for `role`, or None if uncapped.

    Role-level `roles[role].limits.max_mutating_per_session` wins; falls back to
    top-level `policy['limits'][role].max_mutating_per_session`.
    """
    role_limits = ((policy.get("roles") or {}).get(role) or {}).get("limits") or {}
    cap = role_limits.get("max_mutating_per_session")
    if cap is None:
        top = (policy.get("limits") or {}).get(role) or {}
        cap = top.get("max_mutating_per_session")
    return cap

def _check_and_increment(session_id, user_id, cap):
    """flock-guarded read-modify-write of the per-session counts file.

    Returns True (allowed) after incrementing the user's count, or False if the
    user is already at/over `cap` (no increment). Fail-OPEN (return True) on any
    OSError so a lock/IO failure never blocks a tool — this is best-effort, not
    billing-grade.
    """
    try:
        os.makedirs(COUNTS_DIR, exist_ok=True)
        path = _count_path(session_id)
        # Open r+ so we hold a single fd for the whole read-modify-write under
        # the lock; create the file first if it doesn't exist.
        if not os.path.exists(path):
            open(path, "a").close()
        with open(path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                raw = f.read().strip()
                counts = json.loads(raw) if raw else {}
                if not isinstance(counts, dict):
                    counts = {}
                current = counts.get(user_id, 0)
                if current >= cap:
                    return False
                counts[user_id] = current + 1
                f.seek(0)
                f.truncate()
                f.write(json.dumps(counts))
                f.flush()
                return True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        return True  # FAIL-OPEN

def _load_policy():
    with open(POLICY_PATH) as f:
        return json.load(f)

def _audit(user_id, tool_name, decision, reason=""):
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        rec = {"ts": int(time.time()), "user": user_id, "tool": tool_name,
               "decision": "block" if decision else "allow"}
        if reason:
            rec["reason"] = reason
        with open(AUDIT_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass

def handle(payload):
    user_id = (payload.get("extra") or {}).get("user_id", "")
    tool_name = payload.get("tool_name", "")
    # The action lives in tool_input (the shell-hook serializer maps tool args ->
    # tool_input); the creation gate in decide() needs it to distinguish e.g.
    # `cronjob list` (allowed) from `cronjob create` (elevated).
    #
    # A missing/non-string action fails OPEN for cronjob/cron_script (they fall
    # through to normal allow/deny) -- acceptable because the live serializer
    # always sends a string action, and a malformed one is not a valid mutating
    # verb. cron_script_approve is gated at the TOOL level (action ignored), so
    # it fails CLOSED regardless of the action's type.
    action = (payload.get("tool_input") or {}).get("action")
    # session_id is TOP-LEVEL (used by the limit task); read it now for forward-compat.
    session_id = payload.get("session_id", "")  # noqa: F841 (used by a later task)
    try:
        policy = _load_policy()
    except Exception as e:
        # FAIL-OPEN but OBSERVABLE: a broken policy must not brick the bot,
        # but must be visible (audit line + stderr) so it's noticed.
        _audit(user_id, tool_name, None, reason="policy_load_error")
        print(f"team_policy: policy load failed ({e}); failing open (allowing)", file=sys.stderr)
        return {}
    try:
        decision = decide(policy, user_id, tool_name, action)
    except Exception as e:
        # A crash in the decision path must NOT fail OPEN for a gated tool --
        # that would defeat the RBAC gate. Fail CLOSED for gated tools; for
        # everything else preserve the availability-first ethos (fail open).
        if _required_role(tool_name, action if isinstance(action, str) else None) is not None:
            block = {"action": "block",
                     "message": "Policy decision failed; blocking a privileged action (fail-closed)."}
            _audit(user_id, tool_name, block, reason="decide_error")
            return block
        _audit(user_id, tool_name, None, reason="decide_error")
        print(f"team_policy: decide failed ({e}); failing open for non-gated tool", file=sys.stderr)
        return {}
    if decision:
        # Blocked by role allow/deny — it never ran, so no need to count.
        # Distinguish RBAC-gate blocks (role doesn't meet the required rank for
        # an elevated action) so they're greppable in the audit log.
        required = _required_role(tool_name, action)
        if required is not None and not _role_meets(_role_for(policy, user_id), required):
            _audit(user_id, tool_name, decision, reason="rbac_gate")
        else:
            _audit(user_id, tool_name, decision)
        return decision
    # Allowed by role. Enforce the per-session mutating cap (if any) on top.
    if tool_name in _MUTATING:
        role = _role_for(policy, user_id)
        cap = _cap_for(policy, role)
        if cap is not None and not _check_and_increment(session_id, user_id, cap):
            block = {"action": "block",
                     "message": f"Per-session limit reached for '{tool_name}' (max {cap})."}
            _audit(user_id, tool_name, block, reason="rate_limited")
            return block
    _audit(user_id, tool_name, decision)
    return decision or {}

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print("{}")
        return
    out = handle(payload)
    print(json.dumps(out) if out else "{}")

if __name__ == "__main__":
    main()
