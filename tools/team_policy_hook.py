# tools/team_policy_hook.py  (deployed to ~/.hermes/hooks/team_policy.py in a later task)
"""pre_tool_call policy hook for the shared team Hermes. Stdin JSON in, decision JSON out.

This module currently provides the pure decision logic (_role_for, decide).
I/O wrapper (handle/main), audit, and limits are added in later tasks.
"""
import json, sys  # sys/json imported now; used by later tasks' main()

def _role_for(policy, user_id):
    u = (policy.get("users") or {}).get(user_id or "")
    return (u or {}).get("role") or policy.get("default_role") or "member"

def decide(policy, user_id, tool_name):
    """Return a block dict {"action":"block","message":...}, or None to allow.

    Precedence (first match wins):
      1. explicit deny  (tool_name in role.deny)   -> BLOCK
      2. explicit allow (tool_name in role.allow)  -> ALLOW  (overrides wildcard deny)
      3. wildcard deny  ("*" in role.deny)         -> BLOCK
      4. wildcard allow ("*" in role.allow)        -> ALLOW
      5. otherwise (allowlist miss)                -> BLOCK
    """
    role = _role_for(policy, user_id)
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
    decision = decide(policy, user_id, tool_name)
    if decision:
        # Blocked by role allow/deny — it never ran, so no need to count.
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
