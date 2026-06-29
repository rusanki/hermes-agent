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

import os, time

POLICY_PATH = os.path.expanduser("~/.hermes/team_policy.json")
AUDIT_PATH = os.path.expanduser("~/.hermes/logs/team_policy_audit.log")

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
