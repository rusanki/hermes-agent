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
