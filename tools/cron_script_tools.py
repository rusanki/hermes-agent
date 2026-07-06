"""Registered tools over the hash-pinned cron script registry.

Two tools, one per privilege tier (enforced by the RBAC pre_tool_call hook, NOT
here):

  * ``cron_script``          — stage / list_pending / show   (admin)
  * ``cron_script_approve``  — approve / revoke              (superadmin)

These wrappers do exactly two things beyond calling ``cron.script_registry``:
record the *verified* caller uid (from the session context, never from args) and
convert the registry's ``ValueError`` policy violations into structured tool
errors so the model never sees a raw traceback. All real logic — path pinning,
sha256 verification, fail-closed reads — lives in ``cron.script_registry``.
"""
from cron import script_registry
from gateway.session_context import get_session_user_id
from tools.cronjob_tools import check_cronjob_requirements
from tools.registry import registry, tool_error, tool_result


def cron_script(action, name=None, content=None, task_id=None) -> str:
    """Stage a cron script for approval, list pending ones, or show one's body.

    ``task_id`` is accepted for handler compatibility and intentionally unused.
    """
    del task_id  # unused; kept for handler signature parity with cronjob
    try:
        if action == "stage":
            if not name:
                return tool_error("name is required for stage", success=False)
            res = script_registry.stage_script(
                name, content or "", requested_by=get_session_user_id() or ""
            )
            return tool_result(staged=True, pending_approval=True, **res)
        if action == "list_pending":
            return tool_result(pending=script_registry.list_pending())
        if action == "show":
            if not name:
                return tool_error("name is required for show", success=False)
            return tool_result(**script_registry.show_staged(name))
        return tool_error(f"unknown action: {action!r}", success=False)
    except ValueError as e:
        return tool_error(str(e), success=False)


def cron_script_approve(action, name=None, sha256=None, task_id=None) -> str:
    """Approve a staged cron script (sha-pinned) or revoke an approved one.

    ``task_id`` is accepted for handler compatibility and intentionally unused.
    """
    del task_id  # unused; kept for handler signature parity with cronjob
    try:
        if action == "approve":
            if not name:
                return tool_error("name is required for approve", success=False)
            res = script_registry.approve_script(
                name, sha256 or "", approver_uid=get_session_user_id() or ""
            )
            return tool_result(**res)
        if action == "revoke":
            if not name:
                return tool_error("name is required for revoke", success=False)
            res = script_registry.revoke_script(
                name, revoked_by=get_session_user_id() or ""
            )
            return tool_result(**res)
        return tool_error(f"unknown action: {action!r}", success=False)
    except ValueError as e:
        return tool_error(str(e), success=False)


CRON_SCRIPT_SCHEMA = {
    "name": "cron_script",
    "description": (
        "Stage a Python cron script for approval, list pending scripts, or show "
        "a staged script's full content. Staging requires the ADMIN role "
        "(enforced by the security hook, not this tool). A staged script is "
        "inert until a superadmin approves it via cron_script_approve; only "
        "approved, hash-pinned scripts may be run by cron jobs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["stage", "list_pending", "show"],
                "description": (
                    "stage: write a new staging file + pending record. "
                    "list_pending: list scripts awaiting approval. "
                    "show: return a staged script's full content and sha256."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Plain relative script file name (e.g. 'fetch_report.py'); "
                    "required for stage and show. No path separators or absolute "
                    "paths — the name must resolve inside the scripts directory."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full script body to stage (used by action=stage).",
            },
        },
        "required": ["action"],
    },
}


CRON_SCRIPT_APPROVE_SCHEMA = {
    "name": "cron_script_approve",
    "description": (
        "Approve a staged cron script (pinning its sha256) so cron jobs may run "
        "it, or revoke a previously approved script. Both actions require the "
        "SUPERADMIN role (enforced by the security hook, not this tool). Approval "
        "re-hashes the on-disk staging file and fails if the content changed "
        "since staging, so pass the sha256 returned by cron_script show/stage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["approve", "revoke"],
                "description": (
                    "approve: promote a staged script to approved, pinning its "
                    "sha256. revoke: remove an approved script's pin (and file)."
                ),
            },
            "name": {
                "type": "string",
                "description": "Plain relative script file name to approve or revoke.",
            },
            "sha256": {
                "type": "string",
                "description": (
                    "Expected sha256 of the staged content (required for approve); "
                    "must match the current on-disk staging file or approval fails."
                ),
            },
        },
        "required": ["action", "name"],
    },
}


registry.register(
    name="cron_script",
    toolset="cronjob",
    schema=CRON_SCRIPT_SCHEMA,
    handler=lambda args, **kw: cron_script(
        action=args.get("action", ""),
        name=args.get("name"),
        content=args.get("content"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_cronjob_requirements,  # gate on session context, same as `cronjob`
    emoji="📝",
)

registry.register(
    name="cron_script_approve",
    toolset="cronjob",
    schema=CRON_SCRIPT_APPROVE_SCHEMA,
    handler=lambda args, **kw: cron_script_approve(
        action=args.get("action", ""),
        name=args.get("name"),
        sha256=args.get("sha256"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_cronjob_requirements,  # gate on session context, same as `cronjob`
    emoji="✅",
)
