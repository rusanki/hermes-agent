"""Per-turn request trace log (JSONL) for org-wide request debugging.

One JSON record per turn: inbound message, every tool call (name + args +
result), and the final response — all secret-redacted (the "usage" field is
the exception: it's structured telemetry, e.g. token counts, stored verbatim
and not run through _redact()). Always-on (kill-switch env
HERMES_REQUEST_TRACE=0). Failure-isolated: a trace failure NEVER breaks a
turn. Covers the native tool path AND the claude-sdk MCP-proxy path (via a
ContextVar visible on the SDK bridge thread).
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ContextVar so the claude-sdk proxy handler (which dispatches tools on a
# separate bridge thread) can reach the turn's trace ctx. MUST be set at
# turn-start BEFORE the provider's _capture_ctx() snapshot (see the plan /
# spec "LOAD-BEARING ORDERING").
_REQUEST_TRACE_CTX: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "HERMES_REQUEST_TRACE_CTX", default=None)

_WRITE_LOCK = threading.Lock()  # guards the append; cheap, whole-line writes.


def _enabled() -> bool:
    return (os.getenv("HERMES_REQUEST_TRACE", "1").strip().lower() not in ("0", "false", "no", ""))


def _trace_path() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "logs" / "request_trace.jsonl")


def _redact(value: Any) -> str:
    """Serialize (if needed) then secret-redact a field for the trace record."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str,
                                                               ensure_ascii=False)
    except Exception:
        text = str(value)
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        # If redaction is unavailable, DROP the value rather than persist a raw
        # secret. Fail closed.
        return "[unredactable]"


def trace_turn_start(*, session_id: str, user_id: str, platform: str,
                     model: str, provider: str, inbound: str) -> dict | None:
    """Begin a turn's trace. Returns a ctx dict (buffers tool events) or None
    when disabled. Also stores the ctx in the ContextVar for the SDK path."""
    if not _enabled():
        return None
    try:
        ctx = {
            "session_id": session_id or "", "user_id": user_id or "",
            "platform": platform or "", "model": model or "", "provider": provider or "",
            "inbound": _redact(inbound), "tools": [],
        }
        _REQUEST_TRACE_CTX.set(ctx)
        return ctx
    except Exception:
        logger.debug("request_trace: turn_start failed", exc_info=True)
        return None


def trace_tool_call(ctx: dict | None, *, name: str, args: Any, result: Any,
                    duration: float, is_error: bool) -> None:
    """Append one tool event to the in-flight turn. No-op if ctx is None."""
    if ctx is None:
        return
    try:
        ctx["tools"].append({
            "name": name, "args": _redact(args), "result": _redact(result),
            "duration": round(float(duration), 3), "is_error": bool(is_error),
        })
    except Exception:
        logger.debug("request_trace: tool_call failed", exc_info=True)


def trace_turn_end(ctx: dict | None, *, response: str, finish_reason: str,
                   usage: Any) -> None:
    """Flush the turn's record as one JSON line. No-op if ctx is None."""
    if ctx is None:
        return
    try:
        record = {
            "ts": _now_iso(),
            "session_id": ctx["session_id"], "user_id": ctx["user_id"],
            "platform": ctx["platform"], "model": ctx["model"], "provider": ctx["provider"],
            "inbound": ctx["inbound"], "tools": ctx["tools"],
            "response": _redact(response), "finish_reason": finish_reason or "",
            "usage": usage if isinstance(usage, dict) else {},
        }
        _write_record(record)
    except Exception:
        logger.debug("request_trace: turn_end failed", exc_info=True)
    finally:
        try:
            _REQUEST_TRACE_CTX.set(None)
        except Exception:
            pass


def _now_iso() -> str:
    # Late import so a frozen-clock test env can monkeypatch if needed.
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _write_record(record: dict) -> None:
    path = _trace_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
