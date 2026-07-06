"""Cron script staging + approval registry (hash-pinned).

Pure-ish logic over four files under HERMES_HOME:
  scripts/staging/<name>, scripts/approved/<name>,
  cron/pending_scripts.json, cron/approved_scripts.json

No network, no subprocess. The RBAC hook decides authorization; this module
records who approved and enforces content pinning (sha256) + fail-closed reads.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from tools.path_security import validate_within_dir

MAX_SCRIPT_BYTES = 64 * 1024      # 64 KiB content cap
MAX_PENDING_ENTRIES = 20          # pending-queue cap


def _get_hermes_home() -> Path:
    return get_hermes_home()


def _scripts_dir() -> Path:
    return _get_hermes_home() / "scripts"


def _staging_dir() -> Path:
    d = _scripts_dir() / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _approved_dir() -> Path:
    d = _scripts_dir() / "approved"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pending_path() -> Path:
    return _get_hermes_home() / "cron" / "pending_scripts.json"


def _approved_registry_path() -> Path:
    return _get_hermes_home() / "cron" / "approved_scripts.json"


def sha256_of(data: Union[str, bytes]) -> str:
    """sha256 hex of a str (utf-8) or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_name(name: str, base_dir: Path) -> str:
    """Return a safe single-file name resolved within base_dir, or raise ValueError."""
    raw = (name or "").strip()
    if not raw:
        raise ValueError("script name is required")
    if raw.startswith(("/", "~")) or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError(f"script name must be a plain relative name: {name!r}")
    err = validate_within_dir(base_dir / raw, base_dir)
    if err:
        raise ValueError(f"script name escapes the scripts directory: {name!r} ({err})")
    return raw


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def stage_script(name: str, content: str, requested_by: str) -> dict:
    """Write a staging file + pending record. Raises ValueError on policy violation."""
    safe = _validate_name(name, _staging_dir())
    if content is None:
        content = ""
    if len(content.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise ValueError(f"script exceeds {MAX_SCRIPT_BYTES} byte cap")
    pending = _read_json(_pending_path())
    if safe not in pending and len(pending) >= MAX_PENDING_ENTRIES:
        raise ValueError(
            f"pending-approval queue is full ({MAX_PENDING_ENTRIES}); "
            "approve or clear entries"
        )
    digest = sha256_of(content)
    (_staging_dir() / safe).write_text(content, encoding="utf-8")
    preview = content if len(content) <= 800 else content[:800] + "\n... [truncated]"
    ts = _hermes_now()
    pending[safe] = {
        "sha256": digest,
        "authored_at": ts.isoformat(),
        "requested_by": requested_by or "",
        "preview": preview,
    }
    _write_json(_pending_path(), pending)
    return {"name": safe, "sha256": digest}


def list_pending() -> list:
    pending = _read_json(_pending_path())
    return [{"name": n, **rec} for n, rec in sorted(pending.items())]
