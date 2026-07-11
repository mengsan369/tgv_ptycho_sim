"""Run metadata helpers."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def created_at_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def get_git_commit(repo_root: str | Path = ".") -> str | None:
    """Return the current Git commit hash, or None if unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.decode("ascii", errors="ignore").strip()
