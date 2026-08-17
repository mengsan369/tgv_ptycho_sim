"""Run metadata helpers."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path


def created_at_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def get_git_commit(repo_root: str | Path = ".") -> str | None:
    """Return the current Git commit hash, or None if unavailable."""

    root = Path(repo_root).resolve()
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.decode("ascii", errors="ignore").strip()
