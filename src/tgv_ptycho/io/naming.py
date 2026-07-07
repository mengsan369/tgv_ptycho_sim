"""Output naming helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(
    output_root: str | Path,
    run_name: str,
    timestamp: str | None = None,
) -> Path:
    """Create and return a timestamped run directory."""

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = run_name.replace(" ", "_")
    base_dir = Path(output_root) / f"{safe_name}_{timestamp}"
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base_dir}_{suffix:02d}")
        suffix += 1

    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return run_dir
