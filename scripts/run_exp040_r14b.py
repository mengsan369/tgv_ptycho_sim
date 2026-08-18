"""Run the single hash-locked exp040 R14B solver formal benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_exp040_r14 as base_runner  # noqa: E402

REGISTERED_CONFIG_SHA256 = (
    "178C7E64C0E399F38D41821950C962088FD3E86C07CBA1910D6ACF582B29FE67"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def run(config_path: Path) -> Path:
    """Execute R14B through the shared, already-tested R14 runner."""

    return base_runner.run(
        config_path,
        registered_config_sha256=REGISTERED_CONFIG_SHA256,
    )


def main() -> None:
    run(_parse_args().config)


if __name__ == "__main__":
    main()
