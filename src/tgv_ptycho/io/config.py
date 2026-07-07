"""YAML configuration loading and dumping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"Config must contain a YAML mapping: {config_path}"
        raise ValueError(msg)
    return data


def config_to_yaml(config: dict[str, Any]) -> str:
    """Serialize a config dictionary to a YAML string."""

    return yaml.safe_dump(config, sort_keys=False)


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    """Save a config dictionary as YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
