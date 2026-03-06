"""Unified config loader with YAML + CLI override support."""

import yaml
from pathlib import Path
from typing import List, Optional


def load_config(path: str = "configs/default.yaml", overrides: Optional[List[str]] = None) -> dict:
    """
    Load config from YAML file with optional CLI overrides.

    Overrides use dot notation: "ttvla.lr=5e-5", "lora.rank=16"
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    for override in (overrides or []):
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override} (expected key=value)")
        key, value = override.split("=", 1)
        _set_nested(config, key, _parse_value(value))

    return config


def _set_nested(d: dict, key: str, value):
    """Set a nested dict value using dot notation: 'a.b.c' = val."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(value: str):
    """Parse string value to appropriate Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
