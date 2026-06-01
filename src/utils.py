"""Configuration and utility helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
