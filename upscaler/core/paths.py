"""Platform-aware app paths."""

from __future__ import annotations

import os
from pathlib import Path

APP_ID = "realesrgan-upscaler"


def _xdg_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser()
    return fallback


def app_data_dir() -> Path:
    """Return the user-writable app data directory."""

    return _xdg_path("XDG_DATA_HOME", Path.home() / ".local" / "share") / APP_ID


def app_cache_dir() -> Path:
    """Return the user-writable app cache directory."""

    return _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache") / APP_ID


def default_weights_dir() -> Path:
    """Return the default user-writable model weights directory."""

    return app_data_dir() / "weights"
