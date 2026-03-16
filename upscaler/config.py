"""
Configuration management for the RealESRGAN Upscaler.

Settings are persisted to ``~/.config/realesrgan-upscaler/config.json``.

Usage::

    from upscaler.config import Config
    cfg = Config.load()
    cfg.output_dir = "/tmp/output"
    cfg.save()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".config" / "realesrgan-upscaler"
_CONFIG_FILE = _CONFIG_DIR / "config.json"


@dataclass
class Config:
    """Application configuration with JSON persistence."""

    # Paths
    input_dir: str = ""
    output_dir: str = str(Path.home() / "Pictures" / "upscaled")
    weights_dir: str = ""

    # Processing
    model_name: str = "RealESRGAN_x4plus"   # e.g. RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B
    scale: int = 4
    use_half_precision: bool = True
    batch_size: int = 1
    tile_size: int = 0                        # 0 = no tiling (uses more VRAM)
    tile_pad: int = 10

    # GPU
    preferred_gpu_index: int = 0
    use_multi_gpu: bool = False

    # Power management
    enable_power_management: bool = True
    default_power_w: int = 125
    high_temp_power_w: int = 90
    critical_power_w: int = 60
    warm_threshold_c: int = 75
    high_threshold_c: int = 85
    critical_threshold_c: int = 95

    # GUI
    window_width: int = 1200
    window_height: int = 800
    theme: str = "system"                     # "system", "light", "dark"
    recent_input_dirs: List[str] = field(default_factory=list)
    recent_output_dirs: List[str] = field(default_factory=list)

    # Misc
    log_level: str = "INFO"
    check_updates: bool = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from disk, returning defaults if not found."""
        if _CONFIG_FILE.exists():
            try:
                with _CONFIG_FILE.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Build instance with only recognised keys to handle schema changes
                valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
                filtered = {k: v for k, v in data.items() if k in valid_keys}
                return cls(**filtered)
            except Exception as exc:
                logger.warning("Failed to load config from %s: %s. Using defaults.", _CONFIG_FILE, exc)
        return cls()

    def save(self) -> None:
        """Persist configuration to disk."""
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with _CONFIG_FILE.open("w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
        except Exception as exc:
            logger.error("Failed to save config to %s: %s", _CONFIG_FILE, exc)

    def add_recent_input(self, path: str) -> None:
        """Add *path* to the recent input directories list (max 10 entries)."""
        self._add_recent(self.recent_input_dirs, path)

    def add_recent_output(self, path: str) -> None:
        """Add *path* to the recent output directories list (max 10 entries)."""
        self._add_recent(self.recent_output_dirs, path)

    @staticmethod
    def _add_recent(lst: List[str], path: str, max_items: int = 10) -> None:
        if path in lst:
            lst.remove(path)
        lst.insert(0, path)
        del lst[max_items:]

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    MODELS = {
        "RealESRGAN_x2plus": {
            "scale": 2,
            "description": "2× upscaling — general photos",
            "filename": "RealESRGAN_x2plus.pth",
        },
        "RealESRGAN_x4plus": {
            "scale": 4,
            "description": "4× upscaling — general photos (high quality)",
            "filename": "RealESRGAN_x4plus.pth",
        },
        "RealESRGAN_x4plus_anime_6B": {
            "scale": 4,
            "description": "4× upscaling — anime / illustration",
            "filename": "RealESRGAN_x4plus_anime_6B.pth",
        },
        "realesr-general-x4v3": {
            "scale": 4,
            "description": "4× upscaling — general (fast variant)",
            "filename": "realesr-general-x4v3.pth",
        },
    }

    def get_model_info(self) -> dict:
        """Return metadata dict for the currently selected model."""
        return self.MODELS.get(self.model_name, {})

    def resolve_weights_path(self, model_key: Optional[str] = None) -> str:
        """Return the full path to the weights file for *model_key*."""
        key = model_key or self.model_name
        info = self.MODELS.get(key, {})
        filename = info.get("filename", f"{key}.pth")

        if self.weights_dir and os.path.isdir(self.weights_dir):
            return os.path.join(self.weights_dir, filename)

        # Search relative to this file (upscaler/ package) → upscaler/weights/
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pkg_dir, "weights", filename)
