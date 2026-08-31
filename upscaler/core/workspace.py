"""Per-job workspace management."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from upscaler.core.paths import app_cache_dir, app_data_dir


class JobWorkspace:
    """User-cache workspace for one upscaling job."""

    def __init__(self, job_id: str, root: Path | None = None) -> None:
        self.job_id = job_id
        self.root = (root or app_cache_dir() / "jobs") / job_id
        self.extract_dir = self.root / "extract"
        self.upscale_dir = self.root / "upscale"
        self.log_dir = self.root / "logs"
        self.state_path = self.root / "state.json"

    def prepare(self, clean: bool = False) -> None:
        if clean:
            self.cleanup(remove_root=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        self.upscale_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self, remove_root: bool = False) -> None:
        target = self.root if remove_root else None
        if target is not None:
            shutil.rmtree(target, ignore_errors=True)
            return
        shutil.rmtree(self.extract_dir, ignore_errors=True)
        shutil.rmtree(self.upscale_dir, ignore_errors=True)

    def read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        with self.state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_state(self, state: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        tmp.replace(self.state_path)


__all__ = ["JobWorkspace", "app_cache_dir", "app_data_dir"]
