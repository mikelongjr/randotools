"""Model lifecycle wrapper used by pipelines."""

from __future__ import annotations

from typing import Optional

from upscaler.upscale_engine import UpscaleEngine


class ModelManager:
    """Owns one loaded UpscaleEngine and guarantees cleanup."""

    def __init__(
        self,
        device_string: str,
        model_name: str,
        weights_path: str,
        use_half_precision: bool = True,
    ) -> None:
        self.engine = UpscaleEngine(
            device_string=device_string,
            model_name=model_name,
            weights_path=weights_path,
            use_half_precision=use_half_precision,
        )
        self._loaded = False

    def __enter__(self) -> UpscaleEngine:
        self.load()
        return self.engine

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def load(self) -> UpscaleEngine:
        if not self._loaded:
            self.engine.load_model()
            self._loaded = True
        return self.engine

    def unload(self) -> None:
        if self._loaded:
            self.engine.unload_model()
            self._loaded = False


def infer_scale(model_name: str) -> int:
    if "x2" in model_name:
        return 2
    if "x3" in model_name:
        return 3
    return 4
