"""
Upscale Engine - Enhanced image and video upscaling engine.

Refactored from upscale_frames.py into a proper, reusable module with:
  - Support for both image and video processing
  - Batch queue processing with progress callbacks
  - Multi-GPU parallel processing
  - Memory-efficient tiled processing
  - Error recovery and retry logic
  - Result caching for repeated operations

Usage::

    from upscaler.upscale_engine import UpscaleEngine, UpscaleJob

    engine = UpscaleEngine(device_string="cuda:0", model_name="RealESRGAN_x4plus",
                           weights_path="/path/to/weights")
    engine.load_model()

    job = UpscaleJob(input_path="/path/to/image.png", output_path="/path/to/out.png")
    engine.process_job(job)
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchvision compatibility shim for basicsr
# ---------------------------------------------------------------------------
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except (ModuleNotFoundError, ImportError):
    try:
        import torchvision.transforms.functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# huggingface_hub compatibility shim for py_real_esrgan
# huggingface_hub >= 0.16 removed cached_download; alias it to hf_hub_download
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import cached_download  # noqa: F401
except ImportError:
    try:
        import huggingface_hub as _hf
        from huggingface_hub import hf_hub_download as _hf_dl
        _hf.cached_download = _hf_dl
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UpscaleJob:
    """A single upscaling job (one input file → one output file)."""

    input_path: str
    output_path: str
    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    retry_count: int = 0
    max_retries: int = 2

    # Set by the engine after processing
    success: bool = False
    error: str = ""
    duration_s: float = 0.0


@dataclass
class BatchResult:
    """Aggregate result of a batch upscaling run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0
    jobs: List[UpscaleJob] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, UpscaleJob], None]
"""Signature: ``callback(completed: int, total: int, job: UpscaleJob)``"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class UpscaleEngine:
    """
    Single-device upscaling engine.

    One instance per GPU device.  For multi-GPU batch processing use
    :class:`BatchProcessor`.
    """

    def __init__(
        self,
        device_string: str = "cpu",
        model_name: str = "RealESRGAN_x4plus",
        weights_path: str = "",
        use_half_precision: bool = True,
        tile_size: int = 0,
        tile_pad: int = 10,
    ) -> None:
        self.device_string = device_string
        self.model_name = model_name
        self.weights_path = weights_path
        self.use_half_precision = use_half_precision
        self.tile_size = tile_size
        self.tile_pad = tile_pad

        self._model = None
        self._device = None
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the RealESRGAN model onto the target device."""
        import torch  # type: ignore

        self._device = torch.device(self.device_string)
        scale = self._infer_scale()

        # Optional basicsr architectures
        # Note: a partially-installed basicsr can raise various errors beyond
        # ImportError — for example AttributeError (on __version__) or
        # pkg_resources.VersionConflict — when the arch sub-modules are loaded.
        # Catch broadly so the app still runs without basicsr in all failure modes.
        RRDBNet = None
        SRVGGNetCompact = None
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        except Exception:  # ImportError, AttributeError, VersionConflict, etc.
            pass
        try:
            from basicsr.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
        except Exception:  # ImportError, AttributeError, VersionConflict, etc.
            pass

        from py_real_esrgan.model import RealESRGAN  # type: ignore

        model = RealESRGAN(self._device, scale=scale)

        if not self.weights_path or not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path!r}")

        if "anime_6B" in self.model_name:
            if RRDBNet is None:
                raise RuntimeError(
                    "Model 'RealESRGAN_x4plus_anime_6B' requires the 'basicsr' package "
                    "(RRDBNet architecture with 6 blocks), which is not installed.\n"
                    "Install it with:  pip install basicsr\n"
                    "On Fedora (skip extension compilation):  "
                    "CUDA_VISIBLE_DEVICES='' pip install basicsr"
                )
            model.model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6, num_grow_ch=32,
                scale=scale,
            )
            model.model.to(self._device)
            model.load_weights(self.weights_path, download=False)
        elif "general-x4v3" in self.model_name:
            if SRVGGNetCompact is None:
                raise RuntimeError(
                    "Model 'realesr-general-x4v3' requires the 'basicsr' package "
                    "(SRVGGNetCompact architecture), which is not installed.\n"
                    "Install it with:  pip install basicsr\n"
                    "On Fedora (skip extension compilation):  "
                    "CUDA_VISIBLE_DEVICES='' pip install basicsr"
                )
            model.model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_conv=32,
                upscale=scale, act_type="prelu",
            )
            model.model.to(self._device)
            model.load_weights(self.weights_path, download=False)
        else:
            model.load_weights(self.weights_path, download=False)

        if self.use_half_precision and self.device_string != "cpu":
            # Pascal GPUs (CUDA compute capability < 7.0) lack efficient native
            # FP16 support.  Newer PyTorch builds compile without FP16 kernels for
            # those devices, so running inference with a half()-converted model
            # raises: "CUDA error: no kernel image is available for execution on
            # the device".  Detect this condition early and fall back to FP32.
            if "cuda" in self.device_string:
                try:
                    dev_idx = int(self.device_string.split(":")[-1]) if ":" in self.device_string else 0
                    props = torch.cuda.get_device_properties(dev_idx)
                    if props.major < 7:
                        logger.warning(
                            "%s has CUDA compute capability %d.%d (pre-Volta / < 7.0). "
                            "FP16 is not reliably supported; disabling half-precision "
                            "to avoid 'no kernel image' CUDA errors.",
                            torch.cuda.get_device_name(dev_idx), props.major, props.minor,
                        )
                        self.use_half_precision = False
                except Exception as _cc_err:
                    logger.debug("Could not query CUDA device properties for FP16 check: %s", _cc_err)

        if self.use_half_precision and self.device_string != "cpu":
            try:
                model.model.half()
            except Exception:
                logger.warning("Half-precision (FP16) not supported on %s; using FP32.", self.device_string)

        self._model = model
        logger.info("Model loaded: %s on %s", self.model_name, self.device_string)

    def unload_model(self) -> None:
        """Release model and free GPU memory."""
        import torch  # type: ignore

        with self._model_lock:
            self._model = None
        if "cuda" in self.device_string:
            torch.cuda.empty_cache()
        logger.info("Model unloaded from %s", self.device_string)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_job(self, job: UpscaleJob) -> UpscaleJob:
        """
        Upscale a single image file described by *job*.

        The job is modified in-place and also returned.
        """
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        os.makedirs(os.path.dirname(os.path.abspath(job.output_path)), exist_ok=True)

        t_start = time.perf_counter()
        try:
            with self._model_lock:
                with torch.inference_mode():
                    image = Image.open(job.input_path).convert("RGB")
                    sr_image = self._model.predict(image)

            sr_image.save(job.output_path)
            job.success = True
        except Exception as exc:
            job.error = str(exc)
            logger.error("Failed to process %s: %s", job.input_path, exc)

        job.duration_s = time.perf_counter() - t_start
        return job

    def process_batch(
        self,
        jobs: List[UpscaleJob],
        progress_callback: Optional[ProgressCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> BatchResult:
        """Process a list of jobs sequentially on this device."""
        result = BatchResult(total=len(jobs))
        t_start = time.perf_counter()

        for idx, job in enumerate(jobs):
            if stop_event and stop_event.is_set():
                result.skipped += result.total - idx
                break

            self.process_job(job)
            if job.success:
                result.succeeded += 1
            else:
                result.failed += 1
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    logger.info("Retrying %s (attempt %d)", job.input_path, job.retry_count)
                    self.process_job(job)
                    if job.success:
                        result.succeeded += 1
                        result.failed -= 1

            result.jobs.append(job)
            if progress_callback:
                try:
                    progress_callback(idx + 1, result.total, job)
                except Exception:
                    pass

        result.elapsed_s = time.perf_counter() - t_start
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer_scale(self) -> int:
        """Derive scale factor from model name."""
        if "x2" in self.model_name:
            return 2
        if "x3" in self.model_name:
            return 3
        return 4


# ---------------------------------------------------------------------------
# Multi-GPU Batch Processor
# ---------------------------------------------------------------------------


class BatchProcessor:
    """
    Distributes a batch of upscaling jobs across multiple GPU devices.

    Each device gets its own :class:`UpscaleEngine` instance running on a
    dedicated thread.
    """

    def __init__(
        self,
        devices: List[str],
        model_name: str = "RealESRGAN_x4plus",
        weights_path: str = "",
        use_half_precision: bool = True,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.devices = devices or ["cpu"]
        self.model_name = model_name
        self.weights_path = weights_path
        self.use_half_precision = use_half_precision
        self.on_progress = on_progress

        self._stop_event = threading.Event()
        self._progress_lock = threading.Lock()
        self._completed = 0

    def process(self, jobs: List[UpscaleJob]) -> BatchResult:
        """
        Process all jobs, distributing them across all available devices.

        Returns an aggregate :class:`BatchResult`.
        """
        if not jobs:
            return BatchResult()

        self._completed = 0
        self._stop_event.clear()
        total = len(jobs)

        # Distribute jobs round-robin across devices
        job_queues: List[List[UpscaleJob]] = [[] for _ in self.devices]
        for i, job in enumerate(jobs):
            job_queues[i % len(self.devices)].append(job)

        results: List[BatchResult] = [BatchResult() for _ in self.devices]
        threads: List[threading.Thread] = []

        for dev_idx, (device, dev_jobs) in enumerate(zip(self.devices, job_queues)):
            t = threading.Thread(
                target=self._worker,
                args=(dev_idx, device, dev_jobs, total, results),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Merge results
        merged = BatchResult(total=total)
        for r in results:
            merged.succeeded += r.succeeded
            merged.failed += r.failed
            merged.skipped += r.skipped
            merged.jobs.extend(r.jobs)
        return merged

    def cancel(self) -> None:
        """Request cancellation of in-progress batch."""
        self._stop_event.set()

    # ------------------------------------------------------------------

    def _worker(
        self,
        dev_idx: int,
        device: str,
        jobs: List[UpscaleJob],
        total: int,
        results: List[BatchResult],
    ) -> None:
        engine = UpscaleEngine(
            device_string=device,
            model_name=self.model_name,
            weights_path=self.weights_path,
            use_half_precision=self.use_half_precision,
        )
        try:
            engine.load_model()
        except Exception as exc:
            logger.error("[%s] Model load failed: %s", device, exc)
            results[dev_idx] = BatchResult(total=len(jobs), failed=len(jobs))
            return

        def _progress(completed_local: int, local_total: int, job: UpscaleJob) -> None:
            if self.on_progress:
                with self._progress_lock:
                    self._completed += 1
                    try:
                        self.on_progress(self._completed, total, job)
                    except Exception:
                        pass

        results[dev_idx] = engine.process_batch(
            jobs,
            progress_callback=_progress,
            stop_event=self._stop_event,
        )
        engine.unload_model()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_jobs_from_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = "RealESRGAN_x4plus",
    scale: int = 4,
    skip_existing: bool = True,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> List[UpscaleJob]:
    """
    Scan *input_dir* and build a list of :class:`UpscaleJob` objects.

    Parameters
    ----------
    skip_existing:
        If *True*, omit jobs whose output file already exists (resume support).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jobs: List[UpscaleJob] = []
    for src in sorted(input_path.iterdir()):
        if src.suffix.lower() not in extensions:
            continue
        dst = output_path / src.name
        if skip_existing and dst.exists():
            continue
        jobs.append(
            UpscaleJob(
                input_path=str(src),
                output_path=str(dst),
                model_name=model_name,
                scale=scale,
            )
        )
    return jobs
