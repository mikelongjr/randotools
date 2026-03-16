"""
GPU Manager - Unified GPU detection and management for NVIDIA and AMD GPUs.

Supports:
  - NVIDIA GPUs via CUDA
  - AMD GPUs via ROCm/HIP
  - Intel GPUs as fallback via PyTorch XPU (if available)
  - CPU-only mode with performance warnings

Usage::

    from upscaler.gpu_manager import GPUManager
    manager = GPUManager()
    gpus = manager.get_available_gpus()
    for gpu in gpus:
        print(gpu)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU hardware vendor identifiers."""

    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    CPU = auto()
    UNKNOWN = auto()


@dataclass
class GPUInfo:
    """Information about a single GPU device."""

    index: int
    name: str
    vendor: GPUVendor
    device_string: str          # e.g. "cuda:0", "cpu"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    compute_capability: str = ""  # CUDA: "8.6", ROCm: "gfx1030"
    driver_version: str = ""
    temperature_c: Optional[int] = None
    power_draw_w: Optional[float] = None
    is_available: bool = True
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        vram = f" | {self.vram_total_mb} MB VRAM" if self.vram_total_mb else ""
        cc = f" | {self.compute_capability}" if self.compute_capability else ""
        return f"[{self.device_string}] {self.name} ({self.vendor.name}){vram}{cc}"


class GPUManager:
    """
    Unified GPU detection and management for NVIDIA (CUDA) and AMD (ROCm) GPUs.

    Automatically discovers all available GPUs and provides a unified interface
    for device selection, health checks, and diagnostics.
    """

    def __init__(self) -> None:
        self._gpus: List[GPUInfo] = []
        self._torch_available = self._check_torch()
        self._refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_gpus(self) -> List[GPUInfo]:
        """Return list of all detected, usable GPU devices (may include CPU)."""
        return [g for g in self._gpus if g.is_available]

    def get_gpu_by_index(self, index: int) -> Optional[GPUInfo]:
        """Return a *GPUInfo* instance by its list index."""
        available = self.get_available_gpus()
        if 0 <= index < len(available):
            return available[index]
        return None

    def get_preferred_gpu(self) -> GPUInfo:
        """Return the best available GPU, falling back to CPU."""
        gpus = self.get_available_gpus()
        # Prefer NVIDIA > AMD > Intel > CPU
        for vendor in (GPUVendor.NVIDIA, GPUVendor.AMD, GPUVendor.INTEL):
            for gpu in gpus:
                if gpu.vendor == vendor:
                    return gpu
        return gpus[0] if gpus else self._cpu_device()

    def refresh(self) -> None:
        """Re-detect all GPUs (call this if hardware state may have changed)."""
        self._refresh()

    def run_diagnostics(self) -> str:
        """Run GPU diagnostics and return a human-readable report string."""
        lines = ["=" * 60, "GPU DIAGNOSTICS", "=" * 60]

        if self._torch_available:
            import torch  # type: ignore

            lines.append(f"PyTorch version  : {torch.__version__}")
            hip_ver = getattr(getattr(torch, "version", None), "hip", None)
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
            if cuda_ver:
                lines.append(f"CUDA version     : {cuda_ver}")
            if hip_ver:
                lines.append(f"ROCm/HIP version : {hip_ver}")
        else:
            lines.append("PyTorch          : NOT AVAILABLE")

        lines.append("")
        gpus = self.get_available_gpus()
        if gpus:
            lines.append(f"Detected {len(gpus)} device(s):")
            for gpu in gpus:
                lines.append(f"  {gpu}")
        else:
            lines.append("No GPU detected. CPU-only mode.")

        # AMD-specific checks
        if sys.platform == "linux":
            lines.extend(self._amd_linux_diagnostics())

        lines.append("=" * 60)
        return "\n".join(lines)

    def health_check(self, gpu: GPUInfo) -> dict:
        """
        Perform a health check on the specified GPU.

        Returns a dict with keys: ``ok`` (bool), ``temperature``, ``vram_free_mb``,
        ``issues`` (list of str).
        """
        result: dict = {"ok": True, "temperature": None, "vram_free_mb": None, "issues": []}

        if gpu.vendor == GPUVendor.CPU:
            return result

        if self._torch_available:
            import torch  # type: ignore

            try:
                if "cuda" in gpu.device_string:
                    dev_idx = int(gpu.device_string.split(":")[-1]) if ":" in gpu.device_string else 0
                    mem = torch.cuda.mem_get_info(dev_idx)
                    result["vram_free_mb"] = mem[0] // (1024 * 1024)
            except Exception as exc:
                result["issues"].append(f"VRAM check failed: {exc}")

        # Temperature check
        temp = self._read_temperature(gpu)
        if temp is not None:
            result["temperature"] = temp
            if temp > 95:
                result["ok"] = False
                result["issues"].append(f"Critical temperature: {temp}°C")
            elif temp > 85:
                result["issues"].append(f"High temperature warning: {temp}°C")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch  # noqa: F401 # type: ignore
            return True
        except ImportError:
            return False

    def _refresh(self) -> None:
        """Detect all GPUs and populate ``self._gpus``."""
        self._gpus = []

        if not self._torch_available:
            logger.warning("PyTorch is not installed; only CPU mode available.")
            self._gpus.append(self._cpu_device())
            return

        import torch  # type: ignore

        # CUDA / ROCm devices (both use torch.cuda API)
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                self._gpus.append(self._build_cuda_gpu(i))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon
            self._gpus.append(
                GPUInfo(
                    index=0,
                    name="Apple Silicon (MPS)",
                    vendor=GPUVendor.UNKNOWN,
                    device_string="mps",
                )
            )

        # Intel XPU (experimental, PyTorch 2.0+)
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            for i in range(torch.xpu.device_count()):  # type: ignore[attr-defined]
                self._gpus.append(
                    GPUInfo(
                        index=len(self._gpus),
                        name=f"Intel XPU {i}",
                        vendor=GPUVendor.INTEL,
                        device_string=f"xpu:{i}",
                    )
                )

        if not self._gpus:
            self._gpus.append(self._cpu_device())

    def _build_cuda_gpu(self, dev_idx: int) -> GPUInfo:
        """Build a *GPUInfo* for a torch.cuda device (NVIDIA or AMD ROCm)."""
        import torch  # type: ignore

        name = torch.cuda.get_device_name(dev_idx)
        props = torch.cuda.get_device_properties(dev_idx)

        # Detect vendor from name string
        vendor = GPUVendor.NVIDIA
        if any(k in name.lower() for k in ("radeon", "vega", "navi", "amd", "rx", "gfx")):
            vendor = GPUVendor.AMD

        # Compute capability
        if vendor == GPUVendor.NVIDIA:
            cc = f"{props.major}.{props.minor}"
        else:
            # ROCm reports gcnArchName or similar
            cc = getattr(props, "gcnArchName", "") or getattr(props, "name", "")

        vram_total = props.total_memory // (1024 * 1024)

        try:
            free, _ = torch.cuda.mem_get_info(dev_idx)
            vram_free = free // (1024 * 1024)
        except Exception:
            vram_free = 0

        driver = ""
        if vendor == GPUVendor.NVIDIA:
            driver = self._nvidia_driver_version()

        return GPUInfo(
            index=dev_idx,
            name=name,
            vendor=vendor,
            device_string=f"cuda:{dev_idx}",
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            compute_capability=cc,
            driver_version=driver,
        )

    @staticmethod
    def _cpu_device() -> GPUInfo:
        return GPUInfo(
            index=0,
            name="CPU (no GPU detected)",
            vendor=GPUVendor.CPU,
            device_string="cpu",
        )

    @staticmethod
    def _nvidia_driver_version() -> str:
        """Try to read the NVIDIA driver version via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip().splitlines()[0] if result.returncode == 0 else ""
        except Exception:
            return ""

    @staticmethod
    def _read_temperature(gpu: GPUInfo) -> Optional[int]:
        """Read GPU temperature in Celsius, return None if unavailable."""
        if gpu.vendor == GPUVendor.NVIDIA:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader",
                     f"--id={gpu.index}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return int(result.stdout.strip())
            except Exception:
                pass

        elif gpu.vendor == GPUVendor.AMD:
            try:
                import json
                result = subprocess.run(
                    ["rocm-smi", "-d", str(gpu.index), "--showtemp", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    card_key = next(iter(data))
                    card = data[card_key]
                    for key in ("Temperature (Sensor edge) (C)",
                                "Temperature (Sensor junction) (C)",
                                "Temperature (C)"):
                        if key in card:
                            return int(float(str(card[key]).split()[0]))
            except Exception:
                pass

        return None

    @staticmethod
    def _amd_linux_diagnostics() -> List[str]:
        """Return AMD-specific diagnostic lines for Linux."""
        import getpass
        import grp

        lines: List[str] = []
        user = getpass.getuser()
        try:
            groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
            gid = os.getgid()
            groups.append(grp.getgrgid(gid).gr_name)
            for required in ("render", "video"):
                if required not in groups:
                    lines.append(f"WARNING: User '{user}' is NOT in the '{required}' group.")
                    lines.append(f"  Fix: sudo usermod -aG render,video {user} && reboot")
        except Exception as exc:
            lines.append(f"Could not check Linux groups: {exc}")

        for dev, label in [("/dev/kfd", "Kernel Fusion Driver"), ("/dev/dri", "DRI Render")]:
            if os.path.exists(dev):
                if not os.access(dev, os.R_OK | os.W_OK):
                    lines.append(f"PERMISSION ERROR: Cannot access {dev} ({label}).")
            else:
                lines.append(f"NOT FOUND: {dev} ({label}) — ROCm may not be installed.")

        return lines
