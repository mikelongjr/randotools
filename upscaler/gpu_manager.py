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

# ---------------------------------------------------------------------------
# ROCm GFX version mapping
# ---------------------------------------------------------------------------
# PyTorch ROCm wheels include compiled code objects only for specific GFX
# "base" targets.  Variants within the same architecture family that are NOT
# in this set need HSA_OVERRIDE_GFX_VERSION to point at the supported base.
#
# Sources: pytorch.org/whl/rocm6.x wheel metadata + ROCm compatibility docs.
_ROCM_WHEEL_GFX_TARGETS = frozenset({
    "9.0.0",   # gfx900  — Vega 10 (RX Vega 56/64)
    "9.0.2",   # gfx902  — Vega 10 Lite
    "9.0.6",   # gfx906  — Vega 20 / Radeon VII, Instinct MI50/60
    "9.0.8",   # gfx908  — Arcturus / Instinct MI100
    "9.0.a",   # gfx90a  — Aldebaran / Instinct MI200
    "9.4.0",   # gfx940  — Instinct MI300 family
    "9.4.1",   # gfx941
    "9.4.2",   # gfx942
    "10.1.0",  # gfx1010 — RDNA1 base
    "10.3.0",  # gfx1030 — RDNA2 base (Navi 21 / RX 6800 XT)
    "11.0.0",  # gfx1100 — RDNA3 base (Navi 31 / RX 7900 XT)
    "11.0.1",  # gfx1101 — Navi 32 (RX 7700 XT) — present in some wheels
    "11.0.2",  # gfx1102 — Navi 33 (RX 7600)    — present in some wheels
})

# Maps GFX family "major.minor" → the recommended override version when the
# exact GPU variant is not in _ROCM_WHEEL_GFX_TARGETS.
_ROCM_FAMILY_OVERRIDE = {
    "10.1": "10.1.0",  # RDNA1 variants
    "10.3": "10.3.0",  # RDNA2 variants (gfx1031/1032/1033/1034/1035 → gfx1030)
    "11.0": "11.0.0",  # RDNA3 variants not individually packaged
}


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

            # Detect the most common AMD GPU setup failure: the user has installed
            # the CUDA build of PyTorch (e.g. +cu128) on a machine with AMD GPU
            # hardware.  The CUDA build cannot communicate with AMD GPUs at all —
            # no environment variable override can fix this.
            if sys.platform == "linux" and cuda_ver and not hip_ver:
                if self._amd_hardware_present():
                    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                    rocm_wheel_unsupported = sys.version_info >= (3, 13)
                    lines.append("")
                    lines.append("!" * 60)
                    lines.append("CRITICAL: Wrong PyTorch build for this hardware!")
                    lines.append(
                        f"  Installed : torch {torch.__version__}  (CUDA build — for NVIDIA GPUs)"
                    )
                    lines.append(
                        "  Required  : torch with ROCm support (e.g. +rocm6.2)"
                    )
                    lines.append("")
                    lines.append("  AMD GPUs are NOT usable with a CUDA-build PyTorch.")
                    lines.append("  Environment variables (HSA_OVERRIDE_GFX_VERSION, etc.)")
                    lines.append("  have no effect until the correct build is installed.")
                    lines.append("")
                    if rocm_wheel_unsupported:
                        lines.append(
                            f"  NOTE: Python {py_ver} is detected, but PyTorch ROCm wheels"
                        )
                        lines.append(
                            "  are only published for Python 3.9–3.12. The setup script"
                        )
                        lines.append(
                            "  will automatically install Python 3.12 and retry."
                        )
                        lines.append("")
                    lines.append("  Fix — re-run the setup script (recommended):")
                    lines.append("    ./fedora_setup.sh --amd")
                    lines.append("    (tries rocm6.2 → rocm6.1 → rocm6.0 automatically)")
                    lines.append("")
                    lines.append("  Or manually reinstall PyTorch with ROCm support")
                    lines.append("  (requires Python 3.12 or earlier in your venv):")
                    lines.append(
                        "    pip install torch torchvision "
                        "--index-url https://download.pytorch.org/whl/rocm6.2"
                    )
                    lines.append("!" * 60)
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

        # Detect vendor from name string.
        # iGPUs in AMD laptops (e.g. Radeon 680M, 780M) often report generic
        # names like "Radeon Graphics" or include the codename/model number,
        # so we extend the keyword list beyond discrete-GPU-only terms.
        vendor = GPUVendor.NVIDIA
        if any(k in name.lower() for k in (
            "radeon", "vega", "navi", "amd", "rx", "gfx",
            "780m", "680m", "760m",     # common Omen/Ryzen iGPU model strings
            "raphael", "phoenix", "rembrandt",  # AMD CPU/APU codenames
        )):
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
    def _amd_hardware_present() -> bool:
        """Return True if any AMD GPU node is found in the KFD topology.

        This works regardless of which PyTorch build is installed, because it
        reads directly from the kernel's KFD sysfs interface.
        """
        topology_root = "/sys/class/kfd/kfd/topology/nodes"
        if not os.path.exists(topology_root):
            return False
        try:
            node_ids = sorted(
                int(n) for n in os.listdir(topology_root) if n.isdigit()
            )
        except Exception:
            return False
        for node_id in node_ids:
            props_file = os.path.join(topology_root, str(node_id), "properties")
            try:
                props = {}
                with open(props_file) as fh:
                    for line in fh:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            props[parts[0]] = parts[1]
                if props.get("simd_count", "0") != "0":
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _rocm_gfx_override(gfx_raw: str) -> str:
        """Return the correct HSA_OVERRIDE_GFX_VERSION for a KFD gfx_target_version.

        ``gfx_raw`` is the raw decimal string from the KFD topology (e.g.
        ``"100302"`` for gfx1032).

        PyTorch ROCm wheels only include compiled code objects for a small set of
        base GFX targets (see ``_ROCM_WHEEL_GFX_TARGETS``).  GPU variants not in
        that set cause "HIP kernel error: invalid device function" at runtime.
        Setting ``HSA_OVERRIDE_GFX_VERSION`` to the nearest base version makes
        the ROCm runtime load the correct pre-compiled kernels.

        Examples::

            _rocm_gfx_override("100302")  # gfx1032 (RX 6650M) → "10.3.0"
            _rocm_gfx_override("100305")  # gfx1035 (Radeon 680M) → "10.3.0"
            _rocm_gfx_override("110000")  # gfx1100 → "11.0.0" (already supported)
        """
        if not gfx_raw.isdigit() or len(gfx_raw) < 5:
            return gfx_raw
        v = int(gfx_raw)
        major = v // 10000
        minor = (v // 100) % 100
        patch = v % 100
        version_str = f"{major}.{minor}.{patch}"
        if version_str in _ROCM_WHEEL_GFX_TARGETS:
            return version_str  # natively supported — no override needed
        family = f"{major}.{minor}"
        if family in _ROCM_FAMILY_OVERRIDE:
            return _ROCM_FAMILY_OVERRIDE[family]
        # Unknown family — return the raw decoded version (best-effort)
        return version_str

    @staticmethod
    def apply_rocm_gfx_override() -> Optional[str]:
        """Auto-set ``HSA_OVERRIDE_GFX_VERSION`` if the AMD GPU requires it.

        Reads the KFD topology sysfs to find the hardware GFX version(s) and,
        if ``HSA_OVERRIDE_GFX_VERSION`` is not already set in the environment,
        sets it to the nearest PyTorch ROCm wheel-supported target.

        **Must be called before the first HIP kernel is executed** (i.e. before
        any ``torch.cuda.*`` or model inference call) because the ROCm HSA
        runtime reads the variable during device initialisation.

        Returns the override value that was applied, or ``None`` if no change
        was made (either already set, not on Linux, or not needed).
        """
        if sys.platform != "linux":
            return None
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            return None  # already set by user or a previous call

        topology_root = "/sys/class/kfd/kfd/topology/nodes"
        if not os.path.exists(topology_root):
            return None

        overrides: set[str] = set()
        try:
            node_ids = sorted(int(n) for n in os.listdir(topology_root) if n.isdigit())
        except Exception:
            return None

        for node_id in node_ids:
            props_file = os.path.join(topology_root, str(node_id), "properties")
            try:
                props: dict[str, str] = {}
                with open(props_file) as fh:
                    for line in fh:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            props[parts[0]] = parts[1]
                if props.get("simd_count", "0") == "0":
                    continue  # CPU-only node
                gfx_raw = props.get("gfx_target_version", "")
                if gfx_raw:
                    overrides.add(GPUManager._rocm_gfx_override(gfx_raw))
            except Exception:
                pass

        if not overrides:
            return None

        if len(overrides) == 1:
            override_ver = next(iter(overrides))
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = override_ver
            logger.info(
                "Auto-set HSA_OVERRIDE_GFX_VERSION=%s for AMD GPU compatibility.",
                override_ver,
            )
            return override_ver

        # Multiple GPUs need different overrides — cannot set a single global.
        # Log so the user knows; they can use HIP_VISIBLE_DEVICES to isolate.
        logger.warning(
            "AMD GPUs require different GFX overrides (%s). "
            "Cannot auto-set HSA_OVERRIDE_GFX_VERSION. "
            "Use HIP_VISIBLE_DEVICES to target one GPU at a time.",
            ", ".join(sorted(overrides)),
        )
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

        # Check /dev/kfd
        if os.path.exists("/dev/kfd"):
            if not os.access("/dev/kfd", os.R_OK | os.W_OK):
                lines.append("PERMISSION ERROR: Cannot access /dev/kfd (Kernel Fusion Driver).")
        else:
            lines.append("NOT FOUND: /dev/kfd (Kernel Fusion Driver) — ROCm may not be installed.")

        # Enumerate individual DRI render nodes so laptops with both an iGPU
        # and a discrete GPU (e.g. HP Omen) show each device explicitly.
        dri_path = "/dev/dri"
        if os.path.exists(dri_path):
            try:
                render_nodes = sorted(
                    e for e in os.listdir(dri_path) if e.startswith("renderD")
                )
                if render_nodes:
                    lines.append(f"DRI render nodes: {', '.join(render_nodes)}")
                    for rn in render_nodes:
                        full = os.path.join(dri_path, rn)
                        if not os.access(full, os.R_OK | os.W_OK):
                            lines.append(f"  PERMISSION ERROR: Cannot access {full}.")
                else:
                    lines.append("WARNING: No DRI render nodes found under /dev/dri.")
            except Exception as exc:
                lines.append(f"Could not enumerate DRI render nodes: {exc}")
        else:
            lines.append("NOT FOUND: /dev/dri (DRI Render) — ROCm may not be installed.")

        # Scan all KFD topology nodes and report GFX version for each GPU node.
        # On laptops with an iGPU + discrete GPU there will be multiple nodes;
        # the node at index 0 is usually the CPU and should be skipped.
        topology_root = "/sys/class/kfd/kfd/topology/nodes"
        if os.path.exists(topology_root):
            lines.append("KFD topology GPU nodes:")
            found_any = False
            try:
                node_ids = sorted(
                    int(n) for n in os.listdir(topology_root) if n.isdigit()
                )
            except Exception:
                node_ids = []
            gpu_index = 0  # HIP_VISIBLE_DEVICES uses GPU-only ordinals (0, 1, ...)
            for node_id in node_ids:
                props_file = os.path.join(topology_root, str(node_id), "properties")
                try:
                    props = {}
                    with open(props_file) as fh:
                        for line in fh:
                            parts = line.strip().split(None, 1)
                            if len(parts) == 2:
                                props[parts[0]] = parts[1]
                    # Skip CPU-only nodes (simd_count == 0)
                    if props.get("simd_count", "0") == "0":
                        continue
                    found_any = True
                    gfx_raw = props.get("gfx_target_version", "unknown")
                    # Convert raw integer like "100302" → human-readable "10.3.2"
                    gfx_readable = gfx_raw
                    if gfx_raw.isdigit() and len(gfx_raw) >= 5:
                        v = int(gfx_raw)
                        major, minor, patch = v // 10000, (v // 100) % 100, v % 100
                        gfx_readable = f"{major}.{minor}.{patch}"
                    # Use the wheel-aware mapping to determine the correct override.
                    # Directly using the raw GFX version is WRONG for GPU variants
                    # not compiled into the PyTorch ROCm wheel (e.g. gfx1032 →
                    # "10.3.2" would fail; the correct override is "10.3.0").
                    override = GPUManager._rocm_gfx_override(gfx_raw)
                    vendor_id = props.get("vendor_id", "")
                    lines.append(
                        f"  Node {node_id} (HIP device {gpu_index}): "
                        f"gfx_target_version={gfx_raw} ({gfx_readable})"
                        + (f", vendor_id={vendor_id}" if vendor_id else "")
                    )
                    if override != gfx_readable:
                        lines.append(
                            f"    Note: {gfx_readable} is not in PyTorch ROCm wheel; "
                            f"using nearest supported base {override}"
                        )
                    lines.append(
                        f"    Override hint: HSA_OVERRIDE_GFX_VERSION={override} "
                        f"HIP_VISIBLE_DEVICES={gpu_index} python upscale_frames.py"
                    )
                    gpu_index += 1
                except Exception:
                    pass
            if not found_any:
                lines.append("  No GPU nodes found in KFD topology.")
        else:
            lines.append("KFD topology not found — ROCm may not be installed.")

        # Show the current values of relevant ROCm/HIP environment variables so
        # the user can see at a glance whether overrides are already in effect.
        rocm_env_vars = (
            "HSA_OVERRIDE_GFX_VERSION",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "CUDA_VISIBLE_DEVICES",
        )
        env_lines = []
        for var in rocm_env_vars:
            val = os.environ.get(var)
            if val is not None:
                env_lines.append(f"  {var}={val}")
        if env_lines:
            lines.append("Active ROCm/HIP environment overrides:")
            lines.extend(env_lines)

        return lines
