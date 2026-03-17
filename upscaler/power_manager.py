"""
Power Manager - Real-time temperature and power management for NVIDIA and AMD GPUs.

Enhanced version of temperature_watcher.py with:
  - Support for both NVIDIA (nvidia-smi) and AMD (rocm-smi) GPUs
  - Thermal throttling detection
  - Power limit management
  - Automatic processing pause on critical temperatures
  - Temperature history tracking
  - Background monitoring thread

Usage::

    from upscaler.power_manager import PowerManager, ThermalEvent
    pm = PowerManager(gpu_index=0, vendor="nvidia")
    pm.start()
    pm.stop()
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, List, Optional

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """Current thermal state of the GPU."""

    NORMAL = auto()       # Temperature within safe operating range
    WARM = auto()         # Elevated — monitoring closely
    HIGH = auto()         # High temperature — reduced power limit applied
    CRITICAL = auto()     # Critical — processing paused
    UNKNOWN = auto()      # Temperature could not be read


@dataclass
class ThermalEvent:
    """A single temperature reading event."""

    timestamp: float
    temperature_c: int
    power_draw_w: Optional[float]
    state: ThermalState
    action_taken: str = ""


@dataclass
class PowerConfig:
    """Power and thermal configuration thresholds."""

    poll_interval_s: float = 2.0

    # Temperature thresholds (Celsius)
    warm_threshold_c: int = 75
    high_threshold_c: int = 85
    critical_threshold_c: int = 95

    # Power limits (Watts); 0 = don't change
    default_power_w: int = 125
    high_temp_power_w: int = 90
    critical_power_w: int = 60

    # History
    history_size: int = 120   # Keep last 4 minutes at 2s poll


class PowerManager:
    """
    Monitors and manages GPU temperature and power limits.

    Supports NVIDIA GPUs via nvidia-smi and AMD GPUs via rocm-smi.
    Runs a background daemon thread that periodically polls the GPU.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        vendor: str = "auto",
        config: Optional[PowerConfig] = None,
        on_state_change: Optional[Callable[[ThermalEvent], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        gpu_index:
            GPU device index (0-based).
        vendor:
            ``"nvidia"``, ``"amd"``, or ``"auto"`` (auto-detect from available tools).
        config:
            Thermal and power thresholds; defaults to *PowerConfig()*.
        on_state_change:
            Optional callback invoked on every poll cycle with the latest
            *ThermalEvent*.  Called from the background thread.
        """
        self.gpu_index = gpu_index
        self.config = config or PowerConfig()
        self.on_state_change = on_state_change

        self._vendor = self._resolve_vendor(vendor)
        self._state = ThermalState.UNKNOWN
        self._history: Deque[ThermalEvent] = deque(maxlen=self.config.history_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pause_processing = threading.Event()  # Set when processing should pause

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> ThermalState:
        """Current thermal state."""
        return self._state

    @property
    def history(self) -> List[ThermalEvent]:
        """Snapshot of the temperature history list."""
        with self._lock:
            return list(self._history)

    @property
    def latest_temperature(self) -> Optional[int]:
        """Most recently read temperature in Celsius, or *None*."""
        with self._lock:
            if self._history:
                return self._history[-1].temperature_c
        return None

    @property
    def should_pause(self) -> bool:
        """True when temperature is critical and processing should be paused."""
        return self._pause_processing.is_set()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="PowerManager",
            daemon=True,
        )
        self._thread.start()
        logger.info("PowerManager started (vendor=%s, gpu=%d)", self._vendor, self.gpu_index)

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("PowerManager stopped")

    def read_temperature(self) -> Optional[int]:
        """Read current GPU temperature once (outside the monitoring loop)."""
        if self._vendor == "nvidia":
            return self._read_nvidia_temp()
        elif self._vendor == "amd":
            return self._read_amd_temp()
        return None

    def set_power_limit(self, watts: int) -> bool:
        """
        Set GPU power limit.  Requires elevated privileges on most systems.

        Returns *True* on success.
        """
        if self._vendor == "nvidia":
            return self._set_nvidia_power(watts)
        elif self._vendor == "amd":
            return self._set_amd_power(watts)
        return False

    def get_summary(self) -> str:
        """Return a human-readable summary of the current thermal state."""
        temp = self.latest_temperature
        temp_str = f"{temp}°C" if temp is not None else "N/A"
        return (
            f"GPU {self.gpu_index} ({self._vendor.upper()}) | "
            f"Temp: {temp_str} | State: {self._state.name}"
        )

    # ------------------------------------------------------------------
    # Background monitoring loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                logger.error("PowerManager poll error: %s", exc)
            self._stop_event.wait(timeout=self.config.poll_interval_s)

    def _poll_once(self) -> None:
        temp = self.read_temperature()
        power = self._read_power_draw()

        if temp is None:
            state = ThermalState.UNKNOWN
            action = ""
        elif temp >= self.config.critical_threshold_c:
            state = ThermalState.CRITICAL
            action = f"Processing paused; power limit → {self.config.critical_power_w}W"
            self._pause_processing.set()
            if self.config.critical_power_w > 0:
                self.set_power_limit(self.config.critical_power_w)
        elif temp >= self.config.high_threshold_c:
            state = ThermalState.HIGH
            action = f"Power limit → {self.config.high_temp_power_w}W"
            self._pause_processing.clear()
            if self.config.high_temp_power_w > 0:
                self.set_power_limit(self.config.high_temp_power_w)
        elif temp >= self.config.warm_threshold_c:
            state = ThermalState.WARM
            action = "Monitoring closely"
            self._pause_processing.clear()
        else:
            state = ThermalState.NORMAL
            action = ""
            self._pause_processing.clear()
            # Restore default power limit if transitioning from high
            if self._state in (ThermalState.HIGH, ThermalState.CRITICAL):
                action = f"Power limit restored → {self.config.default_power_w}W"
                if self.config.default_power_w > 0:
                    self.set_power_limit(self.config.default_power_w)

        event = ThermalEvent(
            timestamp=time.time(),
            temperature_c=temp if temp is not None else 0,
            power_draw_w=power,
            state=state,
            action_taken=action,
        )

        with self._lock:
            self._history.append(event)
            self._state = state

        if self.on_state_change:
            try:
                self.on_state_change(event)
            except Exception as exc:
                logger.error("on_state_change callback raised: %s", exc)

        if action:
            logger.info("[GPU %d] Temp=%s | %s", self.gpu_index, temp, action)

    # ------------------------------------------------------------------
    # NVIDIA helpers
    # ------------------------------------------------------------------

    def _read_nvidia_temp(self) -> Optional[int]:
        """Read temperature via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as exc:
            logger.debug("nvidia-smi temp read failed: %s", exc)
        return None

    def _read_power_draw(self) -> Optional[float]:
        """Read current power draw in Watts."""
        try:
            if self._vendor == "nvidia":
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    txt = result.stdout.strip()
                    if txt and txt != "N/A":
                        return float(txt)
            elif self._vendor == "amd":
                result = subprocess.run(
                    ["rocm-smi", "-d", str(self.gpu_index), "--showpower", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    card = data.get(next(iter(data)), {})
                    for key in ("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)"):
                        if key in card:
                            return float(str(card[key]).split()[0])
        except Exception as exc:
            logger.debug("Power draw read failed: %s", exc)
        return None

    def _set_nvidia_power(self, watts: int) -> bool:
        """Set NVIDIA GPU power limit via nvidia-smi (requires root on Linux)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}", f"--power-limit={watts}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("NVIDIA power limit set to %dW", watts)
                return True
            logger.warning("nvidia-smi power-limit failed: %s", result.stderr.strip())
        except Exception as exc:
            logger.warning("Failed to set NVIDIA power limit: %s", exc)
        return False

    # ------------------------------------------------------------------
    # AMD helpers
    # ------------------------------------------------------------------

    def _read_amd_temp(self) -> Optional[int]:
        """Read AMD GPU temperature via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.gpu_index), "--showtemp", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                card = data.get(next(iter(data)), {})
                for key in (
                    "Temperature (Sensor edge) (C)",
                    "Temperature (Sensor junction) (C)",
                    "Temperature (C)",
                ):
                    if key in card:
                        match = re.match(r"\s*([0-9.]+)", str(card[key]))
                        if match:
                            return int(float(match.group(1)))
                # Generic search
                for k, v in card.items():
                    if "temp" in k.lower() and "(c)" in k.lower():
                        match = re.match(r"\s*([0-9.]+)", str(v))
                        if match:
                            return int(float(match.group(1)))
        except Exception as exc:
            logger.debug("rocm-smi temp read failed: %s", exc)
        return None

    def _set_amd_power(self, watts: int) -> bool:
        """Set AMD GPU power limit via rocm-smi (may require sudo)."""
        try:
            result = subprocess.run(
                ["sudo", "rocm-smi", "-d", str(self.gpu_index), "--setpower", str(watts)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("AMD power limit set to %dW", watts)
                return True
            logger.warning("rocm-smi setpower failed: %s", result.stderr.strip())
        except Exception as exc:
            logger.warning("Failed to set AMD power limit: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Vendor auto-detect
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_vendor(vendor: str) -> str:
        if vendor in ("nvidia", "amd"):
            return vendor
        # Auto-detect from available CLI tools
        if shutil.which("nvidia-smi"):
            return "nvidia"
        if shutil.which("rocm-smi"):
            return "amd"
        logger.warning("No GPU management tool found (nvidia-smi / rocm-smi). Temperature monitoring disabled.")
        return "unknown"
