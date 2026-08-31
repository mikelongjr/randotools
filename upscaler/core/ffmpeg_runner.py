"""Cancellable ffmpeg/ffprobe subprocess helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


class FfmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


class ProcessCancelled(RuntimeError):
    """Raised when a running media subprocess is cancelled."""


@dataclass
class CommandResult:
    command: List[str]
    stdout: str
    stderr: str
    returncode: int


class FfmpegRunner:
    """Small wrapper around ffmpeg/ffprobe with cancellation support."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            proc = self._process
        if proc and proc.poll() is None:
            self._terminate_process(proc)

    def reset_cancelled(self) -> None:
        self._cancelled.clear()

    def probe_json(self, video_path: str) -> dict:
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            video_path,
        ]
        result = self.run(cmd)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise FfmpegError(f"ffprobe returned invalid JSON for {video_path}: {exc}") from exc

    def run(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        if self.cancelled:
            raise ProcessCancelled("operation cancelled")

        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs,
        )
        with self._lock:
            self._process = proc

        stderr_lines: List[str] = []
        try:
            stdout, stderr = proc.communicate()
            if stderr:
                stderr_lines.extend(stderr.splitlines())
                if on_stderr:
                    for line in stderr_lines:
                        on_stderr(line)
            if self.cancelled:
                raise ProcessCancelled("operation cancelled")
            if proc.returncode != 0:
                tail = "\n".join(stderr_lines[-20:])
                raise FfmpegError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
            return CommandResult(cmd, stdout or "", stderr or "", proc.returncode)
        finally:
            with self._lock:
                if self._process is proc:
                    self._process = None

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass


def require_media_tools(ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
    """Raise a clear error if ffmpeg/ffprobe are not available."""

    import shutil

    missing = [tool for tool in (ffmpeg_path, ffprobe_path) if shutil.which(tool) is None]
    if missing:
        raise FileNotFoundError(
            "Missing required video tool(s): "
            + ", ".join(missing)
            + ". Install ffmpeg, then try again."
        )
