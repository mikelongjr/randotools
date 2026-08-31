"""Typed job and progress objects shared by the GUI and pipeline."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class JobKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class JobStage(str, Enum):
    QUEUED = "queued"
    PROBING = "probing"
    EXTRACTING = "extracting"
    UPSCALING = "upscaling"
    ENCODING = "encoding"
    CLEANING = "cleaning"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioMode(str, Enum):
    COPY = "copy"
    AAC = "aac"
    STRIP = "strip"


@dataclass
class OutputSettings:
    """Settings that affect the final encoded output."""

    container: str = "mp4"
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    audio_mode: AudioMode = AudioMode.COPY
    keep_frames: bool = False
    constant_fps: bool = False


@dataclass
class UpscaleJob:
    """A single source file queued for upscaling."""

    source_path: str
    output_dir: str
    kind: JobKind
    model_name: str
    settings: OutputSettings = field(default_factory=OutputSettings)
    job_id: str = ""
    output_path: str = ""
    status: JobStatus = JobStatus.PENDING
    stage: JobStage = JobStage.QUEUED
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        source = str(Path(self.source_path).expanduser().resolve())
        self.source_path = source
        if not self.job_id:
            digest = hashlib.sha1(f"{source}:{self.created_at}".encode("utf-8")).hexdigest()
            self.job_id = digest[:16]

    @property
    def filename(self) -> str:
        return Path(self.source_path).name

    @property
    def display_name(self) -> str:
        return self.filename


@dataclass
class PipelineEvent:
    """Progress notification emitted by long-running pipeline stages."""

    job_id: str
    stage: JobStage
    message: str = ""
    completed: int = 0
    total: int = 0
    current_file: str = ""
    eta_s: float = -1.0
    avg_s_per_frame: float = 0.0
    output_path: Optional[str] = None
