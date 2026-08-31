"""GUI-independent video upscaling pipeline."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from upscaler.core.ffmpeg_runner import FfmpegRunner, ProcessCancelled, require_media_tools
from upscaler.core.jobs import AudioMode, JobStage, PipelineEvent, UpscaleJob
from upscaler.core.media import VideoMetadata, parse_rate
from upscaler.core.workspace import JobWorkspace

ProgressCallback = Callable[[PipelineEvent], None]
FrameProcessor = Callable[[Path, Path, int, int], None]


class VideoPipeline:
    """Probe, extract, upscale, and encode one video job."""

    def __init__(
        self,
        runner: Optional[FfmpegRunner] = None,
        workspace_root: Optional[Path] = None,
    ) -> None:
        self.runner = runner or FfmpegRunner()
        self.workspace_root = workspace_root

    def cancel(self) -> None:
        self.runner.cancel()

    def probe(self, video_path: str) -> VideoMetadata:
        data = self.runner.probe_json(video_path)
        streams = data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        fps = parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "0/0"))
        duration = float(
            video_stream.get("duration")
            or data.get("format", {}).get("duration")
            or 0.0
        )
        return VideoMetadata(
            fps=fps,
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            duration=duration,
        )

    def workspace_for(self, job: UpscaleJob) -> JobWorkspace:
        return JobWorkspace(job.job_id, root=self.workspace_root)

    def extract_frames(
        self,
        job: UpscaleJob,
        workspace: JobWorkspace,
        metadata: VideoMetadata,
        progress: Optional[ProgressCallback] = None,
        resume: bool = False,
    ) -> List[Path]:
        require_media_tools(self.runner.ffmpeg, self.runner.ffprobe)
        workspace.prepare(clean=not resume)

        existing = sorted(workspace.extract_dir.glob("frame_*.png"))
        if resume and existing:
            return existing

        self._emit(progress, job, JobStage.EXTRACTING, "Extracting video frames")
        cmd = [self.runner.ffmpeg, "-hide_banner", "-i", job.source_path]
        if job.settings.constant_fps and metadata.fps:
            cmd.extend(["-r", f"{metadata.fps:.6f}"])
        else:
            cmd.extend(["-vsync", "0"])
        cmd.extend([
            "-start_number",
            "1",
            "-y",
            str(workspace.extract_dir / "frame_%06d.png"),
        ])
        self.runner.run(cmd)
        frames = sorted(workspace.extract_dir.glob("frame_*.png"))
        if not frames:
            raise ValueError("No frames were extracted from the video.")
        workspace.write_state({
            "stage": JobStage.EXTRACTING.value,
            "source": job.source_path,
            "frames": len(frames),
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
        })
        return frames

    def upscale_frames(
        self,
        job: UpscaleJob,
        workspace: JobWorkspace,
        frames: Iterable[Path],
        processor: FrameProcessor,
        progress: Optional[ProgressCallback] = None,
        resume: bool = True,
    ) -> List[Path]:
        frame_list = list(frames)
        total = len(frame_list)
        times: List[float] = []
        outputs: List[Path] = []

        for idx, src in enumerate(frame_list):
            if self.runner.cancelled:
                raise ProcessCancelled("operation cancelled")
            dst = workspace.upscale_dir / src.name
            outputs.append(dst)
            if resume and dst.exists():
                self._emit(progress, job, JobStage.UPSCALING, f"Skipping {src.name}", idx + 1, total, src.name)
                continue
            started = time.perf_counter()
            processor(src, dst, idx, total)
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            avg = sum(times[-10:]) / min(len(times), 10)
            eta = avg * (total - idx - 1)
            self._emit(progress, job, JobStage.UPSCALING, f"Upscaled {src.name}", idx + 1, total, src.name, eta, avg)

        workspace.write_state({"stage": JobStage.UPSCALING.value, "frames": total})
        return outputs

    def encode_video(
        self,
        job: UpscaleJob,
        workspace: JobWorkspace,
        metadata: VideoMetadata,
        scale: int,
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        self._emit(progress, job, JobStage.ENCODING, "Encoding final video")
        source = Path(job.source_path)
        out_dir = Path(job.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = Path(job.output_path) if job.output_path else out_dir / f"{source.stem}_upscaled.{job.settings.container}"
        if output.suffix.lower() != f".{job.settings.container.lower()}":
            output = output.with_suffix(f".{job.settings.container}")

        cmd = [
            self.runner.ffmpeg,
            "-hide_banner",
            "-framerate",
            f"{metadata.fps:.6f}" if metadata.fps else "30",
            "-i",
            str(workspace.upscale_dir / "frame_%06d.png"),
            "-i",
            job.source_path,
            "-map",
            "0:v:0",
            "-c:v",
            job.settings.video_codec,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(job.settings.crf),
            "-preset",
            job.settings.preset,
        ]

        if job.settings.audio_mode != AudioMode.STRIP:
            cmd.extend(["-map", "1:a:0?"])
            if job.settings.audio_mode == AudioMode.AAC:
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            else:
                cmd.extend(["-c:a", "copy"])

        cmd.extend(["-y", str(output)])
        self.runner.run(cmd)
        job.output_path = str(output)
        workspace.write_state({"stage": JobStage.COMPLETE.value, "output": str(output)})
        return str(output)

    def cleanup(self, workspace: JobWorkspace, keep_frames: bool) -> None:
        if keep_frames:
            shutil.rmtree(workspace.extract_dir, ignore_errors=True)
            return
        workspace.cleanup(remove_root=True)

    def process(
        self,
        job: UpscaleJob,
        processor: FrameProcessor,
        scale: int,
        progress: Optional[ProgressCallback] = None,
        resume: bool = True,
    ) -> str:
        workspace = self.workspace_for(job)
        try:
            self._emit(progress, job, JobStage.PROBING, "Reading video metadata")
            metadata = self.probe(job.source_path)
            frames = self.extract_frames(job, workspace, metadata, progress, resume=resume)
            self.upscale_frames(job, workspace, frames, processor, progress, resume=resume)
            output = self.encode_video(job, workspace, metadata, scale, progress)
            self._emit(progress, job, JobStage.COMPLETE, f"Video complete: {output}", output_path=output)
            return output
        except ProcessCancelled:
            self._emit(progress, job, JobStage.CANCELLED, "Video job cancelled")
            raise
        except Exception as exc:
            self._emit(progress, job, JobStage.FAILED, str(exc))
            raise
        finally:
            if not self.runner.cancelled:
                self.cleanup(workspace, keep_frames=job.settings.keep_frames)

    @staticmethod
    def _emit(
        progress: Optional[ProgressCallback],
        job: UpscaleJob,
        stage: JobStage,
        message: str,
        completed: int = 0,
        total: int = 0,
        current_file: str = "",
        eta_s: float = -1.0,
        avg_s_per_frame: float = 0.0,
        output_path: Optional[str] = None,
    ) -> None:
        if progress:
            progress(PipelineEvent(
                job_id=job.job_id,
                stage=stage,
                message=message,
                completed=completed,
                total=total,
                current_file=current_file,
                eta_s=eta_s,
                avg_s_per_frame=avg_s_per_frame,
                output_path=output_path,
            ))
