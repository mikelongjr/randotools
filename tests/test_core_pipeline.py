from pathlib import Path

import pytest

from upscaler.config import Config
from upscaler.core.ffmpeg_runner import CommandResult, ProcessCancelled
from upscaler.core.jobs import AudioMode, JobKind, JobStage, OutputSettings, UpscaleJob
from upscaler.core.media import parse_rate
from upscaler.core.video_pipeline import VideoPipeline
from upscaler.core.workspace import JobWorkspace


class FakeRunner:
    def __init__(self):
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.cancelled = False
        self.commands = []

    def probe_json(self, video_path):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": "30000/1001",
                    "duration": "2.0",
                }
            ],
            "format": {"duration": "2.0"},
        }

    def run(self, cmd, cwd=None, on_stderr=None):
        self.commands.append(cmd)
        pattern = Path(cmd[-1])
        if pattern.name == "frame_%06d.png":
            for idx in range(1, 4):
                (pattern.parent / f"frame_{idx:06d}.png").write_text("frame", encoding="utf-8")
        return CommandResult(cmd, "", "", 0)

    def cancel(self):
        self.cancelled = True


def make_job(tmp_path, **settings):
    source = tmp_path / "input.mp4"
    source.write_text("video", encoding="utf-8")
    return UpscaleJob(
        source_path=str(source),
        output_dir=str(tmp_path / "out"),
        kind=JobKind.VIDEO,
        model_name="RealESRGAN_x2plus",
        settings=OutputSettings(**settings),
    )


def test_parse_rate_handles_fractional_and_zero_rates():
    assert parse_rate("30000/1001") == pytest.approx(29.970, rel=1e-3)
    assert parse_rate("25") == 25.0
    assert parse_rate("0/0") == 0.0


def test_video_pipeline_extracts_to_clean_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("upscaler.core.video_pipeline.require_media_tools", lambda *args: None)
    runner = FakeRunner()
    pipeline = VideoPipeline(runner=runner, workspace_root=tmp_path / "cache")
    job = make_job(tmp_path)
    workspace = pipeline.workspace_for(job)
    workspace.prepare()
    (workspace.extract_dir / "frame_999999.png").write_text("stale", encoding="utf-8")

    frames = pipeline.extract_frames(job, workspace, pipeline.probe(job.source_path), resume=False)

    assert [p.name for p in frames] == ["frame_000001.png", "frame_000002.png", "frame_000003.png"]
    assert not (workspace.extract_dir / "frame_999999.png").exists()
    assert any("-vsync" in cmd for cmd in runner.commands)


def test_video_pipeline_encode_respects_audio_mode(tmp_path):
    runner = FakeRunner()
    pipeline = VideoPipeline(runner=runner, workspace_root=tmp_path / "cache")
    job = make_job(tmp_path, audio_mode=AudioMode.AAC, container="mkv")
    workspace = pipeline.workspace_for(job)
    workspace.prepare(clean=True)
    for idx in range(1, 3):
        (workspace.upscale_dir / f"frame_{idx:06d}.png").write_text("frame", encoding="utf-8")

    output = pipeline.encode_video(job, workspace, pipeline.probe(job.source_path), scale=2)

    cmd = runner.commands[-1]
    assert output.endswith(".mkv")
    assert "-c:a" in cmd
    assert "aac" in cmd
    assert "-crf" in cmd


def test_video_pipeline_cancellation_stops_frame_processing(tmp_path):
    runner = FakeRunner()
    runner.cancelled = True
    pipeline = VideoPipeline(runner=runner, workspace_root=tmp_path / "cache")
    job = make_job(tmp_path)
    workspace = pipeline.workspace_for(job)
    workspace.prepare(clean=True)
    frame = workspace.extract_dir / "frame_000001.png"
    frame.write_text("frame", encoding="utf-8")

    with pytest.raises(ProcessCancelled):
        pipeline.upscale_frames(job, workspace, [frame], lambda *_: None)


def test_workspace_state_round_trip_and_cleanup(tmp_path):
    workspace = JobWorkspace("abc123", root=tmp_path)
    workspace.prepare(clean=True)
    workspace.write_state({"stage": JobStage.UPSCALING.value, "frames": 3})

    assert workspace.read_state()["frames"] == 3
    workspace.cleanup(remove_root=True)
    assert not workspace.root.exists()


def test_config_prefers_user_weight_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    cfg = Config()
    expected_dir = tmp_path / "data" / "realesrgan-upscaler" / "weights"
    expected_dir.mkdir(parents=True)
    weight = expected_dir / "RealESRGAN_x2plus.pth"
    weight.write_text("weights", encoding="utf-8")

    assert cfg.resolve_weights_path("RealESRGAN_x2plus") == str(weight)
    assert cfg.writable_weights_dir() == str(expected_dir)
