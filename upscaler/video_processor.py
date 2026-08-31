import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from upscaler.core.ffmpeg_runner import FfmpegRunner, require_media_tools
from upscaler.core.media import VideoMetadata, parse_rate


logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video frame extraction, concatenation (via sequential extraction),
    and video encoding using ffmpeg.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self.runner = FfmpegRunner(ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path)

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Extracts FPS, resolution, and duration using ffprobe."""
        try:
            data = self.runner.probe_json(video_path)
            streams = data.get("streams", [])
            stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            if not stream:
                raise ValueError(f"No video stream found in {video_path}")

            fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate", "0/0"))
            duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
            return VideoMetadata(
                fps=fps,
                width=int(stream["width"]),
                height=int(stream["height"]),
                duration=duration,
            )
        except Exception as e:
            logger.error(f"Failed to get metadata for {video_path}: {e}")
            raise

    def extract_frames(self, input_files: List[str], output_dir: str, fps: Optional[float] = None) -> int:
        """
        Extracts frames from a list of video files into output_dir.
        Frames are named sequentially across all input files to facilitate concatenation.
        """
        require_media_tools(self.ffmpeg, self.ffprobe)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        total_frames = 0

        for video_path in input_files:
            logger.info(f"Extracting frames from {video_path}...")
            # We use -start_number to ensure sequential naming across multiple files
            # But we need to know how many frames were extracted from previous files.
            # Actually, ffmpeg's -start_number is for the *current* command.
            # To do this correctly, we can use a pattern that includes the frame index.

            # A better way: use a pattern that starts from total_frames + 1
            # e.g., frame_000001.png, frame_000002.png ...
            # We need to find the next available number.
            
            # However, if we want to be robust, we can just extract them and then
            # sort them. But ffmpeg doesn't easily let us specify the starting index
            # in a way that's easy to track without knowing the count.
            
            # Let's try: ffmpeg -i input -start_number N ...
            # We'll need to count frames after each extraction or use a more clever way.
            
            # Actually, we can just use a pattern like %06d and then rename them,
            # or just use the current total_frames as the start number.
            
            start_num = total_frames + 1
            cmd = [
                self.ffmpeg,
                "-hide_banner",
                "-i",
                video_path,
            ]
            if fps:
                # Use -r for output framerate to ensure consistent frame count
                cmd.extend(["-r", str(fps)])
            else:
                cmd.extend(["-vsync", "0"])
            
            cmd.extend([
                "-start_number",
                str(start_num),
                "-q:v",
                "2",  # High quality for PNG (though PNG is lossless, this is for other formats)
                "-y",  # Overwrite
                os.path.join(output_dir, "frame_%06d.png"),
            ])
            
            try:
                # We don't want to capture all output as it can be huge, 
                # but we need to know if it succeeded.
                subprocess.run(cmd, check=True, capture_output=True)
                
                # After extraction, we need to know how many frames were added.
                # We can count the files in the directory that match the pattern.
                # But that's slow. 
                # Better: use ffprobe to get the number of frames in the input file.
                
                frame_count_cmd = [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_packets",
                    "-show_entries",
                    "stream=nb_read_packets",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ]
                # Note: nb_read_packets can be slow for some formats. 
                # An alternative is to count the files in the output dir.
                
                # Let's just count the files in the output dir that were just added.
                # This is a bit hacky but works.
                # We'll look for files in the range [start_num, end_num]
                
                # Actually, let's just use the number of files in the directory 
                # that match the pattern and are greater than the previous total.
                
                # Let's try a simpler approach: extract all, then count.
                # But we need to know the total count to return it.
                
                # Let's use the number of files in the directory.
                # We'll count all files matching 'frame_*.png'
                
                # Wait, if we have multiple files, we want them to be frame_000001.png, frame_000002.png...
                # across all of them.
                
                # Let's re-run the count.
                # We can use the output of ffmpeg if we parse it, but it's messy.
                
                # Let's use the directory listing.
                files = list(Path(output_dir).glob("frame_*.png"))
                # This is not quite right because it includes old files.
                # We need to know how many were added in THIS step.
                
                # Let's use a more reliable way to get frame count from ffmpeg.
                # ffmpeg -i input -print_format csv -show_entries stream=nb_frames -of csv=p=0
                # This works for many formats but not all.
                
                # Let's try counting the files in the directory that are >= start_num.
                # We'll assume the pattern is frame_XXXXXX.png
                
                current_files = 0
                for f in Path(output_dir).iterdir():
                    if f.name.startswith("frame_") and f.suffix == ".png":
                        try:
                            idx = int(f.stem.split("_")[1])
                            if idx >= start_num:
                                current_files += 1
                        except (ValueError, IndexError):
                            continue
                
                total_frames += current_files
                logger.info(f"Extracted {current_files} frames. Total so far: {total_frames}")

            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg error during extraction of {video_path}: {e.stderr.decode()}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during extraction of {video_path}: {e}")
                raise

        return total_frames

    def encode_video(
        self,
        frame_dir: str,
        output_file: str,
        fps: float,
        width: int,
        height: int,
        original_video_path: Optional[str] = None,
        container: str = "mp4",
        reencode_audio: bool = False,
    ) -> None:
        """
        Encodes the upscaled frames into a video file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Ensure container extension matches
        if not output_file.endswith(f".{container}"):
            output_file = os.path.splitext(output_file)[0] + f".{container}"

        # ffmpeg command for encoding a sequence of images
        # -framerate: input framerate
        # -i: input pattern
        # -c:v libx264: H.264 codec
        # -pix_fmt yuv420p: ensure compatibility with most players
        # -crf 18: high quality (lower is better, 18-23 is standard)
        # -preset medium: encoding speed/compression tradeoff
        
        # Base command for frames
        cmd = [
            self.ffmpeg,
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frame_dir, "frame_%06d.png"),
        ]

        # If original video is provided, add it as a second input to copy audio
        if original_video_path:
            cmd.extend(["-i", original_video_path])

        # Encoding settings
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
        ])

        # Map video from frames (input 0) and audio from original video (input 1)
        if original_video_path:
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
            if reencode_audio:
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            else:
                cmd.extend(["-c:a", "copy"]) # Copy audio without re-encoding
        
        cmd.extend([output_file, "-y"])

        try:
            logger.info(f"Encoding video to {output_file} at {fps} FPS ({width}x{height})...")
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Video encoding complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error during encoding: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during encoding: {e}")
            raise
