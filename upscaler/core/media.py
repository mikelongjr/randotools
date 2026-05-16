"""Shared media metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    fps: float
    width: int
    height: int
    duration: float


def parse_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_i = int(den)
        return int(num) / den_i if den_i else 0.0
    return float(rate)
