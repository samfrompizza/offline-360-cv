from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    """
    Open a video file and return cv2.VideoCapture.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    return cap


def iter_video_frames(
    video_path: str | Path,
    every_n_frames: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_index, frame) for every N-th frame.
    """
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")

    cap = open_video(video_path)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every_n_frames == 0:
                yield frame_idx, frame

            frame_idx += 1
    finally:
        cap.release()


def get_video_metadata(video_path: str | Path) -> dict:
    """
    Return basic metadata for debugging.
    """
    cap = open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec: Optional[float] = None
        if fps and fps > 0:
            duration_sec = frame_count / fps

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
        }
    finally:
        cap.release()


def create_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer


def save_frame(frame: np.ndarray, output_path: str | Path) -> None:
    """
    Save a single frame as an image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        raise RuntimeError(f"Could not write frame to {output_path}")
