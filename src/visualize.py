from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tracker import Track


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    canvas = frame.copy()
    for track in tracks:
        det = track.detection
        color = (0, 255, 255)
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def serialize_frame_tracks(
    frame_idx: int,
    tracks: list[Track],
    frame_width: int,
    frame_height: int,
    fps: float | None,
) -> dict[str, Any]:
    objects = []
    for track in tracks:
        det = track.detection
        center_x, center_y = det.center
        objects.append(
            {
                "track_id": track.track_id,
                "bbox_xyxy": [
                    round(det.x1, 2),
                    round(det.y1, 2),
                    round(det.x2, 2),
                    round(det.y2, 2),
                ],
                "center_xy": [round(center_x, 2), round(center_y, 2)],
                "center_norm": [
                    round(center_x / frame_width, 6),
                    round(center_y / frame_height, 6),
                ],
            }
        )

    frame_data: dict[str, Any] = {"frame_idx": frame_idx, "objects": objects}
    if fps and fps > 0:
        frame_data["time_sec"] = round(frame_idx / fps, 6)
    return frame_data


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
