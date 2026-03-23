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
        color = (100, 100, 255) if track.is_static else (40, 220, 40)
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        text = f"id={track.track_id} {det.source}:{det.label} {det.score:.2f}"
        if track.is_static:
            text += " static"
        cv2.putText(
            canvas,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def serialize_frame_tracks(frame_idx: int, tracks: list[Track]) -> dict[str, Any]:
    objects = []
    for track in tracks:
        det = track.detection
        objects.append(
            {
                "track_id": track.track_id,
                "bbox_xyxy": [det.x1, det.y1, det.x2, det.y2],
                "score": det.score,
                "label": det.label,
                "source": det.source,
                "is_static": track.is_static,
                "hits": track.hits,
                "static_frames": track.static_frames,
            }
        )
    return {"frame_idx": frame_idx, "objects": objects}


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
