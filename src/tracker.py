from __future__ import annotations

from dataclasses import dataclass, field

from src.detector import Detection, iou


@dataclass
class Track:
    track_id: int
    detection: Detection
    age: int = 0
    hits: int = 1
    missed: int = 0
    static_frames: int = 0
    is_static: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)

    def update(self, detection: Detection, static_iou_threshold: float, static_center_threshold: float) -> None:
        previous = self.detection
        self.age += 1
        self.hits += 1
        self.missed = 0
        self.history.append(detection.center)
        self.history = self.history[-20:]

        center_shift = l2_distance(previous.center, detection.center)
        size_shift = 1.0 - iou(previous, detection)

        if center_shift <= static_center_threshold and size_shift <= (1.0 - static_iou_threshold):
            self.static_frames += 1
        else:
            self.static_frames = 0
            self.is_static = False

        self.detection = detection

    def mark_missed(self) -> None:
        self.age += 1
        self.missed += 1


class SimpleTracker:
    """
    Простой IoU + расстояние по центрам трекер для MVP.
    """

    def __init__(
        self,
        max_match_distance: float = 80.0,
        min_iou_for_match: float = 0.01,
        max_missed_frames: int = 8,
        static_center_threshold: float = 4.0,
        static_iou_threshold: float = 0.92,
        static_frame_window: int = 8,
    ) -> None:
        self.max_match_distance = max_match_distance
        self.min_iou_for_match = min_iou_for_match
        self.max_missed_frames = max_missed_frames
        self.static_center_threshold = static_center_threshold
        self.static_iou_threshold = static_iou_threshold
        self.static_frame_window = static_frame_window
        self._next_track_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        scored_pairs: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                distance = l2_distance(track.detection.center, detection.center)
                overlap = iou(track.detection, detection)
                if distance <= self.max_match_distance:
                    score = overlap + max(0.0, 1.0 - distance / max(self.max_match_distance, 1.0))
                    if overlap >= self.min_iou_for_match or distance <= self.max_match_distance * 0.5:
                        scored_pairs.append((score, track_idx, det_idx))

        for _, track_idx, det_idx in sorted(scored_pairs, reverse=True):
            if track_idx not in unmatched_tracks or det_idx not in unmatched_detections:
                continue
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)
            matches.append((track_idx, det_idx))

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            track.update(
                detections[det_idx],
                static_iou_threshold=self.static_iou_threshold,
                static_center_threshold=self.static_center_threshold,
            )
            if track.static_frames >= self.static_frame_window:
                track.is_static = True

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track = Track(
                track_id=self._next_track_id,
                detection=detection,
                history=[detection.center],
            )
            self._next_track_id += 1
            self.tracks.append(track)

        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed_frames]
        return list(self.tracks)


def l2_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5
