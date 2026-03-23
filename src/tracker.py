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

    def predicted_center(self) -> tuple[float, float]:
        if len(self.history) < 2:
            return self.detection.center
        prev_x, prev_y = self.history[-2]
        curr_x, curr_y = self.history[-1]
        return (curr_x + (curr_x - prev_x), curr_y + (curr_y - prev_y))

    def velocity(self) -> tuple[float, float]:
        if len(self.history) < 2:
            return (0.0, 0.0)
        prev_x, prev_y = self.history[-2]
        curr_x, curr_y = self.history[-1]
        return (curr_x - prev_x, curr_y - prev_y)


class SimpleTracker:
    """
    Простой IoU + расстояние по центрам трекер для MVP.

    Опционально включает очень лёгкий "конус" ожидаемых позиций:
    по двум последним центрам оценивается скорость, после чего матчинг
    разрешается только рядом с прогнозируемой точкой. Радиус конуса
    расширяется с ростом скорости и количества пропусков.
    """

    def __init__(
        self,
        max_match_distance: float = 80.0,
        min_iou_for_match: float = 0.01,
        max_missed_frames: int = 4,
        static_center_threshold: float = 4.0,
        static_iou_threshold: float = 0.92,
        static_frame_window: int = 8,
        use_cone_filter: bool = False,
        cone_base_radius: float = 20.0,
        cone_velocity_gain: float = 2.5,
        cone_missed_growth: float = 8.0,
    ) -> None:
        self.max_match_distance = max_match_distance
        self.min_iou_for_match = min_iou_for_match
        self.max_missed_frames = max_missed_frames
        self.static_center_threshold = static_center_threshold
        self.static_iou_threshold = static_iou_threshold
        self.static_frame_window = static_frame_window
        self.use_cone_filter = use_cone_filter
        self.cone_base_radius = cone_base_radius
        self.cone_velocity_gain = cone_velocity_gain
        self.cone_missed_growth = cone_missed_growth
        self._next_track_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        scored_pairs: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            predicted_center = track.predicted_center()
            velocity = track.velocity()
            velocity_norm = l2_distance((0.0, 0.0), velocity)
            cone_radius = self.cone_base_radius + self.cone_velocity_gain * velocity_norm + self.cone_missed_growth * track.missed

            for det_idx, detection in enumerate(detections):
                distance = l2_distance(track.detection.center, detection.center)
                predicted_distance = l2_distance(predicted_center, detection.center)
                overlap = iou(track.detection, detection)

                if distance > self.max_match_distance:
                    continue
                if self.use_cone_filter and predicted_distance > cone_radius:
                    continue

                proximity_score = max(0.0, 1.0 - distance / max(self.max_match_distance, 1.0))
                prediction_score = max(0.0, 1.0 - predicted_distance / max(cone_radius, 1.0))
                score = overlap + 0.7 * proximity_score + 0.5 * prediction_score
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
