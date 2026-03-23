from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from src.tiling import Tile, split_into_tiles, tile_to_global_bbox


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    source: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict:
        return asdict(self)


class ClassicalDroneDetector:
    """
    Простая классическая детекция маленьких движущихся контрастных объектов.

    Идея MVP:
    - ищем движение через разность с предыдущим кадром;
    - усиливаем маленькие яркие/цветные объекты;
    - отсекаем слишком большие и слишком маленькие компоненты.

    Это помогает отделить дроны от статичных цветных ворот.
    """

    def __init__(
        self,
        min_area: int = 4,
        max_area: int = 400,
        motion_threshold: int = 18,
        color_threshold: int = 120,
        max_aspect_ratio: float = 4.0,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.motion_threshold = motion_threshold
        self.color_threshold = color_threshold
        self.max_aspect_ratio = max_aspect_ratio
        self._prev_gray: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections: list[Detection] = []

        if self._prev_gray is None:
            self._prev_gray = gray
            return detections

        prev_blur = cv2.GaussianBlur(self._prev_gray, (5, 5), 0)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray_blur, prev_blur)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        color_mask = cv2.inRange(saturation, self.color_threshold, 255)
        bright_mask = cv2.inRange(value, self.color_threshold, 255)
        combined = cv2.bitwise_and(motion_mask, cv2.bitwise_or(color_mask, bright_mask))

        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._prev_gray = gray

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue

            aspect_ratio = max(w / max(h, 1), h / max(w, 1))
            if aspect_ratio > self.max_aspect_ratio:
                continue

            roi_motion = motion_mask[y : y + h, x : x + w]
            roi_color = color_mask[y : y + h, x : x + w]
            motion_score = float(np.count_nonzero(roi_motion)) / float(area)
            color_score = float(np.count_nonzero(roi_color)) / float(area)
            score = min(0.99, 0.45 + 0.35 * motion_score + 0.2 * color_score)

            detections.append(
                Detection(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    score=score,
                    label="drone_candidate",
                    source="classical",
                )
            )

        return non_max_suppression(detections, iou_threshold=0.25)


class YoloTileDetector:
    """
    Baseline tile-by-tile inference для очень маленьких объектов на широком кадре.

    Использует yolo11n.pt и автоматически скачивает веса через ultralytics,
    если файла нет локально.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        tile_size: int = 640,
        tile_overlap: int = 160,
        imgsz: int = 640,
        conf: float = 0.15,
        iou: float = 0.45,
        max_det: int = 50,
        allowed_class_ids: Sequence[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.allowed_class_ids = list(allowed_class_ids) if allowed_class_ids else None
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tiles = split_into_tiles(frame, tile_size=self.tile_size, overlap=self.tile_overlap)
        detections: list[Detection] = []

        for tile in tiles:
            detections.extend(self._infer_tile(tile))

        return non_max_suppression(detections, iou_threshold=0.35)

    def _infer_tile(self, tile: Tile) -> list[Detection]:
        results = self.model.predict(
            source=tile.image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=True,
            verbose=False,
            max_det=self.max_det,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        names = result.names
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=float)
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

        detections: list[Detection] = []
        for bbox, score, class_id in zip(xyxy, confs, clss):
            if self.allowed_class_ids is not None and class_id not in self.allowed_class_ids:
                continue

            gx1, gy1, gx2, gy2 = tile_to_global_bbox(tuple(float(v) for v in bbox), tile)
            detections.append(
                Detection(
                    x1=gx1,
                    y1=gy1,
                    x2=gx2,
                    y2=gy2,
                    score=float(score),
                    label=str(names.get(class_id, class_id)),
                    source="yolo",
                )
            )
        return detections


def iou(box_a: Detection, box_b: Detection) -> float:
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = box_a.area + box_b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def non_max_suppression(
    detections: Iterable[Detection],
    iou_threshold: float = 0.35,
) -> list[Detection]:
    candidates = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[Detection] = []

    while candidates:
        best = candidates.pop(0)
        kept.append(best)
        candidates = [candidate for candidate in candidates if iou(best, candidate) < iou_threshold]

    return kept
