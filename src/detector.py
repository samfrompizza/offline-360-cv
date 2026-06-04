from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


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
    Классическая детекция любых красных объектов.

    Основная идея:
    - переводим кадр в HSV;
    - оставляем только пиксели из красного HSV-диапазона;
    - строим bounding boxes по компонентам маски без проверки движения.
    """

    RED_HSV_RANGES = (
        (
            np.array([0, 80, 50], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
        ),
        (
            np.array([170, 80, 50], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        ),
    )

    def __init__(
        self,
        min_area: int = 4,
        max_area: int = 400,
        color_threshold: int = 120,
        max_aspect_ratio: float = 4.0,
        morph_kernel_size: int = 3,
        dilate_iterations: int = 0,
    ) -> None:
        if morph_kernel_size < 1:
            raise ValueError("morph_kernel_size must be >= 1")
        if dilate_iterations < 0:
            raise ValueError("dilate_iterations must be >= 0")

        self.min_area = min_area
        self.max_area = max_area
        self.color_threshold = color_threshold
        self.max_aspect_ratio = max_aspect_ratio
        self.morph_kernel_size = morph_kernel_size
        self.dilate_iterations = dilate_iterations

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = self._build_red_mask(hsv)

        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        if self.dilate_iterations > 0:
            red_mask = cv2.dilate(red_mask, kernel, iterations=self.dilate_iterations)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue

            aspect_ratio = max(w / max(h, 1), h / max(w, 1))
            if aspect_ratio > self.max_aspect_ratio:
                continue

            roi_red = red_mask[y : y + h, x : x + w]
            red_score = float(np.count_nonzero(roi_red)) / float(area)
            score = min(0.99, 0.5 + 0.5 * red_score)

            detections.append(
                Detection(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    score=score,
                    label="red_object",
                    source="classical",
                )
            )

        return non_max_suppression(detections, iou_threshold=0.25)

    def _build_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        masks = [cv2.inRange(hsv, lower, upper) for lower, upper in self.RED_HSV_RANGES]
        red_mask = cv2.bitwise_or(masks[0], masks[1])

        if self.color_threshold > 0:
            saturation = cv2.inRange(hsv[:, :, 1], self.color_threshold, 255)
            value = cv2.inRange(hsv[:, :, 2], self.color_threshold, 255)
            red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_and(saturation, value))

        return red_mask


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
