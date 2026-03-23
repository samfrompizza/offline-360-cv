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
    Классическая детекция маленьких движущихся контрастных объектов.

    Основная идея:
    - строим простой фон через exponential moving average;
    - выделяем отличия от фона, а не только от предыдущего кадра;
    - обновляем фон с настраиваемой скоростью, чтобы алгоритм быстрее
      "забывал" старые следы;
    - дополнительно оставляем только яркие/насыщенные небольшие компоненты.
    """

    def __init__(
        self,
        min_area: int = 4,
        max_area: int = 400,
        motion_threshold: int = 18,
        color_threshold: int = 120,
        max_aspect_ratio: float = 4.0,
        background_alpha: float = 0.35,
        morph_kernel_size: int = 3,
        dilate_iterations: int = 0,
    ) -> None:
        if not 0.0 < background_alpha <= 1.0:
            raise ValueError("background_alpha must be in (0, 1]")
        if morph_kernel_size < 1:
            raise ValueError("morph_kernel_size must be >= 1")
        if dilate_iterations < 0:
            raise ValueError("dilate_iterations must be >= 0")

        self.min_area = min_area
        self.max_area = max_area
        self.motion_threshold = motion_threshold
        self.color_threshold = color_threshold
        self.max_aspect_ratio = max_aspect_ratio
        self.background_alpha = background_alpha
        self.morph_kernel_size = morph_kernel_size
        self.dilate_iterations = dilate_iterations
        self._background: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        detections: list[Detection] = []

        if self._background is None:
            self._background = gray_blur.astype(np.float32)
            return detections

        background = cv2.convertScaleAbs(self._background)
        diff = cv2.absdiff(gray_blur, background)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        color_mask = cv2.inRange(saturation, self.color_threshold, 255)
        bright_mask = cv2.inRange(value, self.color_threshold, 255)
        combined = cv2.bitwise_and(motion_mask, cv2.bitwise_or(color_mask, bright_mask))

        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        if self.dilate_iterations > 0:
            combined = cv2.dilate(combined, kernel, iterations=self.dilate_iterations)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.accumulateWeighted(gray_blur, self._background, self.background_alpha)

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
            roi_bright = bright_mask[y : y + h, x : x + w]
            motion_score = float(np.count_nonzero(roi_motion)) / float(area)
            color_score = float(np.count_nonzero(roi_color)) / float(area)
            bright_score = float(np.count_nonzero(roi_bright)) / float(area)
            score = min(0.99, 0.35 + 0.4 * motion_score + 0.15 * color_score + 0.1 * bright_score)

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
