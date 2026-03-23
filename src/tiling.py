from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int
    image: np.ndarray

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def split_into_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: int = 128,
) -> List[Tile]:
    """
    Split image into overlapping square tiles.

    This is a simple baseline for 360/panoramic video.
    Later this can be replaced with cubemap projection.
    """
    if image is None:
        raise ValueError("image is None")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    h, w = image.shape[:2]
    step = tile_size - overlap

    tiles: List[Tile] = []

    if h <= tile_size and w <= tile_size:
        tiles.append(Tile(0, 0, w, h, image.copy()))
        return tiles

    for y1 in range(0, max(h - tile_size + 1, 1), step):
        for x1 in range(0, max(w - tile_size + 1, 1), step):
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)

            tile_img = image[y1:y2, x1:x2].copy()
            tiles.append(Tile(x1, y1, x2, y2, tile_img))

    # Ensure right/bottom borders are covered even when dimensions are not aligned.
    if w > tile_size:
        x1 = w - tile_size
        for y1 in range(0, max(h - tile_size + 1, 1), step):
            y2 = min(y1 + tile_size, h)
            tile_img = image[y1:y2, x1:w].copy()
            tiles.append(Tile(x1, y1, w, y2, tile_img))

    if h > tile_size:
        y1 = h - tile_size
        for x1 in range(0, max(w - tile_size + 1, 1), step):
            x2 = min(x1 + tile_size, w)
            tile_img = image[y1:h, x1:x2].copy()
            tiles.append(Tile(x1, y1, x2, h, tile_img))

    # Bottom-right corner
    if w > tile_size and h > tile_size:
        x1 = w - tile_size
        y1 = h - tile_size
        tiles.append(Tile(x1, y1, w, h, image[y1:h, x1:w].copy()))

    # Deduplicate tiles by coordinates.
    unique = {}
    for t in tiles:
        unique[(t.x1, t.y1, t.x2, t.y2)] = t

    return list(unique.values())


def tile_to_global_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    tile: Tile,
) -> tuple[float, float, float, float]:
    """
    Convert bbox coordinates from tile-local to full-image coordinates.
    """
    x1, y1, x2, y2 = bbox_xyxy
    return (
        x1 + tile.x1,
        y1 + tile.y1,
        x2 + tile.x1,
        y2 + tile.y1,
    )