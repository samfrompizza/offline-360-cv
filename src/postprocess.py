from __future__ import annotations

from src.tracker import Track


def filter_useful_tracks(tracks: list[Track], min_hits: int = 2) -> list[Track]:
    """
    Оставляет только движущиеся и уже немного подтверждённые треки.
    """
    useful: list[Track] = []
    for track in tracks:
        if track.hits < min_hits:
            continue
        if track.is_static:
            continue
        useful.append(track)
    return useful
