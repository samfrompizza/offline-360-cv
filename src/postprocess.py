from __future__ import annotations

from src.tracker import Track


def filter_useful_tracks(tracks: list[Track], min_hits: int = 1) -> list[Track]:
    """
    Оставляет подтверждённые красные треки, не отбрасывая статичные объекты.
    """
    return [
        track
        for track in tracks
        if track.hits >= min_hits and track.missed == 0
    ]
