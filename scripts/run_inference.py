from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from src.detector import ClassicalDroneDetector, Detection, YoloTileDetector, non_max_suppression
from src.postprocess import filter_useful_tracks
from src.tracker import SimpleTracker
from src.video_io import create_video_writer, get_video_metadata, iter_video_frames
from src.visualize import draw_tracks, save_json, serialize_frame_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone detection on panoramic video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to YAML config")
    parser.add_argument(
        "--method",
        choices=["classical", "yolo", "both"],
        default=None,
        help="Override detector method from config",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_detectors(config: dict, method: str) -> list:
    detectors = []
    if method in {"classical", "both"}:
        classical_cfg = config.get("classical", {})
        detectors.append(ClassicalDroneDetector(**classical_cfg))
    if method in {"yolo", "both"}:
        yolo_cfg = config.get("yolo", {})
        detectors.append(YoloTileDetector(**yolo_cfg))
    return detectors


def merge_detections(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []
    return non_max_suppression(detections, iou_threshold=0.3)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    method = args.method or config.get("method", "both")

    metadata = get_video_metadata(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(args.video).stem

    detectors = build_detectors(config, method)
    tracker_cfg = config.get("tracker", {})
    tracker = SimpleTracker(**tracker_cfg)

    annotated_video_path = output_dir / f"{video_stem}_{method}_annotated.mp4"
    json_path = output_dir / f"{video_stem}_{method}_tracks.json"
    writer = create_video_writer(
        annotated_video_path,
        width=metadata["width"],
        height=metadata["height"],
        fps=metadata["fps"] or 25.0,
    )

    frames_payload = []
    every_n_frames = int(config.get("every_n_frames", 1))

    try:
        for frame_idx, frame in tqdm(
            iter_video_frames(args.video, every_n_frames=every_n_frames),
            total=metadata["frame_count"] // every_n_frames if metadata["frame_count"] else None,
            desc=f"Running {method} inference",
        ):
            detections: list[Detection] = []
            for detector in detectors:
                detections.extend(detector.detect(frame))

            merged = merge_detections(detections)
            tracks = tracker.update(merged)
            useful_tracks = filter_useful_tracks(tracks, min_hits=config.get("min_track_hits", 2))

            annotated = draw_tracks(frame, useful_tracks)
            writer.write(annotated)
            frames_payload.append(serialize_frame_tracks(frame_idx, useful_tracks))
    finally:
        writer.release()

    save_json(
        {
            "video": str(Path(args.video)),
            "method": method,
            "metadata": metadata,
            "frames": frames_payload,
        },
        json_path,
    )

    print(f"Annotated video saved to: {annotated_video_path}")
    print(f"Track JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
