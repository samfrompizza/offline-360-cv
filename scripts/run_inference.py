from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
from tqdm import tqdm

from src.detector import ClassicalDroneDetector, Detection, non_max_suppression
from src.postprocess import filter_useful_tracks
from src.tracker import SimpleTracker
from src.video_io import create_video_writer, get_video_metadata, iter_video_frames
from src.visualize import draw_tracks, save_json, serialize_frame_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone detection on panoramic video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=None,
        help="Override detector background update rate in (0, 1]. Bigger values forget old frames faster.",
    )
    parser.add_argument(
        "--use-cone-filter",
        action="store_true",
        help="Enable simple predicted-position cone filter in tracker.",
    )
    parser.add_argument(
        "--disable-cone-filter",
        action="store_true",
        help="Force-disable the cone filter even if it is enabled in config.",
    )
    return parser.parse_args()



def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



def build_detector(config: dict, background_alpha_override: float | None) -> ClassicalDroneDetector:
    classical_cfg = dict(config.get("classical", {}))
    if background_alpha_override is not None:
        classical_cfg["background_alpha"] = background_alpha_override
    return ClassicalDroneDetector(**classical_cfg)



def merge_detections(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []
    return non_max_suppression(detections, iou_threshold=0.3)



def resolve_tracker_config(config: dict, args: argparse.Namespace) -> dict:
    tracker_cfg = dict(config.get("tracker", {}))
    if args.use_cone_filter:
        tracker_cfg["use_cone_filter"] = True
    if args.disable_cone_filter:
        tracker_cfg["use_cone_filter"] = False
    return tracker_cfg



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    method = "classical"

    metadata = get_video_metadata(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(args.video).stem

    detector = build_detector(config, args.background_alpha)
    tracker_cfg = resolve_tracker_config(config, args)
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
            desc="Running classical inference",
        ):
            detections = detector.detect(frame)
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
            "classical": {
                "background_alpha": detector.background_alpha,
            },
            "tracker": {
                "use_cone_filter": tracker.use_cone_filter,
            },
        },
        json_path,
    )

    print(f"Annotated video saved to: {annotated_video_path}")
    print(f"Track JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
