from __future__ import annotations

import argparse
from pathlib import Path

from src.video_io import iter_video_frames, save_frame, get_video_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for drone panorama MVP."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video (.mov / .mp4)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where extracted frames will be saved",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=30,
        help="Save every N-th frame (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on number of saved frames (0 = no cap)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(input_path)
    print("Video metadata:", metadata)

    saved = 0
    for frame_idx, frame in iter_video_frames(input_path, every_n_frames=args.every_n_frames):
        out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        save_frame(frame, out_path)
        saved += 1

        if saved % 10 == 0:
            print(f"Saved {saved} frames...")

        if args.max_frames > 0 and saved >= args.max_frames:
            break

    print(f"Done. Saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    main()