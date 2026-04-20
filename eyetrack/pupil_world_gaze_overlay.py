"""
Overlay Pupil Core gaze samples on each recording's world.mp4.

The script writes outputs into the original Pupil recording folders:
  - world_gaze.mp4
  - world_gaze_points.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:
    from .gaze_extraction import load_gaze
except ImportError:
    from gaze_extraction import load_gaze


@dataclass
class GazeSample:
    timestamp: float
    x: float
    y: float
    confidence: float


def find_recording_dirs(paths: Iterable[Path], recursive: bool) -> List[Path]:
    recording_dirs = []
    for path in paths:
        if (path / "world.mp4").exists():
            recording_dirs.append(path)
            continue

        pattern = "**/world.mp4" if recursive else "*/world.mp4"
        recording_dirs.extend(world_path.parent for world_path in path.glob(pattern))

    return sorted(set(recording_dirs))


def load_world_timestamps(recording_dir: Path):
    import numpy as np

    path = recording_dir / "world_timestamps.npy"
    if not path.exists():
        raise FileNotFoundError(f"World timestamps not found: {path}")
    return np.load(path)


def build_valid_gaze_samples(
    gaze_timestamps,
    gaze_data: list,
    confidence_threshold: float,
) -> List[GazeSample]:
    valid = []
    n = min(len(gaze_timestamps), len(gaze_data))
    for idx in range(n):
        datum = gaze_data[idx]
        confidence = float(datum.get("confidence", 0.0))
        if confidence < confidence_threshold:
            continue

        norm_pos = datum.get("norm_pos") or [None, None]
        if len(norm_pos) < 2 or norm_pos[0] is None or norm_pos[1] is None:
            continue

        x = float(norm_pos[0])
        y = float(norm_pos[1])
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            continue

        valid.append(
            GazeSample(
                timestamp=float(gaze_timestamps[idx]),
                x=x,
                y=y,
                confidence=confidence,
            )
        )
    return valid


def nearest_gaze_sample(
    samples: List[GazeSample],
    sample_timestamps: List[float],
    timestamp: float,
    max_gap: float,
) -> GazeSample | None:
    import bisect

    if not samples:
        return None

    insert_at = bisect.bisect_left(sample_timestamps, timestamp)
    candidates = []
    if insert_at < len(samples):
        candidates.append(samples[insert_at])
    if insert_at > 0:
        candidates.append(samples[insert_at - 1])

    nearest = min(candidates, key=lambda sample: abs(sample.timestamp - timestamp))
    if abs(nearest.timestamp - timestamp) > max_gap:
        return None
    return nearest


def draw_gaze_marker(frame, sample: GazeSample, radius: int, label: bool) -> tuple[int, int]:
    import cv2

    height, width = frame.shape[:2]
    center_x = int(round(sample.x * width))
    center_y = int(round((1.0 - sample.y) * height))
    center_x = min(max(center_x, 0), width - 1)
    center_y = min(max(center_y, 0), height - 1)

    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)
    cv2.circle(frame, (center_x, center_y), max(3, radius // 3), (0, 0, 255), -1)
    cv2.drawMarker(
        frame,
        (center_x, center_y),
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(16, radius * 2),
        thickness=2,
    )
    if label:
        cv2.putText(
            frame,
            f"gaze conf={sample.confidence:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return center_x, center_y


def write_overlay_for_recording(
    recording_dir: Path,
    output_name: str,
    csv_name: str,
    confidence_threshold: float,
    max_gaze_gap: float,
    radius: int,
    overwrite: bool,
    label: bool,
) -> dict[str, object]:
    import cv2

    world_video_path = recording_dir / "world.mp4"
    output_path = recording_dir / output_name
    csv_path = recording_dir / csv_name

    if output_path.exists() and not overwrite:
        logging.info("Skipping existing overlay: %s", output_path)
        return {"recording_dir": str(recording_dir), "output_path": str(output_path), "skipped": True}

    world_timestamps = load_world_timestamps(recording_dir)
    gaze_timestamps, gaze_data = load_gaze(recording_dir)
    gaze_samples = build_valid_gaze_samples(gaze_timestamps, gaze_data, confidence_threshold)
    gaze_sample_timestamps = [sample.timestamp for sample in gaze_samples]

    capture = cv2.VideoCapture(str(world_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open world video: {world_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid world video metadata: {world_video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create overlay video writer: {output_path}")

    matched_frames = 0
    written_frames = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "world_timestamp",
                "gaze_timestamp",
                "gaze_time_gap",
                "norm_pos_x",
                "norm_pos_y",
                "pixel_x",
                "pixel_y",
                "confidence",
            ],
        )
        csv_writer.writeheader()

        frame_index = 0
        while frame_index < total_frames:
            ok, frame = capture.read()
            if not ok:
                break

            world_timestamp = (
                float(world_timestamps[frame_index])
                if frame_index < len(world_timestamps)
                else None
            )
            sample = (
                nearest_gaze_sample(
                    gaze_samples,
                    gaze_sample_timestamps,
                    world_timestamp,
                    max_gaze_gap,
                )
                if world_timestamp is not None
                else None
            )

            pixel_x = ""
            pixel_y = ""
            if sample is not None:
                pixel_x, pixel_y = draw_gaze_marker(frame, sample, radius, label)
                matched_frames += 1

            writer.write(frame)
            csv_writer.writerow(
                {
                    "frame_index": frame_index,
                    "world_timestamp": "" if world_timestamp is None else world_timestamp,
                    "gaze_timestamp": "" if sample is None else sample.timestamp,
                    "gaze_time_gap": (
                        ""
                        if sample is None or world_timestamp is None
                        else sample.timestamp - world_timestamp
                    ),
                    "norm_pos_x": "" if sample is None else sample.x,
                    "norm_pos_y": "" if sample is None else sample.y,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "confidence": "" if sample is None else sample.confidence,
                }
            )

            frame_index += 1
            written_frames += 1

    writer.release()
    capture.release()

    return {
        "recording_dir": str(recording_dir),
        "world_video_path": str(world_video_path),
        "output_path": str(output_path),
        "csv_path": str(csv_path),
        "world_frame_count": total_frames,
        "written_frame_count": written_frames,
        "valid_gaze_sample_count": len(gaze_samples),
        "matched_frame_count": matched_frames,
        "skipped": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Pupil Core gaze samples onto world.mp4 recordings."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Recording folder(s), or parent folder(s) containing recording folders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search parent folders for world.mp4.",
    )
    parser.add_argument(
        "--output-name",
        default="world_gaze.mp4",
        help="Overlay video filename written inside each recording folder.",
    )
    parser.add_argument(
        "--csv-name",
        default="world_gaze_points.csv",
        help="Per-frame gaze CSV filename written inside each recording folder.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum gaze confidence threshold.",
    )
    parser.add_argument(
        "--max-gaze-gap",
        type=float,
        default=0.05,
        help="Maximum seconds between world frame timestamp and nearest gaze sample.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=24,
        help="Gaze marker radius in pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing overlay videos.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Do not draw the confidence text label.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV summary path for all processed recordings.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    recording_dirs = find_recording_dirs(args.paths, args.recursive)
    if not recording_dirs:
        raise FileNotFoundError("No recording folders with world.mp4 were found.")

    logging.info("found %s recording folders", len(recording_dirs))
    rows = []
    for recording_dir in recording_dirs:
        try:
            logging.info("processing %s", recording_dir)
            row = write_overlay_for_recording(
                recording_dir=recording_dir,
                output_name=args.output_name,
                csv_name=args.csv_name,
                confidence_threshold=args.confidence,
                max_gaze_gap=args.max_gaze_gap,
                radius=args.radius,
                overwrite=args.overwrite,
                label=not args.no_label,
            )
            rows.append(row)
            logging.info(
                "wrote %s; matched %s/%s frames",
                row["output_path"],
                row.get("matched_frame_count", 0),
                row.get("written_frame_count", 0),
            )
        except Exception as exc:
            logging.error("failed %s: %s", recording_dir, exc)
            rows.append({"recording_dir": str(recording_dir), "error": str(exc)})

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logging.info("wrote summary CSV: %s", args.summary_csv)


if __name__ == "__main__":
    main()
