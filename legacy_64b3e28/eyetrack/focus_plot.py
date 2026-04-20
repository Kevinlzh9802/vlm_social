from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Tuple

try:
    from .gaze_extraction import VideoGazeData
except ImportError:
    from gaze_extraction import VideoGazeData


VIDEO_SCREEN_RATIO = 0.7


def map_screen_focus_to_video(samples: List[dict], video_screen_ratio: float) -> List[Tuple[float, float]]:
    rect_min = 1.0 - video_screen_ratio
    eps = 1e-9
    points = []
    for sample in samples:
        x = sample["norm_pos_x"]
        y = sample["norm_pos_y"]
        if x is None or y is None:
            continue
        video_x = (float(x) - rect_min) / video_screen_ratio
        video_y = (float(y) - rect_min) / video_screen_ratio
        if -eps <= video_x <= 1.0 + eps and -eps <= video_y <= 1.0 + eps:
            points.append((min(max(video_x, 0.0), 1.0), min(max(video_y, 0.0), 1.0)))
    return points


def read_video_frame(video_path: Path, video_length: float) -> Any:
    try:
        import cv2
    except ModuleNotFoundError:
        logging.warning("cv2 is not installed; plotting focus points without video frames.")
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logging.warning("Could not open video; plotting focus points without frame: %s", video_path)
        return None

    capture.set(cv2.CAP_PROP_POS_MSEC, max(video_length * 500.0, 0.0))
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        logging.warning("Could not read a frame; plotting focus points without frame: %s", video_path)
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_focus_points(
    video_gaze: VideoGazeData,
    points: List[Tuple[float, float]],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    timing = video_gaze.timing
    video_entry = video_gaze.video_entry
    frame = read_video_frame(video_entry.video_path, timing.video_length)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f"Annotator {video_gaze.annotator_number}, video {timing.video_number}: "
        f"{len(points)}/{len(video_gaze.samples)} gaze samples in video region"
    )

    if frame is not None:
        height, width = frame.shape[:2]
        ax.imshow(frame)
        ax.axis("off")
    else:
        width = 1.0
        height = 1.0
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("video x")
        ax.set_ylabel("video y")
        ax.grid(True, alpha=0.25)

    if points:
        xs = np.array([point[0] * width for point in points])
        ys = (
            np.array([(1.0 - point[1]) * height for point in points])
            if frame is not None
            else np.array([point[1] for point in points])
        )
        ax.scatter(xs, ys, s=12, c="red", alpha=0.35, edgecolors="none")
        ax.scatter([float(np.mean(xs))], [float(np.mean(ys))], s=90, c="yellow", marker="x")
    else:
        ax.text(
            0.5,
            0.5,
            "No gaze samples inside assumed video region",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="yellow",
            fontsize=12,
        )

    output_path = output_dir / (
        f"annotator_{video_gaze.annotator_number}_video_{timing.video_number:03d}_"
        f"node_key_{timing.annotation_key}_{safe_name(video_entry.video_path.stem)}.png"
    )
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_focus_for_video_gaze_data(
    video_gaze_data: List[VideoGazeData],
    output_dir: Path,
    video_screen_ratio: float,
    annotator_dir_template: str = "annotator_{annotator_number}",
) -> List[dict]:
    rows = []
    for video_gaze in video_gaze_data:
        points = map_screen_focus_to_video(video_gaze.samples, video_screen_ratio)
        annotator_output_dir = output_dir / annotator_dir_template.format(
            annotator_number=video_gaze.annotator_number
        )
        output_path = plot_focus_points(video_gaze, points, annotator_output_dir)
        logging.info("wrote focus plot: %s", output_path)
        row = {
            "annotator_number": video_gaze.annotator_number,
            "video_number": video_gaze.timing.video_number,
            "annotation_key": video_gaze.timing.annotation_key,
            "video_path": str(video_gaze.video_entry.video_path),
            "plot_path": str(output_path),
            "raw_gaze_sample_count": len(video_gaze.samples),
            "mapped_gaze_sample_count": len(points),
            "focus_mapping": "legacy-extraction",
            "video_screen_ratio": video_screen_ratio,
            "mean_mapped_gaze_x": "",
            "mean_mapped_gaze_y": "",
        }
        if points:
            row["mean_mapped_gaze_x"] = sum(point[0] for point in points) / len(points)
            row["mean_mapped_gaze_y"] = sum(point[1] for point in points) / len(points)
        rows.append(row)
    return rows
