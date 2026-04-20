from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Tuple

try:
    from .gaze_extraction import VideoGazeData
except ImportError:
    from gaze_extraction import VideoGazeData


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
PLAYER_LEFT = 480
PLAYER_TOP = 147
PLAYER_WIDTH = 1440
PLAYER_HEIGHT = 700
VIDEO_ASPECT_RATIO = 16 / 9

# Kept for CLI compatibility. The measured geometry above is now used for
# mapping instead of a single square-region ratio.
VIDEO_SCREEN_RATIO = PLAYER_HEIGHT / SCREEN_HEIGHT


def video_display_bounds() -> Tuple[float, float, float, float]:
    player_aspect = PLAYER_WIDTH / PLAYER_HEIGHT
    if VIDEO_ASPECT_RATIO >= player_aspect:
        display_width = float(PLAYER_WIDTH)
        display_height = display_width / VIDEO_ASPECT_RATIO
        display_left = float(PLAYER_LEFT)
        display_top = PLAYER_TOP + (PLAYER_HEIGHT - display_height) / 2.0
    else:
        display_height = float(PLAYER_HEIGHT)
        display_width = display_height * VIDEO_ASPECT_RATIO
        display_left = PLAYER_LEFT + (PLAYER_WIDTH - display_width) / 2.0
        display_top = float(PLAYER_TOP)
    return display_left, display_top, display_width, display_height


def map_screen_sample_to_video_point(sample: dict) -> Tuple[float, float] | None:
    x = sample["norm_pos_x"]
    y = sample["norm_pos_y"]
    if x is None or y is None:
        return None

    screen_x = float(x) * SCREEN_WIDTH
    screen_y_from_top = (1.0 - float(y)) * SCREEN_HEIGHT
    display_left, display_top, display_width, display_height = video_display_bounds()
    video_x = (screen_x - display_left) / display_width
    video_y = 1.0 - ((screen_y_from_top - display_top) / display_height)
    eps = 1e-9
    if -eps <= video_x <= 1.0 + eps and -eps <= video_y <= 1.0 + eps:
        return min(max(video_x, 0.0), 1.0), min(max(video_y, 0.0), 1.0)
    return None


def map_screen_focus_to_video(samples: List[dict], video_screen_ratio: float) -> List[Tuple[float, float]]:
    points = []
    for sample in samples:
        point = map_screen_sample_to_video_point(sample)
        if point is not None:
            points.append(point)
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
) -> None:
    for video_gaze in video_gaze_data:
        points = map_screen_focus_to_video(video_gaze.samples, video_screen_ratio)
        annotator_output_dir = output_dir / annotator_dir_template.format(
            annotator_number=video_gaze.annotator_number
        )
        output_path = plot_focus_points(video_gaze, points, annotator_output_dir)
        logging.info("wrote focus plot: %s", output_path)
