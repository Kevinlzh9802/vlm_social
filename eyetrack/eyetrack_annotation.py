"""
Extract and plot eye-focus data for annotated video-watching intervals.

This CLI coordinates three pieces:
  1. Read video time intervals from the annotation JSON.
  2. Extract Pupil Core gaze samples for those intervals and videos.
  3. Map screen gaze coordinates to the assumed video rectangle and plot them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .annotation_intervals import (
        TIME_TOLERANCE_SECONDS,
        load_all_video_timings,
        print_timing_table,
    )
    from .focus_plot import VIDEO_SCREEN_RATIO, plot_focus_for_video_gaze_data
    from .gaze_extraction import (
        MEDIA_URL_PREFIX,
        extract_video_gaze_data,
        load_gaze,
        load_recording_time_offset,
        load_video_entries,
    )
except ImportError:
    from annotation_intervals import (
        TIME_TOLERANCE_SECONDS,
        load_all_video_timings,
        print_timing_table,
    )
    from focus_plot import VIDEO_SCREEN_RATIO, plot_focus_for_video_gaze_data
    from gaze_extraction import (
        MEDIA_URL_PREFIX,
        extract_video_gaze_data,
        load_gaze,
        load_recording_time_offset,
        load_video_entries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot eye-focus data for annotated videos."
    )
    parser.add_argument("recording_dir", type=Path, help="Path to the Pupil Core recording folder.")
    parser.add_argument("annotation_json", type=Path, help="Path to the annotation JSON file.")
    parser.add_argument("video_json", type=Path, help="JSON file containing the ordered video list.")
    parser.add_argument(
        "data_parent",
        type=Path,
        help="Local folder replacing the http://localhost:5000/api/media/ prefix.",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=TIME_TOLERANCE_SECONDS,
        help="Allowed seconds of difference between end-start and current_video_time.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eyetrack_focus_plots"),
        help="Directory where focus plots will be written.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum gaze confidence threshold.",
    )
    parser.add_argument(
        "--media-url-prefix",
        default=MEDIA_URL_PREFIX,
        help="URL prefix to replace with data_parent.",
    )
    parser.add_argument(
        "--video-screen-ratio",
        type=float,
        default=VIDEO_SCREEN_RATIO,
        help="Assumed width/height ratio of the video rectangle on the screen.",
    )
    parser.add_argument(
        "--system-to-pupil-offset",
        type=float,
        default=None,
        help="Optional offset added to UNIX annotation timestamps to get Pupil Time.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if not 0.0 < args.video_screen_ratio <= 1.0:
        raise ValueError("--video-screen-ratio must be in the range (0, 1].")

    all_timings = load_all_video_timings(args.annotation_json, args.duration_tolerance)
    for annotator_timings in all_timings:
        print_timing_table(annotator_timings)

    video_entries = load_video_entries(args.video_json, args.data_parent, args.media_url_prefix)
    logging.info("loaded %s video entries from %s", len(video_entries), args.video_json)

    timestamps, gaze_data = load_gaze(args.recording_dir)
    system_to_pupil_offset = (
        args.system_to_pupil_offset
        if args.system_to_pupil_offset is not None
        else load_recording_time_offset(args.recording_dir)
    )
    video_gaze_data = extract_video_gaze_data(
        all_timings=all_timings,
        video_entries=video_entries,
        timestamps=timestamps,
        gaze_data=gaze_data,
        system_to_pupil_offset=system_to_pupil_offset,
        confidence_threshold=args.confidence,
    )
    plot_focus_for_video_gaze_data(
        video_gaze_data=video_gaze_data,
        output_dir=args.output_dir,
        video_screen_ratio=args.video_screen_ratio,
    )


if __name__ == "__main__":
    main()
