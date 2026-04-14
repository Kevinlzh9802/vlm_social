"""
Extract and plot eye-focus data for annotated video-watching intervals.

This CLI coordinates three pieces:
  1. Read video time intervals from each T{x}_{y}.json annotation file.
  2. Match each annotation/annotator to recording_parent/T{x}_{y}_annotator{1,2}
     and extract Pupil Core gaze samples for those intervals and videos.
  3. Map screen gaze coordinates to the assumed video rectangle and plot them under
     output_dir/T{x}_{y}_annotator{1,2}.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple

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


ANNOTATION_FILE_RE = re.compile(r"^T(?P<task_number>\d+)_(?P<task_instance_id>\d+)\.json$")


def find_annotation_jsons(annotation_dir: Path) -> List[Tuple[int, int, Path]]:
    annotation_files = []
    for path in annotation_dir.iterdir():
        if not path.is_file():
            continue
        match = ANNOTATION_FILE_RE.match(path.name)
        if match is None:
            continue
        annotation_files.append(
            (
                int(match.group("task_number")),
                int(match.group("task_instance_id")),
                path,
            )
        )
    return sorted(annotation_files, key=lambda item: (item[0], item[1], item[2].name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot eye-focus data for annotated videos."
    )
    parser.add_argument(
        "recording_parent_dir",
        type=Path,
        help="Parent folder containing Pupil Core recording folders named T{x}_{y}_annotator1/2.",
    )
    parser.add_argument(
        "annotation_dir",
        type=Path,
        help="Folder containing annotation JSON files named like T{x}_{y}.json.",
    )
    parser.add_argument("video_json", type=Path, help="JSON file containing the ordered video list.")
    parser.add_argument(
        "local_path_prefix",
        type=Path,
        help=f"Local path prefix replacing the {MEDIA_URL_PREFIX} URL prefix.",
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
        help="URL prefix to replace with local_path_prefix.",
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
    if not args.recording_parent_dir.is_dir():
        raise NotADirectoryError(
            f"Recording parent folder not found: {args.recording_parent_dir}"
        )
    if not args.annotation_dir.is_dir():
        raise NotADirectoryError(f"Annotation folder not found: {args.annotation_dir}")

    annotation_files = find_annotation_jsons(args.annotation_dir)
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation JSON files named like T{{x}}_{{y}}.json found in {args.annotation_dir}"
        )
    logging.info("found %s annotation files in %s", len(annotation_files), args.annotation_dir)

    video_entries = load_video_entries(
        args.video_json,
        args.local_path_prefix,
        args.media_url_prefix,
    )
    logging.info("loaded %s video entries from %s", len(video_entries), args.video_json)

    for task_number, task_instance_id, annotation_json in annotation_files:
        task_instance_name = f"T{task_number}_{task_instance_id}"
        logging.info("processing annotation file: %s", annotation_json)

        all_timings = load_all_video_timings(annotation_json, args.duration_tolerance)
        for annotator_timings in all_timings:
            annotator_output_dir = (
                args.output_dir
                / f"{task_instance_name}_annotator{annotator_timings.annotator_number}"
            )
            annotator_output_dir.mkdir(parents=True, exist_ok=True)
            print_timing_table(annotator_timings)

            recording_dir = args.recording_parent_dir / annotator_output_dir.name
            if not recording_dir.is_dir():
                raise NotADirectoryError(f"Pupil recording folder not found: {recording_dir}")

            logging.info(
                "using recording folder for %s annotator %s: %s",
                task_instance_name,
                annotator_timings.annotator_number,
                recording_dir,
            )
            timestamps, gaze_data = load_gaze(recording_dir)
            system_to_pupil_offset = (
                args.system_to_pupil_offset
                if args.system_to_pupil_offset is not None
                else load_recording_time_offset(recording_dir)
            )

            video_gaze_data = extract_video_gaze_data(
                all_timings=[annotator_timings],
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
                annotator_dir_template=f"{task_instance_name}_annotator{{annotator_number}}",
            )


if __name__ == "__main__":
    main()
