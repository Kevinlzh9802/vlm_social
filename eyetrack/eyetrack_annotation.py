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
import csv
import logging
import re
from pathlib import Path
from typing import List, Tuple

try:
    from .annotation_intervals import (
        RESPONSE_SELECTION_CHOICES,
        TIME_TOLERANCE_SECONDS,
        load_all_video_timings,
        print_timing_table,
    )
    from .focus_plot import (
        LEGACY_EXTRACTION_VIDEO_SCREEN_RATIO,
        plot_focus_for_video_gaze_data,
    )
    from .gaze_extraction import (
        MEDIA_URL_PREFIX,
        extract_video_gaze_data,
        load_gaze,
        load_recording_time_offset,
        load_video_entries,
    )
except ImportError:
    from annotation_intervals import (
        RESPONSE_SELECTION_CHOICES,
        TIME_TOLERANCE_SECONDS,
        load_all_video_timings,
        print_timing_table,
    )
    from focus_plot import (
        LEGACY_EXTRACTION_VIDEO_SCREEN_RATIO,
        plot_focus_for_video_gaze_data,
    )
    from gaze_extraction import (
        MEDIA_URL_PREFIX,
        extract_video_gaze_data,
        load_gaze,
        load_recording_time_offset,
        load_video_entries,
    )


ANNOTATION_FILE_RE = re.compile(r"^T(?P<task_number>\d+)_(?P<task_instance_id>\d+)\.json$")
TIMING_CSV_FIELDS = (
    "task_number",
    "task_instance_id",
    "task_instance",
    "annotator_number",
    "node_id",
    "global_unique_id",
    "response_selection",
    "response_index",
    "response_created",
    "response_submitted",
    "video_number",
    "annotation_key",
    "video_start_time",
    "video_end_time",
    "video_path",
    "audio_path",
    "video_length",
    "current_video_time",
    "time_annot",
)
EXTRACTION_SUMMARY_FIELDS = (
    "task_number",
    "task_instance_id",
    "task_instance",
    "annotator_number",
    "node_id",
    "global_unique_id",
    "response_selection",
    "response_index",
    "response_created",
    "response_submitted",
    "video_number",
    "annotation_key",
    "video_start_time",
    "video_end_time",
    "video_length",
    "current_video_time",
    "time_annot",
    "annotation_video_path",
    "resolved_video_path",
    "plot_path",
    "recording_dir",
    "gaze_timestamps_path",
    "gaze_pldata_path",
    "system_to_pupil_offset",
    "confidence_threshold",
    "focus_mapping",
    "video_screen_ratio",
    "raw_gaze_sample_count",
    "mapped_gaze_sample_count",
    "mean_mapped_gaze_x",
    "mean_mapped_gaze_y",
)


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


def write_timing_csv_rows(
    writer: csv.DictWriter,
    task_number: int,
    task_instance_id: int,
    task_instance_name: str,
    annotator_timings,
) -> None:
    for timing in annotator_timings.timings:
        writer.writerow(
            {
                "task_number": task_number,
                "task_instance_id": task_instance_id,
                "task_instance": task_instance_name,
                "annotator_number": annotator_timings.annotator_number,
                "node_id": annotator_timings.node_id,
                "global_unique_id": annotator_timings.global_unique_id,
                "response_selection": annotator_timings.response_selection,
                "response_index": annotator_timings.response_index,
                "response_created": annotator_timings.response_created,
                "response_submitted": annotator_timings.response_submitted,
                "video_number": timing.video_number,
                "annotation_key": timing.annotation_key,
                "video_start_time": timing.video_start_time,
                "video_end_time": timing.video_end_time,
                "video_path": timing.video_path,
                "audio_path": timing.audio_path,
                "video_length": timing.video_length,
                "current_video_time": timing.current_video_time,
                "time_annot": timing.time_annot,
            }
        )


def write_extraction_summary_rows(
    writer: csv.DictWriter,
    task_number: int,
    task_instance_id: int,
    task_instance_name: str,
    annotator_timings,
    plot_rows: list[dict],
    recording_dir: Path,
    system_to_pupil_offset: float,
    confidence_threshold: float,
    focus_mapping: str,
    video_screen_ratio: float,
) -> None:
    timing_by_key = {
        (timing.video_number, timing.annotation_key): timing
        for timing in annotator_timings.timings
    }
    for plot_row in plot_rows:
        timing = timing_by_key.get(
            (plot_row["video_number"], plot_row["annotation_key"])
        )
        if timing is None:
            logging.warning(
                "Could not find timing row for annotator %s video %s annotation %s.",
                annotator_timings.annotator_number,
                plot_row["video_number"],
                plot_row["annotation_key"],
            )
            continue
        writer.writerow(
            {
                "task_number": task_number,
                "task_instance_id": task_instance_id,
                "task_instance": task_instance_name,
                "annotator_number": annotator_timings.annotator_number,
                "node_id": annotator_timings.node_id,
                "global_unique_id": annotator_timings.global_unique_id,
                "response_selection": annotator_timings.response_selection,
                "response_index": annotator_timings.response_index,
                "response_created": annotator_timings.response_created,
                "response_submitted": annotator_timings.response_submitted,
                "video_number": timing.video_number,
                "annotation_key": timing.annotation_key,
                "video_start_time": timing.video_start_time,
                "video_end_time": timing.video_end_time,
                "video_length": timing.video_length,
                "current_video_time": timing.current_video_time,
                "time_annot": timing.time_annot,
                "annotation_video_path": timing.video_path,
                "resolved_video_path": plot_row["video_path"],
                "plot_path": plot_row["plot_path"],
                "recording_dir": str(recording_dir),
                "gaze_timestamps_path": str(recording_dir / "gaze_timestamps.npy"),
                "gaze_pldata_path": str(recording_dir / "gaze.pldata"),
                "system_to_pupil_offset": system_to_pupil_offset,
                "confidence_threshold": confidence_threshold,
                "focus_mapping": focus_mapping,
                "video_screen_ratio": video_screen_ratio,
                "raw_gaze_sample_count": plot_row["raw_gaze_sample_count"],
                "mapped_gaze_sample_count": plot_row["mapped_gaze_sample_count"],
                "mean_mapped_gaze_x": plot_row["mean_mapped_gaze_x"],
                "mean_mapped_gaze_y": plot_row["mean_mapped_gaze_y"],
            }
        )


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
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=(
            f"Local path prefix replacing the {MEDIA_URL_PREFIX} URL prefix. "
            "Legacy positional form is also supported: video_json local_path_prefix."
        ),
    )
    parser.add_argument(
        "--video-json",
        type=Path,
        default=None,
        help="Optional fallback JSON file containing the ordered video list.",
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
        "--timing-csv",
        type=Path,
        default=None,
        help="Optional CSV file for annotation start/end timing rows.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV file with one row per generated gaze plot, including "
            "Pupil files, sample counts, mapped gaze counts, and plot path. "
            "Defaults to output_dir/extraction_summary.csv."
        ),
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
        default=LEGACY_EXTRACTION_VIDEO_SCREEN_RATIO,
        help=(
            "Video screen ratio used only with --focus-mapping legacy-extraction. "
            "Commit 64b3e28 used 0.7."
        ),
    )
    parser.add_argument(
        "--focus-mapping",
        choices=("legacy-extraction", "measured-player"),
        default="measured-player",
        help=(
            "Screen-to-video focus mapping. 'legacy-extraction' reproduces the "
            "64b3e28 focus plots; 'measured-player' uses focus_plot.py player geometry."
        ),
    )
    parser.add_argument(
        "--response-selection",
        choices=RESPONSE_SELECTION_CHOICES,
        default="latest-submitted",
        help=(
            "Which response to read from each annotator node. "
            "'first-response' matches the 64b3e28 behavior."
        ),
    )
    parser.add_argument(
        "--system-to-pupil-offset",
        type=float,
        default=None,
        help="Optional offset added to UNIX annotation timestamps to get Pupil Time.",
    )
    args = parser.parse_args()
    if len(args.paths) == 1:
        args.local_path_prefix = args.paths[0]
    elif len(args.paths) == 2:
        if args.video_json is not None:
            parser.error("Use either --video-json or legacy positional video_json, not both.")
        args.video_json = args.paths[0]
        args.local_path_prefix = args.paths[1]
    else:
        parser.error("Expected local_path_prefix, or legacy: video_json local_path_prefix.")
    return args


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

    video_entries = None
    if args.video_json is not None:
        video_entries = load_video_entries(
            args.video_json,
            args.local_path_prefix,
            args.media_url_prefix,
        )
        logging.info("loaded %s video entries from %s", len(video_entries), args.video_json)

    timing_csv_file = None
    timing_writer = None
    summary_csv_file = None
    summary_writer = None
    try:
        summary_csv = args.summary_csv or (args.output_dir / "extraction_summary.csv")
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_csv_file = summary_csv.open("w", encoding="utf-8", newline="")
        summary_writer = csv.DictWriter(
            summary_csv_file,
            fieldnames=EXTRACTION_SUMMARY_FIELDS,
        )
        summary_writer.writeheader()

        if args.timing_csv is not None:
            args.timing_csv.parent.mkdir(parents=True, exist_ok=True)
            timing_csv_file = args.timing_csv.open("w", encoding="utf-8", newline="")
            timing_writer = csv.DictWriter(timing_csv_file, fieldnames=TIMING_CSV_FIELDS)
            timing_writer.writeheader()

        for task_number, task_instance_id, annotation_json in annotation_files:
            task_instance_name = f"T{task_number}_{task_instance_id}"
            logging.info("processing annotation file: %s", annotation_json)

            all_timings = load_all_video_timings(
                annotation_json,
                args.duration_tolerance,
                response_selection=args.response_selection,
            )
            for annotator_timings in all_timings:
                annotator_output_dir = (
                    args.output_dir
                    / f"{task_instance_name}_annotator{annotator_timings.annotator_number}"
                )
                annotator_output_dir.mkdir(parents=True, exist_ok=True)
                print_timing_table(annotator_timings)
                if timing_writer is not None:
                    write_timing_csv_rows(
                        timing_writer,
                        task_number,
                        task_instance_id,
                        task_instance_name,
                        annotator_timings,
                    )

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
                    local_path_prefix=args.local_path_prefix,
                    media_url_prefix=args.media_url_prefix,
                )
                plot_rows = plot_focus_for_video_gaze_data(
                    video_gaze_data=video_gaze_data,
                    output_dir=args.output_dir,
                    video_screen_ratio=args.video_screen_ratio,
                    annotator_dir_template=f"{task_instance_name}_annotator{{annotator_number}}",
                    focus_mapping=args.focus_mapping,
                )
                write_extraction_summary_rows(
                    writer=summary_writer,
                    task_number=task_number,
                    task_instance_id=task_instance_id,
                    task_instance_name=task_instance_name,
                    annotator_timings=annotator_timings,
                    plot_rows=plot_rows,
                    recording_dir=recording_dir,
                    system_to_pupil_offset=system_to_pupil_offset,
                    confidence_threshold=args.confidence,
                    focus_mapping=args.focus_mapping,
                    video_screen_ratio=args.video_screen_ratio,
                )
    finally:
        if timing_csv_file is not None:
            logging.info("wrote timing CSV: %s", args.timing_csv)
            timing_csv_file.close()
        if summary_csv_file is not None:
            logging.info("wrote extraction summary CSV: %s", summary_csv_file.name)
            summary_csv_file.close()


if __name__ == "__main__":
    main()
