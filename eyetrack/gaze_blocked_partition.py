from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
for import_root in (REPO_ROOT, DATASET_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset.dialogue_partition import (  # noqa: E402
    GROUP_FOLDER_NAMES,
    FailedTriplet,
    PartitionGroup,
    build_summary,
    collect_video_records,
    concatenate_group_video,
    get_video_properties,
    parse_utt_group_selection,
    partition_dialogue,
    run_ffmpeg_with_video_codec_fallback,
    validate_materialized_group,
    write_summary_files,
)
from eyetrack.annotation_intervals import (  # noqa: E402
    RESPONSE_SELECTION_CHOICES,
    TIME_TOLERANCE_SECONDS,
    load_all_video_timings,
)
from eyetrack.eyetrack_annotation import find_annotation_jsons  # noqa: E402
from eyetrack.focus_plot import (  # noqa: E402
    map_screen_sample_to_video_point,
    map_screen_sample_to_video_point_legacy_extraction,
)
from eyetrack.gaze_extraction import (  # noqa: E402
    MEDIA_URL_PREFIX,
    annotation_time_to_pupil_time,
    extract_gaze_in_interval,
    gaze_source_metadata,
    get_video_entry_for_timing,
    load_gaze,
    load_recording_time_offset,
    load_video_entries,
    resolve_optional_media_path,
)


DEFAULT_FOCUS_BOX_RATIO = 0.18
GAZE_MAPPING_CHOICES = ("legacy-extraction", "measured-player")
CLIP_STEM_RE = re.compile(r"^(?P<prefix>.+)_clip(?P<index>\d+)$")
GROUP_BATCH_RE = re.compile(r"^(?P<dataset>.+)_u(?P<size>[123])b(?P<batch>\d+)$")


@dataclass(frozen=True)
class GazePoint:
    time_seconds: float
    x: float
    y: float


@dataclass(frozen=True)
class AnnotationClipGroup:
    annotator_number: int
    dataset: str
    group_size: int
    batch_name: str | None
    group_name: str
    clip_prefix: str
    final_clip_path: Path
    gaze_points: list[GazePoint]
    annotation_resolved_video_path: str | None = None
    annotation_resolved_audio_path: str | None = None
    # Provenance metadata
    task_number: int | None = None
    task_instance_id: int | None = None
    task_instance: str | None = None
    video_number: int | None = None
    annotation_key: int | None = None
    annotation_video_path: str | None = None
    annotation_json_path: str | None = None
    recording_dir: str | None = None
    gaze_mapping: str = "measured-player"
    response_selection: str | None = None
    response_index: int | None = None
    response_created: str | None = None
    response_submitted: bool | None = None
    annotation_sources: list[dict[str, object]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create dialogue-partition-style cumulative clips with the final "
            "segment blurred or blocked around human eye focus."
        )
    )
    parser.add_argument(
        "output_parent",
        type=Path,
        help="Output parent. The script writes dataset/context/{1,2,3}-utt_group under this folder.",
    )
    parser.add_argument(
        "recording_parent_dir",
        type=Path,
        help="Parent folder containing Pupil Core recordings named T{x}_{y}_annotator1/2.",
    )
    parser.add_argument(
        "annotation_dir",
        type=Path,
        help="Folder containing annotation JSON files named like T{x}_{y}.json.",
    )
    parser.add_argument(
        "local_path_prefix",
        type=Path,
        help=f"Local path prefix replacing the {MEDIA_URL_PREFIX} URL prefix.",
    )
    parser.add_argument(
        "--original-output-parent",
        type=Path,
        default=None,
        help=(
            "Optional output parent for unmanipulated clips copied into the same "
            "dataset/context/{1,2,3}-utt_group structure."
        ),
    )
    parser.add_argument(
        "--full-corruption-output-parent",
        type=Path,
        default=None,
        help=(
            "Optional output parent for clips where gaze corruption is applied "
            "throughout each whole clip instead of only the final segment."
        ),
    )
    parser.add_argument(
        "--focus-plot-output-parent",
        type=Path,
        default=None,
        help=(
            "Optional output parent for static gaze plots generated from the "
            "same mapped gaze points used by this script."
        ),
    )
    parser.add_argument(
        "--debug-overlay-output-parent",
        type=Path,
        default=None,
        help=(
            "Optional output parent for videos that overlay the exact per-frame "
            "gaze point used for corruption."
        ),
    )
    parser.add_argument(
        "--skip-final-segment-output",
        action="store_true",
        help="Skip generating the final-0.5s corrupted cumulative clips.",
    )
    parser.add_argument(
        "--gaze-mapping",
        choices=GAZE_MAPPING_CHOICES,
        default="measured-player",
        help=(
            "Screen-to-video gaze mapping. 'legacy-extraction' reproduces the "
            "64b3e28 eyetrack_annotation focus plots; 'measured-player' uses "
            "the measured 1920x1080 browser/player geometry."
        ),
    )
    parser.add_argument(
        "--response-selection",
        choices=RESPONSE_SELECTION_CHOICES,
        default="latest-submitted",
        help=(
            "Which response to read from each annotator node. "
            "'first-response' matches the 64b3e28 extraction behavior."
        ),
    )
    parser.add_argument(
        "--video-json",
        type=Path,
        default=None,
        help="JSON file containing the ordered task-instance video lists.",
    )
    parser.add_argument(
        "--full-corruption-localization-source",
        choices=("annotation-media", "video-json"),
        default="annotation-media",
        help=(
            "Which media reference to use when grouping and sourcing the full-corruption "
            "outputs. 'annotation-media' uses press_data video_path/audio_path; "
            "'video-json' uses task2.json indexing."
        ),
    )
    parser.add_argument(
        "--use-annotation-video-path",
        action="store_true",
        help=(
            "Use video_path from each annotation JSON when video-json lookup fails. "
            "By default, annotation video_path is ignored because frontend exports "
            "can point to the wrong first video."
        ),
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default=None,
        help=(
            "Optional legacy mode: partition source videos named dia{n}_utt{m}.mp4. "
            "When omitted, the script uses annotation video_path clips directly."
        ),
    )
    parser.add_argument(
        "--media-url-prefix",
        default=MEDIA_URL_PREFIX,
        help="URL prefix to replace with local_path_prefix.",
    )
    parser.add_argument(
        "--clip-length",
        type=float,
        default=0.5,
        help="Length in seconds for cumulative target-utterance segments.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan source videos recursively.",
    )
    parser.add_argument(
        "--dialogue-range",
        type=int,
        default=None,
        help="1-based hundred-range index. E.g. 1 -> dialogues [0,100).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Number of utterances to skip between consecutive 3-utterance windows.",
    )
    parser.add_argument(
        "--cut",
        type=int,
        default=None,
        help="Optional max seconds of the target utterance before cumulative clipping.",
    )
    parser.add_argument(
        "--utt",
        type=str,
        default=None,
        help="Comma-separated group sizes to materialize, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output folders for generated groups before writing.",
    )
    parser.add_argument(
        "--effect",
        choices=("blur", "block"),
        default="blur",
        help="How to hide the focus region in the final segment.",
    )
    parser.add_argument(
        "--focus-box-ratio",
        type=float,
        default=DEFAULT_FOCUS_BOX_RATIO,
        help="Focus box size as a fraction of min(frame_width, frame_height).",
    )
    parser.add_argument(
        "--max-gaze-gap",
        type=float,
        default=0.5,
        help="Maximum seconds between a frame and nearest gaze sample to apply masking.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum gaze confidence threshold.",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=TIME_TOLERANCE_SECONDS,
        help="Allowed seconds of difference between end-start and current_video_time.",
    )
    parser.add_argument(
        "--system-to-pupil-offset",
        type=float,
        default=None,
        help="Optional fixed offset added to UNIX annotation timestamps to get Pupil Time.",
    )
    parser.add_argument(
        "--write-frame-focus-csv",
        action="store_true",
        help="Write output_parent/frame_focus_points.csv with one nearest focus point per source frame.",
    )
    return parser.parse_args()


def map_sample_to_video_point(
    sample: dict,
    gaze_mapping: str,
) -> tuple[float, float] | None:
    if gaze_mapping == "legacy-extraction":
        return map_screen_sample_to_video_point_legacy_extraction(sample)
    if gaze_mapping == "measured-player":
        return map_screen_sample_to_video_point(sample)
    raise ValueError(f"Unsupported gaze mapping: {gaze_mapping}")


def canonical_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def add_gaze_points(
    exact_index: dict[str, list[GazePoint]],
    name_index: dict[str, list[GazePoint]],
    video_path: Path,
    points: Iterable[GazePoint],
) -> None:
    point_list = list(points)
    if not point_list:
        return

    exact_index[canonical_path(video_path)].extend(point_list)
    name_index[video_path.name].extend(point_list)


def build_gaze_index(
    recording_parent_dir: Path,
    annotation_dir: Path,
    local_path_prefix: Path,
    media_url_prefix: str,
    confidence_threshold: float,
    duration_tolerance: float,
    system_to_pupil_offset: float | None,
    video_json: Path | None,
    gaze_mapping: str,
    response_selection: str,
    use_annotation_video_path: bool,
) -> tuple[dict[str, list[GazePoint]], dict[str, list[GazePoint]]]:
    annotation_files = find_annotation_jsons(annotation_dir)
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation JSON files named like T{{x}}_{{y}}.json found in {annotation_dir}"
        )

    video_entries = None
    if video_json is not None:
        video_entries = load_video_entries(video_json, local_path_prefix, media_url_prefix)

    exact_index: dict[str, list[GazePoint]] = defaultdict(list)
    name_index: dict[str, list[GazePoint]] = defaultdict(list)

    for task_number, task_instance_id, annotation_json in annotation_files:
        task_instance_name = f"T{task_number}_{task_instance_id}"
        all_timings = load_all_video_timings(
            annotation_json,
            duration_tolerance,
            response_selection=response_selection,
        )

        for annotator_timings in all_timings:
            recording_dir = (
                recording_parent_dir
                / f"{task_instance_name}_annotator{annotator_timings.annotator_number}"
            )
            if not recording_dir.is_dir():
                logging.warning("Skipping missing Pupil recording folder: %s", recording_dir)
                continue

            timestamps, gaze_data = load_gaze(recording_dir)
            offset = (
                system_to_pupil_offset
                if system_to_pupil_offset is not None
                else load_recording_time_offset(recording_dir)
            )

            for timing in annotator_timings.timings:
                if timing.video_start_time is None or timing.video_end_time is None:
                    continue

                video_entry = get_video_entry_for_timing(
                    timing=timing,
                    video_entries=video_entries,
                    local_path_prefix=local_path_prefix,
                    media_url_prefix=media_url_prefix,
                    task_instance_id=task_instance_id,
                    use_annotation_video_path=use_annotation_video_path,
                )
                if video_entry is None:
                    continue

                start_time = annotation_time_to_pupil_time(timing.video_start_time, offset)
                end_time = annotation_time_to_pupil_time(timing.video_end_time, offset)
                samples = extract_gaze_in_interval(
                    timestamps=timestamps,
                    gaze_data=gaze_data,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_threshold=confidence_threshold,
                )

                points: list[GazePoint] = []
                for sample in samples:
                    mapped = map_sample_to_video_point(sample, gaze_mapping)
                    if mapped is None:
                        continue
                    points.append(
                        GazePoint(
                            time_seconds=float(sample["timestamp"]) - start_time,
                            x=mapped[0],
                            y=mapped[1],
                        )
                    )

                add_gaze_points(
                    exact_index=exact_index,
                    name_index=name_index,
                    video_path=video_entry.video_path,
                    points=points,
                )

    for points in exact_index.values():
        points.sort(key=lambda point: point.time_seconds)
    for points in name_index.values():
        points.sort(key=lambda point: point.time_seconds)

    return dict(exact_index), dict(name_index)


def parse_annotation_clip_layout(final_clip_path: Path) -> tuple[str, int, str | None, str, str, int]:
    stem_match = CLIP_STEM_RE.fullmatch(final_clip_path.stem)
    if stem_match is None:
        raise ValueError(f"Annotation video path is not named like *_clipN.mp4: {final_clip_path}")

    clip_prefix = stem_match.group("prefix")
    clip_index = int(stem_match.group("index"))
    group_name = final_clip_path.parent.name
    parts = final_clip_path.parts

    for index, part in enumerate(parts):
        if part not in {"u1", "u2", "u3"}:
            continue
        if index + 1 >= len(parts):
            continue
        batch_match = GROUP_BATCH_RE.fullmatch(parts[index + 1])
        if batch_match is None:
            continue
        group_size = int(part[1:])
        if int(batch_match.group("size")) != group_size:
            logging.warning(
                "Group size mismatch between %s and %s in %s",
                part,
                parts[index + 1],
                final_clip_path,
            )
        return (
            batch_match.group("dataset"),
            group_size,
            f"batch{int(batch_match.group('batch')):02d}",
            group_name,
            clip_prefix,
            clip_index,
        )

    raise ValueError(
        "Could not parse dataset/group layout from annotation clip path "
        f"(expected .../u1/meld_u1b1/group/group_clipN.mp4): {final_clip_path}"
    )


def merge_gaze_points(existing: list[GazePoint], new_points: Iterable[GazePoint]) -> list[GazePoint]:
    merged = [*existing, *new_points]
    merged.sort(key=lambda point: point.time_seconds)
    return merged


def build_annotation_source_record(
    task_number: int,
    task_instance_id: int,
    task_instance_name: str,
    annotator_timings,
    timing,
    annotation_json: Path,
    recording_dir: Path,
    video_path: Path,
    annotation_video_path: str | None,
    start_time: float,
    end_time: float,
    system_to_pupil_offset: float,
    raw_sample_count: int,
    mapped_sample_count: int,
    gaze_mapping: str,
    gaze_source: dict[str, str],
) -> dict[str, object]:
    return {
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
        "annotation_video_path": annotation_video_path,
        "resolved_video_path": str(video_path),
        "annotation_json": str(annotation_json),
        "recording_dir": str(recording_dir),
        "gaze_source_type": gaze_source["gaze_source_type"],
        "gaze_source_path": gaze_source["gaze_source_path"],
        "gaze_timestamps_path": gaze_source["gaze_timestamps_path"],
        "gaze_pldata_path": gaze_source["gaze_pldata_path"],
        "system_to_pupil_offset": system_to_pupil_offset,
        "pupil_start_time": start_time,
        "pupil_end_time": end_time,
        "raw_gaze_sample_count": raw_sample_count,
        "mapped_gaze_sample_count": mapped_sample_count,
        "gaze_mapping": gaze_mapping,
    }


def resolve_annotation_media_paths(
    timing,
    local_path_prefix: Path,
    media_url_prefix: str,
) -> tuple[Path | None, Path | None]:
    return (
        resolve_optional_media_path(
            timing.video_path,
            local_path_prefix,
            media_url_prefix,
        ),
        resolve_optional_media_path(
            timing.audio_path,
            local_path_prefix,
            media_url_prefix,
        ),
    )


def choose_clip_localization_video_path(
    clip_localization_source: str,
    annotation_media_video_path: Path | None,
    indexed_video_path: Path,
    task_instance_name: str,
    annotator_number: int,
    video_number: int,
) -> Path:
    if clip_localization_source != "annotation-media":
        return indexed_video_path
    if annotation_media_video_path is not None:
        return annotation_media_video_path

    logging.warning(
        "Falling back to video-json localization for %s annotator %s video %s "
        "because press_data video_path is unavailable.",
        task_instance_name,
        annotator_number,
        video_number,
    )
    return indexed_video_path


def build_annotation_clip_groups(
    recording_parent_dir: Path,
    annotation_dir: Path,
    local_path_prefix: Path,
    media_url_prefix: str,
    confidence_threshold: float,
    duration_tolerance: float,
    system_to_pupil_offset: float | None,
    video_json: Path | None,
    gaze_mapping: str,
    response_selection: str,
    use_annotation_video_path: bool,
    clip_localization_source: str = "video-json",
) -> list[AnnotationClipGroup]:
    annotation_files = find_annotation_jsons(annotation_dir)
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation JSON files named like T{{x}}_{{y}}.json found in {annotation_dir}"
        )

    video_entries = None
    if video_json is not None:
        video_entries = load_video_entries(video_json, local_path_prefix, media_url_prefix)

    grouped: dict[tuple[int, str, int, str | None, str, str, str], AnnotationClipGroup] = {}

    for task_number, task_instance_id, annotation_json in annotation_files:
        task_instance_name = f"T{task_number}_{task_instance_id}"
        all_timings = load_all_video_timings(
            annotation_json,
            duration_tolerance,
            response_selection=response_selection,
        )

        for annotator_timings in all_timings:
            recording_dir = (
                recording_parent_dir
                / f"{task_instance_name}_annotator{annotator_timings.annotator_number}"
            )
            if not recording_dir.is_dir():
                logging.warning("Skipping missing Pupil recording folder: %s", recording_dir)
                continue

            timestamps, gaze_data = load_gaze(recording_dir)
            gaze_source = gaze_source_metadata(recording_dir)
            offset = (
                system_to_pupil_offset
                if system_to_pupil_offset is not None
                else load_recording_time_offset(recording_dir)
            )

            for timing in annotator_timings.timings:
                if timing.video_start_time is None or timing.video_end_time is None:
                    continue

                video_entry = get_video_entry_for_timing(
                    timing=timing,
                    video_entries=video_entries,
                    local_path_prefix=local_path_prefix,
                    media_url_prefix=media_url_prefix,
                    task_instance_id=task_instance_id,
                    use_annotation_video_path=use_annotation_video_path,
                )
                if video_entry is None:
                    continue

                annotation_media_video_path, annotation_media_audio_path = (
                    resolve_annotation_media_paths(
                        timing,
                        local_path_prefix,
                        media_url_prefix,
                    )
                )
                localization_video_path = choose_clip_localization_video_path(
                    clip_localization_source=clip_localization_source,
                    annotation_media_video_path=annotation_media_video_path,
                    indexed_video_path=video_entry.video_path,
                    task_instance_name=task_instance_name,
                    annotator_number=annotator_timings.annotator_number,
                    video_number=timing.video_number,
                )

                try:
                    dataset, group_size, batch_name, group_name, clip_prefix, _ = (
                        parse_annotation_clip_layout(localization_video_path)
                    )
                except ValueError as exc:
                    if (
                        clip_localization_source == "annotation-media"
                        and localization_video_path != video_entry.video_path
                    ):
                        logging.warning(
                            "Annotation media layout parse failed for %s; falling back to "
                            "video-json path %s",
                            localization_video_path,
                            video_entry.video_path,
                        )
                        localization_video_path = video_entry.video_path
                        try:
                            dataset, group_size, batch_name, group_name, clip_prefix, _ = (
                                parse_annotation_clip_layout(localization_video_path)
                            )
                        except ValueError:
                            logging.warning(
                                "Skipping annotation clip with unsupported layout after fallback: %s",
                                exc,
                            )
                            continue
                    else:
                        logging.warning("Skipping annotation clip with unsupported layout: %s", exc)
                        continue

                start_time = annotation_time_to_pupil_time(timing.video_start_time, offset)
                end_time = annotation_time_to_pupil_time(timing.video_end_time, offset)
                samples = extract_gaze_in_interval(
                    timestamps=timestamps,
                    gaze_data=gaze_data,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_threshold=confidence_threshold,
                )
                points = []
                for sample in samples:
                    mapped = map_sample_to_video_point(sample, gaze_mapping)
                    if mapped is None:
                        continue
                    points.append(
                        GazePoint(
                            time_seconds=float(sample["timestamp"]) - start_time,
                            x=mapped[0],
                            y=mapped[1],
                        )
                    )
                annotation_source = build_annotation_source_record(
                    task_number=task_number,
                    task_instance_id=task_instance_id,
                    task_instance_name=task_instance_name,
                    annotator_timings=annotator_timings,
                    timing=timing,
                    annotation_json=annotation_json,
                    recording_dir=recording_dir,
                    video_path=localization_video_path,
                    annotation_video_path=timing.video_path,
                    start_time=start_time,
                    end_time=end_time,
                    system_to_pupil_offset=offset,
                    raw_sample_count=len(samples),
                    mapped_sample_count=len(points),
                    gaze_mapping=gaze_mapping,
                    gaze_source=gaze_source,
                )

                key = (
                    annotator_timings.annotator_number,
                    dataset,
                    group_size,
                    batch_name,
                    group_name,
                    clip_prefix,
                    canonical_path(localization_video_path),
                )
                existing = grouped.get(key)
                if existing is None:
                    grouped[key] = AnnotationClipGroup(
                        annotator_number=annotator_timings.annotator_number,
                        dataset=dataset,
                        group_size=group_size,
                        batch_name=batch_name,
                        group_name=group_name,
                        clip_prefix=clip_prefix,
                        final_clip_path=localization_video_path,
                        gaze_points=merge_gaze_points([], points),
                        annotation_resolved_video_path=(
                            None if annotation_media_video_path is None else str(annotation_media_video_path)
                        ),
                        annotation_resolved_audio_path=(
                            None if annotation_media_audio_path is None else str(annotation_media_audio_path)
                        ),
                        task_number=task_number,
                        task_instance_id=task_instance_id,
                        task_instance=task_instance_name,
                        video_number=timing.video_number,
                        annotation_key=timing.annotation_key,
                        annotation_video_path=timing.video_path,
                        annotation_json_path=str(annotation_json),
                        recording_dir=str(recording_dir),
                        gaze_mapping=gaze_mapping,
                        response_selection=annotator_timings.response_selection,
                        response_index=annotator_timings.response_index,
                        response_created=annotator_timings.response_created,
                        response_submitted=annotator_timings.response_submitted,
                        annotation_sources=[annotation_source],
                    )
                else:
                    grouped[key] = AnnotationClipGroup(
                        annotator_number=existing.annotator_number,
                        dataset=existing.dataset,
                        group_size=existing.group_size,
                        batch_name=existing.batch_name,
                        group_name=existing.group_name,
                        clip_prefix=existing.clip_prefix,
                        final_clip_path=existing.final_clip_path,
                        gaze_points=merge_gaze_points(existing.gaze_points, points),
                        annotation_resolved_video_path=existing.annotation_resolved_video_path,
                        annotation_resolved_audio_path=existing.annotation_resolved_audio_path,
                        task_number=existing.task_number,
                        task_instance_id=existing.task_instance_id,
                        task_instance=existing.task_instance,
                        video_number=existing.video_number,
                        annotation_key=existing.annotation_key,
                        annotation_video_path=existing.annotation_video_path,
                        annotation_json_path=existing.annotation_json_path,
                        recording_dir=existing.recording_dir,
                        gaze_mapping=existing.gaze_mapping,
                        response_selection=existing.response_selection,
                        response_index=existing.response_index,
                        response_created=existing.response_created,
                        response_submitted=existing.response_submitted,
                        annotation_sources=[
                            *(existing.annotation_sources or []),
                            annotation_source,
                        ],
                    )

    return sorted(
        grouped.values(),
        key=lambda item: (
            item.annotator_number,
            item.dataset,
            item.group_size,
            item.batch_name or "",
            item.group_name,
            item.clip_prefix,
        ),
    )


def discover_sibling_clips(final_clip_path: Path, clip_prefix: str) -> list[Path]:
    clips = []
    for path in final_clip_path.parent.glob(f"{clip_prefix}_clip*.mp4"):
        match = CLIP_STEM_RE.fullmatch(path.stem)
        if match is None or match.group("prefix") != clip_prefix:
            continue
        clips.append(path)
    return sorted(clips, key=lambda path: int(CLIP_STEM_RE.fullmatch(path.stem).group("index")))


def output_dir_for_annotation_group(group: AnnotationClipGroup, output_parent: Path) -> Path:
    output_dir = (
        output_parent
        / f"annotator{group.annotator_number}"
        / group.dataset
        / "context"
        / GROUP_FOLDER_NAMES[group.group_size]
    )
    if group.batch_name is not None:
        output_dir /= group.batch_name
    return output_dir / group.group_name


def lookup_gaze_points(
    source_path: str | Path,
    exact_index: dict[str, list[GazePoint]],
    name_index: dict[str, list[GazePoint]],
) -> list[GazePoint]:
    path = Path(source_path)
    points = exact_index.get(canonical_path(path))
    if points is not None:
        return points
    return name_index.get(path.name, [])


def nearest_gaze_point(
    points: list[GazePoint],
    time_seconds: float,
    max_gap: float,
) -> GazePoint | None:
    if not points:
        return None

    lo = 0
    hi = len(points)
    while lo < hi:
        mid = (lo + hi) // 2
        if points[mid].time_seconds < time_seconds:
            lo = mid + 1
        else:
            hi = mid

    candidates = []
    if lo < len(points):
        candidates.append(points[lo])
    if lo > 0:
        candidates.append(points[lo - 1])
    if not candidates:
        return None

    nearest = min(candidates, key=lambda point: abs(point.time_seconds - time_seconds))
    if abs(nearest.time_seconds - time_seconds) > max_gap:
        return None
    return nearest


def apply_focus_mask(
    frame,
    point: GazePoint,
    effect: str,
    focus_box_ratio: float,
) -> None:
    height, width = frame.shape[:2]
    box_size = max(8, int(min(width, height) * focus_box_ratio))
    center_x = int(point.x * width)
    center_y = int((1.0 - point.y) * height)
    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(width, center_x + box_size // 2)
    y2 = min(height, center_y + box_size // 2)
    if x2 <= x1 or y2 <= y1:
        return

    if effect == "block":
        frame[y1:y2, x1:x2] = (0, 0, 0)
        return

    roi = frame[y1:y2, x1:x2]
    kernel = max(15, (box_size // 2) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)


def encode_video_with_fallback(raw_video_path: Path, output_path: Path) -> None:
    run_ffmpeg_with_video_codec_fallback(
        lambda video_codec: [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video_path),
            "-map",
            "0:v",
            "-c:v",
            video_codec,
            "-an",
            str(output_path),
        ],
        context=f"Encoding masked video {output_path.stem}",
    )


def strip_audio_video(source_path: str | Path, output_path: Path) -> None:
    run_ffmpeg_with_video_codec_fallback(
        lambda video_codec: [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v",
            "-c:v",
            video_codec,
            "-an",
            str(output_path),
        ],
        context=f"Creating visual-only context video {output_path.stem}",
    )


def write_masked_target_prefix_video(
    source_path: str | Path,
    output_path: Path,
    duration_seconds: float,
    clip_length: float,
    gaze_points: list[GazePoint],
    effect: str,
    focus_box_ratio: float,
    max_gaze_gap: float,
    mask_full_video: bool = False,
) -> dict[str, int]:
    source_path = Path(source_path)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata for {source_path}")

    frame_limit = min(total_frames, max(1, int(round(duration_seconds * fps))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked_frame_count = 0

    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_") as tmpdir:
        raw_video_path = Path(tmpdir) / "masked_raw.mp4"
        writer = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create temporary video writer for {raw_video_path}")

        mask_start_time = 0.0 if mask_full_video else max(0.0, duration_seconds - clip_length)
        frame_index = 0
        while frame_index < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break

            frame_time = frame_index / fps
            if frame_time >= mask_start_time:
                point = nearest_gaze_point(gaze_points, frame_time, max_gaze_gap)
                if point is not None:
                    apply_focus_mask(frame, point, effect, focus_box_ratio)
                    masked_frame_count += 1

            writer.write(frame)
            frame_index += 1

        writer.release()
        cap.release()
        encode_video_with_fallback(raw_video_path, output_path)

    return {"frames": frame_limit, "masked_frames": masked_frame_count}


def extract_audio_segment(source_path: str | Path, output_wav_path: Path, duration_seconds: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-t",
        str(duration_seconds),
        "-i",
        str(source_path),
        "-map",
        "0:a",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]
    subprocess_result = subprocess.run(command, capture_output=True, text=True)
    if subprocess_result.returncode != 0:
        raise RuntimeError(
            f"Extracting audio failed for {source_path}\n"
            f"Command: {' '.join(command)}\n"
            f"ffmpeg stderr: {(subprocess_result.stderr or '')[-1000:]}"
        )


def copy_file_contents(source_path: str | Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_path)


def move_file_contents(source_path: str | Path, output_path: Path) -> None:
    copy_file_contents(source_path, output_path)
    Path(source_path).unlink()


def copy_or_extract_clip_audio(
    source_clip_path: Path,
    output_wav_path: Path,
    preferred_audio_source_path: Path | None = None,
) -> bool:
    if preferred_audio_source_path is not None and preferred_audio_source_path.exists():
        copy_file_contents(preferred_audio_source_path, output_wav_path)
        return True

    source_wav_path = source_clip_path.with_suffix(".wav")
    if source_wav_path.exists():
        copy_file_contents(source_wav_path, output_wav_path)
        return True

    _, _, duration_seconds = get_video_properties(source_clip_path)
    extract_audio_segment(source_clip_path, output_wav_path, duration_seconds)
    return output_wav_path.exists()


def write_group_audio(
    context_paths: list[str],
    target_source_path: str,
    output_wav_path: Path,
    target_duration_seconds: float,
    temp_dir: Path,
) -> None:
    if not context_paths:
        extract_audio_segment(target_source_path, output_wav_path, target_duration_seconds)
        return

    target_cut_path = temp_dir / "target_audio_window.mp4"
    run_ffmpeg_with_video_codec_fallback(
        lambda video_codec: [
            "ffmpeg",
            "-y",
            "-t",
            str(target_duration_seconds),
            "-i",
            str(target_source_path),
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:v",
            video_codec,
            "-c:a",
            "aac",
            "-shortest",
            str(target_cut_path),
        ],
        context="Cutting target audio/video window",
    )

    audio_context_path = temp_dir / "audio_context.mp4"
    concatenate_group_video([*context_paths, str(target_cut_path)], audio_context_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_context_path),
        "-map",
        "0:a",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]
    subprocess_result = subprocess.run(command, capture_output=True, text=True)
    if subprocess_result.returncode != 0:
        raise RuntimeError(
            f"Extracting contextual audio failed.\n"
            f"Command: {' '.join(command)}\n"
            f"ffmpeg stderr: {(subprocess_result.stderr or '')[-1000:]}"
        )


def concatenate_context_with_masked_target(
    context_paths: list[str],
    masked_target_path: Path,
    output_path: Path,
    temp_dir: Path,
) -> None:
    if not context_paths:
        move_file_contents(masked_target_path, output_path)
        return

    visual_context_paths: list[str] = []
    for index, context_path in enumerate(context_paths):
        visual_context_path = temp_dir / f"visual_context_{index}.mp4"
        strip_audio_video(context_path, visual_context_path)
        visual_context_paths.append(str(visual_context_path))
    concatenate_group_video([*visual_context_paths, str(masked_target_path)], output_path)


def materialize_gaze_group_context(
    group: PartitionGroup,
    output_parent: Path,
    clip_length: float,
    gaze_points: list[GazePoint],
    effect: str,
    focus_box_ratio: float,
    max_gaze_gap: float,
    cut: int | None,
    overwrite: bool,
) -> dict[str, object] | None:
    group_root = output_parent / GROUP_FOLDER_NAMES[group.group_size] / group.group_name
    clip_prefix = group.group_name

    if group_root.exists() and any(group_root.glob(f"{clip_prefix}_clip*.mp4")):
        if not overwrite:
            return None
        shutil.rmtree(group_root)
    group_root.mkdir(parents=True, exist_ok=True)

    target_fps, target_total_frames, target_duration = get_video_properties(group.source_paths[-1])
    if cut is not None:
        target_total_frames = min(target_total_frames, int(float(cut) * target_fps))
        target_duration = min(target_duration, float(cut))
    clip_frames = int(clip_length * target_fps)
    if clip_frames <= 0:
        raise RuntimeError(
            f"Invalid clip length ({clip_length}) or FPS ({target_fps}) for {group.source_paths[-1]}"
        )
    clip_count = target_total_frames // clip_frames
    if clip_count <= 0:
        raise RuntimeError(f"Target video is shorter than clip length: {group.source_paths[-1]}")

    clip_mp4_count = 0
    clip_wav_count = 0
    total_masked_frames = 0
    context_paths = group.source_paths[:-1]

    with tempfile.TemporaryDirectory(prefix=f"{group.group_name}_", dir=str(group_root)) as tmpdir:
        temp_dir = Path(tmpdir)
        for clip_index in range(1, clip_count + 1):
            target_clip_duration = clip_index * clip_frames / target_fps
            output_mp4_path = group_root / f"{clip_prefix}_clip{clip_index}.mp4"
            output_wav_path = group_root / f"{clip_prefix}_clip{clip_index}.wav"
            masked_target_path = temp_dir / f"{clip_prefix}_target_clip{clip_index}.mp4"

            stats = write_masked_target_prefix_video(
                source_path=group.source_paths[-1],
                output_path=masked_target_path,
                duration_seconds=target_clip_duration,
                clip_length=clip_length,
                gaze_points=gaze_points,
                effect=effect,
                focus_box_ratio=focus_box_ratio,
                max_gaze_gap=max_gaze_gap,
            )
            concatenate_context_with_masked_target(
                context_paths=context_paths,
                masked_target_path=masked_target_path,
                output_path=output_mp4_path,
                temp_dir=temp_dir,
            )
            write_group_audio(
                context_paths=context_paths,
                target_source_path=group.source_paths[-1],
                output_wav_path=output_wav_path,
                target_duration_seconds=target_clip_duration,
                temp_dir=temp_dir,
            )
            if output_mp4_path.exists():
                clip_mp4_count += 1
            if output_wav_path.exists():
                clip_wav_count += 1
            total_masked_frames += stats["masked_frames"]

    return {
        "dialogue_id": group.dialogue_id,
        "group_size": group.group_size,
        "group_name": group.group_name,
        "utterance_ids": group.utterance_ids,
        "mode": "context",
        "effect": effect,
        "whole_videos": [],
        "clip_output_folder": str(group_root),
        "clip_mp4_count": clip_mp4_count,
        "clip_wav_count": clip_wav_count,
        "masked_frame_count": total_masked_frames,
        "gaze_point_count": len(gaze_points),
    }


def materialize_gaze_triplet(
    triplet,
    output_parent: Path,
    clip_length: float,
    selected_group_sizes: tuple[int, ...],
    exact_gaze_index: dict[str, list[GazePoint]],
    name_gaze_index: dict[str, list[GazePoint]],
    effect: str,
    focus_box_ratio: float,
    max_gaze_gap: float,
    cut: int | None,
    overwrite: bool,
) -> tuple[list[dict[str, object]], FailedTriplet | None]:
    results: list[dict[str, object]] = []
    touched_groups: list[PartitionGroup] = []

    for group in sorted(triplet.groups, key=lambda item: item.group_size):
        if group.group_size not in selected_group_sizes:
            continue

        try:
            gaze_points = lookup_gaze_points(
                group.source_paths[-1],
                exact_gaze_index,
                name_gaze_index,
            )
            result = materialize_gaze_group_context(
                group=group,
                output_parent=output_parent,
                clip_length=clip_length,
                gaze_points=gaze_points,
                effect=effect,
                focus_box_ratio=focus_box_ratio,
                max_gaze_gap=max_gaze_gap,
                cut=cut,
                overwrite=overwrite,
            )
            if result is None:
                result = {
                    "dialogue_id": group.dialogue_id,
                    "group_size": group.group_size,
                    "group_name": group.group_name,
                    "utterance_ids": group.utterance_ids,
                    "mode": "context",
                    "clip_output_folder": str(
                        output_parent / GROUP_FOLDER_NAMES[group.group_size] / group.group_name
                    ),
                    "clip_mp4_count": len(
                        list((output_parent / GROUP_FOLDER_NAMES[group.group_size] / group.group_name).glob("*.mp4"))
                    ),
                    "clip_wav_count": len(
                        list((output_parent / GROUP_FOLDER_NAMES[group.group_size] / group.group_name).glob("*.wav"))
                    ),
                    "reused": True,
                }
            else:
                touched_groups.append(group)
            validate_materialized_group(result)
            results.append(result)
        except Exception as exc:
            for rollback_group in touched_groups:
                group_root = (
                    output_parent
                    / GROUP_FOLDER_NAMES[rollback_group.group_size]
                    / rollback_group.group_name
                )
                if group_root.exists():
                    shutil.rmtree(group_root)
            return [], FailedTriplet(
                dialogue_id=triplet.dialogue_id,
                utterance_ids=triplet.utterance_ids,
                failed_group_name=group.group_name,
                error=str(exc),
            )

    return results, None


def _write_provenance_metadata(
    output_dir: Path,
    group: AnnotationClipGroup,
    source_clips: list[Path],
) -> None:
    """Write ``provenance.json`` and a gaze scatter plot into *output_dir*."""
    avg_x: float | None = None
    avg_y: float | None = None
    if group.gaze_points:
        avg_x = sum(p.x for p in group.gaze_points) / len(group.gaze_points)
        avg_y = sum(p.y for p in group.gaze_points) / len(group.gaze_points)

    metadata = {
        "task_number": group.task_number,
        "task_instance_id": group.task_instance_id,
        "task_instance": group.task_instance,
        "video_number": group.video_number,
        "annotation_key": group.annotation_key,
        "resolved_video_path": str(group.final_clip_path),
        "annotation_video_path": group.annotation_video_path,
        "annotation_resolved_video_path": group.annotation_resolved_video_path,
        "annotation_resolved_audio_path": group.annotation_resolved_audio_path,
        "source_clips": [str(p) for p in source_clips],
        "recording_dir": group.recording_dir,
        "annotation_json": group.annotation_json_path,
        "annotator_number": group.annotator_number,
        "gaze_mapping": group.gaze_mapping,
        "response_selection": group.response_selection,
        "response_index": group.response_index,
        "response_created": group.response_created,
        "response_submitted": group.response_submitted,
        "annotation_source_count": len(group.annotation_sources or []),
        "annotation_sources": group.annotation_sources or [],
        "gaze_point_count": len(group.gaze_points),
        "avg_gaze_x": avg_x,
        "avg_gaze_y": avg_y,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Generate aggregated gaze scatter plot on a video frame
    _write_annotation_sources_csv(output_dir, group)
    _write_gaze_plot(output_dir, group, source_clips)


def _write_annotation_sources_csv(output_dir: Path, group: AnnotationClipGroup) -> None:
    fieldnames = [
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
        "gaze_source_type",
        "gaze_source_path",
        "annotation_json",
        "recording_dir",
        "gaze_timestamps_path",
        "gaze_pldata_path",
        "system_to_pupil_offset",
        "pupil_start_time",
        "pupil_end_time",
        "raw_gaze_sample_count",
        "mapped_gaze_sample_count",
        "gaze_mapping",
    ]
    with (output_dir / "annotation_sources.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in group.annotation_sources or []:
            writer.writerow({field: source.get(field, "") for field in fieldnames})


def _write_gaze_plot(
    output_dir: Path,
    group: AnnotationClipGroup,
    source_clips: list[Path],
) -> None:
    """Render a scatter plot of all gaze points over a mid-video frame."""
    try:
        if "MPLCONFIGDIR" not in os.environ:
            mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
            mpl_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
        if "XDG_CACHE_HOME" not in os.environ:
            xdg_cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
            xdg_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.warning("matplotlib/numpy not available; skipping gaze plot")
        return

    # Read a frame from the final source clip
    video_path = source_clips[-1] if source_clips else group.final_clip_path
    frame = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
            ok, bgr = cap.read()
            if ok:
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()
    except Exception:
        logging.warning("Could not read frame from %s for gaze plot", video_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f"Annotator {group.annotator_number}, video {group.video_number}: "
        f"{len(group.gaze_points)} gaze points"
    )

    if frame is not None:
        height, width = frame.shape[:2]
        ax.imshow(frame)
        ax.axis("off")
    else:
        width, height = 1.0, 1.0
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    if group.gaze_points:
        xs = np.array([p.x * width for p in group.gaze_points])
        ys = np.array([(1.0 - p.y) * height for p in group.gaze_points])
        ax.scatter(xs, ys, s=12, c="red", alpha=0.35, edgecolors="none")
        ax.scatter([float(np.mean(xs))], [float(np.mean(ys))], s=90, c="yellow", marker="x")

    plot_path = output_dir / "gaze_plot.png"
    fig.tight_layout(pad=0)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_debug_focus_marker(frame, point: GazePoint, focus_box_ratio: float) -> None:
    height, width = frame.shape[:2]
    box_size = max(8, int(min(width, height) * focus_box_ratio))
    center_x = int(point.x * width)
    center_y = int((1.0 - point.y) * height)
    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(width, center_x + box_size // 2)
    y2 = min(height, center_y + box_size // 2)
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.drawMarker(
        frame,
        (center_x, center_y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(12, box_size // 2),
        thickness=2,
    )
    cv2.circle(frame, (center_x, center_y), max(4, box_size // 12), (0, 255, 0), -1)


def write_gaze_overlay_video(
    source_path: str | Path,
    output_path: Path,
    frame_csv_path: Path,
    gaze_points: list[GazePoint],
    focus_box_ratio: float,
    max_gaze_gap: float,
) -> dict[str, int]:
    source_path = Path(source_path)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata for {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_csv_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_frame_count = 0

    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_overlay_") as tmpdir:
        raw_video_path = Path(tmpdir) / "overlay_raw.mp4"
        writer = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create temporary video writer for {raw_video_path}")

        with frame_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame_index",
                    "frame_time_seconds",
                    "nearest_gaze_time_seconds",
                    "gaze_x",
                    "gaze_y",
                    "pixel_x",
                    "pixel_y",
                ],
            )
            csv_writer.writeheader()

            frame_index = 0
            while frame_index < total_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_time = frame_index / fps
                point = nearest_gaze_point(gaze_points, frame_time, max_gaze_gap)
                if point is not None:
                    draw_debug_focus_marker(frame, point, focus_box_ratio)
                    overlay_frame_count += 1
                    csv_writer.writerow(
                        {
                            "frame_index": frame_index,
                            "frame_time_seconds": frame_time,
                            "nearest_gaze_time_seconds": point.time_seconds,
                            "gaze_x": point.x,
                            "gaze_y": point.y,
                            "pixel_x": int(point.x * width),
                            "pixel_y": int((1.0 - point.y) * height),
                        }
                    )
                else:
                    csv_writer.writerow(
                        {
                            "frame_index": frame_index,
                            "frame_time_seconds": frame_time,
                            "nearest_gaze_time_seconds": "",
                            "gaze_x": "",
                            "gaze_y": "",
                            "pixel_x": "",
                            "pixel_y": "",
                        }
                    )

                writer.write(frame)
                frame_index += 1

        writer.release()
        cap.release()
        encode_video_with_fallback(raw_video_path, output_path)

    return {"frames": total_frames, "overlay_frames": overlay_frame_count}


def materialize_annotation_focus_plot_group(
    group: AnnotationClipGroup,
    output_parent: Path,
    overwrite: bool,
) -> dict[str, object]:
    source_clips = discover_sibling_clips(group.final_clip_path, group.clip_prefix)
    if not source_clips:
        raise FileNotFoundError(f"No sibling clips found for {group.final_clip_path}")
    source_clips = [source_clips[-1]]

    output_dir = output_dir_for_annotation_group(group, output_parent)
    output_path = output_dir / "gaze_plot.png"
    if output_path.exists() and not overwrite:
        return {
            "task_instance": group.task_instance,
            "annotator_number": group.annotator_number,
            "dataset": group.dataset,
            "group_size": group.group_size,
            "batch_name": group.batch_name,
            "group_name": group.group_name,
            "clip_prefix": group.clip_prefix,
            "final_clip_path": str(group.final_clip_path),
            "clip_output_folder": str(output_dir),
            "source_clip_count": len(source_clips),
            "clip_mp4_count": 0,
            "clip_wav_count": 0,
            "gaze_point_count": len(group.gaze_points),
            "annotation_source_count": len(group.annotation_sources or []),
            "reused": True,
        }
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_provenance_metadata(output_dir, group, source_clips)
    return {
        "task_instance": group.task_instance,
        "annotator_number": group.annotator_number,
        "dataset": group.dataset,
        "group_size": group.group_size,
        "batch_name": group.batch_name,
        "group_name": group.group_name,
        "clip_prefix": group.clip_prefix,
        "final_clip_path": str(group.final_clip_path),
        "clip_output_folder": str(output_dir),
        "source_clip_count": len(source_clips),
        "clip_mp4_count": 0,
        "clip_wav_count": 0,
        "gaze_point_count": len(group.gaze_points),
        "annotation_source_count": len(group.annotation_sources or []),
    }


def materialize_debug_overlay_group(
    group: AnnotationClipGroup,
    output_parent: Path,
    focus_box_ratio: float,
    max_gaze_gap: float,
    overwrite: bool,
) -> dict[str, object]:
    source_clips = discover_sibling_clips(group.final_clip_path, group.clip_prefix)
    if not source_clips:
        raise FileNotFoundError(f"No sibling clips found for {group.final_clip_path}")
    source_clip_path = source_clips[-1]

    output_dir = output_dir_for_annotation_group(group, output_parent)
    output_mp4_path = output_dir / source_clip_path.name
    output_csv_path = output_dir / f"{source_clip_path.stem}_frame_gaze.csv"
    if output_mp4_path.exists() and not overwrite:
        return {
            "annotator_number": group.annotator_number,
            "dataset": group.dataset,
            "group_size": group.group_size,
            "batch_name": group.batch_name,
            "group_name": group.group_name,
            "clip_prefix": group.clip_prefix,
            "final_clip_path": str(group.final_clip_path),
            "clip_output_folder": str(output_dir),
            "source_clip_count": 1,
            "clip_mp4_count": 1,
            "clip_wav_count": 0,
            "gaze_point_count": len(group.gaze_points),
            "reused": True,
        }
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not group.gaze_points:
        return {
            "annotator_number": group.annotator_number,
            "dataset": group.dataset,
            "group_size": group.group_size,
            "batch_name": group.batch_name,
            "group_name": group.group_name,
            "clip_prefix": group.clip_prefix,
            "final_clip_path": str(group.final_clip_path),
            "clip_output_folder": str(output_dir),
            "source_clip_count": 1,
            "clip_mp4_count": 0,
            "clip_wav_count": 0,
            "gaze_point_count": 0,
            "skipped": True,
            "skip_reason": "no_gaze_points",
        }

    stats = write_gaze_overlay_video(
        source_path=source_clip_path,
        output_path=output_mp4_path,
        frame_csv_path=output_csv_path,
        gaze_points=group.gaze_points,
        focus_box_ratio=focus_box_ratio,
        max_gaze_gap=max_gaze_gap,
    )
    _write_provenance_metadata(output_dir, group, [source_clip_path])
    return {
        "annotator_number": group.annotator_number,
        "dataset": group.dataset,
        "group_size": group.group_size,
        "batch_name": group.batch_name,
        "group_name": group.group_name,
        "clip_prefix": group.clip_prefix,
        "final_clip_path": str(group.final_clip_path),
        "clip_output_folder": str(output_dir),
        "source_clip_count": 1,
        "clip_mp4_count": 1 if output_mp4_path.exists() else 0,
        "clip_wav_count": 0,
        "masked_frame_count": stats["overlay_frames"],
        "gaze_point_count": len(group.gaze_points),
    }


def materialize_annotation_clip_group(
    group: AnnotationClipGroup,
    output_parent: Path,
    clip_length: float,
    effect: str,
    focus_box_ratio: float,
    max_gaze_gap: float,
    overwrite: bool,
    mask_full_video: bool = False,
    use_annotation_media_for_full_video: bool = False,
) -> dict[str, object]:
    source_clips = discover_sibling_clips(group.final_clip_path, group.clip_prefix)
    if not source_clips:
        raise FileNotFoundError(f"No sibling clips found for {group.final_clip_path}")
    preferred_audio_source_path = None
    if mask_full_video:
        if use_annotation_media_for_full_video and group.annotation_resolved_video_path is not None:
            source_clips = [Path(group.annotation_resolved_video_path)]
        else:
            source_clips = [source_clips[-1]]
        if use_annotation_media_for_full_video and group.annotation_resolved_audio_path is not None:
            preferred_audio_source_path = Path(group.annotation_resolved_audio_path)

    output_dir = output_dir_for_annotation_group(group, output_parent)
    mask_mode = "full_video" if mask_full_video else "final_segment"
    if not group.gaze_points:
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        return {
            "annotator_number": group.annotator_number,
            "dataset": group.dataset,
            "group_size": group.group_size,
            "batch_name": group.batch_name,
            "group_name": group.group_name,
            "clip_prefix": group.clip_prefix,
            "final_clip_path": str(group.final_clip_path),
            "clip_output_folder": str(output_dir),
            "source_clip_count": len(source_clips),
            "mask_mode": mask_mode,
            "clip_mp4_count": 0,
            "clip_wav_count": 0,
            "masked_frame_count": 0,
            "gaze_point_count": 0,
            "skipped": True,
            "skip_reason": "no_gaze_points",
        }

    if output_dir.exists() and any(output_dir.glob(f"{group.clip_prefix}_clip*.mp4")):
        if not overwrite:
            mp4_count = len(list(output_dir.glob("*.mp4")))
            wav_count = len(list(output_dir.glob("*.wav")))
            return {
                "annotator_number": group.annotator_number,
                "dataset": group.dataset,
                "group_size": group.group_size,
                "batch_name": group.batch_name,
                "group_name": group.group_name,
                "clip_prefix": group.clip_prefix,
                "final_clip_path": str(group.final_clip_path),
                "clip_output_folder": str(output_dir),
                "source_clip_count": len(source_clips),
                "mask_mode": mask_mode,
                "clip_mp4_count": mp4_count,
                "clip_wav_count": wav_count,
                "masked_frame_count": "",
                "gaze_point_count": len(group.gaze_points),
                "reused": True,
            }
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_count = 0
    wav_count = 0
    masked_frame_count = 0
    with tempfile.TemporaryDirectory(prefix=f"{group.group_name}_") as tmpdir:
        temp_dir = Path(tmpdir)
        for source_clip_path in source_clips:
            _, _, duration_seconds = get_video_properties(source_clip_path)
            output_mp4_path = output_dir / source_clip_path.name
            output_wav_path = output_dir / source_clip_path.with_suffix(".wav").name
            temp_mp4_path = temp_dir / source_clip_path.name
            stats = write_masked_target_prefix_video(
                source_path=source_clip_path,
                output_path=temp_mp4_path,
                duration_seconds=duration_seconds,
                clip_length=clip_length,
                gaze_points=group.gaze_points,
                effect=effect,
                focus_box_ratio=focus_box_ratio,
                max_gaze_gap=max_gaze_gap,
                mask_full_video=mask_full_video,
            )
            masked_frame_count += stats["masked_frames"]
            if stats["masked_frames"] <= 0:
                temp_mp4_path.unlink(missing_ok=True)
                continue

            move_file_contents(temp_mp4_path, output_mp4_path)
            if copy_or_extract_clip_audio(
                source_clip_path,
                output_wav_path,
                preferred_audio_source_path=preferred_audio_source_path,
            ):
                wav_count += 1
            if output_mp4_path.exists():
                mp4_count += 1

    if mp4_count == 0:
        shutil.rmtree(output_dir)
        return {
            "annotator_number": group.annotator_number,
            "dataset": group.dataset,
            "group_size": group.group_size,
            "batch_name": group.batch_name,
            "group_name": group.group_name,
            "clip_prefix": group.clip_prefix,
            "final_clip_path": str(group.final_clip_path),
            "clip_output_folder": str(output_dir),
            "source_clip_count": len(source_clips),
            "mask_mode": mask_mode,
            "clip_mp4_count": 0,
            "clip_wav_count": 0,
            "masked_frame_count": masked_frame_count,
            "gaze_point_count": len(group.gaze_points),
            "skipped": True,
            "skip_reason": "no_masked_frames",
        }

    # Write provenance metadata files
    _write_provenance_metadata(output_dir, group, source_clips)

    return {
        "annotator_number": group.annotator_number,
        "dataset": group.dataset,
        "group_size": group.group_size,
        "batch_name": group.batch_name,
        "group_name": group.group_name,
        "clip_prefix": group.clip_prefix,
        "final_clip_path": str(group.final_clip_path),
        "clip_output_folder": str(output_dir),
        "source_clip_count": len(source_clips),
        "mask_mode": mask_mode,
        "clip_mp4_count": mp4_count,
        "clip_wav_count": wav_count,
        "masked_frame_count": masked_frame_count,
        "gaze_point_count": len(group.gaze_points),
    }


def materialize_original_annotation_clip_group(
    group: AnnotationClipGroup,
    output_parent: Path,
    overwrite: bool,
) -> dict[str, object]:
    source_clips = discover_sibling_clips(group.final_clip_path, group.clip_prefix)
    if not source_clips:
        raise FileNotFoundError(f"No sibling clips found for {group.final_clip_path}")

    output_dir = output_dir_for_annotation_group(group, output_parent)
    if output_dir.exists() and any(output_dir.glob(f"{group.clip_prefix}_clip*.mp4")):
        if not overwrite:
            mp4_count = len(list(output_dir.glob("*.mp4")))
            wav_count = len(list(output_dir.glob("*.wav")))
            return {
                "annotator_number": group.annotator_number,
                "dataset": group.dataset,
                "group_size": group.group_size,
                "batch_name": group.batch_name,
                "group_name": group.group_name,
                "clip_prefix": group.clip_prefix,
                "final_clip_path": str(group.final_clip_path),
                "clip_output_folder": str(output_dir),
                "source_clip_count": len(source_clips),
                "clip_mp4_count": mp4_count,
                "clip_wav_count": wav_count,
                "gaze_point_count": len(group.gaze_points),
                "reused": True,
            }
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_count = 0
    wav_count = 0
    for source_clip_path in source_clips:
        output_mp4_path = output_dir / source_clip_path.name
        output_wav_path = output_dir / source_clip_path.with_suffix(".wav").name
        copy_file_contents(source_clip_path, output_mp4_path)
        mp4_count += 1
        if copy_or_extract_clip_audio(source_clip_path, output_wav_path):
            wav_count += 1

    return {
        "annotator_number": group.annotator_number,
        "dataset": group.dataset,
        "group_size": group.group_size,
        "batch_name": group.batch_name,
        "group_name": group.group_name,
        "clip_prefix": group.clip_prefix,
        "final_clip_path": str(group.final_clip_path),
        "clip_output_folder": str(output_dir),
        "source_clip_count": len(source_clips),
        "clip_mp4_count": mp4_count,
        "clip_wav_count": wav_count,
        "gaze_point_count": len(group.gaze_points),
    }


def write_annotation_clip_summary(
    output_parent: Path,
    results: list[dict[str, object]],
    stem: str = "gaze_blocked_annotation_clip",
    pipeline: str = "annotation_clip_gaze_blocked_partition",
) -> None:
    output_parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "pipeline": pipeline,
        "group_count": len(results),
        "skipped_group_count": sum(1 for result in results if result.get("skipped")),
        "clip_mp4_count": sum(int(result.get("clip_mp4_count", 0)) for result in results),
        "clip_wav_count": sum(int(result.get("clip_wav_count", 0)) for result in results),
        "masked_frame_count": sum(
            int(result.get("masked_frame_count") or 0) for result in results
        ),
        "groups": results,
    }
    (output_parent / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (output_parent / f"{stem}_groups.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "task_instance",
            "annotator_number",
            "dataset",
            "group_size",
            "batch_name",
            "group_name",
            "clip_prefix",
            "final_clip_path",
            "clip_output_folder",
            "source_clip_count",
            "gaze_mapping",
            "mask_mode",
            "clip_mp4_count",
            "clip_wav_count",
            "masked_frame_count",
            "gaze_point_count",
            "annotation_source_count",
            "skipped",
            "skip_reason",
            "reused",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})


def write_frame_focus_csv(
    records,
    output_csv_path: Path,
    exact_gaze_index: dict[str, list[GazePoint]],
    name_gaze_index: dict[str, list[GazePoint]],
    max_gaze_gap: float,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "video_path",
                "frame_index",
                "time_seconds",
                "focus_x",
                "focus_y",
                "nearest_gaze_time_seconds",
            ],
        )
        writer.writeheader()
        for record in records:
            points = lookup_gaze_points(record.path, exact_gaze_index, name_gaze_index)
            fps, total_frames, _ = get_video_properties(record.path)
            for frame_index in range(total_frames):
                frame_time = frame_index / fps
                point = nearest_gaze_point(points, frame_time, max_gaze_gap)
                writer.writerow(
                    {
                        "video_path": record.path,
                        "frame_index": frame_index,
                        "time_seconds": frame_time,
                        "focus_x": "" if point is None else point.x,
                        "focus_y": "" if point is None else point.y,
                        "nearest_gaze_time_seconds": "" if point is None else point.time_seconds,
                    }
                )


def build_gaze_partition_summary(
    base_summary: dict[str, object],
    effect: str,
    focus_box_ratio: float,
    max_gaze_gap: float,
) -> dict[str, object]:
    return {
        **base_summary,
        "pipeline": "gaze_blocked_partition",
        "effect": effect,
        "focus_box_ratio": focus_box_ratio,
        "max_gaze_gap_seconds": max_gaze_gap,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    selected_group_sizes = parse_utt_group_selection(args.utt)
    if args.video_folder is None:
        logging.info("building annotation-clip gaze-blocked dataset")
        annotation_groups = build_annotation_clip_groups(
            recording_parent_dir=args.recording_parent_dir,
            annotation_dir=args.annotation_dir,
            local_path_prefix=args.local_path_prefix,
            media_url_prefix=args.media_url_prefix,
            confidence_threshold=args.confidence,
            duration_tolerance=args.duration_tolerance,
            system_to_pupil_offset=args.system_to_pupil_offset,
            video_json=args.video_json,
            gaze_mapping=args.gaze_mapping,
            response_selection=args.response_selection,
            use_annotation_video_path=args.use_annotation_video_path,
            clip_localization_source="video-json",
        )
        selected_annotation_groups = [
            group for group in annotation_groups if group.group_size in selected_group_sizes
        ]

        if args.focus_plot_output_parent is not None:
            focus_plot_results = []
            logging.info(
                "writing partition-aligned gaze focus plots under %s",
                args.focus_plot_output_parent,
            )
            for group in selected_annotation_groups:
                try:
                    result = materialize_annotation_focus_plot_group(
                        group=group,
                        output_parent=args.focus_plot_output_parent,
                        overwrite=args.overwrite,
                    )
                    result.setdefault("gaze_mapping", group.gaze_mapping)
                    focus_plot_results.append(result)
                    logging.info(
                        "wrote focus plot for %s/%s/%s",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                    )
                except Exception as exc:
                    logging.warning(
                        "failed focus plot annotation clip group %s/%s/%s: %s",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                        exc,
                    )
            write_annotation_clip_summary(
                args.focus_plot_output_parent,
                focus_plot_results,
                stem="annotation_focus_plot",
                pipeline="annotation_clip_focus_plot",
            )

        if args.debug_overlay_output_parent is not None:
            overlay_results = []
            logging.info(
                "writing per-frame gaze debug overlays under %s",
                args.debug_overlay_output_parent,
            )
            for group in selected_annotation_groups:
                try:
                    result = materialize_debug_overlay_group(
                        group=group,
                        output_parent=args.debug_overlay_output_parent,
                        focus_box_ratio=args.focus_box_ratio,
                        max_gaze_gap=args.max_gaze_gap,
                        overwrite=args.overwrite,
                    )
                    result.setdefault("gaze_mapping", group.gaze_mapping)
                    overlay_results.append(result)
                    if result.get("skipped"):
                        logging.info(
                            "skipped debug overlay for %s/%s/%s: %s",
                            group.dataset,
                            GROUP_FOLDER_NAMES[group.group_size],
                            group.group_name,
                            result.get("skip_reason"),
                        )
                        continue
                    logging.info(
                        "wrote debug overlay for %s/%s/%s with %s marked frames",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                        result.get("masked_frame_count", 0),
                    )
                except Exception as exc:
                    logging.warning(
                        "failed debug overlay annotation clip group %s/%s/%s: %s",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                        exc,
                    )
            write_annotation_clip_summary(
                args.debug_overlay_output_parent,
                overlay_results,
                stem="gaze_debug_overlay",
                pipeline="annotation_clip_gaze_debug_overlay",
            )

        if args.original_output_parent is not None:
            original_results = []
            logging.info(
                "copying unmanipulated annotation clips under %s",
                args.original_output_parent,
            )
            for group in selected_annotation_groups:
                try:
                    result = materialize_original_annotation_clip_group(
                        group=group,
                        output_parent=args.original_output_parent,
                        overwrite=args.overwrite,
                    )
                    result.setdefault("gaze_mapping", group.gaze_mapping)
                    original_results.append(result)
                    logging.info(
                        "copied %s original clips for %s/%s/%s",
                        result["clip_mp4_count"],
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                    )
                except Exception as exc:
                    logging.warning(
                        "failed original annotation clip group %s/%s/%s: %s",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                        exc,
                    )
            write_annotation_clip_summary(
                args.original_output_parent,
                original_results,
                stem="original_annotation_clip",
                pipeline="annotation_clip_original_copy",
            )

        if args.full_corruption_output_parent is not None:
            full_corruption_groups = selected_annotation_groups
            if args.full_corruption_localization_source != "video-json":
                full_corruption_groups = [
                    group
                    for group in build_annotation_clip_groups(
                        recording_parent_dir=args.recording_parent_dir,
                        annotation_dir=args.annotation_dir,
                        local_path_prefix=args.local_path_prefix,
                        media_url_prefix=args.media_url_prefix,
                        confidence_threshold=args.confidence,
                        duration_tolerance=args.duration_tolerance,
                        system_to_pupil_offset=args.system_to_pupil_offset,
                        video_json=args.video_json,
                        gaze_mapping=args.gaze_mapping,
                        response_selection=args.response_selection,
                        use_annotation_video_path=args.use_annotation_video_path,
                        clip_localization_source=args.full_corruption_localization_source,
                    )
                    if group.group_size in selected_group_sizes
                ]
            full_corruption_results = []
            logging.info(
                "building full-video gaze-corrupted clips under %s using %s localization",
                args.full_corruption_output_parent,
                args.full_corruption_localization_source,
            )
            for group in full_corruption_groups:
                try:
                    result = materialize_annotation_clip_group(
                        group=group,
                        output_parent=args.full_corruption_output_parent,
                        clip_length=args.clip_length,
                        effect=args.effect,
                        focus_box_ratio=args.focus_box_ratio,
                        max_gaze_gap=args.max_gaze_gap,
                        overwrite=args.overwrite,
                        mask_full_video=True,
                    )
                    result.setdefault("gaze_mapping", group.gaze_mapping)
                    full_corruption_results.append(result)
                    if result.get("skipped"):
                        logging.info(
                            "skipped full-video corrupted clips for %s/%s/%s: %s",
                            group.dataset,
                            GROUP_FOLDER_NAMES[group.group_size],
                            group.group_name,
                            result.get("skip_reason"),
                        )
                        continue
                    logging.info(
                        "wrote %s full-video corrupted clips for %s/%s/%s",
                        result["clip_mp4_count"],
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                    )
                except Exception as exc:
                    logging.warning(
                        "failed full-video corrupted annotation clip group %s/%s/%s: %s",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                        exc,
                    )
            write_annotation_clip_summary(
                args.full_corruption_output_parent,
                full_corruption_results,
                stem="gaze_corrupted_full_annotation_clip",
                pipeline="annotation_clip_gaze_corrupted_full",
            )

        if args.skip_final_segment_output:
            logging.info("skipping final-segment gaze-blocked output")
        else:
            results = []
            for group in selected_annotation_groups:
                try:
                    result = materialize_annotation_clip_group(
                        group=group,
                        output_parent=args.output_parent,
                        clip_length=args.clip_length,
                        effect=args.effect,
                        focus_box_ratio=args.focus_box_ratio,
                        max_gaze_gap=args.max_gaze_gap,
                        overwrite=args.overwrite,
                    )
                    result.setdefault("gaze_mapping", group.gaze_mapping)
                    results.append(result)
                    if result.get("skipped"):
                        logging.info(
                            "skipped manipulated clips for %s/%s/%s: %s",
                            group.dataset,
                            GROUP_FOLDER_NAMES[group.group_size],
                            group.group_name,
                            result.get("skip_reason"),
                        )
                        continue
                    logging.info(
                        "wrote %s clips for %s/%s/%s",
                        result["clip_mp4_count"],
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                    )
                except Exception as exc:
                    logging.warning(
                        "failed annotation clip group %s/%s/%s: %s",
                        group.dataset,
                        GROUP_FOLDER_NAMES[group.group_size],
                        group.group_name,
                        exc,
                    )
            write_annotation_clip_summary(args.output_parent, results)
            logging.info("wrote annotation-clip gaze-blocked data under %s", args.output_parent)
        return

    records, skipped_files = collect_video_records(args.video_folder, recursive=args.recursive)
    if args.dialogue_range is not None:
        lo = (args.dialogue_range - 1) * 100
        hi = args.dialogue_range * 100
        records = [record for record in records if lo <= record.dialogue_id < hi]
    if not records:
        raise ValueError("No matching videos found for gaze-blocked partitioning.")

    logging.info("building gaze index from annotations")
    exact_gaze_index, name_gaze_index = build_gaze_index(
        recording_parent_dir=args.recording_parent_dir,
        annotation_dir=args.annotation_dir,
        local_path_prefix=args.local_path_prefix,
        media_url_prefix=args.media_url_prefix,
        confidence_threshold=args.confidence,
        duration_tolerance=args.duration_tolerance,
        system_to_pupil_offset=args.system_to_pupil_offset,
        video_json=args.video_json,
        gaze_mapping=args.gaze_mapping,
        response_selection=args.response_selection,
        use_annotation_video_path=args.use_annotation_video_path,
    )

    if args.write_frame_focus_csv:
        write_frame_focus_csv(
            records=records,
            output_csv_path=args.output_parent / "frame_focus_points.csv",
            exact_gaze_index=exact_gaze_index,
            name_gaze_index=name_gaze_index,
            max_gaze_gap=args.max_gaze_gap,
        )

    output_context_parent = args.output_parent / "context"
    records_by_dialogue: dict[int, list] = defaultdict(list)
    for record in records:
        records_by_dialogue[record.dialogue_id].append(record)

    all_triplets = []
    skipped_windows = []
    materialized_groups = []
    failed_triplets = []

    for dialogue_id in sorted(records_by_dialogue):
        triplets, skipped = partition_dialogue(records_by_dialogue[dialogue_id], stride=args.stride)
        all_triplets.extend(triplets)
        skipped_windows.extend(skipped)
        for triplet in triplets:
            results, failure = materialize_gaze_triplet(
                triplet=triplet,
                output_parent=output_context_parent,
                clip_length=args.clip_length,
                selected_group_sizes=selected_group_sizes,
                exact_gaze_index=exact_gaze_index,
                name_gaze_index=name_gaze_index,
                effect=args.effect,
                focus_box_ratio=args.focus_box_ratio,
                max_gaze_gap=args.max_gaze_gap,
                cut=args.cut,
                overwrite=args.overwrite,
            )
            materialized_groups.extend(results)
            if failure is not None:
                failed_triplets.append(failure)
                logging.warning(
                    "failed triplet dia%s %s on %s: %s",
                    failure.dialogue_id,
                    failure.utterance_ids,
                    failure.failed_group_name,
                    failure.error,
                )

    base_summary = build_summary(
        video_folder=args.video_folder,
        output_parent=output_context_parent,
        records=records,
        skipped_files=skipped_files,
        triplets=all_triplets,
        skipped_windows=skipped_windows,
        materialized_groups=materialized_groups,
        failed_triplets=failed_triplets,
        clip_length=args.clip_length,
        mode="context",
        cut=args.cut,
        selected_group_sizes=selected_group_sizes,
        stride=args.stride,
    )
    output_summary = build_gaze_partition_summary(
        base_summary=base_summary,
        effect=args.effect,
        focus_box_ratio=args.focus_box_ratio,
        max_gaze_gap=args.max_gaze_gap,
    )
    write_summary_files(output_summary, output_context_parent)
    (args.output_parent / "gaze_blocked_partition_summary.json").write_text(
        json.dumps(output_summary, indent=2),
        encoding="utf-8",
    )
    logging.info("wrote gaze-blocked context data under %s", output_context_parent)


if __name__ == "__main__":
    main()
