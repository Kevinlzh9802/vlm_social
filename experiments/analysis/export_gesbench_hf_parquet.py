from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis.parse_human_annotations import (  # noqa: E402
    extract_press_data,
    find_annotation_jsons,
    flatten_task_media_lists,
    get_dict,
    get_journey_for_annotator,
    get_node_id_for_journey,
    link_rows_to_media,
    load_json,
    normalize_task_payload,
    parse_annotation_file_for_annotator,
    resolve_task_json_path,
    select_latest_submitted_response,
    sorted_items,
)
from eyetrack.annotation_intervals import (  # noqa: E402
    TIME_TOLERANCE_SECONDS,
    get_annotations,
    get_annotator_node_ids,
    parse_video_timings,
    select_latest_press_data,
    sorted_annotation_items,
)
from eyetrack.gaze_extraction import (  # noqa: E402
    find_latest_surface_gaze_csv,
    load_recording_time_offset,
)


TASK2_ANNOTATION_FILE_RE = re.compile(r"^T(?P<task_number>\d+)_(?P<task_instance_id>\d+)\.json$")
PUPIL_RECORDING_DIR_RE = re.compile(
    r"^T(?P<task_number>\d+)_(?P<task_instance_id>\d+)_annotator(?P<annotator_number>\d+)$"
)
CLIP_FILE_RE = re.compile(r"^(?P<prefix>.+)_clip_?(?P<index>\d+)\.[^.]+$")
BATCH_FOLDER_RE = re.compile(r"^(?P<dataset>.+)_u(?P<size>[123])b(?P<batch>\d+)$")


COLUMNS: tuple[str, ...] = (
    "record_id",
    "source_task",
    "task_number",
    "task_instance_id",
    "batch_number",
    "annotation_file",
    "task_json_file",
    "source_file_relpath",
    "annotator_number",
    "journey_index",
    "node_id",
    "global_unique_id",
    "response_index",
    "response_created",
    "response_submitted",
    "annotation_key",
    "video_number",
    "participant",
    "category",
    "annotation_type",
    "speaker_intention",
    "response",
    "video_path",
    "audio_path",
    "media_annotation_set",
    "utt_count",
    "dataset_name",
    "batch_name",
    "group_name",
    "clip_prefix",
    "clip_index",
    "group_identifier",
    "sequence_id",
    "clip_count",
    "clip_position",
    "progress_ratio",
    "is_final_clip",
    "video_start_time",
    "video_end_time",
    "video_length_seconds",
    "current_video_time",
    "time_annot",
    "has_valid_timing",
    "pupil_recording_dir",
    "pupil_info_path",
    "gaze_csv_path",
    "has_pupil_recording",
    "recording_start_time_system_s",
    "recording_start_time_synced_s",
    "system_to_pupil_offset_seconds",
)

INT_COLUMNS = {
    "task_number",
    "task_instance_id",
    "batch_number",
    "annotator_number",
    "journey_index",
    "response_index",
    "video_number",
    "annotation_key",
    "utt_count",
    "clip_index",
    "clip_count",
    "clip_position",
}
FLOAT_COLUMNS = {
    "progress_ratio",
    "video_length_seconds",
    "current_video_time",
    "time_annot",
    "recording_start_time_system_s",
    "recording_start_time_synced_s",
    "system_to_pupil_offset_seconds",
}
BOOL_COLUMNS = {
    "response_submitted",
    "is_final_clip",
    "has_valid_timing",
    "has_pupil_recording",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the GesBench human-evaluation annotation and gaze provenance "
            "files as one flat Hugging Face-friendly parquet table."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("GesBench_data"),
        help="Root containing task1, task2, and pupil folders. Default: GesBench_data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Parquet output path. Default: "
            "<data-root>/gesbench_human_eval-train.parquet."
        ),
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Optional row-count assertion for reproducibility checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate rows without writing parquet or importing pyarrow.",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=TIME_TOLERANCE_SECONDS,
        help="Allowed seconds between annotation duration and current_video_time.",
    )
    return parser.parse_args()


def relpath(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def clean_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def normalize_value(column: str, value: Any) -> Any:
    if column in INT_COLUMNS:
        return clean_int(value)
    if column in FLOAT_COLUMNS:
        return clean_float(value)
    if column in BOOL_COLUMNS:
        return clean_bool(value)
    return clean_text(value)


def empty_row() -> dict[str, Any]:
    return {column: None for column in COLUMNS}


def parse_media_layout(media_path: str | None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "media_annotation_set": None,
        "utt_count": None,
        "dataset_name": None,
        "batch_name": None,
        "group_name": None,
        "clip_prefix": None,
        "clip_index": None,
        "group_identifier": None,
    }
    if not media_path:
        return fields

    parsed = urlparse(media_path)
    path_text = parsed.path if parsed.scheme else media_path
    parts = PurePosixPath(path_text).parts
    if not parts:
        return fields

    filename = parts[-1]
    clip_match = CLIP_FILE_RE.fullmatch(filename)
    if clip_match is not None:
        fields["clip_prefix"] = clip_match.group("prefix")
        fields["clip_index"] = int(clip_match.group("index"))

    for index, part in enumerate(parts):
        if part == "gestalt_bench" and index + 1 < len(parts):
            fields["media_annotation_set"] = parts[index + 1]
        if not re.fullmatch(r"u[123]", part):
            continue
        if index + 1 >= len(parts):
            continue
        batch_match = BATCH_FOLDER_RE.fullmatch(parts[index + 1])
        if batch_match is None:
            continue
        fields["utt_count"] = int(part[1:])
        fields["dataset_name"] = batch_match.group("dataset")
        fields["batch_name"] = parts[index + 1]
        if index + 2 < len(parts) - 1:
            fields["group_name"] = parts[index + 2]
        if fields["clip_prefix"] is not None:
            fields["group_identifier"] = "/".join(parts[index + 2 : -1] + (fields["clip_prefix"],))
        return fields

    return fields


def sequence_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_task"),
        row.get("dataset_name"),
        row.get("utt_count"),
        row.get("annotator_number"),
        row.get("source_file_relpath"),
        row.get("group_identifier"),
    )


def finalize_sequences(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        clip_index = clean_int(row.get("clip_index"))
        if clip_index is None:
            continue
        key = sequence_key(row)
        if any(item is None for item in key):
            continue
        grouped[key].add(clip_index)

    clip_positions: dict[tuple[Any, ...], dict[int, int]] = {}
    for key, clip_indices in grouped.items():
        ordered = sorted(clip_indices)
        clip_positions[key] = {clip_index: pos for pos, clip_index in enumerate(ordered, start=1)}

    for row in rows:
        key = sequence_key(row)
        clip_index = clean_int(row.get("clip_index"))
        positions = clip_positions.get(key)
        if clip_index is None or not positions or clip_index not in positions:
            continue
        clip_count = len(positions)
        clip_position = positions[clip_index]
        row["sequence_id"] = "|".join(str(item) for item in key)
        row["clip_count"] = clip_count
        row["clip_position"] = clip_position
        row["progress_ratio"] = clip_position / clip_count
        row["is_final_clip"] = clip_position == clip_count


def task1_press_data_by_annotator(
    payload: dict[str, Any],
    annotator_number: int,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    journey_index, journey = get_journey_for_annotator(payload, annotator_number)
    node_id = get_node_id_for_journey(journey, journey_index)
    nodes = get_dict(payload.get("nodes"), "nodes")
    node = get_dict(nodes.get(node_id), f"nodes[{node_id}]")
    response_index, response = select_latest_submitted_response(node.get("responses"), node_id)
    annotations = get_dict(
        response.get("annotations"),
        f"nodes[{node_id}].responses[{response_index}].annotations",
    )
    press_by_key: dict[str, dict[str, Any]] = {}
    for annotation_key, annotation_value in sorted_items(annotations):
        annotation = get_dict(annotation_value, f"annotations[{annotation_key}]")
        press_by_key[annotation_key] = extract_press_data(annotation, annotation_key)
    return clean_text(journey.get("global_unique_id") or payload.get("global_unique_id")), press_by_key


def valid_timing(start: Any, end: Any) -> bool:
    return isinstance(start, str) and bool(start) and isinstance(end, str) and bool(end)


def build_task1_rows(data_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    task1_root = data_root / "task1"
    annotation_dir = task1_root / "annotations"
    task_json_source = task1_root
    task_items_by_path: dict[Path, list[Any]] = {}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not annotation_dir.is_dir():
        warnings.append(f"Task 1 annotation directory does not exist: {annotation_dir}")
        return rows, warnings

    for annotation_file in find_annotation_jsons(annotation_dir, task_number=1):
        try:
            payload = get_dict(load_json(annotation_file.path), annotation_file.path.name)
            task_json_path = resolve_task_json_path(
                task_json_source,
                task_number=annotation_file.task_number,
                batch_number=annotation_file.batch_number,
            )
            task_items = task_items_by_path.get(task_json_path)
            if task_items is None:
                task_items = normalize_task_payload(load_json(task_json_path))
                task_items_by_path[task_json_path] = task_items
            video_paths, audio_paths = flatten_task_media_lists(
                task_items,
                annotation_file.task_instance_id,
            )
        except Exception as exc:
            warnings.append(f"{annotation_file.path.name}: {exc}")
            continue

        for annotator_number in (1, 2):
            try:
                annotator_rows = parse_annotation_file_for_annotator(
                    payload=payload,
                    task_number=annotation_file.task_number,
                    task_instance_id=annotation_file.task_instance_id,
                    annotator_number=annotator_number,
                    path=annotation_file.path,
                )
                annotator_rows = link_rows_to_media(annotator_rows, video_paths, audio_paths)
                global_unique_id, press_by_key = task1_press_data_by_annotator(
                    payload,
                    annotator_number,
                )
            except Exception as exc:
                warnings.append(f"{annotation_file.path.name} annotator {annotator_number}: {exc}")
                continue

            for video_number, parsed_row in enumerate(annotator_rows):
                annotation_key = clean_text(parsed_row.get("annotation_key"))
                press_data = press_by_key.get(annotation_key or "", {})
                media_fields = parse_media_layout(parsed_row.get("video_path"))
                row = empty_row()
                row.update(
                    {
                        "source_task": "task1",
                        "task_number": annotation_file.task_number,
                        "task_instance_id": annotation_file.task_instance_id,
                        "batch_number": annotation_file.batch_number,
                        "annotation_file": annotation_file.path.name,
                        "task_json_file": relpath(task_json_path, data_root),
                        "source_file_relpath": relpath(annotation_file.path, data_root),
                        "annotator_number": parsed_row.get("annotator_number"),
                        "journey_index": parsed_row.get("journey_index"),
                        "node_id": parsed_row.get("node_id"),
                        "global_unique_id": global_unique_id,
                        "response_index": parsed_row.get("response_index"),
                        "response_created": parsed_row.get("response_created"),
                        "response_submitted": parsed_row.get("response_submitted"),
                        "annotation_key": parsed_row.get("annotation_key"),
                        "video_number": video_number,
                        "participant": parsed_row.get("participant"),
                        "category": parsed_row.get("category"),
                        "annotation_type": press_data.get("annotation_type"),
                        "speaker_intention": parsed_row.get("speaker_intention"),
                        "response": parsed_row.get("response"),
                        "video_path": parsed_row.get("video_path"),
                        "audio_path": parsed_row.get("audio_path"),
                        "video_start_time": press_data.get("video_start_time"),
                        "video_end_time": press_data.get("video_end_time"),
                        "has_valid_timing": valid_timing(
                            press_data.get("video_start_time"),
                            press_data.get("video_end_time"),
                        ),
                        "has_pupil_recording": False,
                    }
                )
                row.update(media_fields)
                rows.append(row)

    return rows, warnings


def read_recording_info(info_path: Path) -> dict[str, Any]:
    if not info_path.is_file():
        return {}
    with info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_pupil_recordings(data_root: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    recordings: dict[tuple[int, int, int], dict[str, Any]] = {}
    pupil_root = data_root / "pupil"
    if not pupil_root.is_dir():
        return recordings

    for path in sorted(pupil_root.iterdir()):
        if not path.is_dir():
            continue
        match = PUPIL_RECORDING_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (
            int(match.group("task_number")),
            int(match.group("task_instance_id")),
            int(match.group("annotator_number")),
        )
        info_path = path / "info.player.json"
        info = read_recording_info(info_path)
        gaze_csv_path: Path | None = None
        system_to_pupil_offset: float | None = None
        try:
            gaze_csv_path = find_latest_surface_gaze_csv(path)
        except Exception as exc:
            logging.warning("Could not find surface gaze CSV for %s: %s", path, exc)
        try:
            system_to_pupil_offset = load_recording_time_offset(path)
        except Exception as exc:
            logging.warning("Could not read recording time offset for %s: %s", path, exc)

        recordings[key] = {
            "recording_dir": path,
            "info_path": info_path if info_path.exists() else None,
            "gaze_csv_path": gaze_csv_path,
            "recording_start_time_system_s": info.get("start_time_system_s"),
            "recording_start_time_synced_s": info.get("start_time_synced_s"),
            "system_to_pupil_offset_seconds": system_to_pupil_offset,
        }

    return recordings


def build_task2_rows(data_root: Path, duration_tolerance: float) -> tuple[list[dict[str, Any]], list[str]]:
    annotation_dir = data_root / "task2" / "results"
    task_json_path = data_root / "task2" / "task2.json"
    recordings = discover_pupil_recordings(data_root)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not annotation_dir.is_dir():
        warnings.append(f"Task 2 annotation directory does not exist: {annotation_dir}")
        return rows, warnings

    for annotation_path in sorted(annotation_dir.glob("T*.json")):
        match = TASK2_ANNOTATION_FILE_RE.fullmatch(annotation_path.name)
        if match is None:
            warnings.append(f"Skipping nonstandard Task 2 annotation file name: {annotation_path.name}")
            continue
        task_number = int(match.group("task_number"))
        task_instance_id = int(match.group("task_instance_id"))

        try:
            payload = get_dict(load_json(annotation_path), annotation_path.name)
            annotator_nodes = get_annotator_node_ids(payload)
        except Exception as exc:
            warnings.append(f"{annotation_path.name}: {exc}")
            continue

        for annotator_number, node_id, global_unique_id in annotator_nodes:
            try:
                selected = get_annotations(payload, node_id, response_selection="latest-submitted")
            except Exception as exc:
                warnings.append(f"{annotation_path.name} annotator {annotator_number}: {exc}")
                continue
            if selected is None:
                warnings.append(
                    f"{annotation_path.name} annotator {annotator_number}: no submitted response found"
                )
                continue

            annotations, selected_response = selected
            timing_rows = parse_video_timings(
                annotations,
                tolerance=duration_tolerance,
                annotator_number=annotator_number,
                node_id=node_id,
            )
            annotation_items = sorted_annotation_items(annotations)
            recording = recordings.get((task_number, task_instance_id, annotator_number))

            for timing, (annotation_key, annotation_value) in zip(timing_rows, annotation_items):
                annotation = get_dict(annotation_value, f"annotations[{annotation_key}]")
                press_data = select_latest_press_data(annotation_key, annotation) or {}
                video_path = timing.video_path or press_data.get("video_path")
                audio_path = timing.audio_path or press_data.get("audio_path")
                media_fields = parse_media_layout(video_path)
                row = empty_row()
                row.update(
                    {
                        "source_task": "task2",
                        "task_number": task_number,
                        "task_instance_id": task_instance_id,
                        "annotation_file": annotation_path.name,
                        "task_json_file": relpath(task_json_path, data_root)
                        if task_json_path.exists()
                        else None,
                        "source_file_relpath": relpath(annotation_path, data_root),
                        "annotator_number": annotator_number,
                        "node_id": node_id,
                        "global_unique_id": global_unique_id or payload.get("global_unique_id"),
                        "response_index": selected_response.response_index,
                        "response_created": selected_response.created,
                        "response_submitted": selected_response.submitted,
                        "annotation_key": timing.annotation_key,
                        "video_number": timing.video_number,
                        "participant": annotation.get("participant"),
                        "category": annotation.get("category"),
                        "annotation_type": press_data.get("annotation_type"),
                        "speaker_intention": press_data.get("speaker_intention"),
                        "response": press_data.get("response"),
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "video_start_time": timing.video_start_time,
                        "video_end_time": timing.video_end_time,
                        "video_length_seconds": timing.video_length,
                        "current_video_time": timing.current_video_time,
                        "time_annot": timing.time_annot,
                        "has_valid_timing": valid_timing(
                            timing.video_start_time,
                            timing.video_end_time,
                        ),
                        "has_pupil_recording": recording is not None,
                    }
                )
                if recording is not None:
                    row.update(
                        {
                            "pupil_recording_dir": relpath(recording.get("recording_dir"), data_root),
                            "pupil_info_path": relpath(recording.get("info_path"), data_root),
                            "gaze_csv_path": relpath(recording.get("gaze_csv_path"), data_root),
                            "recording_start_time_system_s": recording.get(
                                "recording_start_time_system_s"
                            ),
                            "recording_start_time_synced_s": recording.get(
                                "recording_start_time_synced_s"
                            ),
                            "system_to_pupil_offset_seconds": recording.get(
                                "system_to_pupil_offset_seconds"
                            ),
                        }
                    )
                row.update(media_fields)
                rows.append(row)

    return rows, warnings


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row = {column: normalize_value(column, row.get(column)) for column in COLUMNS}
        row["record_id"] = f"{row['source_task']}-{index:06d}"
        normalized.append(row)
    return normalized


def build_rows(data_root: Path, duration_tolerance: float) -> tuple[list[dict[str, Any]], list[str]]:
    task1_rows, task1_warnings = build_task1_rows(data_root)
    task2_rows, task2_warnings = build_task2_rows(data_root, duration_tolerance)
    rows = [*task1_rows, *task2_rows]
    finalize_sequences(rows)
    return normalize_rows(rows), [*task1_warnings, *task2_warnings]


def pyarrow_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to write parquet. Install it with "
            "`python -m pip install pyarrow`, then rerun this exporter."
        ) from exc

    fields = []
    for column in COLUMNS:
        if column in INT_COLUMNS:
            field_type = pa.int64()
        elif column in FLOAT_COLUMNS:
            field_type = pa.float64()
        elif column in BOOL_COLUMNS:
            field_type = pa.bool_()
        else:
            field_type = pa.string()
        fields.append(pa.field(column, field_type, nullable=True))
    return pa.schema(fields)


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to write parquet. Install it with "
            "`python -m pip install pyarrow`, then rerun this exporter."
        ) from exc

    schema = pyarrow_schema()
    arrays = [
        pa.array([row.get(column) for row in rows], type=schema.field(column).type)
        for column in COLUMNS
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    nested_fields = [
        field.name
        for field in table.schema
        if pa.types.is_list(field.type)
        or pa.types.is_large_list(field.type)
        or pa.types.is_struct(field.type)
        or pa.types.is_map(field.type)
    ]
    if nested_fields:
        raise ValueError(f"Nested parquet fields are not allowed: {nested_fields}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")


def validate_written_parquet(output_path: Path, expected_rows: int | None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(output_path)
    missing_columns = [column for column in COLUMNS if column not in table.column_names]
    if missing_columns:
        raise ValueError(f"Parquet is missing columns: {missing_columns}")
    nested_fields = [
        field.name
        for field in table.schema
        if pa.types.is_list(field.type)
        or pa.types.is_large_list(field.type)
        or pa.types.is_struct(field.type)
        or pa.types.is_map(field.type)
    ]
    if nested_fields:
        raise ValueError(f"Parquet contains nested fields: {nested_fields}")
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {table.num_rows}")


def print_summary(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    task_counts: dict[str, int] = defaultdict(int)
    valid_timing_count = 0
    pupil_count = 0
    for row in rows:
        task_counts[str(row.get("source_task"))] += 1
        if row.get("has_valid_timing") is True:
            valid_timing_count += 1
        if row.get("has_pupil_recording") is True:
            pupil_count += 1

    print(f"[INFO] Built {len(rows)} rows")
    for task_name in sorted(task_counts):
        print(f"[INFO]   {task_name}: {task_counts[task_name]}")
    print(f"[INFO]   valid timing rows: {valid_timing_count}")
    print(f"[INFO]   rows linked to Pupil recordings: {pupil_count}")
    for warning in warnings:
        print(f"[WARN] {warning}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    output_path = args.output
    if output_path is None:
        output_path = data_root / "gesbench_human_eval-train.parquet"
    else:
        output_path = output_path.expanduser().resolve()

    rows, warnings = build_rows(data_root, duration_tolerance=args.duration_tolerance)
    print_summary(rows, warnings)

    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, built {len(rows)}")

    if args.dry_run:
        print("[INFO] Dry run complete; parquet was not written.")
        return

    write_parquet(rows, output_path)
    validate_written_parquet(output_path, args.expected_rows)
    print(f"[INFO] Wrote parquet: {output_path}")


if __name__ == "__main__":
    main()
