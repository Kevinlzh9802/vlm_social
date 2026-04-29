from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ANNOTATION_FILE_RE = re.compile(
    r"^T(?P<task_number>\d+)(?:_b(?P<batch_number>\d+))?_(?P<task_instance_id>\d+)\.json$"
)
TASK_JSON_STEM_RE = re.compile(r"^(?P<prefix>.+?)(?:_b(?P<batch_number>\d+))?$")


@dataclass(frozen=True)
class AnnotationFile:
    task_number: int
    task_instance_id: int
    path: Path
    batch_number: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse human annotation JSON files, extract annotator 1/2 intention "
            "and response pairs, and optionally link them to task media lists."
        )
    )
    parser.add_argument(
        "annotation_dir",
        type=Path,
        help=(
            "Directory containing annotation JSON files named like T{x}_{y}.json "
            "or T{x}_b{batch}_{y}.json."
        ),
    )
    parser.add_argument(
        "--task-json",
        type=Path,
        default=None,
        help=(
            "Optional task JSON file or directory used to link media by task-instance "
            "index y. Batched annotations like T1_b2_7.json resolve to task1_b2.json."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to annotation_dir/human_annotations.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. Defaults to "
            "annotation_dir/human_annotations_linked.json when --task-json is set, "
            "otherwise annotation_dir/human_annotations.json."
        ),
    )
    parser.add_argument(
        "--task-number",
        type=int,
        default=None,
        help="Optional task number filter.",
    )
    return parser.parse_args()


def annotation_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def find_annotation_jsons(
    annotation_dir: Path,
    task_number: int | None,
) -> list[AnnotationFile]:
    annotation_files: list[AnnotationFile] = []
    for path in sorted(annotation_dir.iterdir()):
        if not path.is_file():
            continue
        match = ANNOTATION_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        parsed_task_number = int(match.group("task_number"))
        batch_number = match.group("batch_number")
        parsed_task_instance_id = int(match.group("task_instance_id"))
        if task_number is not None and parsed_task_number != task_number:
            continue
        annotation_files.append(
            AnnotationFile(
                task_number=parsed_task_number,
                task_instance_id=parsed_task_instance_id,
                path=path,
                batch_number=None if batch_number is None else int(batch_number),
            )
        )
    return annotation_files


def resolve_task_json_path(
    task_json_source: Path,
    task_number: int,
    batch_number: int | None,
) -> Path:
    source = task_json_source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Task JSON path does not exist: {source}")

    if source.is_dir():
        candidate_names = []
        if batch_number is not None:
            candidate_names.append(f"task{task_number}_b{batch_number}.json")
        candidate_names.append(f"task{task_number}.json")
        for candidate_name in candidate_names:
            candidate = source / candidate_name
            if candidate.is_file():
                return candidate.resolve()
        requested = f"task{task_number}_b{batch_number}.json" if batch_number is not None else f"task{task_number}.json"
        raise FileNotFoundError(
            f"Could not find {requested} under task JSON directory: {source}"
        )

    if not source.is_file():
        raise FileNotFoundError(f"Task JSON is not a file: {source}")
    if batch_number is None:
        return source

    stem_match = TASK_JSON_STEM_RE.fullmatch(source.stem)
    prefix = stem_match.group("prefix") if stem_match is not None else source.stem
    candidate = source.with_name(f"{prefix}_b{batch_number}{source.suffix}")
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find batch-aligned task JSON for batch b{batch_number}: {candidate}"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object, got {type(value).__name__}")
    return value


def get_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list, got {type(value).__name__}")
    return value


def sorted_items(mapping: dict[str, Any]) -> list[tuple[str, Any]]:
    return sorted(
        ((str(key), value) for key, value in mapping.items()),
        key=lambda item: annotation_sort_key(item[0]),
    )


def iter_indexed_values(values: Any, context: str) -> list[tuple[int, Any]]:
    if isinstance(values, list):
        return list(enumerate(values))
    if isinstance(values, dict):
        return [(int(key), value) for key, value in sorted_items(values)]
    raise ValueError(f"{context} must be a list or object, got {type(values).__name__}")


def parse_created_timestamp(value: Any) -> datetime:
    if value is None:
        return datetime.min
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None)
    except ValueError:
        return datetime.min


def response_created_value(response: dict[str, Any]) -> Any:
    created = response.get("created")
    if created is not None:
        return created
    annotations = response.get("annotations")
    if isinstance(annotations, dict):
        annotation_created = annotations.get("created")
        if annotation_created is not None:
            return annotation_created
    return None


def get_journey_for_annotator(
    payload: dict[str, Any],
    annotator_number: int,
) -> tuple[int, dict[str, Any]]:
    journeys = get_list(payload.get("journeys"), "journeys")
    journey_index = annotator_number - 1
    if journey_index < 0 or journey_index >= len(journeys):
        raise ValueError(
            f"annotator {annotator_number} maps to journeys[{journey_index}], "
            f"but only {len(journeys)} journey entries exist"
        )
    journey = get_dict(journeys[journey_index], f"journeys[{journey_index}]")
    return journey_index, journey


def get_node_id_for_journey(journey: dict[str, Any], journey_index: int) -> str:
    nodes = get_list(journey.get("nodes"), f"journeys[{journey_index}].nodes")
    if not nodes:
        raise ValueError(f"journeys[{journey_index}].nodes is empty")
    return str(nodes[0])


def select_latest_submitted_response(
    responses: Any,
    node_id: str,
) -> tuple[int, dict[str, Any]]:
    submitted_candidates: list[tuple[datetime, int, dict[str, Any]]] = []
    for response_index, response_value in iter_indexed_values(
        responses, f"nodes[{node_id}].responses"
    ):
        response = get_dict(
            response_value, f"nodes[{node_id}].responses[{response_index}]"
        )
        if coerce_bool(response.get("submitted")):
            submitted_candidates.append(
                (
                    parse_created_timestamp(response_created_value(response)),
                    response_index,
                    response,
                )
            )
    if not submitted_candidates:
        raise ValueError("no submitted response found")
    submitted_candidates.sort(key=lambda item: (item[0], item[1]))
    _, response_index, response = submitted_candidates[-1]
    return response_index, response


def last_ordered_value(value: Any, context: str) -> Any:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{context} is empty")
        return value[-1]
    if isinstance(value, dict):
        ordered = sorted_items(value)
        if not ordered:
            raise ValueError(f"{context} is empty")
        return ordered[-1][1]
    raise ValueError(f"{context} must be a list or object, got {type(value).__name__}")


def extract_press_data(annotation_value: dict[str, Any], annotation_key: str) -> dict[str, Any]:
    data = annotation_value.get("data")
    if data is None:
        raise ValueError(f"annotation {annotation_key} is missing data")
    outer_value = last_ordered_value(data, f"annotations[{annotation_key}].data")
    inner_value = last_ordered_value(
        outer_value, f"annotations[{annotation_key}].data[last]"
    )
    inner_entry = get_dict(
        inner_value, f"annotations[{annotation_key}].data[last][last]"
    )
    return get_dict(
        inner_entry.get("press_data"),
        f"annotations[{annotation_key}].data[last][last].press_data",
    )


def parse_annotation_file_for_annotator(
    payload: dict[str, Any],
    task_number: int,
    task_instance_id: int,
    annotator_number: int,
    path: Path,
) -> list[dict[str, Any]]:
    journey_index, journey = get_journey_for_annotator(payload, annotator_number)
    node_id = get_node_id_for_journey(journey, journey_index)

    nodes = get_dict(payload.get("nodes"), "nodes")
    node = get_dict(nodes.get(node_id), f"nodes[{node_id}]")
    responses = node.get("responses")
    response_index, response = select_latest_submitted_response(responses, node_id)
    annotations = get_dict(
        response.get("annotations"),
        f"nodes[{node_id}].responses[{response_index}].annotations",
    )

    rows: list[dict[str, Any]] = []
    for annotation_key, annotation_value in sorted_items(annotations):
        annotation = get_dict(annotation_value, f"annotations[{annotation_key}]")
        press_data = extract_press_data(annotation, annotation_key)
        rows.append(
            {
                "task_number": task_number,
                "task_instance_id": task_instance_id,
                "annotator_number": annotator_number,
                "journey_index": journey_index,
                "node_id": node_id,
                "response_index": response_index,
                "response_created": response_created_value(response),
                "response_submitted": response.get("submitted"),
                "annotation_key": annotation_key,
                "participant": annotation.get("participant"),
                "category": annotation.get("category"),
                "speaker_intention": press_data.get("speaker_intention"),
                "response": press_data.get("response"),
                "source_file": str(path.resolve()),
            }
        )
    return rows


def normalize_task_payload(task_payload: Any) -> list[Any]:
    if isinstance(task_payload, list):
        return task_payload
    if isinstance(task_payload, dict):
        if "task1" in task_payload:
            return get_list(task_payload["task1"], "task_json.task1")
        if "data" in task_payload:
            return get_list(task_payload["data"], "task_json.data")
    raise ValueError("task JSON must be a list, or an object with a list under 'task1' or 'data'")


def get_media_list(entry: dict[str, Any], *field_names: str) -> list[str]:
    for field_name in field_names:
        if field_name not in entry:
            continue
        values = get_list(entry[field_name], field_name)
        return [str(value) for value in values]
    raise ValueError(
        f"entry is missing any of the expected fields: {', '.join(field_names)}"
    )


def flatten_task_media_lists(
    task_items: list[Any],
    task_instance_id: int,
) -> tuple[list[str], list[str]]:
    if task_instance_id < 0 or task_instance_id >= len(task_items):
        raise ValueError(
            f"task instance index {task_instance_id} is out of range for task JSON "
            f"with {len(task_items)} top-level items"
        )
    instance_items = get_list(task_items[task_instance_id], f"task_json[{task_instance_id}]")

    video_paths: list[str] = []
    audio_paths: list[str] = []
    for index, item in enumerate(instance_items):
        entry = get_dict(item, f"task_json[{task_instance_id}][{index}]")
        video_paths.extend(get_media_list(entry, "video", "video_path"))
        audio_paths.extend(get_media_list(entry, "audio", "audio_path"))
    return video_paths, audio_paths


def link_rows_to_media(
    rows: list[dict[str, Any]],
    video_paths: list[str],
    audio_paths: list[str],
) -> list[dict[str, Any]]:
    if len(video_paths) != len(audio_paths):
        raise ValueError(
            f"video/audio length mismatch after flattening: "
            f"{len(video_paths)} videos vs {len(audio_paths)} audios"
        )
    if len(rows) != len(video_paths):
        raise ValueError(
            f"annotation/media length mismatch: {len(rows)} annotations vs "
            f"{len(video_paths)} flattened media entries"
        )

    linked_rows: list[dict[str, Any]] = []
    for row, video_path, audio_path in zip(rows, video_paths, audio_paths):
        linked_rows.append(
            {
                **row,
                "video_path": video_path,
                "audio_path": audio_path,
            }
        )
    return linked_rows


def build_grouped_json(
    grouped_records: list[dict[str, Any]],
    task_json_path: Path | None,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    total_records = 0
    for grouped in grouped_records:
        annotators = {
            str(annotator_number): [
                {
                    "speaker_intention": row["speaker_intention"],
                    "response": row["response"],
                    "video_path": row.get("video_path"),
                    "audio_path": row.get("audio_path"),
                }
                for row in grouped["annotator_rows"].get(annotator_number, [])
            ]
            for annotator_number in (1, 2)
            if grouped["annotator_rows"].get(annotator_number)
        }
        total_records += sum(len(entries) for entries in annotators.values())
        tasks.append(
            {
                "task_number": grouped["task_number"],
                "task_instance_id": grouped["task_instance_id"],
                "source_file": grouped["source_file"],
                "annotators": annotators,
            }
        )

    return {
        "task_json": None if task_json_path is None else str(task_json_path.resolve()),
        "task_count": len(tasks),
        "record_count": total_records,
        "tasks": tasks,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_number",
        "task_instance_id",
        "annotator_number",
        "journey_index",
        "node_id",
        "response_index",
        "response_created",
        "response_submitted",
        "annotation_key",
        "participant",
        "category",
        "speaker_intention",
        "response",
        "video_path",
        "audio_path",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotation_dir = args.annotation_dir.expanduser().resolve()
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    task_json_path: Path | None = None
    if args.task_json is not None:
        task_json_path = args.task_json.expanduser().resolve()
        if not task_json_path.exists():
            raise FileNotFoundError(f"Task JSON path does not exist: {task_json_path}")

    output_csv = args.output_csv or (annotation_dir / "human_annotations.csv")
    output_json = args.output_json or (
        annotation_dir / (
            "human_annotations_linked.json"
            if task_json_path is not None
            else "human_annotations.json"
        )
    )

    flat_rows: list[dict[str, Any]] = []
    grouped_records: list[dict[str, Any]] = []
    errors: list[str] = []

    task_items_by_path: dict[Path, list[Any]] = {}
    for annotation_file in find_annotation_jsons(annotation_dir, args.task_number):
        try:
            payload = get_dict(load_json(annotation_file.path), annotation_file.path.name)
            video_paths: list[str] | None = None
            audio_paths: list[str] | None = None
            if task_json_path is not None:
                resolved_task_json_path = resolve_task_json_path(
                    task_json_path,
                    task_number=annotation_file.task_number,
                    batch_number=annotation_file.batch_number,
                )
                current_task_items = task_items_by_path.get(resolved_task_json_path)
                if current_task_items is None:
                    current_task_items = normalize_task_payload(
                        load_json(resolved_task_json_path)
                    )
                    task_items_by_path[resolved_task_json_path] = current_task_items
                video_paths, audio_paths = flatten_task_media_lists(
                    current_task_items,
                    annotation_file.task_instance_id,
                )
        except Exception as exc:
            errors.append(f"{annotation_file.path.name}: {exc}")
            continue

        annotator_rows: dict[int, list[dict[str, Any]]] = {}
        for annotator_number in (1, 2):
            try:
                rows = parse_annotation_file_for_annotator(
                    payload=payload,
                    task_number=annotation_file.task_number,
                    task_instance_id=annotation_file.task_instance_id,
                    annotator_number=annotator_number,
                    path=annotation_file.path,
                )
                if task_json_path is not None:
                    rows = link_rows_to_media(rows, video_paths or [], audio_paths or [])
            except Exception as exc:
                errors.append(
                    f"{annotation_file.path.name} annotator {annotator_number}: {exc}"
                )
                continue

            annotator_rows[annotator_number] = rows
            flat_rows.extend(rows)

        if annotator_rows:
            grouped_records.append(
                {
                    "task_number": annotation_file.task_number,
                    "task_instance_id": annotation_file.task_instance_id,
                    "source_file": str(annotation_file.path.resolve()),
                    "annotator_rows": annotator_rows,
                }
            )

    write_csv(output_csv, flat_rows)
    write_json(output_json, build_grouped_json(grouped_records, task_json_path))

    if errors:
        for error in errors:
            print(f"[WARN] {error}")
    print(f"[INFO] Wrote {len(flat_rows)} records to {output_csv}")
    print(f"[INFO] Wrote grouped records to {output_json}")


if __name__ == "__main__":
    main()
