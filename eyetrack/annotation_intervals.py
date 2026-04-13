from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Tuple


TIME_TOLERANCE_SECONDS = 0.5


@dataclass
class VideoTiming:
    video_number: int
    annotation_key: int
    video_start_time: str
    video_end_time: str
    video_length: float
    current_video_time: float | None
    time_annot: float | None


@dataclass
class AnnotatorTimings:
    annotator_number: int
    node_id: str
    global_unique_id: str | None
    timings: List[VideoTiming]


def get_indexed_value(container: Any, index: int) -> Any:
    if isinstance(container, list):
        return container[index]
    if isinstance(container, dict):
        return container[str(index)]
    raise TypeError(f"Expected list or dict, got {type(container).__name__}.")


def get_first_node_id(journey_entry: dict) -> str:
    nodes = journey_entry["nodes"]
    return str(get_indexed_value(nodes, 0))


def get_annotator_node_ids(payload: dict) -> List[Tuple[int, str, str | None]]:
    journeys = payload.get("journeys", payload.get("journey"))
    if journeys is None:
        raise KeyError("Annotation JSON must contain a journeys field.")

    annotators = []
    for journey_index in (0, 1):
        journey_entry = get_indexed_value(journeys, journey_index)
        annotators.append(
            (
                journey_index + 1,
                get_first_node_id(journey_entry),
                journey_entry.get("global_unique_id"),
            )
        )
    return annotators


def get_annotations(payload: dict, node_id: str) -> dict:
    node = payload["nodes"][node_id]
    responses = node["responses"]
    response = responses[0] if isinstance(responses, list) else responses["0"]
    annotations = response["annotations"]
    if not isinstance(annotations, dict):
        raise TypeError(
            f"nodes -> {node_id} -> responses -> 0 -> annotations must be a dict."
        )
    return annotations


def sorted_annotation_items(annotations: dict) -> List[Tuple[int, Any]]:
    items = [(int(key), value) for key, value in annotations.items()]
    keys = [key for key, _ in items]
    if keys != sorted(keys):
        logging.error("annotation keys are not in ascending order: %s", keys)
    return sorted(items, key=lambda item: item[0])


def iter_press_data(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        press_data = node.get("press_data")
        if isinstance(press_data, dict):
            yield press_data
        for value in node.values():
            yield from iter_press_data(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_press_data(value)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_latest_press_data(annotation_key: int, annotation: dict) -> dict:
    candidates = list(iter_press_data(annotation.get("data", {})))
    if not candidates:
        raise ValueError(f"annotation {annotation_key} has no press_data entry under data.")
    if len(candidates) > 1:
        logging.warning(
            "annotation %s has %s press_data entries; using the latest time_annot.",
            annotation_key,
            len(candidates),
        )
    return max(
        candidates,
        key=lambda item: (
            time_annot
            if (time_annot := parse_optional_float(item.get("time_annot"))) is not None
            else -1.0
        ),
    )


def parse_video_timings(
    annotations: dict,
    tolerance: float,
    annotator_number: int,
) -> List[VideoTiming]:
    timings = []
    for video_number, (annotation_key, annotation) in enumerate(sorted_annotation_items(annotations)):
        press_data = select_latest_press_data(annotation_key, annotation)
        start_raw = press_data["video_start_time"]
        end_raw = press_data["video_end_time"]
        video_length = (parse_timestamp(end_raw) - parse_timestamp(start_raw)).total_seconds()
        current_video_time = parse_optional_float(press_data.get("current_video_time"))
        time_annot = parse_optional_float(press_data.get("time_annot"))

        if current_video_time is None:
            logging.error(
                "annotator %s annotation %s is missing a valid current_video_time.",
                annotator_number,
                annotation_key,
            )
        elif abs(video_length - current_video_time) > tolerance:
            logging.error(
                "annotator %s annotation %s duration mismatch: "
                "end-start=%.3fs, current_video_time=%.3fs.",
                annotator_number,
                annotation_key,
                video_length,
                current_video_time,
            )

        timings.append(
            VideoTiming(
                video_number=video_number,
                annotation_key=annotation_key,
                video_start_time=start_raw,
                video_end_time=end_raw,
                video_length=video_length,
                current_video_time=current_video_time,
                time_annot=time_annot,
            )
        )
    return timings


def load_all_video_timings(json_path: Path, tolerance: float) -> List[AnnotatorTimings]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    all_timings = []
    for annotator_number, node_id, global_unique_id in get_annotator_node_ids(payload):
        annotations = get_annotations(payload, node_id)
        print(
            f"Annotator {annotator_number}: node {node_id}, "
            f"videos under annotations: {len(annotations)}"
        )
        all_timings.append(
            AnnotatorTimings(
                annotator_number=annotator_number,
                node_id=node_id,
                global_unique_id=global_unique_id,
                timings=parse_video_timings(annotations, tolerance, annotator_number),
            )
        )
    return all_timings


def print_timing_table(annotator_timings: AnnotatorTimings) -> None:
    timings = annotator_timings.timings
    print()
    print(
        f"Annotator {annotator_timings.annotator_number} "
        f"(node {annotator_timings.node_id}, {annotator_timings.global_unique_id})"
    )
    headers = ("video", "video_start_time", "video_end_time", "video_length")
    rows = [
        (
            str(timing.video_number),
            timing.video_start_time,
            timing.video_end_time,
            f"{timing.video_length:.3f}",
        )
        for timing in timings
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read video time intervals from an annotation JSON and print timing tables."
    )
    parser.add_argument("annotation_json", type=Path, help="Path to the annotation JSON file.")
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=TIME_TOLERANCE_SECONDS,
        help="Allowed seconds of difference between end-start and current_video_time.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    all_timings = load_all_video_timings(args.annotation_json, args.duration_tolerance)
    for annotator_timings in all_timings:
        print_timing_table(annotator_timings)


if __name__ == "__main__":
    main()
