from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import urlparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .parse_human_annotations import (
        build_grouped_json,
        find_annotation_jsons,
        flatten_task_media_lists,
        get_dict,
        link_rows_to_media,
        load_json,
        normalize_task_payload,
        parse_annotation_file_for_annotator,
        write_csv as write_extracted_csv,
        write_json as write_extracted_json,
    )
except ImportError:
    from parse_human_annotations import (
        build_grouped_json,
        find_annotation_jsons,
        flatten_task_media_lists,
        get_dict,
        link_rows_to_media,
        load_json,
        normalize_task_payload,
        parse_annotation_file_for_annotator,
        write_csv as write_extracted_csv,
        write_json as write_extracted_json,
    )


DEFAULT_PROGRESS_PARTITIONS = 20
PROMPT_FIELD_MAP = {
    "intention": "speaker_intention",
    "affordance": "response",
}
CLIP_FILE_RE = re.compile(r"^(?P<prefix>.+)_clip_?(?P<index>\d+)\.[^.]+$")
BATCH_FOLDER_RE = re.compile(r"^(?P<dataset>.+)_u(?P<size>[123])b\d+$")


@dataclass(frozen=True)
class ClipMetadata:
    dataset_name: str
    utt_count: int
    clip_prefix: str
    clip_index: int
    group_identifier: str


@dataclass(frozen=True)
class UtteranceMetrics:
    clip_count: int
    clip_to_final_similarities: list[float]
    neighboring_similarities: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze human annotations by comparing partial clip text to full "
            "clip text. The input can be a pre-extracted human_annotations.csv "
            "or a directory containing raw T{x}_{y}.json files."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to human_annotations.csv or a directory containing T{x}_{y}.json files.",
    )
    parser.add_argument(
        "--task-json",
        type=Path,
        default=None,
        help=(
            "Required when input_path is an annotation directory. Used to link "
            "task-instance media lists, e.g. task1.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for extracted CSV/JSON plus plots and summaries. "
            "Defaults to input_path.parent/human_annotation_similarity for CSV "
            "input, or input_path.parent/annotation_extracted for annotation dirs."
        ),
    )
    parser.add_argument(
        "--extraction-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for extracted human_annotations.csv and "
            "human_annotations_linked.json when input_path is an annotation dir. "
            "Defaults to --output-dir."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for plot image outputs. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--plot-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory for reusable plot data summaries and point-level "
            "similarity data. Defaults to --output-dir."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (used when --model-path is not set).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local SentenceTransformer model directory.",
    )
    parser.add_argument(
        "--progress-partitions",
        type=int,
        default=DEFAULT_PROGRESS_PARTITIONS,
        help="Number of progress bins. Default: 20.",
    )
    parser.add_argument(
        "--task-number",
        type=int,
        default=None,
        help="Optional task number filter when input_path is an annotation directory.",
    )
    return parser.parse_args()


def parse_clip_metadata(video_path: str) -> ClipMetadata:
    parsed = urlparse(video_path)
    path_text = parsed.path if parsed.scheme else video_path
    parts = PurePosixPath(path_text).parts
    if not parts:
        raise ValueError(f"empty video path: {video_path}")

    filename = parts[-1]
    clip_match = CLIP_FILE_RE.fullmatch(filename)
    if clip_match is None:
        raise ValueError(f"video path is not a clip file: {video_path}")

    clip_prefix = clip_match.group("prefix")
    clip_index = int(clip_match.group("index"))

    for index, part in enumerate(parts[:-1]):
        if not re.fullmatch(r"u[123]", part):
            continue
        if index + 1 >= len(parts):
            continue
        batch_match = BATCH_FOLDER_RE.fullmatch(parts[index + 1])
        if batch_match is None:
            continue

        utt_count = int(part[1:])
        batch_utt_count = int(batch_match.group("size"))
        if utt_count != batch_utt_count:
            raise ValueError(
                f"utterance-count mismatch in path {video_path}: {part} vs {parts[index + 1]}"
            )

        group_identifier = "/".join(parts[index + 2 : -1] + (clip_prefix,))
        return ClipMetadata(
            dataset_name=batch_match.group("dataset"),
            utt_count=utt_count,
            clip_prefix=clip_prefix,
            clip_index=clip_index,
            group_identifier=group_identifier,
        )

    raise ValueError(
        "could not infer dataset/utt structure from video path "
        f"(expected .../u1/meld_u1b1/.../..._clipN.*): {video_path}"
    )


def load_annotation_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_rows_from_annotation_dir(
    annotation_dir: Path,
    task_json_path: Path,
    task_number: int | None,
    output_dir: Path,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    task_items = normalize_task_payload(load_json(task_json_path))
    flat_rows: list[dict[str, Any]] = []
    grouped_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for parsed_task_number, task_instance_id, path in find_annotation_jsons(
        annotation_dir, task_number
    ):
        try:
            payload = get_dict(load_json(path), path.name)
            annotator_rows: dict[int, list[dict[str, Any]]] = {}
            video_paths, audio_paths = flatten_task_media_lists(task_items, task_instance_id)

            for annotator_number in (1, 2):
                rows = parse_annotation_file_for_annotator(
                    payload=payload,
                    task_number=parsed_task_number,
                    task_instance_id=task_instance_id,
                    annotator_number=annotator_number,
                    path=path,
                )
                rows = link_rows_to_media(rows, video_paths, audio_paths)
                annotator_rows[annotator_number] = rows
                flat_rows.extend(rows)

            grouped_records.append(
                {
                    "task_number": parsed_task_number,
                    "task_instance_id": task_instance_id,
                    "source_file": str(path.resolve()),
                    "annotator_rows": annotator_rows,
                }
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    extracted_csv_path = output_dir / "human_annotations.csv"
    extracted_json_path = output_dir / "human_annotations_linked.json"
    write_extracted_csv(extracted_csv_path, flat_rows)
    write_extracted_json(
        extracted_json_path,
        build_grouped_json(grouped_records, task_json_path),
    )
    return extracted_csv_path, flat_rows, errors


def build_grouped_clip_sequences(
    rows: Sequence[dict[str, Any]],
    text_field: str,
) -> tuple[dict[tuple[str, int], list[list[tuple[int, str]]]], list[str]]:
    grouped_rows: dict[tuple[str, int, str, str, str], dict[int, str]] = defaultdict(dict)
    case_by_group: dict[tuple[str, int, str, str, str], tuple[str, int]] = {}
    warnings: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        video_path = str(row.get("video_path") or "").strip()
        text_value = str(row.get(text_field) or "").strip()
        if not video_path:
            warnings.append(f"row {row_index}: missing video_path, skipping")
            continue
        if not text_value:
            warnings.append(f"row {row_index}: missing {text_field}, skipping")
            continue

        try:
            clip = parse_clip_metadata(video_path)
        except ValueError as exc:
            warnings.append(f"row {row_index}: {exc}")
            continue

        group_key = (
            clip.dataset_name,
            clip.utt_count,
            str(row.get("annotator_number", "")),
            str(row.get("source_file", "")),
            clip.group_identifier,
        )
        case_by_group[group_key] = (clip.dataset_name, clip.utt_count)
        grouped_rows[group_key][clip.clip_index] = text_value

    case_sequences: dict[tuple[str, int], list[list[tuple[int, str]]]] = defaultdict(list)
    for group_key, clip_texts in grouped_rows.items():
        ordered = sorted(clip_texts.items(), key=lambda item: item[0])
        if len(ordered) < 2:
            warnings.append(
                f"group {group_key[-1]} annotator={group_key[2]} has fewer than 2 clips, skipping"
            )
            continue
        case_sequences[case_by_group[group_key]].append(ordered)

    return dict(case_sequences), warnings


def quantize_progress_ratio(progress_ratio: float, partitions: int) -> float:
    bucket_index = round(progress_ratio * partitions)
    bucket_index = max(1, min(partitions, bucket_index))
    return bucket_index / partitions


def collect_clip_to_final_bins(
    utterance_metrics: Sequence[UtteranceMetrics],
    progress_partitions: int,
) -> dict[float, list[float]]:
    grouped_values: dict[float, list[float]] = defaultdict(list)
    for metrics in utterance_metrics:
        for position, similarity in enumerate(metrics.clip_to_final_similarities, start=1):
            progress_ratio = quantize_progress_ratio(
                position / metrics.clip_count,
                partitions=progress_partitions,
            )
            grouped_values[progress_ratio].append(similarity)
    return grouped_values


def mean_similarity_by_ratio(grouped_values: dict[float, list[float]]) -> tuple[list[float], list[float]]:
    ratios = sorted(grouped_values)
    means = [float(np.mean(grouped_values[ratio])) for ratio in ratios]
    return ratios, means


def plot_case_percentiles(
    dataset_name: str,
    utt_count: int,
    prompt_name: str,
    grouped_values: dict[float, list[float]],
    output_path: Path,
    progress_partitions: int,
) -> None:
    ratios = sorted(grouped_values)
    percentile_25 = [float(np.percentile(grouped_values[ratio], 25)) for ratio in ratios]
    percentile_50 = [float(np.percentile(grouped_values[ratio], 50)) for ratio in ratios]
    percentile_75 = [float(np.percentile(grouped_values[ratio], 75)) for ratio in ratios]

    plt.figure(figsize=(8, 6))
    plt.plot(ratios, percentile_25, color="#E45756", linewidth=1.8, label="25th percentile")
    plt.plot(ratios, percentile_50, color="#4C78A8", linewidth=1.8, label="50th percentile")
    plt.plot(ratios, percentile_75, color="#72B7B2", linewidth=1.8, label="75th percentile")
    plt.title(
        f"Human Partial-to-Full Similarity | {dataset_name} | {utt_count}-utt | {prompt_name}"
    )
    plt.xlabel(
        f"Observed clip ratio (rounded to nearest 1/{progress_partitions})"
    )
    plt.ylabel("Cosine similarity to full clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_case_average(
    dataset_name: str,
    utt_count: int,
    prompt_name: str,
    grouped_values: dict[float, list[float]],
    output_path: Path,
    progress_partitions: int,
) -> None:
    ratios, mean_values = mean_similarity_by_ratio(grouped_values)

    plt.figure(figsize=(8, 6))
    plt.plot(ratios, mean_values, color="#4C78A8", linewidth=1.8, marker="o", label="Mean")
    plt.title(
        f"Human Partial-to-Full Similarity | {dataset_name} | {utt_count}-utt | {prompt_name} | Mean"
    )
    plt.xlabel(
        f"Observed clip ratio (rounded to nearest 1/{progress_partitions})"
    )
    plt.ylabel("Average cosine similarity to full clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_overall_percentiles(
    prompt_name: str,
    grouped_values: dict[float, list[float]],
    output_path: Path,
    progress_partitions: int,
) -> None:
    ratios = sorted(grouped_values)
    percentile_25 = [float(np.percentile(grouped_values[ratio], 25)) for ratio in ratios]
    percentile_50 = [float(np.percentile(grouped_values[ratio], 50)) for ratio in ratios]
    percentile_75 = [float(np.percentile(grouped_values[ratio], 75)) for ratio in ratios]

    plt.figure(figsize=(8, 6))
    plt.plot(ratios, percentile_25, color="#E45756", linewidth=1.8, label="25th percentile")
    plt.plot(ratios, percentile_50, color="#4C78A8", linewidth=1.8, label="50th percentile")
    plt.plot(ratios, percentile_75, color="#72B7B2", linewidth=1.8, label="75th percentile")
    plt.title(f"Human Partial-to-Full Similarity | All Datasets | {prompt_name}")
    plt.xlabel(
        f"Observed clip ratio (rounded to nearest 1/{progress_partitions})"
    )
    plt.ylabel("Cosine similarity to full clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_overall_average(
    prompt_name: str,
    grouped_values: dict[float, list[float]],
    output_path: Path,
    progress_partitions: int,
) -> None:
    ratios, mean_values = mean_similarity_by_ratio(grouped_values)

    plt.figure(figsize=(8, 6))
    plt.plot(ratios, mean_values, color="#4C78A8", linewidth=1.8, marker="o", label="Mean")
    plt.title(f"Human Partial-to-Full Similarity | All Datasets | {prompt_name} | Mean")
    plt.xlabel(
        f"Observed clip ratio (rounded to nearest 1/{progress_partitions})"
    )
    plt.ylabel("Average cosine similarity to full clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_case_bins(
    dataset_name: str,
    utt_count: int,
    prompt_name: str,
    utterance_metrics: Sequence[UtteranceMetrics],
    progress_partitions: int,
) -> list[dict[str, object]]:
    grouped_values = collect_clip_to_final_bins(
        utterance_metrics=utterance_metrics,
        progress_partitions=progress_partitions,
    )
    rows: list[dict[str, object]] = []
    for ratio in sorted(grouped_values):
        bin_index = int(round(ratio * progress_partitions))
        values = grouped_values[ratio]
        rows.append(
            {
                "prompt": prompt_name,
                "dataset": dataset_name,
                "utt_count": utt_count,
                "bin_index": bin_index,
                "progress_ratio": ratio,
                "sample_count": len(values),
                "mean_similarity": float(np.mean(values)),
                "percentile_25": float(np.percentile(values, 25)),
                "percentile_50": float(np.percentile(values, 50)),
                "percentile_75": float(np.percentile(values, 75)),
            }
        )
    return rows


def summarize_overall_bins(
    prompt_name: str,
    utterance_metrics: Sequence[UtteranceMetrics],
    progress_partitions: int,
) -> list[dict[str, object]]:
    grouped_values = collect_clip_to_final_bins(
        utterance_metrics=utterance_metrics,
        progress_partitions=progress_partitions,
    )
    rows: list[dict[str, object]] = []
    for ratio in sorted(grouped_values):
        bin_index = int(round(ratio * progress_partitions))
        values = grouped_values[ratio]
        rows.append(
            {
                "prompt": prompt_name,
                "dataset": "all",
                "utt_count": "all",
                "bin_index": bin_index,
                "progress_ratio": ratio,
                "sample_count": len(values),
                "mean_similarity": float(np.mean(values)),
                "percentile_25": float(np.percentile(values, 25)),
                "percentile_50": float(np.percentile(values, 50)),
                "percentile_75": float(np.percentile(values, 75)),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "prompt",
        "dataset",
        "utt_count",
        "bin_index",
        "progress_ratio",
        "sample_count",
        "mean_similarity",
        "percentile_25",
        "percentile_50",
        "percentile_75",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(
    path: Path,
    source_path: Path,
    prompt_case_metrics: dict[str, dict[tuple[str, int], list[UtteranceMetrics]]],
    bin_rows: Sequence[dict[str, object]],
    warnings: Sequence[str],
) -> None:
    payload = {
        "source": str(source_path.resolve()),
        "prompt_count": len(prompt_case_metrics),
        "prompts": [],
        "warnings": list(warnings),
    }

    for prompt_name in sorted(prompt_case_metrics):
        case_metrics = prompt_case_metrics[prompt_name]
        overall_bin_rows = [
            row
            for row in bin_rows
            if row["prompt"] == prompt_name
            and row["dataset"] == "all"
            and row["utt_count"] == "all"
        ]
        payload["prompts"].append(
            {
                "prompt": prompt_name,
                "case_count": len(case_metrics),
                "overall_bin_rows": overall_bin_rows,
                "cases": [
                    {
                        "dataset": dataset_name,
                        "utt_count": utt_count,
                        "utterance_count": len(metrics_list),
                        "bin_rows": [
                            row
                            for row in bin_rows
                            if row["prompt"] == prompt_name
                            and row["dataset"] == dataset_name
                            and row["utt_count"] == utt_count
                        ],
                    }
                    for (dataset_name, utt_count), metrics_list in sorted(case_metrics.items())
                ],
            }
        )

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_plot_point_rows(
    prompt_name: str,
    dataset_name: str,
    utt_count: int,
    utterance_metrics: Sequence[UtteranceMetrics],
    progress_partitions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for utterance_index, metrics in enumerate(utterance_metrics, start=1):
        for clip_position, similarity in enumerate(
            metrics.clip_to_final_similarities, start=1
        ):
            raw_progress_ratio = clip_position / metrics.clip_count
            rows.append(
                {
                    "prompt": prompt_name,
                    "dataset": dataset_name,
                    "utt_count": utt_count,
                    "utterance_index": utterance_index,
                    "clip_position": clip_position,
                    "clip_count": metrics.clip_count,
                    "progress_ratio_raw": raw_progress_ratio,
                    "progress_ratio_binned": quantize_progress_ratio(
                        raw_progress_ratio,
                        progress_partitions,
                    ),
                    "similarity_to_full": similarity,
                }
            )
    return rows


def build_overall_plot_point_rows(
    prompt_name: str,
    utterance_metrics: Sequence[UtteranceMetrics],
    progress_partitions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for utterance_index, metrics in enumerate(utterance_metrics, start=1):
        for clip_position, similarity in enumerate(
            metrics.clip_to_final_similarities, start=1
        ):
            raw_progress_ratio = clip_position / metrics.clip_count
            rows.append(
                {
                    "prompt": prompt_name,
                    "dataset": "all",
                    "utt_count": "all",
                    "utterance_index": utterance_index,
                    "clip_position": clip_position,
                    "clip_count": metrics.clip_count,
                    "progress_ratio_raw": raw_progress_ratio,
                    "progress_ratio_binned": quantize_progress_ratio(
                        raw_progress_ratio,
                        progress_partitions,
                    ),
                    "similarity_to_full": similarity,
                }
            )
    return rows


def write_plot_points_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "prompt",
        "dataset",
        "utt_count",
        "utterance_index",
        "clip_position",
        "clip_count",
        "progress_ratio_raw",
        "progress_ratio_binned",
        "similarity_to_full",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot_points_json(
    path: Path,
    source_path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    payload = {
        "source": str(source_path.resolve()),
        "point_count": len(rows),
        "points": list(rows),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir(input_path: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.expanduser().resolve()
    if input_path.is_dir():
        return (input_path.parent / "annotation_extracted").resolve()
    return (input_path.parent / "human_annotation_similarity").resolve()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_dir = resolve_output_dir(input_path, args.output_dir)
    extraction_output_dir = (
        args.extraction_output_dir.expanduser().resolve()
        if args.extraction_output_dir is not None
        else output_dir
    )
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else output_dir
    )
    plot_data_dir = (
        args.plot_data_dir.expanduser().resolve()
        if args.plot_data_dir is not None
        else output_dir
    )
    extraction_output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        if args.task_json is None:
            raise ValueError("--task-json is required when input_path is an annotation directory")
        task_json_path = args.task_json.expanduser().resolve()
        if not task_json_path.is_file():
            raise FileNotFoundError(f"Task JSON does not exist: {task_json_path}")
        csv_path, rows, extraction_errors = extract_rows_from_annotation_dir(
            annotation_dir=input_path,
            task_json_path=task_json_path,
            task_number=args.task_number,
            output_dir=extraction_output_dir,
        )
        analysis_source_path = input_path
    else:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        csv_path = input_path
        rows = load_annotation_rows(csv_path)
        extraction_errors = []
        analysis_source_path = csv_path

    from sentence_transformers import SentenceTransformer

    try:
        from .metrics import compute_utterance_metrics
    except ImportError:
        from metrics import compute_utterance_metrics

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    model = SentenceTransformer(model_loc)

    prompt_case_metrics: dict[str, dict[tuple[str, int], list[UtteranceMetrics]]] = {}
    all_bin_rows: list[dict[str, object]] = []
    all_point_rows: list[dict[str, object]] = []
    all_warnings: list[str] = list(extraction_errors)

    for prompt_name, text_field in PROMPT_FIELD_MAP.items():
        case_sequences, warnings = build_grouped_clip_sequences(
            rows=rows,
            text_field=text_field,
        )
        all_warnings.extend(warnings)

        case_metrics: dict[tuple[str, int], list[UtteranceMetrics]] = defaultdict(list)
        for case_key, sequences in sorted(case_sequences.items()):
            for ordered_clips in sequences:
                metrics = compute_utterance_metrics(model=model, ordered_clips=ordered_clips)
                if metrics is not None:
                    case_metrics[case_key].append(metrics)
        prompt_case_metrics[prompt_name] = dict(case_metrics)

        prompt_output_dir = plot_dir / prompt_name
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        for (dataset_name, utt_count), metrics_list in sorted(case_metrics.items()):
            if not metrics_list:
                print(
                    f"[WARN] No usable utterances for prompt={prompt_name} "
                    f"dataset={dataset_name} utt={utt_count}"
                )
                continue

            case_bin_rows = summarize_case_bins(
                dataset_name=dataset_name,
                utt_count=utt_count,
                prompt_name=prompt_name,
                utterance_metrics=metrics_list,
                progress_partitions=args.progress_partitions,
            )
            all_bin_rows.extend(case_bin_rows)
            all_point_rows.extend(
                build_plot_point_rows(
                    prompt_name=prompt_name,
                    dataset_name=dataset_name,
                    utt_count=utt_count,
                    utterance_metrics=metrics_list,
                    progress_partitions=args.progress_partitions,
                )
            )

            grouped_values = collect_clip_to_final_bins(
                utterance_metrics=metrics_list,
                progress_partitions=args.progress_partitions,
            )
            plot_path = (
                prompt_output_dir
                / f"{dataset_name}_{utt_count}utt_partial_to_full_percentiles.png"
            )
            plot_case_percentiles(
                dataset_name=dataset_name,
                utt_count=utt_count,
                prompt_name=prompt_name,
                grouped_values=grouped_values,
                output_path=plot_path,
                progress_partitions=args.progress_partitions,
            )
            print(f"[INFO] Saved {plot_path}")

            mean_plot_path = (
                prompt_output_dir
                / f"{dataset_name}_{utt_count}utt_partial_to_full_mean.png"
            )
            plot_case_average(
                dataset_name=dataset_name,
                utt_count=utt_count,
                prompt_name=prompt_name,
                grouped_values=grouped_values,
                output_path=mean_plot_path,
                progress_partitions=args.progress_partitions,
            )
            print(f"[INFO] Saved {mean_plot_path}")

        overall_metrics = [
            metric
            for metrics_list in case_metrics.values()
            for metric in metrics_list
        ]
        if overall_metrics:
            all_bin_rows.extend(
                summarize_overall_bins(
                    prompt_name=prompt_name,
                    utterance_metrics=overall_metrics,
                    progress_partitions=args.progress_partitions,
                )
            )
            all_point_rows.extend(
                build_overall_plot_point_rows(
                    prompt_name=prompt_name,
                    utterance_metrics=overall_metrics,
                    progress_partitions=args.progress_partitions,
                )
            )
            overall_grouped_values = collect_clip_to_final_bins(
                utterance_metrics=overall_metrics,
                progress_partitions=args.progress_partitions,
            )
            overall_plot_path = plot_dir / prompt_name / "all_datasets_partial_to_full_percentiles.png"
            plot_overall_percentiles(
                prompt_name=prompt_name,
                grouped_values=overall_grouped_values,
                output_path=overall_plot_path,
                progress_partitions=args.progress_partitions,
            )
            print(f"[INFO] Saved {overall_plot_path}")

            overall_mean_plot_path = plot_dir / prompt_name / "all_datasets_partial_to_full_mean.png"
            plot_overall_average(
                prompt_name=prompt_name,
                grouped_values=overall_grouped_values,
                output_path=overall_mean_plot_path,
                progress_partitions=args.progress_partitions,
            )
            print(f"[INFO] Saved {overall_mean_plot_path}")

    summary_csv_path = plot_data_dir / "partial_to_full_percentiles.csv"
    summary_json_path = plot_data_dir / "partial_to_full_percentiles.json"
    plot_points_csv_path = plot_data_dir / "partial_to_full_points.csv"
    plot_points_json_path = plot_data_dir / "partial_to_full_points.json"
    write_summary_csv(summary_csv_path, all_bin_rows)
    write_summary_json(
        summary_json_path,
        source_path=analysis_source_path,
        prompt_case_metrics=prompt_case_metrics,
        bin_rows=all_bin_rows,
        warnings=all_warnings,
    )
    write_plot_points_csv(plot_points_csv_path, all_point_rows)
    write_plot_points_json(plot_points_json_path, analysis_source_path, all_point_rows)
    print(f"[INFO] Saved extracted/loaded CSV source {csv_path}")
    print(f"[INFO] Saved {summary_csv_path}")
    print(f"[INFO] Saved {summary_json_path}")
    print(f"[INFO] Saved {plot_points_csv_path}")
    print(f"[INFO] Saved {plot_points_json_path}")

    for warning in all_warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
