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
DATASET_NAMES = ("mintrec2", "meld", "seamless_interaction")
TASK_NAMES = ("affordance", "intention")
PROMPT_FIELD_MAP = {
    "intention": "speaker_intention",
    "affordance": "response",
}
CLIP_FILE_RE = re.compile(r"^(?P<prefix>.+)_clip(?P<index>\d+)\.[^.]+$")
BATCH_FOLDER_RE = re.compile(r"^(?P<dataset>.+)_u(?P<size>[123])b\d+$")
DIALOGUE_KEY_PATTERN = re.compile(r"^d\d+u\d+$")
FILE_CLIP_PATTERN = re.compile(r"^(?P<prefix>d\d+u\d+)_clip(?P<index>\d+)$")
UTT_GROUP_PATTERN = re.compile(r"^(?P<size>[123])-utt_group$")
MODEL_SIZE_PATTERN = re.compile(r"(?P<size>\d+[Bb])(?:[_-]|$)")
ERROR_TEXT_PATTERN = re.compile(
    r"(cuda.*out of memory|outofmemoryerror|torch\.cuda\..*outofmemory|cuda out of memory)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClipMetadata:
    dataset_name: str
    utt_count: int
    clip_prefix: str
    clip_index: int
    group_identifier: str


@dataclass(frozen=True)
class HumanClipSequence:
    prompt_name: str
    dataset_name: str
    utt_count: int
    clip_prefix: str
    group_identifier: str
    annotator_number: str
    source_file: str
    ordered_clips: list[tuple[int, str]]


@dataclass(frozen=True)
class SequenceSimilarity:
    model_label: str
    prompt_name: str
    dataset_name: str
    utt_count: int
    clip_prefix: str
    annotator_number: str
    source_file: str
    group_identifier: str
    clip_count: int
    similarities: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare human annotation text against model answers clip by clip. "
            "The input can be human_annotations.csv or a directory containing "
            "raw T{x}_{y}.json files."
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
        "--results-root",
        type=Path,
        default=None,
        help="Root results folder containing model subfolders such as qwen2.5 or ming-lite-omni.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default parent for derived outputs. Defaults to "
            "input_path.parent/human_model_similarity for CSV input, or "
            "input_path.parent/human_model_similarity for annotation dirs."
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
            "Directory for reusable plot-data summaries and point-level "
            "human-vs-model similarity data. Defaults to --output-dir."
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


def parse_clip_metadata(media_path: str) -> ClipMetadata:
    parsed = urlparse(media_path)
    path_text = parsed.path if parsed.scheme else media_path
    parts = PurePosixPath(path_text).parts
    if not parts:
        raise ValueError(f"empty media path: {media_path}")

    filename = parts[-1]
    clip_match = CLIP_FILE_RE.fullmatch(filename)
    if clip_match is None:
        raise ValueError(f"media path is not a clip file: {media_path}")

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
        if int(batch_match.group("size")) != utt_count:
            raise ValueError(
                f"utterance-count mismatch in path {media_path}: {part} vs {parts[index + 1]}"
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
        "could not infer dataset/utt structure from media path "
        f"(expected .../u1/meld_u1b1/.../..._clipN.*): {media_path}"
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


def resolve_results_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Results root does not exist: {root}")
        return root

    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "results",
        repo_root.parent / "results",
        repo_root / "gestalt_bench" / "results",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find results automatically. Pass --results-root explicitly."
    )


def has_dataset_structure(root: Path) -> bool:
    return any((root / dataset_name).is_dir() for dataset_name in DATASET_NAMES)


def iter_model_roots(results_root: Path) -> list[Path]:
    if has_dataset_structure(results_root):
        return [results_root]

    model_roots: list[Path] = []
    for path in sorted(results_root.iterdir()):
        if path.name == "plots" or not path.is_dir():
            continue
        if has_dataset_structure(path):
            model_roots.append(path)
    return model_roots


def iter_result_folders(dataset_root: Path) -> list[Path]:
    result_folders: list[Path] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_dir() and list(path.glob("batch*.json")):
            result_folders.append(path)
    return result_folders


def find_dialogue_entries(payload: object) -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and DIALOGUE_KEY_PATTERN.fullmatch(key):
                    found.append((key, value))
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return found


def sort_key_for_clip_identifier(identifier: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", identifier)
    if match:
        return int(match.group(1)), identifier
    return 10**9, identifier


def extract_assistant_text(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    for field_name in ("assistant", "response", "response_text"):
        field_value = value.get(field_name)
        if isinstance(field_value, str) and field_value.strip():
            return field_value.strip()
    return None


def normalize_clip_dict_from_sequence(dialogue_key: str, value: Sequence[object]) -> dict[int, str]:
    clips: dict[int, str] = {}
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue

        text = extract_assistant_text(item)
        if text is None:
            continue

        clip_index = index
        file_name = item.get("file")
        if isinstance(file_name, str):
            match = FILE_CLIP_PATTERN.fullmatch(file_name)
            if match and match.group("prefix") == dialogue_key:
                clip_index = int(match.group("index"))

        clips[clip_index] = text

    return clips


def extract_ordered_clips(dialogue_key: str, dialogue_value: object) -> list[tuple[int, str]]:
    clip_dict: dict[int, str] = {}
    if isinstance(dialogue_value, Sequence) and not isinstance(dialogue_value, (str, bytes)):
        clip_dict = normalize_clip_dict_from_sequence(dialogue_key, dialogue_value)
    return sorted(clip_dict.items(), key=lambda item: item[0])


def parse_group_size(result_folder: Path) -> int | None:
    for part in result_folder.parts:
        match = UTT_GROUP_PATTERN.fullmatch(part)
        if match:
            return int(match.group("size"))
    return None


def parse_task_name(result_folder: Path) -> str | None:
    leaf_name = result_folder.name.lower()
    for task_name in TASK_NAMES:
        if task_name in leaf_name:
            return task_name
    return None


def build_model_label(model_root: Path, result_folder: Path) -> str:
    model_name = model_root.name
    if "ming-lite-omni" in model_name.lower():
        return model_name

    size_match = MODEL_SIZE_PATTERN.search(result_folder.name)
    if size_match is not None:
        return f"{model_name}-{size_match.group('size').upper()}"
    return model_name


def result_folder_priority(result_folder: Path) -> tuple[int, str]:
    leaf_name = result_folder.name.lower()
    return (1 if "single-turn" in leaf_name else 0, str(result_folder))


def is_error_text(text: str) -> bool:
    normalized = text.strip()
    return normalized.startswith("[ERROR]") or ERROR_TEXT_PATTERN.search(normalized) is not None


def quantize_progress_ratio(progress_ratio: float, partitions: int) -> float:
    bucket_index = round(progress_ratio * partitions)
    bucket_index = max(1, min(partitions, bucket_index))
    return bucket_index / partitions


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def build_human_sequences(
    rows: Sequence[dict[str, Any]],
    text_field: str,
    prompt_name: str,
) -> tuple[list[HumanClipSequence], list[str]]:
    grouped_rows: dict[tuple[str, int, str, str, str, str], dict[int, str]] = defaultdict(dict)
    warnings: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        media_path = str(row.get("video_path") or row.get("audio_path") or "").strip()
        text_value = str(row.get(text_field) or "").strip()
        if not media_path:
            warnings.append(f"{prompt_name} row {row_index}: missing media path, skipping")
            continue
        if not text_value:
            warnings.append(f"{prompt_name} row {row_index}: missing {text_field}, skipping")
            continue

        try:
            clip = parse_clip_metadata(media_path)
        except ValueError as exc:
            warnings.append(f"{prompt_name} row {row_index}: {exc}")
            continue

        group_key = (
            clip.dataset_name,
            clip.utt_count,
            clip.clip_prefix,
            str(row.get("annotator_number", "")),
            str(row.get("source_file", "")),
            clip.group_identifier,
        )
        grouped_rows[group_key][clip.clip_index] = text_value

    sequences: list[HumanClipSequence] = []
    for group_key, clip_texts in sorted(grouped_rows.items()):
        ordered = sorted(clip_texts.items(), key=lambda item: item[0])
        if len(ordered) < 2:
            warnings.append(
                f"{prompt_name} group {group_key[-1]} annotator={group_key[3]} has fewer than 2 clips, skipping"
            )
            continue
        sequences.append(
            HumanClipSequence(
                prompt_name=prompt_name,
                dataset_name=group_key[0],
                utt_count=group_key[1],
                clip_prefix=group_key[2],
                annotator_number=group_key[3],
                source_file=group_key[4],
                group_identifier=group_key[5],
                ordered_clips=ordered,
            )
        )

    return sequences, warnings


def index_model_sequences(
    results_root: Path,
) -> tuple[dict[tuple[str, str, str, int, str], list[tuple[int, str]]], list[str]]:
    indexed_sequences: dict[tuple[str, str, str, int, str], tuple[tuple[int, str], list[tuple[int, str]]]] = {}
    warnings: list[str] = []

    model_roots = iter_model_roots(results_root)
    if not model_roots:
        raise FileNotFoundError(
            f"No model result folders with dataset structure found under {results_root}"
        )

    for model_root in model_roots:
        for dataset_name in DATASET_NAMES:
            dataset_root = model_root / dataset_name
            if not dataset_root.is_dir():
                continue

            for result_folder in iter_result_folders(dataset_root):
                group_size = parse_group_size(result_folder)
                prompt_name = parse_task_name(result_folder)
                if group_size is None or prompt_name is None:
                    continue

                model_label = build_model_label(model_root, result_folder)
                folder_priority = result_folder_priority(result_folder)

                json_paths = sorted(
                    result_folder.glob("batch*.json"),
                    key=lambda path: sort_key_for_clip_identifier(path.stem),
                )
                for json_path in json_paths:
                    payload = load_json(json_path)
                    for dialogue_key, dialogue_value in find_dialogue_entries(payload):
                        ordered_clips = extract_ordered_clips(dialogue_key, dialogue_value)
                        if len(ordered_clips) < 2:
                            continue

                        key = (
                            model_label,
                            prompt_name,
                            dataset_name,
                            group_size,
                            dialogue_key,
                        )
                        existing = indexed_sequences.get(key)
                        if existing is None or folder_priority > existing[0]:
                            if existing is not None:
                                warnings.append(
                                    "Replacing duplicate model sequence for "
                                    f"{key} with higher-priority folder {result_folder}"
                                )
                            indexed_sequences[key] = (folder_priority, ordered_clips)
                        elif folder_priority == existing[0]:
                            warnings.append(
                                f"Duplicate model sequence for {key} in {result_folder}; keeping first match"
                            )

    return {key: value[1] for key, value in indexed_sequences.items()}, warnings


def compute_aligned_sequence_similarity(
    model: Any,
    human_sequence: HumanClipSequence,
    model_label: str,
    model_clips: list[tuple[int, str]],
) -> tuple[SequenceSimilarity | None, str | None]:
    human_indices = [index for index, _ in human_sequence.ordered_clips]
    model_indices = [index for index, _ in model_clips]
    if human_indices != model_indices:
        return None, (
            f"{model_label} {human_sequence.prompt_name} {human_sequence.dataset_name} "
            f"{human_sequence.utt_count}-utt {human_sequence.clip_prefix}: "
            f"clip index mismatch human={human_indices} model={model_indices}"
        )

    human_texts = [text.strip() for _, text in human_sequence.ordered_clips]
    model_texts = [text.strip() for _, text in model_clips]

    if any(not text or is_error_text(text) for text in human_texts):
        return None, (
            f"{model_label} {human_sequence.prompt_name} {human_sequence.clip_prefix}: "
            "invalid human text encountered"
        )
    if any(not text or is_error_text(text) for text in model_texts):
        return None, (
            f"{model_label} {human_sequence.prompt_name} {human_sequence.clip_prefix}: "
            "invalid model text encountered"
        )

    embeddings = model.encode(human_texts + model_texts, convert_to_numpy=True)
    clip_count = len(human_texts)
    human_embeddings = embeddings[:clip_count]
    model_embeddings = embeddings[clip_count:]

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for cosine similarity") from exc

    similarities = cosine_similarity(human_embeddings, model_embeddings).diagonal().tolist()
    return (
        SequenceSimilarity(
            model_label=model_label,
            prompt_name=human_sequence.prompt_name,
            dataset_name=human_sequence.dataset_name,
            utt_count=human_sequence.utt_count,
            clip_prefix=human_sequence.clip_prefix,
            annotator_number=human_sequence.annotator_number,
            source_file=human_sequence.source_file,
            group_identifier=human_sequence.group_identifier,
            clip_count=clip_count,
            similarities=[float(value) for value in similarities],
        ),
        None,
    )


def collect_similarity_bins(
    sequences: Sequence[SequenceSimilarity],
    progress_partitions: int,
) -> dict[float, list[float]]:
    grouped_values: dict[float, list[float]] = defaultdict(list)
    for sequence in sequences:
        for position, similarity in enumerate(sequence.similarities, start=1):
            progress_ratio = quantize_progress_ratio(
                position / sequence.clip_count,
                progress_partitions,
            )
            grouped_values[progress_ratio].append(similarity)
    return grouped_values


def mean_and_median_by_ratio(
    grouped_values: dict[float, list[float]],
) -> tuple[list[float], list[float], list[float]]:
    ratios = sorted(grouped_values)
    means = [float(np.mean(grouped_values[ratio])) for ratio in ratios]
    medians = [float(np.median(grouped_values[ratio])) for ratio in ratios]
    return ratios, means, medians


def percentile_by_ratio(
    grouped_values: dict[float, list[float]],
    percentile: int,
) -> tuple[list[float], list[float]]:
    ratios = sorted(grouped_values)
    values = [float(np.percentile(grouped_values[ratio], percentile)) for ratio in ratios]
    return ratios, values


def plot_similarity_stat_multi_model(
    case_sequences: dict[str, Sequence[SequenceSimilarity]],
    prompt_name: str,
    dataset_name: str,
    utt_count: int,
    stat_name: str,
    output_path: Path,
    progress_partitions: int,
    percentile: int | None = None,
) -> bool:
    plt.figure(figsize=(9, 6))
    plotted_any = False
    color_map = plt.cm.get_cmap("tab10")

    for model_index, model_label in enumerate(sorted(case_sequences)):
        sequences = case_sequences[model_label]
        if not sequences:
            continue
        grouped_values = collect_similarity_bins(sequences, progress_partitions)
        if not grouped_values:
            continue

        if stat_name == "mean":
            ratios = sorted(grouped_values)
            stat_values = [float(np.mean(grouped_values[ratio])) for ratio in ratios]
            legend_suffix = "mean"
            ylabel = "Average cosine similarity"
            title_suffix = "Mean"
        elif stat_name == "median":
            ratios = sorted(grouped_values)
            stat_values = [float(np.median(grouped_values[ratio])) for ratio in ratios]
            legend_suffix = "median"
            ylabel = "Median cosine similarity"
            title_suffix = "Median"
        elif stat_name == "percentile" and percentile is not None:
            ratios, stat_values = percentile_by_ratio(grouped_values, percentile)
            legend_suffix = f"p{percentile}"
            ylabel = f"{percentile}th percentile cosine similarity"
            title_suffix = f"p{percentile}"
        else:
            raise ValueError(f"Unsupported stat_name={stat_name} percentile={percentile}")

        color = color_map(model_index % 10)
        plt.plot(
            ratios,
            stat_values,
            color=color,
            linewidth=1.8,
            marker="o",
            label=f"{model_label} {legend_suffix}",
        )
        plotted_any = True

    if not plotted_any:
        plt.close()
        return False

    plt.title(
        f"Human vs Model Similarity | {dataset_name} | {utt_count}-utt | {prompt_name} | {title_suffix}"
    )
    plt.xlabel(f"Observed clip ratio (rounded to nearest 1/{progress_partitions})")
    plt.ylabel(ylabel)
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def summarize_bins(
    model_label: str,
    prompt_name: str,
    dataset_name: str,
    utt_count: int,
    sequences: Sequence[SequenceSimilarity],
    progress_partitions: int,
) -> list[dict[str, object]]:
    grouped_values = collect_similarity_bins(sequences, progress_partitions)
    rows: list[dict[str, object]] = []
    for ratio in sorted(grouped_values):
        values = grouped_values[ratio]
        rows.append(
            {
                "model": model_label,
                "prompt": prompt_name,
                "dataset": dataset_name,
                "utt_count": utt_count,
                "bin_index": int(round(ratio * progress_partitions)),
                "progress_ratio": ratio,
                "sample_count": len(values),
                "mean_similarity": float(np.mean(values)),
                "median_similarity": float(np.median(values)),
                "percentile_25": float(np.percentile(values, 25)),
                "percentile_50": float(np.percentile(values, 50)),
                "percentile_75": float(np.percentile(values, 75)),
            }
        )
    return rows


def build_point_rows(
    sequences: Sequence[SequenceSimilarity],
    progress_partitions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence in sequences:
        for clip_position, similarity in enumerate(sequence.similarities, start=1):
            raw_progress_ratio = clip_position / sequence.clip_count
            rows.append(
                {
                    "model": sequence.model_label,
                    "prompt": sequence.prompt_name,
                    "dataset": sequence.dataset_name,
                    "utt_count": sequence.utt_count,
                    "annotator_number": sequence.annotator_number,
                    "source_file": sequence.source_file,
                    "group_identifier": sequence.group_identifier,
                    "clip_prefix": sequence.clip_prefix,
                    "clip_position": clip_position,
                    "clip_count": sequence.clip_count,
                    "progress_ratio_raw": raw_progress_ratio,
                    "progress_ratio_binned": quantize_progress_ratio(
                        raw_progress_ratio,
                        progress_partitions,
                    ),
                    "similarity": similarity,
                }
            )
    return rows


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "prompt",
        "dataset",
        "utt_count",
        "bin_index",
        "progress_ratio",
        "sample_count",
        "mean_similarity",
        "median_similarity",
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
    results_root: Path,
    rows: Sequence[dict[str, object]],
    warnings: Sequence[str],
) -> None:
    payload = {
        "source": str(source_path.resolve()),
        "results_root": str(results_root.resolve()),
        "row_count": len(rows),
        "rows": list(rows),
        "warnings": list(warnings),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_points_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "prompt",
        "dataset",
        "utt_count",
        "annotator_number",
        "source_file",
        "group_identifier",
        "clip_prefix",
        "clip_position",
        "clip_count",
        "progress_ratio_raw",
        "progress_ratio_binned",
        "similarity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_points_json(
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
    return (input_path.parent / "human_model_similarity").resolve()


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

    results_root = resolve_results_root(args.results_root)

    from sentence_transformers import SentenceTransformer

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    embedding_model = SentenceTransformer(model_loc)

    model_sequences, model_index_warnings = index_model_sequences(results_root)
    model_labels = sorted({key[0] for key in model_sequences})

    all_warnings: list[str] = list(extraction_errors) + model_index_warnings
    missing_sequence_warnings: set[str] = set()
    alignment_warnings: set[str] = set()
    all_summary_rows: list[dict[str, object]] = []
    all_point_rows: list[dict[str, object]] = []

    sequences_by_case: dict[tuple[str, str, str, int], list[SequenceSimilarity]] = defaultdict(list)

    for prompt_name, text_field in PROMPT_FIELD_MAP.items():
        human_sequences, human_warnings = build_human_sequences(
            rows=rows,
            text_field=text_field,
            prompt_name=prompt_name,
        )
        all_warnings.extend(human_warnings)

        for human_sequence in human_sequences:
            for model_label in model_labels:
                key = (
                    model_label,
                    prompt_name,
                    human_sequence.dataset_name,
                    human_sequence.utt_count,
                    human_sequence.clip_prefix,
                )
                model_clips = model_sequences.get(key)
                if model_clips is None:
                    missing_sequence_warnings.add(
                        f"missing model sequence for {key}"
                    )
                    continue

                similarity_sequence, warning = compute_aligned_sequence_similarity(
                    model=embedding_model,
                    human_sequence=human_sequence,
                    model_label=model_label,
                    model_clips=model_clips,
                )
                if warning is not None:
                    alignment_warnings.add(warning)
                    continue
                if similarity_sequence is None:
                    continue

                case_key = (
                    model_label,
                    prompt_name,
                    human_sequence.dataset_name,
                    human_sequence.utt_count,
                )
                sequences_by_case[case_key].append(similarity_sequence)

    all_warnings.extend(sorted(missing_sequence_warnings))
    all_warnings.extend(sorted(alignment_warnings))

    plot_sequences_by_case: dict[tuple[str, str, int], dict[str, list[SequenceSimilarity]]] = defaultdict(
        lambda: defaultdict(list)
    )
    overall_plot_sequences_by_case: dict[tuple[str, int], dict[str, list[SequenceSimilarity]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (model_label, prompt_name, dataset_name, utt_count), sequences in sorted(sequences_by_case.items()):
        if not sequences:
            continue

        plot_sequences_by_case[(prompt_name, dataset_name, utt_count)][model_label].extend(sequences)
        overall_plot_sequences_by_case[(prompt_name, utt_count)][model_label].extend(sequences)
        all_summary_rows.extend(
            summarize_bins(
                model_label=model_label,
                prompt_name=prompt_name,
                dataset_name=dataset_name,
                utt_count=utt_count,
                sequences=sequences,
                progress_partitions=args.progress_partitions,
            )
        )
        all_point_rows.extend(build_point_rows(sequences, args.progress_partitions))

    for (prompt_name, dataset_name, utt_count), case_sequences in sorted(plot_sequences_by_case.items()):
        case_plot_dir = plot_dir / prompt_name / f"{utt_count}utt"
        case_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_specs = [
            ("mean", None, case_plot_dir / f"{dataset_name}_human_vs_model_similarity_mean.png"),
            ("median", None, case_plot_dir / f"{dataset_name}_human_vs_model_similarity_median.png"),
            ("percentile", 25, case_plot_dir / f"{dataset_name}_human_vs_model_similarity_p25.png"),
            ("percentile", 50, case_plot_dir / f"{dataset_name}_human_vs_model_similarity_p50.png"),
            ("percentile", 75, case_plot_dir / f"{dataset_name}_human_vs_model_similarity_p75.png"),
        ]
        for stat_name, percentile, plot_path in plot_specs:
            if plot_similarity_stat_multi_model(
                case_sequences=case_sequences,
                prompt_name=prompt_name,
                dataset_name=dataset_name,
                utt_count=utt_count,
                stat_name=stat_name,
                output_path=plot_path,
                progress_partitions=args.progress_partitions,
                percentile=percentile,
            ):
                print(f"[INFO] Saved {plot_path}")

    for (prompt_name, utt_count), case_sequences in sorted(overall_plot_sequences_by_case.items()):
        for model_label, sequences in sorted(case_sequences.items()):
            if not sequences:
                continue
            all_summary_rows.extend(
                summarize_bins(
                    model_label=model_label,
                    prompt_name=prompt_name,
                    dataset_name="all",
                    utt_count=utt_count,
                    sequences=sequences,
                    progress_partitions=args.progress_partitions,
                )
            )

        case_plot_dir = plot_dir / prompt_name / f"{utt_count}utt"
        case_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_specs = [
            ("mean", None, case_plot_dir / "all_datasets_human_vs_model_similarity_mean.png"),
            ("median", None, case_plot_dir / "all_datasets_human_vs_model_similarity_median.png"),
            ("percentile", 25, case_plot_dir / "all_datasets_human_vs_model_similarity_p25.png"),
            ("percentile", 50, case_plot_dir / "all_datasets_human_vs_model_similarity_p50.png"),
            ("percentile", 75, case_plot_dir / "all_datasets_human_vs_model_similarity_p75.png"),
        ]
        for stat_name, percentile, plot_path in plot_specs:
            if plot_similarity_stat_multi_model(
                case_sequences=case_sequences,
                prompt_name=prompt_name,
                dataset_name="all",
                utt_count=utt_count,
                stat_name=stat_name,
                output_path=plot_path,
                progress_partitions=args.progress_partitions,
                percentile=percentile,
            ):
                print(f"[INFO] Saved {plot_path}")

    summary_csv_path = plot_data_dir / "human_model_similarity_summary.csv"
    summary_json_path = plot_data_dir / "human_model_similarity_summary.json"
    point_csv_path = plot_data_dir / "human_model_similarity_points.csv"
    point_json_path = plot_data_dir / "human_model_similarity_points.json"

    write_summary_csv(summary_csv_path, all_summary_rows)
    write_summary_json(
        summary_json_path,
        source_path=analysis_source_path,
        results_root=results_root,
        rows=all_summary_rows,
        warnings=all_warnings,
    )
    write_points_csv(point_csv_path, all_point_rows)
    write_points_json(point_json_path, analysis_source_path, all_point_rows)

    print(f"[INFO] Saved extracted/loaded CSV source {csv_path}")
    print(f"[INFO] Saved {summary_csv_path}")
    print(f"[INFO] Saved {summary_json_path}")
    print(f"[INFO] Saved {point_csv_path}")
    print(f"[INFO] Saved {point_json_path}")

    for warning in all_warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
