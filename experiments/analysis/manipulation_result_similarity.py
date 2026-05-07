from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .human_model_similarity import (
        DATASET_NAMES,
        build_model_label,
        extract_assistant_text,
        find_dialogue_entries,
        is_error_text,
        iter_model_roots,
        iter_result_folders,
        parse_group_size,
        parse_task_name,
        result_folder_priority,
        sort_key_for_clip_identifier,
    )
except ImportError:
    from human_model_similarity import (
        DATASET_NAMES,
        build_model_label,
        extract_assistant_text,
        find_dialogue_entries,
        is_error_text,
        iter_model_roots,
        iter_result_folders,
        parse_group_size,
        parse_task_name,
        result_folder_priority,
        sort_key_for_clip_identifier,
    )


DEFAULT_MODEL = "all-MiniLM-L6-v2"
PROMPT_DISPLAY_LABELS = {
    "intention": "Q1",
    "affordance": "Q2",
}
PROMPT_SORT_ORDER = {
    "intention": 0,
    "affordance": 1,
}
AUDIO_CONDITION_ORDER = {
    "with_audio": 0,
    "no_audio": 1,
}
METRIC_ORDER = {
    "HCS": 0,
    "NHCS": 1,
    "r_H": 2,
}
TABLE_FIELDNAMES = [
    "question",
    "model",
    "with_audio_HCS",
    "with_audio_NHCS",
    "with_audio_r_H",
    "no_audio_HCS",
    "no_audio_NHCS",
    "no_audio_r_H",
]
RATIO_EPSILON = 1e-12


@dataclass(frozen=True)
class SourceVariant:
    slug: str
    source_results_root: Path
    audio_condition: str
    comparison_mode: bool

    @property
    def metric_name(self) -> str:
        return "NHCS" if self.comparison_mode else "HCS"


@dataclass(frozen=True)
class ClipResult:
    model_label: str
    prompt_name: str
    dataset_name: str
    utt_count: int
    clip_identifier: str
    text: str
    batch_json: str
    result_folder: str


@dataclass(frozen=True)
class ClipComparison:
    variant_slug: str
    source_results_root: str
    audio_condition: str
    comparison_mode: bool
    metric_name: str
    model_label: str
    prompt_name: str
    dataset_name: str
    utt_count: int
    clip_identifier: str
    source_text: str
    reference_text: str
    source_batch_json: str
    reference_batch_json: str
    similarity: float

    @property
    def prompt_label(self) -> str:
        return prompt_display_label(self.prompt_name)

    @property
    def sensitivity(self) -> float:
        return 1.0 - self.similarity


@dataclass(frozen=True)
class RatioPoint:
    model_label: str
    prompt_name: str
    dataset_name: str
    utt_count: int
    clip_identifier: str
    audio_condition: str
    hcs_value: float
    nhcs_value: float
    ratio_value: float

    @property
    def prompt_label(self) -> str:
        return prompt_display_label(self.prompt_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare task-2 manipulated-result answers against the corresponding "
            "full benchmark answers for all four corruption settings and compute "
            "HCS/NHCS summaries."
        )
    )
    parser.add_argument(
        "--source-results-root",
        type=Path,
        required=True,
        help="Task-2 manipulated result tree with audio and without comparison corruption.",
    )
    parser.add_argument(
        "--comparison-source-results-root",
        type=Path,
        required=True,
        help="Task-2 manipulated result tree with audio and comparison corruption.",
    )
    parser.add_argument(
        "--no-audio-source-results-root",
        type=Path,
        required=True,
        help="Task-2 manipulated result tree without audio and without comparison corruption.",
    )
    parser.add_argument(
        "--no-audio-comparison-source-results-root",
        type=Path,
        required=True,
        help="Task-2 manipulated result tree without audio and with comparison corruption.",
    )
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        required=True,
        help="Full benchmark result tree to search across, e.g. gestalt_bench/results.",
    )
    parser.add_argument(
        "--additional-reference-results-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional full benchmark result tree to search across. May be "
            "passed multiple times, e.g. the Gemini tree from gemini_retrieve_daic.sh."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where comparison tables and point-level data will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name (used when --model-path is not set).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local SentenceTransformer model directory.",
    )
    return parser.parse_args()


def prompt_display_label(prompt_name: str) -> str:
    return PROMPT_DISPLAY_LABELS.get(prompt_name, prompt_name)


def prompt_sort_key(prompt_name: str) -> tuple[int, str]:
    return (PROMPT_SORT_ORDER.get(prompt_name, len(PROMPT_SORT_ORDER)), prompt_name)


def model_sort_key(model_label: str) -> tuple[int, int, str]:
    normalized = model_label.lower()
    family_rank = 99
    size_rank = 99
    if normalized.startswith("qwen2.5"):
        family_rank = 0
        if "7b" in normalized:
            size_rank = 0
        elif "3b" in normalized:
            size_rank = 1
    elif normalized.startswith("ming-lite-omni"):
        family_rank = 1
        size_rank = 0
    elif normalized.startswith("gemma"):
        family_rank = 2
        size_rank = 0
    elif normalized.startswith("gemini"):
        family_rank = 3
        size_rank = 0
    return family_rank, size_rank, normalized


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_clip_entries(dialogue_key: str, dialogue_value: object) -> list[tuple[str, str]]:
    if not isinstance(dialogue_value, Sequence) or isinstance(dialogue_value, (str, bytes)):
        return []

    entries: dict[str, str] = {}
    for index, item in enumerate(dialogue_value, start=1):
        if not isinstance(item, dict):
            continue

        text = extract_assistant_text(item)
        if text is None:
            continue

        clip_index = index
        file_name = item.get("file")
        if isinstance(file_name, str):
            match = re.fullmatch(rf"{re.escape(dialogue_key)}_clip(?P<index>\d+)", file_name)
            if match is not None:
                clip_index = int(match.group("index"))

        clip_identifier = f"{dialogue_key}_clip{clip_index}"
        entries[clip_identifier] = text.strip()

    return sorted(entries.items(), key=lambda item: sort_key_for_clip_identifier(item[0]))


def index_result_clips(
    results_root: Path,
) -> tuple[dict[tuple[str, str, str, int, str], ClipResult], list[str]]:
    indexed: dict[tuple[str, str, str, int, str], tuple[tuple[int, str], ClipResult]] = {}
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
                utt_count = parse_group_size(result_folder)
                prompt_name = parse_task_name(result_folder)
                if utt_count is None or prompt_name is None:
                    continue

                model_label = build_model_label(model_root, result_folder)
                folder_rank = result_folder_priority(result_folder)

                for json_path in sorted(
                    result_folder.glob("batch*.json"),
                    key=lambda path: sort_key_for_clip_identifier(path.stem),
                ):
                    payload = load_json(json_path)
                    for dialogue_key, dialogue_value in find_dialogue_entries(payload):
                        for clip_identifier, text in iter_clip_entries(dialogue_key, dialogue_value):
                            key = (
                                model_label,
                                prompt_name,
                                dataset_name,
                                utt_count,
                                clip_identifier,
                            )
                            clip_result = ClipResult(
                                model_label=model_label,
                                prompt_name=prompt_name,
                                dataset_name=dataset_name,
                                utt_count=utt_count,
                                clip_identifier=clip_identifier,
                                text=text,
                                batch_json=str(json_path.resolve()),
                                result_folder=str(result_folder.resolve()),
                            )
                            existing = indexed.get(key)
                            if existing is None or folder_rank > existing[0]:
                                if existing is not None:
                                    warnings.append(
                                        "Replacing duplicate clip result for "
                                        f"{key} with higher-priority folder {result_folder}"
                                    )
                                indexed[key] = (folder_rank, clip_result)
                            elif folder_rank == existing[0]:
                                warnings.append(
                                    f"Duplicate clip result for {key} in {json_path}; keeping first match"
                                )

    return {key: value[1] for key, value in indexed.items()}, warnings


def index_result_clips_from_roots(
    results_roots: Sequence[Path],
    root_label: str,
) -> tuple[dict[tuple[str, str, str, int, str], ClipResult], list[str]]:
    merged: dict[tuple[str, str, str, int, str], ClipResult] = {}
    warnings: list[str] = []

    for results_root in results_roots:
        clips, root_warnings = index_result_clips(results_root)
        warnings.extend(root_warnings)
        for key, clip_result in clips.items():
            existing = merged.get(key)
            if existing is not None:
                warnings.append(
                    f"Duplicate {root_label} clip result for {key} in "
                    f"{clip_result.batch_json}; keeping first match from {existing.batch_json}"
                )
                continue
            merged[key] = clip_result

    return merged, warnings


def compute_clip_similarities(
    embedding_model: Any,
    source_variant: SourceVariant,
    source_clips: dict[tuple[str, str, str, int, str], ClipResult],
    reference_clips: dict[tuple[str, str, str, int, str], ClipResult],
) -> tuple[list[ClipComparison], list[str]]:
    pending_pairs: list[tuple[ClipResult, ClipResult]] = []
    warnings: list[str] = []

    for key, source_clip in sorted(source_clips.items()):
        reference_clip = reference_clips.get(key)
        if reference_clip is None:
            warnings.append(
                f"[{source_variant.slug}] Missing reference clip for {key}"
            )
            continue
        if not source_clip.text or is_error_text(source_clip.text):
            warnings.append(f"[{source_variant.slug}] Invalid source text for {key}")
            continue
        if not reference_clip.text or is_error_text(reference_clip.text):
            warnings.append(f"[{source_variant.slug}] Invalid reference text for {key}")
            continue
        pending_pairs.append((source_clip, reference_clip))

    if not pending_pairs:
        return [], warnings

    source_texts = [source_clip.text for source_clip, _ in pending_pairs]
    reference_texts = [reference_clip.text for _, reference_clip in pending_pairs]

    source_embeddings = embedding_model.encode(source_texts, convert_to_numpy=True)
    reference_embeddings = embedding_model.encode(reference_texts, convert_to_numpy=True)

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for cosine similarity") from exc

    similarities = cosine_similarity(source_embeddings, reference_embeddings).diagonal().tolist()
    comparisons: list[ClipComparison] = []
    for (source_clip, reference_clip), similarity in zip(pending_pairs, similarities):
        comparisons.append(
            ClipComparison(
                variant_slug=source_variant.slug,
                source_results_root=str(source_variant.source_results_root),
                audio_condition=source_variant.audio_condition,
                comparison_mode=source_variant.comparison_mode,
                metric_name=source_variant.metric_name,
                model_label=source_clip.model_label,
                prompt_name=source_clip.prompt_name,
                dataset_name=source_clip.dataset_name,
                utt_count=source_clip.utt_count,
                clip_identifier=source_clip.clip_identifier,
                source_text=source_clip.text,
                reference_text=reference_clip.text,
                source_batch_json=source_clip.batch_json,
                reference_batch_json=reference_clip.batch_json,
                similarity=float(similarity),
            )
        )

    return comparisons, warnings


def build_raw_point_rows(comparisons: Sequence[ClipComparison]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        rows.append(
            {
                "variant": comparison.variant_slug,
                "source_results_root": comparison.source_results_root,
                "audio_condition": comparison.audio_condition,
                "comparison_mode": comparison.comparison_mode,
                "metric": comparison.metric_name,
                "model": comparison.model_label,
                "prompt": comparison.prompt_name,
                "question": comparison.prompt_label,
                "dataset": comparison.dataset_name,
                "utt_count": comparison.utt_count,
                "clip_identifier": comparison.clip_identifier,
                "source_batch_json": comparison.source_batch_json,
                "reference_batch_json": comparison.reference_batch_json,
                "similarity": comparison.similarity,
                "sensitivity": comparison.sensitivity,
            }
        )
    return rows


def summarize_comparisons(
    comparisons: Sequence[ClipComparison],
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[ClipComparison]] = defaultdict(list)
    for comparison in comparisons:
        key = tuple(getattr(comparison, field) for field in group_fields)
        grouped[key].append(comparison)

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=_comparison_group_sort_key(group_fields)):
        similarities = np.asarray([value.similarity for value in values], dtype=float)
        sensitivities = 1.0 - similarities
        row = {
            field: value for field, value in zip(group_fields, key)
        }
        row["question"] = prompt_display_label(str(row["prompt_name"]))
        row["sample_count"] = int(len(values))
        row["mean_similarity"] = float(np.mean(similarities))
        row["mean_sensitivity"] = float(np.mean(sensitivities))
        row["std_sensitivity"] = sample_std(sensitivities)
        row["std_err_sensitivity"] = std_err(sensitivities)
        row["formatted_sensitivity"] = format_mean_std_err(
            row["mean_sensitivity"], row["std_err_sensitivity"]
        )
        rows.append(normalize_summary_row(row))
    return rows


def sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def std_err(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(sample_std(values) / math.sqrt(values.size))


def build_ratio_points(
    comparisons: Sequence[ClipComparison],
) -> tuple[list[RatioPoint], list[str]]:
    warnings: list[str] = []
    hcs_map: dict[tuple[str, str, str, int, str, str], ClipComparison] = {}
    nhcs_map: dict[tuple[str, str, str, int, str, str], ClipComparison] = {}

    for comparison in comparisons:
        key = (
            comparison.model_label,
            comparison.prompt_name,
            comparison.dataset_name,
            comparison.utt_count,
            comparison.clip_identifier,
            comparison.audio_condition,
        )
        if comparison.metric_name == "HCS":
            hcs_map[key] = comparison
        elif comparison.metric_name == "NHCS":
            nhcs_map[key] = comparison

    ratio_points: list[RatioPoint] = []
    for key in sorted(hcs_map):
        hcs_point = hcs_map[key]
        nhcs_point = nhcs_map.get(key)
        if nhcs_point is None:
            warnings.append(f"Missing NHCS point for ratio key {key}")
            continue

        nhcs_value = nhcs_point.sensitivity
        if abs(nhcs_value) <= RATIO_EPSILON:
            warnings.append(f"Skipping ratio for near-zero NHCS at key {key}")
            continue

        ratio_points.append(
            RatioPoint(
                model_label=hcs_point.model_label,
                prompt_name=hcs_point.prompt_name,
                dataset_name=hcs_point.dataset_name,
                utt_count=hcs_point.utt_count,
                clip_identifier=hcs_point.clip_identifier,
                audio_condition=hcs_point.audio_condition,
                hcs_value=hcs_point.sensitivity,
                nhcs_value=nhcs_value,
                ratio_value=hcs_point.sensitivity / nhcs_value,
            )
        )

    return ratio_points, warnings


def build_ratio_point_rows(ratio_points: Sequence[RatioPoint]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in ratio_points:
        rows.append(
            {
                "audio_condition": point.audio_condition,
                "model": point.model_label,
                "prompt": point.prompt_name,
                "question": point.prompt_label,
                "dataset": point.dataset_name,
                "utt_count": point.utt_count,
                "clip_identifier": point.clip_identifier,
                "HCS": point.hcs_value,
                "NHCS": point.nhcs_value,
                "r_H": point.ratio_value,
            }
        )
    return rows


def summarize_metric_points(
    comparisons: Sequence[ClipComparison],
    ratio_points: Sequence[RatioPoint],
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)

    for comparison in comparisons:
        key_parts = [getattr(comparison, field) for field in group_fields]
        key_parts.append(comparison.metric_name)
        grouped[tuple(key_parts)].append(comparison.sensitivity)

    for ratio_point in ratio_points:
        key_parts = [getattr(ratio_point, field) for field in group_fields]
        key_parts.append("r_H")
        grouped[tuple(key_parts)].append(ratio_point.ratio_value)

    rows: list[dict[str, object]] = []
    for key, values_list in sorted(grouped.items(), key=_metric_group_sort_key(group_fields)):
        row = {
            field: value for field, value in zip(group_fields, key[:-1])
        }
        row["metric"] = str(key[-1])
        row["question"] = prompt_display_label(str(row["prompt_name"]))
        values = np.asarray(values_list, dtype=float)
        row["sample_count"] = int(values.size)
        row["mean_value"] = float(np.mean(values))
        row["std_value"] = sample_std(values)
        row["std_err_value"] = std_err(values)
        row["formatted_value"] = format_mean_std_err(row["mean_value"], row["std_err_value"])
        rows.append(normalize_summary_row(row))
    return rows


def normalize_summary_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    if "model_label" in normalized:
        normalized["model"] = normalized.pop("model_label")
    if "prompt_name" in normalized:
        normalized["prompt"] = normalized.pop("prompt_name")
    if "dataset_name" in normalized:
        normalized["dataset"] = normalized.pop("dataset_name")
    if "metric_name" in normalized:
        normalized["metric"] = normalized.pop("metric_name")
    return normalized


def format_mean_std_err(mean_value: object, std_err_value: object) -> str:
    return f"{float(mean_value):.3f} ± {float(std_err_value):.3f}"


def format_ratio_from_means(numerator: float, denominator: float) -> str:
    if abs(denominator) <= RATIO_EPSILON:
        return ""
    return f"{(numerator / denominator):.3f}"


def build_table_rows(summary_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summary_map = {
        (
            str(row["model"]),
            str(row["prompt"]),
            str(row["audio_condition"]),
            str(row["metric"]),
        ): row
        for row in summary_rows
    }

    models_and_prompts = sorted(
        {
            (str(row["model"]), str(row["prompt"]))
            for row in summary_rows
        },
        key=lambda item: (prompt_sort_key(item[1]), model_sort_key(item[0])),
    )

    rows: list[dict[str, object]] = []
    for model_label, prompt_name in models_and_prompts:
        row: dict[str, object] = {
            "question": prompt_display_label(prompt_name),
            "model": model_label,
        }
        for audio_condition in ("with_audio", "no_audio"):
            hcs_row = summary_map.get((model_label, prompt_name, audio_condition, "HCS"))
            nhcs_row = summary_map.get((model_label, prompt_name, audio_condition, "NHCS"))
            row[f"{audio_condition}_HCS"] = (
                str(hcs_row["formatted_value"]) if hcs_row is not None else ""
            )
            row[f"{audio_condition}_NHCS"] = (
                str(nhcs_row["formatted_value"]) if nhcs_row is not None else ""
            )
            if hcs_row is not None and nhcs_row is not None:
                row[f"{audio_condition}_r_H"] = format_ratio_from_means(
                    float(hcs_row["mean_value"]),
                    float(nhcs_row["mean_value"]),
                )
            else:
                row[f"{audio_condition}_r_H"] = ""
        rows.append(row)
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_table(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_group_sort_key(group_fields: Sequence[str]):
    def _key(item: tuple[tuple[object, ...], list[ClipComparison]]) -> tuple[object, ...]:
        key = item[0]
        sort_parts: list[object] = []
        for field_name, value in zip(group_fields, key):
            sort_parts.extend(_field_sort_parts(field_name, value))
        return tuple(sort_parts)

    return _key


def _metric_group_sort_key(group_fields: Sequence[str]):
    def _key(item: tuple[tuple[object, ...], list[float]]) -> tuple[object, ...]:
        key = item[0]
        sort_parts: list[object] = []
        for field_name, value in zip(group_fields, key[:-1]):
            sort_parts.extend(_field_sort_parts(field_name, value))
        sort_parts.extend(_field_sort_parts("metric", key[-1]))
        return tuple(sort_parts)

    return _key


def _field_sort_parts(field_name: str, value: object) -> list[object]:
    text_value = str(value)
    if field_name == "prompt_name":
        return list(prompt_sort_key(text_value))
    if field_name == "model_label":
        return list(model_sort_key(text_value))
    if field_name == "audio_condition":
        return [AUDIO_CONDITION_ORDER.get(text_value, len(AUDIO_CONDITION_ORDER)), text_value]
    if field_name == "metric":
        return [METRIC_ORDER.get(text_value, len(METRIC_ORDER)), text_value]
    if field_name == "utt_count":
        return [int(value)]
    return [text_value]


def build_source_variants(args: argparse.Namespace) -> list[SourceVariant]:
    return [
        SourceVariant(
            slug="with_audio",
            source_results_root=args.source_results_root.expanduser().resolve(),
            audio_condition="with_audio",
            comparison_mode=False,
        ),
        SourceVariant(
            slug="with_audio_comparison",
            source_results_root=args.comparison_source_results_root.expanduser().resolve(),
            audio_condition="with_audio",
            comparison_mode=True,
        ),
        SourceVariant(
            slug="no_audio",
            source_results_root=args.no_audio_source_results_root.expanduser().resolve(),
            audio_condition="no_audio",
            comparison_mode=False,
        ),
        SourceVariant(
            slug="no_audio_comparison",
            source_results_root=args.no_audio_comparison_source_results_root.expanduser().resolve(),
            audio_condition="no_audio",
            comparison_mode=True,
        ),
    ]


def table_path_stem(utt_scope: str, dataset_name: str | None) -> str:
    if dataset_name is None:
        return f"manipulation_sensitivity_table_{utt_scope}_all_datasets"
    return f"manipulation_sensitivity_table_{utt_scope}_{dataset_name}"


def main() -> None:
    args = parse_args()
    source_variants = build_source_variants(args)
    reference_results_root = args.reference_results_root.expanduser().resolve()
    additional_reference_results_roots = [
        root.expanduser().resolve() for root in args.additional_reference_results_root
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_variant in source_variants:
        if not source_variant.source_results_root.is_dir():
            raise FileNotFoundError(
                f"Source results root does not exist: {source_variant.source_results_root}"
            )
    reference_results_roots = [reference_results_root, *additional_reference_results_roots]
    for root in reference_results_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Reference results root does not exist: {root}")

    from sentence_transformers import SentenceTransformer

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    embedding_model = SentenceTransformer(model_loc)

    print(
        "[INFO] Indexing reference results from "
        f"{', '.join(str(root) for root in reference_results_roots)}"
    )
    reference_clips, reference_warnings = index_result_clips_from_roots(
        reference_results_roots,
        root_label="reference",
    )
    print(f"[INFO] Indexed {len(reference_clips)} reference clips")

    all_comparisons: list[ClipComparison] = []
    source_warnings: list[str] = []
    comparison_warnings: list[str] = []
    for source_variant in source_variants:
        print(
            "[INFO] Indexing source results for "
            f"{source_variant.slug} from {source_variant.source_results_root}"
        )
        source_clips, variant_source_warnings = index_result_clips_from_roots(
            [source_variant.source_results_root],
            root_label=f"source:{source_variant.slug}",
        )
        source_warnings.extend(variant_source_warnings)
        print(
            f"[INFO] Indexed {len(source_clips)} source clips for {source_variant.slug}"
        )

        variant_comparisons, variant_comparison_warnings = compute_clip_similarities(
            embedding_model=embedding_model,
            source_variant=source_variant,
            source_clips=source_clips,
            reference_clips=reference_clips,
        )
        all_comparisons.extend(variant_comparisons)
        comparison_warnings.extend(variant_comparison_warnings)
        print(
            "[INFO] Computed "
            f"{len(variant_comparisons)} clip comparisons for {source_variant.slug}"
        )

    ratio_points, ratio_warnings = build_ratio_points(all_comparisons)
    raw_point_rows = build_raw_point_rows(all_comparisons)
    ratio_point_rows = build_ratio_point_rows(ratio_points)

    comparison_dataset_utt_rows = summarize_comparisons(
        all_comparisons,
        ["model_label", "prompt_name", "dataset_name", "utt_count", "audio_condition", "metric_name"],
    )
    comparison_utt_rows = summarize_comparisons(
        all_comparisons,
        ["model_label", "prompt_name", "utt_count", "audio_condition", "metric_name"],
    )
    comparison_dataset_rows = summarize_comparisons(
        all_comparisons,
        ["model_label", "prompt_name", "dataset_name", "audio_condition", "metric_name"],
    )
    comparison_overall_rows = summarize_comparisons(
        all_comparisons,
        ["model_label", "prompt_name", "audio_condition", "metric_name"],
    )

    metric_dataset_utt_rows = summarize_metric_points(
        all_comparisons,
        ratio_points,
        ["model_label", "prompt_name", "dataset_name", "utt_count", "audio_condition"],
    )
    metric_utt_rows = summarize_metric_points(
        all_comparisons,
        ratio_points,
        ["model_label", "prompt_name", "utt_count", "audio_condition"],
    )
    metric_dataset_rows = summarize_metric_points(
        all_comparisons,
        ratio_points,
        ["model_label", "prompt_name", "dataset_name", "audio_condition"],
    )
    metric_overall_rows = summarize_metric_points(
        all_comparisons,
        ratio_points,
        ["model_label", "prompt_name", "audio_condition"],
    )

    table_specs: list[tuple[str, list[dict[str, object]]]] = []
    table_specs.append(
        (
            table_path_stem("iutt", None),
            build_table_rows(metric_overall_rows),
        )
    )
    for dataset_name in DATASET_NAMES:
        dataset_rows = [
            row
            for row in metric_dataset_utt_rows
            if str(row["dataset"]) == dataset_name and int(row["utt_count"]) == 1
        ]
        table_specs.append(
            (
                table_path_stem("1utt", dataset_name),
                build_table_rows(dataset_rows),
            )
        )

    warnings = source_warnings + reference_warnings + comparison_warnings + ratio_warnings

    raw_point_csv_path = output_dir / "manipulation_result_similarity_points.csv"
    raw_point_json_path = output_dir / "manipulation_result_similarity_points.json"
    ratio_point_csv_path = output_dir / "manipulation_sensitivity_ratio_points.csv"
    comparison_dataset_utt_csv_path = output_dir / "manipulation_result_similarity_by_dataset_utt.csv"
    comparison_utt_csv_path = output_dir / "manipulation_result_similarity_by_utt.csv"
    comparison_dataset_csv_path = output_dir / "manipulation_result_similarity_by_dataset.csv"
    comparison_overall_csv_path = output_dir / "manipulation_result_similarity_overall.csv"
    metric_dataset_utt_csv_path = output_dir / "manipulation_sensitivity_by_dataset_utt.csv"
    metric_utt_csv_path = output_dir / "manipulation_sensitivity_by_utt.csv"
    metric_dataset_csv_path = output_dir / "manipulation_sensitivity_by_dataset.csv"
    metric_overall_csv_path = output_dir / "manipulation_sensitivity_overall.csv"
    summary_json_path = output_dir / "manipulation_result_similarity_summary.json"

    write_csv(
        raw_point_csv_path,
        [
            "variant",
            "source_results_root",
            "audio_condition",
            "comparison_mode",
            "metric",
            "model",
            "prompt",
            "question",
            "dataset",
            "utt_count",
            "clip_identifier",
            "source_batch_json",
            "reference_batch_json",
            "similarity",
            "sensitivity",
        ],
        raw_point_rows,
    )
    write_json(
        raw_point_json_path,
        {
            "source_variants": [
                {
                    "slug": variant.slug,
                    "source_results_root": str(variant.source_results_root),
                    "audio_condition": variant.audio_condition,
                    "comparison_mode": variant.comparison_mode,
                    "metric": variant.metric_name,
                }
                for variant in source_variants
            ],
            "reference_results_root": str(reference_results_root),
            "additional_reference_results_roots": [
                str(root) for root in additional_reference_results_roots
            ],
            "point_count": len(raw_point_rows),
            "points": raw_point_rows,
        },
    )
    write_csv(
        ratio_point_csv_path,
        [
            "audio_condition",
            "model",
            "prompt",
            "question",
            "dataset",
            "utt_count",
            "clip_identifier",
            "HCS",
            "NHCS",
            "r_H",
        ],
        ratio_point_rows,
    )
    write_csv(
        comparison_dataset_utt_csv_path,
        [
            "model",
            "prompt",
            "question",
            "dataset",
            "utt_count",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_similarity",
            "mean_sensitivity",
            "std_sensitivity",
            "std_err_sensitivity",
            "formatted_sensitivity",
        ],
        comparison_dataset_utt_rows,
    )
    write_csv(
        comparison_utt_csv_path,
        [
            "model",
            "prompt",
            "question",
            "utt_count",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_similarity",
            "mean_sensitivity",
            "std_sensitivity",
            "std_err_sensitivity",
            "formatted_sensitivity",
        ],
        comparison_utt_rows,
    )
    write_csv(
        comparison_dataset_csv_path,
        [
            "model",
            "prompt",
            "question",
            "dataset",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_similarity",
            "mean_sensitivity",
            "std_sensitivity",
            "std_err_sensitivity",
            "formatted_sensitivity",
        ],
        comparison_dataset_rows,
    )
    write_csv(
        comparison_overall_csv_path,
        [
            "model",
            "prompt",
            "question",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_similarity",
            "mean_sensitivity",
            "std_sensitivity",
            "std_err_sensitivity",
            "formatted_sensitivity",
        ],
        comparison_overall_rows,
    )
    write_csv(
        metric_dataset_utt_csv_path,
        [
            "model",
            "prompt",
            "question",
            "dataset",
            "utt_count",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_value",
            "std_value",
            "std_err_value",
            "formatted_value",
        ],
        metric_dataset_utt_rows,
    )
    write_csv(
        metric_utt_csv_path,
        [
            "model",
            "prompt",
            "question",
            "utt_count",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_value",
            "std_value",
            "std_err_value",
            "formatted_value",
        ],
        metric_utt_rows,
    )
    write_csv(
        metric_dataset_csv_path,
        [
            "model",
            "prompt",
            "question",
            "dataset",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_value",
            "std_value",
            "std_err_value",
            "formatted_value",
        ],
        metric_dataset_rows,
    )
    write_csv(
        metric_overall_csv_path,
        [
            "model",
            "prompt",
            "question",
            "audio_condition",
            "metric",
            "sample_count",
            "mean_value",
            "std_value",
            "std_err_value",
            "formatted_value",
        ],
        metric_overall_rows,
    )

    for table_stem, table_rows in table_specs:
        table_csv_path = output_dir / f"{table_stem}.csv"
        table_md_path = output_dir / f"{table_stem}.md"
        write_csv(table_csv_path, TABLE_FIELDNAMES, table_rows)
        write_markdown_table(table_md_path, TABLE_FIELDNAMES, table_rows)

    write_json(
        summary_json_path,
        {
            "source_variants": [
                {
                    "slug": variant.slug,
                    "source_results_root": str(variant.source_results_root),
                    "audio_condition": variant.audio_condition,
                    "comparison_mode": variant.comparison_mode,
                    "metric": variant.metric_name,
                }
                for variant in source_variants
            ],
            "reference_results_root": str(reference_results_root),
            "additional_reference_results_roots": [
                str(root) for root in additional_reference_results_roots
            ],
            "comparison_count": len(all_comparisons),
            "ratio_point_count": len(ratio_points),
            "comparison_dataset_utt_rows": comparison_dataset_utt_rows,
            "comparison_utt_rows": comparison_utt_rows,
            "comparison_dataset_rows": comparison_dataset_rows,
            "comparison_overall_rows": comparison_overall_rows,
            "metric_dataset_utt_rows": metric_dataset_utt_rows,
            "metric_utt_rows": metric_utt_rows,
            "metric_dataset_rows": metric_dataset_rows,
            "metric_overall_rows": metric_overall_rows,
            "tables": [table_stem for table_stem, _ in table_specs],
            "warnings": warnings,
        },
    )

    print(f"[INFO] Saved {raw_point_csv_path}")
    print(f"[INFO] Saved {raw_point_json_path}")
    print(f"[INFO] Saved {ratio_point_csv_path}")
    print(f"[INFO] Saved {comparison_dataset_utt_csv_path}")
    print(f"[INFO] Saved {comparison_utt_csv_path}")
    print(f"[INFO] Saved {comparison_dataset_csv_path}")
    print(f"[INFO] Saved {comparison_overall_csv_path}")
    print(f"[INFO] Saved {metric_dataset_utt_csv_path}")
    print(f"[INFO] Saved {metric_utt_csv_path}")
    print(f"[INFO] Saved {metric_dataset_csv_path}")
    print(f"[INFO] Saved {metric_overall_csv_path}")
    for table_stem, _ in table_specs:
        print(f"[INFO] Saved {output_dir / f'{table_stem}.csv'}")
        print(f"[INFO] Saved {output_dir / f'{table_stem}.md'}")
    print(f"[INFO] Saved {summary_json_path}")

    for warning in warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
