from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .human_model_similarity import (
        DATASET_NAMES,
        TASK_NAMES,
        build_model_label,
        extract_assistant_text,
        find_dialogue_entries,
        iter_model_roots,
        iter_result_folders,
        parse_group_size,
        parse_task_name,
        result_folder_priority,
        sort_key_for_clip_identifier,
        is_error_text,
    )
except ImportError:
    from human_model_similarity import (
        DATASET_NAMES,
        TASK_NAMES,
        build_model_label,
        extract_assistant_text,
        find_dialogue_entries,
        iter_model_roots,
        iter_result_folders,
        parse_group_size,
        parse_task_name,
        result_folder_priority,
        sort_key_for_clip_identifier,
        is_error_text,
    )


DEFAULT_MODEL = "all-MiniLM-L6-v2"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare task-2 manipulated-result answers against the corresponding "
            "full benchmark answers for the same models and clip ids."
        )
    )
    parser.add_argument(
        "--source-results-root",
        type=Path,
        required=True,
        help="Smaller result tree to iterate from, e.g. human_eval/task2/manipulation_full/results.",
    )
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        required=True,
        help="Full benchmark result tree to search across, e.g. gestalt_bench/results.",
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


def compute_clip_similarities(
    embedding_model: Any,
    source_clips: dict[tuple[str, str, str, int, str], ClipResult],
    reference_clips: dict[tuple[str, str, str, int, str], ClipResult],
) -> tuple[list[ClipComparison], list[str]]:
    pending_pairs: list[tuple[ClipResult, ClipResult]] = []
    warnings: list[str] = []

    for key, source_clip in sorted(source_clips.items()):
        reference_clip = reference_clips.get(key)
        if reference_clip is None:
            warnings.append(f"Missing reference clip for {key}")
            continue
        if not source_clip.text or is_error_text(source_clip.text):
            warnings.append(f"Invalid source text for {key}")
            continue
        if not reference_clip.text or is_error_text(reference_clip.text):
            warnings.append(f"Invalid reference text for {key}")
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


def build_point_rows(comparisons: Sequence[ClipComparison]) -> list[dict[str, object]]:
    return [
        {
            "model": comparison.model_label,
            "prompt": comparison.prompt_name,
            "dataset": comparison.dataset_name,
            "utt_count": comparison.utt_count,
            "clip_identifier": comparison.clip_identifier,
            "source_batch_json": comparison.source_batch_json,
            "reference_batch_json": comparison.reference_batch_json,
            "similarity": comparison.similarity,
        }
        for comparison in comparisons
    ]


def summarize_by_dataset_utt(
    comparisons: Sequence[ClipComparison],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for comparison in comparisons:
        grouped[
            (
                comparison.model_label,
                comparison.prompt_name,
                comparison.dataset_name,
                comparison.utt_count,
            )
        ].append(comparison.similarity)

    rows: list[dict[str, object]] = []
    for (model_label, prompt_name, dataset_name, utt_count), values in sorted(grouped.items()):
        rows.append(
            {
                "model": model_label,
                "prompt": prompt_name,
                "dataset": dataset_name,
                "utt_count": utt_count,
                "sample_count": len(values),
                "mean_similarity": float(np.mean(values)),
            }
        )
    return rows


def summarize_by_dataset(
    comparisons: Sequence[ClipComparison],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for comparison in comparisons:
        grouped[
            (
                comparison.model_label,
                comparison.prompt_name,
                comparison.dataset_name,
            )
        ].append(comparison.similarity)

    rows: list[dict[str, object]] = []
    for (model_label, prompt_name, dataset_name), values in sorted(grouped.items()):
        rows.append(
            {
                "model": model_label,
                "prompt": prompt_name,
                "dataset": dataset_name,
                "sample_count": len(values),
                "mean_similarity": float(np.mean(values)),
            }
        )
    return rows


def summarize_overall_weighted(
    comparisons: Sequence[ClipComparison],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for comparison in comparisons:
        grouped[(comparison.model_label, comparison.prompt_name)].append(comparison.similarity)

    rows: list[dict[str, object]] = []
    for (model_label, prompt_name), values in sorted(grouped.items()):
        rows.append(
            {
                "model": model_label,
                "prompt": prompt_name,
                "dataset": "all",
                "sample_count": len(values),
                "weighted_mean_similarity": float(np.mean(values)),
            }
        )
    return rows


def build_table_rows(
    dataset_rows: Sequence[dict[str, object]],
    overall_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    dataset_map = {
        (str(row["model"]), str(row["prompt"]), str(row["dataset"])): row
        for row in dataset_rows
    }
    overall_map = {
        (str(row["model"]), str(row["prompt"])): row
        for row in overall_rows
    }

    models_and_prompts = sorted(
        {
            (str(row["model"]), str(row["prompt"]))
            for row in dataset_rows
        }
        | {
            (str(row["model"]), str(row["prompt"]))
            for row in overall_rows
        }
    )

    rows: list[dict[str, object]] = []
    for model_label, prompt_name in models_and_prompts:
        row: dict[str, object] = {
            "model": model_label,
            "prompt": prompt_name,
        }
        for dataset_name in DATASET_NAMES:
            dataset_row = dataset_map.get((model_label, prompt_name, dataset_name))
            row[f"{dataset_name}_mean_similarity"] = (
                dataset_row["mean_similarity"] if dataset_row is not None else ""
            )
            row[f"{dataset_name}_sample_count"] = (
                dataset_row["sample_count"] if dataset_row is not None else ""
            )

        overall_row = overall_map.get((model_label, prompt_name))
        row["all_weighted_mean_similarity"] = (
            overall_row["weighted_mean_similarity"] if overall_row is not None else ""
        )
        row["all_sample_count"] = (
            overall_row["sample_count"] if overall_row is not None else ""
        )
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


def write_markdown_table(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_results_root = args.source_results_root.expanduser().resolve()
    reference_results_root = args.reference_results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_results_root.is_dir():
        raise FileNotFoundError(f"Source results root does not exist: {source_results_root}")
    if not reference_results_root.is_dir():
        raise FileNotFoundError(f"Reference results root does not exist: {reference_results_root}")

    from sentence_transformers import SentenceTransformer

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    embedding_model = SentenceTransformer(model_loc)

    print(f"[INFO] Indexing source results from {source_results_root}")
    source_clips, source_warnings = index_result_clips(source_results_root)
    print(f"[INFO] Indexed {len(source_clips)} source clips")

    print(f"[INFO] Indexing reference results from {reference_results_root}")
    reference_clips, reference_warnings = index_result_clips(reference_results_root)
    print(f"[INFO] Indexed {len(reference_clips)} reference clips")

    comparisons, comparison_warnings = compute_clip_similarities(
        embedding_model=embedding_model,
        source_clips=source_clips,
        reference_clips=reference_clips,
    )
    print(f"[INFO] Computed {len(comparisons)} clip comparisons")

    point_rows = build_point_rows(comparisons)
    dataset_utt_rows = summarize_by_dataset_utt(comparisons)
    dataset_rows = summarize_by_dataset(comparisons)
    overall_rows = summarize_overall_weighted(comparisons)
    table_rows = build_table_rows(dataset_rows, overall_rows)

    warnings = source_warnings + reference_warnings + comparison_warnings

    point_csv_path = output_dir / "manipulation_result_similarity_points.csv"
    point_json_path = output_dir / "manipulation_result_similarity_points.json"
    dataset_utt_csv_path = output_dir / "manipulation_result_similarity_by_dataset_utt.csv"
    dataset_csv_path = output_dir / "manipulation_result_similarity_by_dataset.csv"
    overall_csv_path = output_dir / "manipulation_result_similarity_overall_weighted.csv"
    table_csv_path = output_dir / "manipulation_result_similarity_table.csv"
    table_md_path = output_dir / "manipulation_result_similarity_table.md"
    summary_json_path = output_dir / "manipulation_result_similarity_summary.json"

    write_csv(
        point_csv_path,
        [
            "model",
            "prompt",
            "dataset",
            "utt_count",
            "clip_identifier",
            "source_batch_json",
            "reference_batch_json",
            "similarity",
        ],
        point_rows,
    )
    write_json(
        point_json_path,
        {
            "source_results_root": str(source_results_root),
            "reference_results_root": str(reference_results_root),
            "point_count": len(point_rows),
            "points": point_rows,
        },
    )
    write_csv(
        dataset_utt_csv_path,
        ["model", "prompt", "dataset", "utt_count", "sample_count", "mean_similarity"],
        dataset_utt_rows,
    )
    write_csv(
        dataset_csv_path,
        ["model", "prompt", "dataset", "sample_count", "mean_similarity"],
        dataset_rows,
    )
    write_csv(
        overall_csv_path,
        ["model", "prompt", "dataset", "sample_count", "weighted_mean_similarity"],
        overall_rows,
    )
    table_fieldnames = [
        "model",
        "prompt",
        "mintrec2_mean_similarity",
        "mintrec2_sample_count",
        "meld_mean_similarity",
        "meld_sample_count",
        "seamless_interaction_mean_similarity",
        "seamless_interaction_sample_count",
        "all_weighted_mean_similarity",
        "all_sample_count",
    ]
    write_csv(table_csv_path, table_fieldnames, table_rows)
    write_markdown_table(table_md_path, table_fieldnames, table_rows)
    write_json(
        summary_json_path,
        {
            "source_results_root": str(source_results_root),
            "reference_results_root": str(reference_results_root),
            "comparison_count": len(comparisons),
            "dataset_utt_rows": dataset_utt_rows,
            "dataset_rows": dataset_rows,
            "overall_rows": overall_rows,
            "table_rows": table_rows,
            "warnings": warnings,
        },
    )

    print(f"[INFO] Saved {point_csv_path}")
    print(f"[INFO] Saved {point_json_path}")
    print(f"[INFO] Saved {dataset_utt_csv_path}")
    print(f"[INFO] Saved {dataset_csv_path}")
    print(f"[INFO] Saved {overall_csv_path}")
    print(f"[INFO] Saved {table_csv_path}")
    print(f"[INFO] Saved {table_md_path}")
    print(f"[INFO] Saved {summary_json_path}")

    for warning in warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
