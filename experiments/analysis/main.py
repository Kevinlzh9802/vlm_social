import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from metrics import (
    DEFAULT_TURNOVER_THRESHOLDS,
    UtteranceMetrics,
    compute_semantic_turnover_ratio,
    compute_utterance_metrics,
    compute_weighted_average_st_position,
)


DATASET_NAMES = ("mintrec2", "meld", "seamless_interaction")
TASK_NAMES = ("affordance", "intention")
UTTERANCE_GROUP_SIZES = (1, 2, 3)
DIALOGUE_KEY_PATTERN = re.compile(r"^d\d+u\d+$")
FILE_CLIP_PATTERN = re.compile(r"^(?P<prefix>.+)_clip_?(?P<index>\d+)$")
UTT_GROUP_PATTERN = re.compile(r"^(?P<size>[123])-utt_group$")
MODEL_SIZE_PATTERN = re.compile(r"(?P<size>\d+[Bb])(?:[_-]|$)")
DEFAULT_PROGRESS_PARTITIONS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute similarity-based analysis for Qwen2.5 batch results and save "
            "plots under each dataset root."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Path to results/qwen2.5. Defaults to an auto-discovered repo-relative path.",
    )
    parser.add_argument(
        "--additional-results-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional result root to include in aggregate plots. May be passed "
            "multiple times, e.g. a Gemini root stored outside --results-root."
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
        help=(
            "Path to a local directory containing a pre-downloaded "
            "SentenceTransformer model. When set, --model is ignored."
        ),
    )
    parser.add_argument(
        "--turnover-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_TURNOVER_THRESHOLDS),
        help=(
            "Semantic turnover thresholds t. Neighboring similarity below t counts "
            "as one turnover event. Default: 0.3 0.5 0.7 0.9."
        ),
    )
    parser.add_argument(
        "--progress-partitions",
        type=int,
        default=DEFAULT_PROGRESS_PARTITIONS,
        help=(
            "Number of fixed x-axis partitions for clip-progress aggregation in "
            "clip-to-final plots. Default: 20."
        ),
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Skip scatter points in per-folder clip-to-final plots and draw percentile lines only.",
    )
    parser.add_argument(
        "--human-annotation-summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional partial_to_full_percentiles.csv from "
            "human_annotation_similarity.py. When set, additional aggregate "
            "plots overlay human annotation partial-to-full similarity with "
            "model clip-to-final similarity under each utterance/task condition."
        ),
    )
    return parser.parse_args()


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


def resolve_additional_results_roots(explicit_roots: Sequence[Path]) -> list[Path]:
    roots: list[Path] = []
    for explicit_root in explicit_roots:
        root = explicit_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Additional results root does not exist: {root}")
        roots.append(root)
    return roots


def has_dataset_structure(root: Path) -> bool:
    return any((root / dataset_name).is_dir() for dataset_name in DATASET_NAMES)


def iter_model_roots(results_root: Path) -> List[Path]:
    if has_dataset_structure(results_root):
        return [results_root]

    model_roots: List[Path] = []
    for path in sorted(results_root.iterdir()):
        if path.name == "plots":
            continue
        if path.is_dir() and has_dataset_structure(path):
            model_roots.append(path)
    return model_roots


def resolve_plots_root(results_root: Path) -> Path:
    if has_dataset_structure(results_root):
        return (results_root.parent / "plots").resolve()
    return (results_root / "plots").resolve()


def iter_result_folders(dataset_root: Path) -> List[Path]:
    result_folders: List[Path] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_dir() and list(path.glob("batch*.json")):
            result_folders.append(path)
    return result_folders


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_dialogue_entries(payload: object) -> List[Tuple[str, object]]:
    found: List[Tuple[str, object]] = []

    def _walk(node: object, parent_key: str | None = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and DIALOGUE_KEY_PATTERN.fullmatch(key):
                    found.append((key, value))
                _walk(value, key if isinstance(key, str) else None)
        elif isinstance(node, list):
            if parent_key is None or DIALOGUE_KEY_PATTERN.fullmatch(parent_key) is None:
                grouped_by_file_prefix: dict[str, list[object]] = defaultdict(list)
                for item in node:
                    if not isinstance(item, dict):
                        continue
                    file_name = item.get("file")
                    if not isinstance(file_name, str):
                        continue
                    match = FILE_CLIP_PATTERN.fullmatch(file_name)
                    if match is not None:
                        grouped_by_file_prefix[match.group("prefix")].append(item)
                found.extend(sorted(grouped_by_file_prefix.items()))

            for item in node:
                _walk(item)

    _walk(payload)
    return found


def sort_key_for_clip_identifier(identifier: str) -> Tuple[int, str]:
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


def normalize_clip_dict_from_sequence(dialogue_key: str, value: Sequence[object]) -> Dict[int, str]:
    clips: Dict[int, str] = {}
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


def extract_ordered_clips(dialogue_key: str, dialogue_value: object) -> List[Tuple[int, str]]:
    clip_dict: Dict[int, str] = {}
    if isinstance(dialogue_value, Sequence) and not isinstance(dialogue_value, (str, bytes)):
        clip_dict = normalize_clip_dict_from_sequence(dialogue_key, dialogue_value)
    return sorted(clip_dict.items(), key=lambda item: item[0])


def build_plot_title(model_root: Path, dataset_root: Path, result_folder: Path) -> str:
    relative_path = str(result_folder.relative_to(dataset_root)).replace("\\", " / ")
    return f"{model_root.name} | {relative_path}"


def build_plot_output_dir(
    plots_root: Path,
    dataset_root: Path,
    result_folder: Path,
) -> Path:
    relative_parts = list(result_folder.relative_to(dataset_root).parts)
    if relative_parts and relative_parts[0] == "context":
        relative_parts = relative_parts[1:]

    output_dir = plots_root / dataset_root.name
    for part in relative_parts:
        output_dir /= part
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    if model_name.lower() == "gemini":
        leaf_name = result_folder.name.lower()
        for task_name in TASK_NAMES:
            suffix = f"_{task_name}_single-turn"
            if leaf_name.endswith(suffix):
                gemini_mode = result_folder.name[: -len(suffix)]
                return f"{model_name}-{gemini_mode}"
        return model_name

    if "ming-lite-omni" in model_name.lower():
        return model_name

    size_match = MODEL_SIZE_PATTERN.search(result_folder.name)
    if size_match is not None:
        return f"{model_name}-{size_match.group('size').upper()}"
    return model_name


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
        for position, similarity in enumerate(
            metrics.clip_to_final_similarities, start=1
        ):
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


def load_human_annotation_summary_csv(
    summary_csv_path: Path,
) -> dict[tuple[int, str], dict[str, list[tuple[float, float]]]]:
    summary_path = summary_csv_path.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Human annotation summary CSV does not exist: {summary_path}")

    human_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("dataset") != "all":
                continue

            prompt_name = str(row.get("prompt", "")).strip()
            utt_count_text = str(row.get("utt_count", "")).strip()
            if prompt_name not in TASK_NAMES or not utt_count_text.isdigit():
                continue

            case_key = (int(utt_count_text), prompt_name)
            try:
                progress_ratio = float(row["progress_ratio"])
                stat_values = {
                    "mean": float(row["mean_similarity"]),
                    "p25": float(row["percentile_25"]),
                    "p50": float(row["percentile_50"]),
                    "p75": float(row["percentile_75"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[WARN] Skipping malformed human summary row {row}: {exc}")
                continue

            for stat_name, stat_value in stat_values.items():
                human_summary[case_key][stat_name].append((progress_ratio, stat_value))

    return {
        case_key: {
            stat_name: sorted(points, key=lambda item: item[0])
            for stat_name, points in stat_rows.items()
        }
        for case_key, stat_rows in human_summary.items()
    }


def plot_clip_to_final_scatter(
    utterance_metrics: Sequence[UtteranceMetrics],
    title: str,
    progress_partitions: int,
    output_path: Path,
    include_scatter: bool,
) -> None:
    x_values: list[float] = []
    y_values: list[float] = []
    grouped_values = collect_clip_to_final_bins(
        utterance_metrics=utterance_metrics,
        progress_partitions=progress_partitions,
    )

    for metrics in utterance_metrics:
        for position, similarity in enumerate(
            metrics.clip_to_final_similarities, start=1
        ):
            progress_ratio = quantize_progress_ratio(
                position / metrics.clip_count,
                partitions=progress_partitions,
            )
            x_values.append(progress_ratio)
            y_values.append(similarity)

    if not x_values:
        return

    plt.figure(figsize=(8, 6))
    if include_scatter:
        plt.scatter(x_values, y_values, s=18, alpha=0.65, edgecolors="none")

    sorted_ratios = sorted(grouped_values)
    percentile_25 = [
        float(np.percentile(grouped_values[ratio], 25)) for ratio in sorted_ratios
    ]
    percentile_50 = [
        float(np.percentile(grouped_values[ratio], 50)) for ratio in sorted_ratios
    ]
    percentile_75 = [
        float(np.percentile(grouped_values[ratio], 75)) for ratio in sorted_ratios
    ]

    plt.plot(
        sorted_ratios,
        percentile_25,
        color="#E45756",
        linewidth=1.8,
        label="25th percentile",
    )
    plt.plot(
        sorted_ratios,
        percentile_50,
        color="#4C78A8",
        linewidth=1.8,
        label="50th percentile",
    )
    plt.plot(
        sorted_ratios,
        percentile_75,
        color="#72B7B2",
        linewidth=1.8,
        label="75th percentile",
    )

    plt.title(title)
    plt.xlabel(f"Observed clip ratio (rounded to nearest 1/{progress_partitions})")
    plt.ylabel("Cosine similarity to final clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_clip_to_final_mean(
    utterance_metrics: Sequence[UtteranceMetrics],
    title: str,
    progress_partitions: int,
    output_path: Path,
) -> None:
    grouped_values = collect_clip_to_final_bins(
        utterance_metrics=utterance_metrics,
        progress_partitions=progress_partitions,
    )
    if not grouped_values:
        return

    ratios, mean_values = mean_similarity_by_ratio(grouped_values)

    plt.figure(figsize=(8, 6))
    plt.plot(
        ratios,
        mean_values,
        color="#4C78A8",
        marker="o",
        linewidth=1.8,
        label="Mean",
    )
    plt.title(title)
    plt.xlabel(f"Observed clip ratio (rounded to nearest 1/{progress_partitions})")
    plt.ylabel("Average cosine similarity to final clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_clip_to_final_percentile_lines(
    case_metrics: dict[str, list[UtteranceMetrics]],
    percentile: int,
    utt_group_size: int,
    task_name: str,
    progress_partitions: int,
    output_path: Path,
) -> bool:
    plt.figure(figsize=(8, 6))
    plotted_any = False

    for model_label in sorted(case_metrics):
        grouped_values = collect_clip_to_final_bins(
            utterance_metrics=case_metrics[model_label],
            progress_partitions=progress_partitions,
        )
        if not grouped_values:
            print(
                f"[WARN] No usable utterances for combined percentile plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}"
            )
            continue

        ratios = sorted(grouped_values)
        percentile_values = [
            float(np.percentile(grouped_values[ratio], percentile))
            for ratio in ratios
        ]
        plt.plot(ratios, percentile_values, marker="o", linewidth=1.8, label=model_label)
        plotted_any = True

    if not plotted_any:
        plt.close()
        print(
            f"[WARN] No usable utterances for combined clip-to-final percentile plot: "
            f"utt={utt_group_size}, task={task_name}, percentile={percentile}"
        )
        return False

    plt.title(
        f"Combined Clip-to-Final Similarity | {utt_group_size}-utt | {task_name} | p{percentile}"
    )
    plt.xlabel(f"Observed clip ratio (rounded to nearest 1/{progress_partitions})")
    plt.ylabel("Cosine similarity to final clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_combined_clip_to_final_mean_lines(
    case_metrics: dict[str, list[UtteranceMetrics]],
    utt_group_size: int,
    task_name: str,
    progress_partitions: int,
    output_path: Path,
) -> bool:
    plt.figure(figsize=(8, 6))
    plotted_any = False

    for model_label in sorted(case_metrics):
        grouped_values = collect_clip_to_final_bins(
            utterance_metrics=case_metrics[model_label],
            progress_partitions=progress_partitions,
        )
        if not grouped_values:
            print(
                f"[WARN] No usable utterances for combined mean plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}"
            )
            continue

        ratios, mean_values = mean_similarity_by_ratio(grouped_values)
        plt.plot(ratios, mean_values, marker="o", linewidth=1.8, label=model_label)
        plotted_any = True

    if not plotted_any:
        plt.close()
        print(
            f"[WARN] No usable utterances for combined clip-to-final mean plot: "
            f"utt={utt_group_size}, task={task_name}"
        )
        return False

    plt.title(
        f"Combined Clip-to-Final Similarity | {utt_group_size}-utt | {task_name} | Mean"
    )
    plt.xlabel(f"Observed clip ratio (rounded to nearest 1/{progress_partitions})")
    plt.ylabel("Average cosine similarity to final clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_combined_human_model_partial_to_full_lines(
    case_metrics: dict[str, list[UtteranceMetrics]],
    human_case_summary: dict[str, list[tuple[float, float]]],
    stat_name: str,
    utt_group_size: int,
    task_name: str,
    progress_partitions: int,
    output_path: Path,
    percentile: int | None = None,
) -> bool:
    plt.figure(figsize=(9, 6))
    plotted_any = False
    color_map = plt.cm.get_cmap("tab10")
    if stat_name == "mean":
        title_suffix = "Mean"
        ylabel = "Average cosine similarity to full clip"
        legend_suffix = "mean"
    elif stat_name == "percentile" and percentile is not None:
        title_suffix = f"p{percentile}"
        ylabel = f"{percentile}th percentile cosine similarity to full clip"
        legend_suffix = f"p{percentile}"
    else:
        raise ValueError(f"Unsupported stat_name={stat_name} percentile={percentile}")

    for model_index, model_label in enumerate(sorted(case_metrics)):
        grouped_values = collect_clip_to_final_bins(
            utterance_metrics=case_metrics[model_label],
            progress_partitions=progress_partitions,
        )
        if not grouped_values:
            print(
                f"[WARN] No usable utterances for human/model aggregate plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}, stat={stat_name}"
            )
            continue

        if stat_name == "mean":
            ratios, stat_values = mean_similarity_by_ratio(grouped_values)
        else:
            ratios = sorted(grouped_values)
            stat_values = [
                float(np.percentile(grouped_values[ratio], percentile))
                for ratio in ratios
            ]

        plt.plot(
            ratios,
            stat_values,
            color=color_map(model_index % 10),
            marker="o",
            linewidth=1.8,
            label=f"{model_label} {legend_suffix}",
        )
        plotted_any = True

    human_stat_key = "mean" if stat_name == "mean" else f"p{percentile}"
    human_points = human_case_summary.get(human_stat_key, [])
    if human_points:
        ratios = [ratio for ratio, _ in human_points]
        values = [value for _, value in human_points]
        plt.plot(
            ratios,
            values,
            color="#111111",
            linestyle="--",
            marker="s",
            linewidth=2.2,
            label=f"Human annotations {human_stat_key}",
        )
        plotted_any = True
    else:
        print(
            f"[WARN] No human annotation summary for aggregate plot: "
            f"utt={utt_group_size}, task={task_name}, stat={human_stat_key}"
        )

    if not plotted_any:
        plt.close()
        print(
            f"[WARN] No usable data for combined human/model partial-to-full plot: "
            f"utt={utt_group_size}, task={task_name}, stat={human_stat_key}"
        )
        return False

    plt.title(
        f"Human and Model Partial-to-Full Similarity | {utt_group_size}-utt | {task_name} | {title_suffix}"
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


def plot_neighbor_similarity_by_clip_count(
    utterance_metrics: Sequence[UtteranceMetrics],
    title: str,
    output_path: Path,
) -> None:
    grouped_similarities: dict[int, list[float]] = defaultdict(list)

    for metrics in utterance_metrics:
        grouped_similarities[metrics.clip_count].extend(metrics.neighboring_similarities)

    if not grouped_similarities:
        return

    clip_counts = sorted(grouped_similarities)
    values = [grouped_similarities[clip_count] for clip_count in clip_counts]

    plt.figure(figsize=(9, 6))
    plt.boxplot(values, labels=[str(value) for value in clip_counts], showfliers=False)

    for position, clip_count in enumerate(clip_counts, start=1):
        plt.scatter(
            [position] * len(grouped_similarities[clip_count]),
            grouped_similarities[clip_count],
            s=12,
            alpha=0.45,
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("Utterance clip count")
    plt.ylabel("Neighboring cosine similarity")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_semantic_turnover_by_clip_count(
    utterance_metrics: Sequence[UtteranceMetrics],
    turnover_threshold: float,
    title: str,
    output_path: Path,
) -> None:
    grouped_turnover: dict[int, list[float]] = defaultdict(list)

    for metrics in utterance_metrics:
        grouped_turnover[metrics.clip_count].append(
            compute_semantic_turnover_ratio(
                clip_count=metrics.clip_count,
                neighboring_similarities=metrics.neighboring_similarities,
                turnover_threshold=turnover_threshold,
            )
        )

    if not grouped_turnover:
        return

    clip_counts = sorted(grouped_turnover)
    values = [grouped_turnover[clip_count] for clip_count in clip_counts]

    plt.figure(figsize=(9, 6))
    plt.boxplot(values, labels=[str(value) for value in clip_counts], showfliers=False)

    for position, clip_count in enumerate(clip_counts, start=1):
        plt.scatter(
            [position] * len(grouped_turnover[clip_count]),
            grouped_turnover[clip_count],
            s=16,
            alpha=0.5,
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("Utterance clip count")
    plt.ylabel("Semantic turnover / number of clips")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_st_threshold_lines(
    case_metrics: dict[str, list[UtteranceMetrics]],
    turnover_thresholds: Sequence[float],
    utt_group_size: int,
    task_name: str,
    output_path: Path,
) -> bool:
    plt.figure(figsize=(8, 6))
    plotted_any = False

    for model_label in sorted(case_metrics):
        model_metrics = case_metrics[model_label]
        if not model_metrics:
            print(
                f"[WARN] No usable utterances for ST-threshold plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}"
            )
            continue

        averages = []
        for threshold in turnover_thresholds:
            values = [
                compute_semantic_turnover_ratio(
                    clip_count=metrics.clip_count,
                    neighboring_similarities=metrics.neighboring_similarities,
                    turnover_threshold=threshold,
                )
                for metrics in model_metrics
            ]
            averages.append(float(np.mean(values)))

        plt.plot(
            turnover_thresholds,
            averages,
            marker="o",
            linewidth=1.8,
            label=model_label,
        )
        plotted_any = True

    if not plotted_any:
        plt.close()
        print(
            f"[WARN] No usable utterances for combined ST-threshold plot: "
            f"utt={utt_group_size}, task={task_name}"
        )
        return False

    plt.title(f"Average ST/Clip Count vs Threshold | {utt_group_size}-utt | {task_name}")
    plt.xlabel("Threshold t")
    plt.ylabel("Average(ST / number of clips)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def write_wastp_table(
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    output_csv_path: Path,
    output_md_path: Path,
) -> None:
    case_order = [
        (utt_group_size, task_name)
        for utt_group_size in UTTERANCE_GROUP_SIZES
        for task_name in TASK_NAMES
    ]

    model_labels = sorted(
        {
            model_label
            for case_metrics in combined_case_metrics.values()
            for model_label in case_metrics
        }
    )

    fieldnames = ["model"] + [
        f"{utt_group_size}utt_{task_name}" for utt_group_size, task_name in case_order
    ]

    rows: list[dict[str, str]] = []
    for model_label in model_labels:
        row: dict[str, str] = {"model": model_label}
        for utt_group_size, task_name in case_order:
            case_metrics = combined_case_metrics.get((utt_group_size, task_name), {})
            utterance_metrics = case_metrics.get(model_label, [])
            wastp_values = [
                compute_weighted_average_st_position(
                    clip_count=metrics.clip_count,
                    neighboring_similarities=metrics.neighboring_similarities,
                )
                for metrics in utterance_metrics
            ]
            valid_values = [value for value in wastp_values if value is not None]
            row[f"{utt_group_size}utt_{task_name}"] = (
                f"{float(np.mean(valid_values)):.4f}" if valid_values else ""
            )
        rows.append(row)

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    output_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_combined_outputs(
    plots_root: Path,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    turnover_thresholds: Sequence[float],
    progress_partitions: int,
    human_annotation_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None = None,
) -> None:
    for utt_group_size in UTTERANCE_GROUP_SIZES:
        for task_name in TASK_NAMES:
            case_key = (utt_group_size, task_name)
            case_metrics = combined_case_metrics.get(case_key, {})
            human_case_summary = (
                human_annotation_summary.get(case_key, {})
                if human_annotation_summary is not None
                else {}
            )

            for percentile in (25, 50, 75):
                output_path = (
                    plots_root
                    / f"combined_clip_to_final_percentile_p{percentile}_{utt_group_size}utt_{task_name}.png"
                )
                plot_combined_clip_to_final_percentile_lines(
                    case_metrics=case_metrics,
                    percentile=percentile,
                    utt_group_size=utt_group_size,
                    task_name=task_name,
                    progress_partitions=progress_partitions,
                    output_path=output_path,
                )

                if human_annotation_summary is not None:
                    human_model_output_path = (
                        plots_root
                        / f"combined_human_model_partial_to_full_p{percentile}_{utt_group_size}utt_{task_name}.png"
                    )
                    if plot_combined_human_model_partial_to_full_lines(
                        case_metrics=case_metrics,
                        human_case_summary=human_case_summary,
                        stat_name="percentile",
                        percentile=percentile,
                        utt_group_size=utt_group_size,
                        task_name=task_name,
                        progress_partitions=progress_partitions,
                        output_path=human_model_output_path,
                    ):
                        print(f"[INFO] Saved {human_model_output_path}")

            mean_output_path = (
                plots_root
                / f"combined_clip_to_final_mean_{utt_group_size}utt_{task_name}.png"
            )
            plot_combined_clip_to_final_mean_lines(
                case_metrics=case_metrics,
                utt_group_size=utt_group_size,
                task_name=task_name,
                progress_partitions=progress_partitions,
                output_path=mean_output_path,
            )

            if human_annotation_summary is not None:
                human_model_mean_output_path = (
                    plots_root
                    / f"combined_human_model_partial_to_full_mean_{utt_group_size}utt_{task_name}.png"
                )
                if plot_combined_human_model_partial_to_full_lines(
                    case_metrics=case_metrics,
                    human_case_summary=human_case_summary,
                    stat_name="mean",
                    utt_group_size=utt_group_size,
                    task_name=task_name,
                    progress_partitions=progress_partitions,
                    output_path=human_model_mean_output_path,
                ):
                    print(f"[INFO] Saved {human_model_mean_output_path}")

            st_output_path = (
                plots_root
                / f"combined_st_vs_threshold_{utt_group_size}utt_{task_name}.png"
            )
            plot_combined_st_threshold_lines(
                case_metrics=case_metrics,
                turnover_thresholds=turnover_thresholds,
                utt_group_size=utt_group_size,
                task_name=task_name,
                output_path=st_output_path,
            )

    write_wastp_table(
        combined_case_metrics=combined_case_metrics,
        output_csv_path=plots_root / "wastp_summary.csv",
        output_md_path=plots_root / "wastp_summary.md",
    )


def analyze_result_folder(
    model: SentenceTransformer,
    result_folder: Path,
) -> List[UtteranceMetrics]:
    utterance_metrics: List[UtteranceMetrics] = []
    json_paths = sorted(
        result_folder.glob("batch*.json"),
        key=lambda path: sort_key_for_clip_identifier(path.stem),
    )

    for json_path in json_paths:
        payload = load_json(json_path)
        for dialogue_key, dialogue_value in find_dialogue_entries(payload):
            ordered_clips = extract_ordered_clips(dialogue_key, dialogue_value)
            metrics = compute_utterance_metrics(model=model, ordered_clips=ordered_clips)
            if metrics is not None:
                utterance_metrics.append(metrics)

    return utterance_metrics


def analyze_dataset(
    model: SentenceTransformer,
    model_root: Path,
    dataset_root: Path,
    plots_root: Path,
    turnover_thresholds: Sequence[float],
    progress_partitions: int,
    include_scatter: bool,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
) -> None:
    result_folders = iter_result_folders(dataset_root)
    if not result_folders:
        print(f"[WARN] No folders with batch*.json found under {dataset_root}")
        return

    for result_folder in result_folders:
        utterance_metrics = analyze_result_folder(
            model=model,
            result_folder=result_folder,
        )
        if not utterance_metrics:
            print(f"[WARN] No usable utterances found in {result_folder}")
            continue

        title = build_plot_title(model_root=model_root, dataset_root=dataset_root, result_folder=result_folder)
        output_dir = build_plot_output_dir(
            plots_root=plots_root,
            dataset_root=dataset_root,
            result_folder=result_folder,
        )
        group_size = parse_group_size(result_folder)
        task_name = parse_task_name(result_folder)
        model_label = build_model_label(model_root=model_root, result_folder=result_folder)
        if group_size is not None and task_name is not None:
            combined_case_metrics[(group_size, task_name)][model_label].extend(utterance_metrics)

        if include_scatter:
            clip_to_final_path = output_dir / "clip_to_final_similarity_with_scatter.png"
            plot_clip_to_final_scatter(
                utterance_metrics=utterance_metrics,
                title=title,
                progress_partitions=progress_partitions,
                output_path=clip_to_final_path,
                include_scatter=True,
            )
            print(f"[INFO] Saved {clip_to_final_path}")

        clip_to_final_percentile_path = output_dir / "clip_to_final_similarity_percentiles_only.png"
        plot_clip_to_final_scatter(
            utterance_metrics=utterance_metrics,
            title=f"{title} | Percentiles Only",
            progress_partitions=progress_partitions,
            output_path=clip_to_final_percentile_path,
            include_scatter=False,
        )
        print(f"[INFO] Saved {clip_to_final_percentile_path}")

        clip_to_final_mean_path = output_dir / "clip_to_final_similarity_mean_only.png"
        plot_clip_to_final_mean(
            utterance_metrics=utterance_metrics,
            title=f"{title} | Mean Only",
            progress_partitions=progress_partitions,
            output_path=clip_to_final_mean_path,
        )
        print(f"[INFO] Saved {clip_to_final_mean_path}")

        neighbor_path = output_dir / "neighbor_similarity_by_clip_count.png"
        plot_neighbor_similarity_by_clip_count(
            utterance_metrics=utterance_metrics,
            title=f"{title} | Neighbor Similarity by Clip Count",
            output_path=neighbor_path,
        )
        print(f"[INFO] Saved {neighbor_path}")

        for turnover_threshold in turnover_thresholds:
            threshold_tag = f"{turnover_threshold:.2f}".replace(".", "p")
            turnover_path = output_dir / f"semantic_turnover_by_clip_count_t{threshold_tag}.png"
            plot_semantic_turnover_by_clip_count(
                utterance_metrics=utterance_metrics,
                turnover_threshold=turnover_threshold,
                title=f"{title} | Semantic Turnover by Clip Count (t={turnover_threshold:.2f})",
                output_path=turnover_path,
            )
            print(f"[INFO] Saved {turnover_path}")


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args.results_root)
    additional_results_roots = resolve_additional_results_roots(
        args.additional_results_root
    )
    plots_root = resolve_plots_root(results_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    human_annotation_summary = None
    if args.human_annotation_summary_csv is not None:
        human_annotation_summary = load_human_annotation_summary_csv(
            args.human_annotation_summary_csv
        )
        print(
            "[INFO] Loaded human annotation summary cases: "
            f"{len(human_annotation_summary)} from {args.human_annotation_summary_csv}"
        )

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    model = SentenceTransformer(model_loc)

    model_roots: list[Path] = []
    for current_results_root in [results_root, *additional_results_roots]:
        current_model_roots = iter_model_roots(current_results_root)
        if not current_model_roots:
            print(
                f"[WARN] No model result folders with dataset structure found under "
                f"{current_results_root}"
            )
            continue
        model_roots.extend(current_model_roots)

    if not model_roots:
        raise FileNotFoundError(
            "No model result folders with dataset structure found under "
            f"{[str(root) for root in [results_root, *additional_results_roots]]}"
        )

    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for model_root in model_roots:
        print(f"[INFO] Processing model: {model_root}")
        for dataset_name in DATASET_NAMES:
            dataset_root = model_root / dataset_name
            if not dataset_root.is_dir():
                print(f"[WARN] Dataset folder not found, skipping: {dataset_root}")
                continue
            print(f"[INFO] Processing dataset: {dataset_root}")
            analyze_dataset(
                model=model,
                model_root=model_root,
                dataset_root=dataset_root,
                plots_root=plots_root,
                turnover_thresholds=args.turnover_thresholds,
                progress_partitions=args.progress_partitions,
                include_scatter=not args.no_scatter,
                combined_case_metrics=combined_case_metrics,
            )

    generate_combined_outputs(
        plots_root=plots_root,
        combined_case_metrics=combined_case_metrics,
        turnover_thresholds=args.turnover_thresholds,
        progress_partitions=args.progress_partitions,
        human_annotation_summary=human_annotation_summary,
    )


if __name__ == "__main__":
    main()
