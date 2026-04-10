import argparse
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
)


DATASET_NAMES = ("mintrec2", "meld", "seamless_interaction")
DIALOGUE_KEY_PATTERN = re.compile(r"^d\d+u\d+$")
FILE_CLIP_PATTERN = re.compile(r"^(?P<prefix>d\d+u\d+)_clip(?P<index>\d+)$")
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


def quantize_progress_ratio(progress_ratio: float, partitions: int) -> float:
    bucket_index = round(progress_ratio * partitions)
    bucket_index = max(1, min(partitions, bucket_index))
    return bucket_index / partitions


def plot_clip_to_final_scatter(
    utterance_metrics: Sequence[UtteranceMetrics],
    title: str,
    progress_partitions: int,
    output_path: Path,
    include_scatter: bool,
) -> None:
    x_values: list[float] = []
    y_values: list[float] = []
    grouped_values: dict[float, list[float]] = defaultdict(list)

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
            grouped_values[progress_ratio].append(similarity)

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
    plots_root = resolve_plots_root(results_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    model = SentenceTransformer(model_loc)

    model_roots = iter_model_roots(results_root)
    if not model_roots:
        raise FileNotFoundError(
            f"No model result folders with dataset structure found under {results_root}"
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
            )


if __name__ == "__main__":
    main()
