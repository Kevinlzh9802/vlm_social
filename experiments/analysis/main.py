import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from metrics import (
    DEFAULT_TURNOVER_THRESHOLDS,
    UtteranceMetrics,
    compute_semantic_turnover_ratio,
    compute_utterance_metrics,
)


DATASET_NAMES = ("mintrec2", "seamless_interaction")
DIALOGUE_KEY_PATTERN = re.compile(r"^d\d+u\d+$")
FILE_CLIP_PATTERN = re.compile(r"^(?P<prefix>d\d+u\d+)_clip(?P<index>\d+)$")


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
    return parser.parse_args()


def resolve_results_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Results root does not exist: {root}")
        return root

    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "results" / "qwen2.5",
        repo_root.parent / "results" / "qwen2.5",
        repo_root / "gestalt_bench" / "results" / "qwen2.5",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find results/qwen2.5 automatically. Pass --results-root explicitly."
    )


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
    assistant = value.get("assistant")
    if isinstance(assistant, str) and assistant.strip():
        return assistant.strip()
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


def build_base_figure_name(result_folder: Path, dataset_root: Path) -> str:
    return "__".join(result_folder.relative_to(dataset_root).parts)


def plot_clip_to_final_scatter(
    utterance_metrics: Sequence[UtteranceMetrics],
    title: str,
    output_path: Path,
) -> None:
    x_values: list[float] = []
    y_values: list[float] = []

    for metrics in utterance_metrics:
        x_values.extend(
            position / metrics.clip_count
            for position in range(1, metrics.clip_count + 1)
        )
        y_values.extend(metrics.clip_to_final_similarities)

    if not x_values:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, s=18, alpha=0.65, edgecolors="none")
    plt.title(title)
    plt.xlabel("Observed clip ratio")
    plt.ylabel("Cosine similarity to final clip")
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
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
    dataset_root: Path,
    turnover_thresholds: Sequence[float],
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

        base_name = build_base_figure_name(result_folder, dataset_root)
        title = str(result_folder.relative_to(dataset_root)).replace("\\", " / ")

        clip_to_final_path = dataset_root / f"{base_name}_clip_to_final_similarity.png"
        plot_clip_to_final_scatter(
            utterance_metrics=utterance_metrics,
            title=title,
            output_path=clip_to_final_path,
        )
        print(f"[INFO] Saved {clip_to_final_path}")

        neighbor_path = dataset_root / f"{base_name}_neighbor_similarity_by_clip_count.png"
        plot_neighbor_similarity_by_clip_count(
            utterance_metrics=utterance_metrics,
            title=f"{title} | Neighbor Similarity by Clip Count",
            output_path=neighbor_path,
        )
        print(f"[INFO] Saved {neighbor_path}")

        for turnover_threshold in turnover_thresholds:
            threshold_tag = f"{turnover_threshold:.2f}".replace(".", "p")
            turnover_path = (
                dataset_root
                / f"{base_name}_semantic_turnover_by_clip_count_t{threshold_tag}.png"
            )
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

    if args.model_path is not None:
        model_loc = str(args.model_path.expanduser().resolve())
        print(f"[INFO] Loading embedding model from local path: {model_loc}")
    else:
        model_loc = args.model
        print(f"[INFO] Loading embedding model by name: {model_loc}")
    model = SentenceTransformer(model_loc)

    for dataset_name in DATASET_NAMES:
        dataset_root = results_root / dataset_name
        if not dataset_root.is_dir():
            print(f"[WARN] Dataset folder not found, skipping: {dataset_root}")
            continue
        print(f"[INFO] Processing dataset: {dataset_root}")
        analyze_dataset(
            model=model,
            dataset_root=dataset_root,
            turnover_thresholds=args.turnover_thresholds,
        )


if __name__ == "__main__":
    main()
