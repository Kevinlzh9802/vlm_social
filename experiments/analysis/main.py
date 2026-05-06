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
DEFAULT_MIN_BIN_SAMPLES = 5
PFS_ERROR_BAR_OFFSET_FRACTION = 0.12
STR_ERROR_BAR_OFFSET_FRACTION = 0.08
PFS_TASK_QUESTION_LABELS = {
    "intention": "Q_1",
    "affordance": "Q_2",
}
PFS_TITLE_FONT_SIZE = 25
PFS_AXIS_LABEL_FONT_SIZE = 20
PFS_TICK_LABEL_FONT_SIZE = 20
PFS_LEGEND_FONT_SIZE = 18
STR_TITLE_FONT_SIZE = 25
STR_AXIS_LABEL_FONT_SIZE = 20
STR_TICK_LABEL_FONT_SIZE = 20
STR_LEGEND_FONT_SIZE = 18


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
            "model clip-to-final similarity under each utterance/task condition. "
            "The sibling partial_to_full_points.csv is used for human ST overlay "
            "when it includes neighboring_similarity_to_next."
        ),
    )
    parser.add_argument(
        "--save-plot-data",
        action="store_true",
        help=(
            "Save computed aggregate similarity data so combined plots can be "
            "regenerated without reading result JSON files or embedding text again."
        ),
    )
    parser.add_argument(
        "--plot-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory for --save-plot-data outputs. Defaults to "
            "<plots-root>/plot_data."
        ),
    )
    parser.add_argument(
        "--from-plot-data",
        type=Path,
        default=None,
        help=(
            "Regenerate combined plots from analysis_plot_data.json and skip "
            "reading result JSON files or embedding text."
        ),
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=None,
        help=(
            "Directory for plot outputs. Defaults to the resolved results-root "
            "plot directory, or the plots_root stored in --from-plot-data."
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


def filter_grouped_values_for_plot(
    grouped_values: dict[float, list[float]],
    min_bin_samples: int = DEFAULT_MIN_BIN_SAMPLES,
) -> dict[float, list[float]]:
    return {
        ratio: values
        for ratio, values in grouped_values.items()
        if len(values) >= min_bin_samples
    }


def mean_similarity_by_ratio(grouped_values: dict[float, list[float]]) -> tuple[list[float], list[float]]:
    ratios = sorted(grouped_values)
    means = [float(np.mean(grouped_values[ratio])) for ratio in ratios]
    return ratios, means


def mean_and_std_by_ratio(
    grouped_values: dict[float, list[float]],
) -> tuple[list[float], list[float], list[float]]:
    ratios = sorted(grouped_values)
    means = [float(np.mean(grouped_values[ratio])) for ratio in ratios]
    stds = [float(np.std(grouped_values[ratio])) for ratio in ratios]
    return ratios, means, stds


def resolve_mean_variant(
    sigma_multiplier: float | None,
) -> tuple[str, str]:
    if sigma_multiplier is None:
        return "", "Mean"
    if sigma_multiplier == 1:
        return "_1sigma", "Mean +/- 1 sigma"
    if sigma_multiplier == 2:
        return "_2sigma", "Mean +/- 2 sigma"
    raise ValueError(f"Unsupported sigma multiplier: {sigma_multiplier}")


def format_pfs_plot_title(
    task_name: str,
    stat_name: str,
    percentile: int | None,
    dataset_name: str | None = None,
) -> str:
    question_label = PFS_TASK_QUESTION_LABELS.get(task_name, task_name)
    if stat_name == "mean":
        title = f"Mean PFS for ${question_label}$"
    elif stat_name == "percentile" and percentile is not None:
        title = f"p{percentile} PFS for ${question_label}$"
    else:
        raise ValueError(f"Unsupported PFS stat_name={stat_name} percentile={percentile}")
    if dataset_name is not None:
        return f"{title} | {dataset_name}"
    return title


def format_str_plot_title(task_name: str, dataset_name: str | None = None) -> str:
    question_label = PFS_TASK_QUESTION_LABELS.get(task_name, task_name)
    title = f"Average STR for ${question_label}$"
    if dataset_name is not None:
        return f"{title} | {dataset_name}"
    return title


def compute_series_x_offset(
    series_index: int,
    series_count: int,
    x_step: float,
    sigma_multiplier: float | None,
    offset_fraction: float,
) -> float:
    if sigma_multiplier is None or series_count <= 1:
        return 0.0
    return (
        series_index - (series_count - 1) / 2
    ) * x_step * offset_fraction


def offset_x_values(x_values: Sequence[float], x_offset: float) -> list[float]:
    if x_offset == 0.0:
        return [float(value) for value in x_values]
    return [float(value) + x_offset for value in x_values]


def min_positive_x_step(x_values: Sequence[float]) -> float:
    sorted_values = sorted({float(value) for value in x_values})
    steps = [
        next_value - current_value
        for current_value, next_value in zip(sorted_values, sorted_values[1:])
        if next_value > current_value
    ]
    return min(steps) if steps else 1.0


def plot_line_with_optional_error_bars(
    x_values: Sequence[float],
    y_values: Sequence[float],
    std_values: Sequence[float] | None,
    sigma_multiplier: float | None,
    **plot_kwargs: object,
) -> None:
    x_offset = float(plot_kwargs.pop("x_offset", 0.0))
    plot_kwargs.pop("error_color", None)
    plotted_x_values = offset_x_values(x_values, x_offset)
    (line,) = plt.plot(plotted_x_values, y_values, **plot_kwargs)
    if sigma_multiplier is None or std_values is None:
        return

    yerr = np.asarray(std_values, dtype=float) * float(sigma_multiplier)
    plt.errorbar(
        plotted_x_values,
        y_values,
        yerr=yerr,
        fmt="none",
        ecolor=line.get_color(),
        capsize=3,
        elinewidth=1.1,
    )


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


def load_human_annotation_summary_csv_by_dataset(
    summary_csv_path: Path,
) -> dict[str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]]:
    summary_path = summary_csv_path.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Human annotation summary CSV does not exist: {summary_path}")

    human_summary_by_dataset: dict[
        str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            dataset_name = str(row.get("dataset", "")).strip()
            if dataset_name not in DATASET_NAMES:
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
                human_summary_by_dataset[dataset_name][case_key][stat_name].append(
                    (progress_ratio, stat_value)
                )

    return {
        dataset_name: {
            case_key: {
                stat_name: sorted(points, key=lambda item: item[0])
                for stat_name, points in stat_rows.items()
            }
            for case_key, stat_rows in dataset_rows.items()
        }
        for dataset_name, dataset_rows in human_summary_by_dataset.items()
    }


def infer_human_annotation_points_csv(summary_csv_path: Path) -> Path:
    return summary_csv_path.expanduser().resolve().parent / "partial_to_full_points.csv"


def load_human_annotation_turnover_points_csv(
    summary_csv_path: Path,
) -> dict[tuple[int, str], list[UtteranceMetrics]]:
    points_path = infer_human_annotation_points_csv(summary_csv_path)
    if not points_path.is_file():
        print(
            "[WARN] Human annotation point CSV does not exist; "
            f"skipping human ST overlay: {points_path}"
        )
        return {}

    grouped_rows: dict[
        tuple[int, str, int, int],
        list[tuple[int, float, float | None]],
    ] = defaultdict(list)
    with points_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "neighboring_similarity_to_next" not in (reader.fieldnames or []):
            print(
                "[WARN] Human annotation point CSV has no "
                "neighboring_similarity_to_next column; rerun "
                "human_annotation_similarity.py to enable human ST overlay."
            )
            return {}

        for row in reader:
            if row.get("dataset") != "all":
                continue
            prompt_name = str(row.get("prompt", "")).strip()
            utt_count_text = str(row.get("utt_count", "")).strip()
            if prompt_name not in TASK_NAMES or not utt_count_text.isdigit():
                continue

            try:
                utt_count = int(utt_count_text)
                utterance_index = int(row["utterance_index"])
                clip_count = int(row["clip_count"])
                clip_position = int(row["clip_position"])
                similarity_to_full = float(row["similarity_to_full"])
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[WARN] Skipping malformed human point row {row}: {exc}")
                continue

            neighboring_text = str(row.get("neighboring_similarity_to_next", "")).strip()
            neighboring_similarity = None
            if neighboring_text:
                try:
                    neighboring_similarity = float(neighboring_text)
                except ValueError as exc:
                    print(f"[WARN] Skipping malformed human neighbor value {row}: {exc}")
                    continue

            grouped_rows[(utt_count, prompt_name, utterance_index, clip_count)].append(
                (clip_position, similarity_to_full, neighboring_similarity)
            )

    case_metrics: dict[tuple[int, str], list[UtteranceMetrics]] = defaultdict(list)
    for (utt_count, prompt_name, _utterance_index, clip_count), point_rows in sorted(
        grouped_rows.items()
    ):
        ordered_rows = sorted(point_rows, key=lambda item: item[0])
        clip_to_final_similarities = [similarity for _, similarity, _ in ordered_rows]
        neighboring_similarities = [
            similarity
            for _, _, similarity in ordered_rows
            if similarity is not None
        ]
        if len(clip_to_final_similarities) != clip_count:
            print(
                "[WARN] Skipping incomplete human annotation utterance for ST overlay: "
                f"utt={utt_count}, task={prompt_name}, expected={clip_count}, "
                f"actual={len(clip_to_final_similarities)}"
            )
            continue
        if len(neighboring_similarities) != clip_count - 1:
            print(
                "[WARN] Skipping human annotation utterance with incomplete neighbor "
                f"similarities: utt={utt_count}, task={prompt_name}, "
                f"expected={clip_count - 1}, actual={len(neighboring_similarities)}"
            )
            continue
        case_metrics[(utt_count, prompt_name)].append(
            UtteranceMetrics(
                clip_count=clip_count,
                clip_to_final_similarities=clip_to_final_similarities,
                neighboring_similarities=neighboring_similarities,
            )
        )

    return dict(case_metrics)


def load_human_annotation_turnover_points_csv_by_dataset(
    summary_csv_path: Path,
) -> dict[str, dict[tuple[int, str], list[UtteranceMetrics]]]:
    points_path = infer_human_annotation_points_csv(summary_csv_path)
    if not points_path.is_file():
        print(
            "[WARN] Human annotation point CSV does not exist; "
            f"skipping per-dataset human ST overlay: {points_path}"
        )
        return {}

    grouped_rows: dict[
        tuple[str, int, str, int, int],
        list[tuple[int, float, float | None]],
    ] = defaultdict(list)
    with points_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "neighboring_similarity_to_next" not in (reader.fieldnames or []):
            print(
                "[WARN] Human annotation point CSV has no "
                "neighboring_similarity_to_next column; rerun "
                "human_annotation_similarity.py to enable per-dataset human ST overlay."
            )
            return {}

        for row in reader:
            dataset_name = str(row.get("dataset", "")).strip()
            if dataset_name not in DATASET_NAMES:
                continue
            prompt_name = str(row.get("prompt", "")).strip()
            utt_count_text = str(row.get("utt_count", "")).strip()
            if prompt_name not in TASK_NAMES or not utt_count_text.isdigit():
                continue

            try:
                utt_count = int(utt_count_text)
                utterance_index = int(row["utterance_index"])
                clip_count = int(row["clip_count"])
                clip_position = int(row["clip_position"])
                similarity_to_full = float(row["similarity_to_full"])
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[WARN] Skipping malformed human point row {row}: {exc}")
                continue

            neighboring_text = str(row.get("neighboring_similarity_to_next", "")).strip()
            neighboring_similarity = None
            if neighboring_text:
                try:
                    neighboring_similarity = float(neighboring_text)
                except ValueError as exc:
                    print(f"[WARN] Skipping malformed human neighbor value {row}: {exc}")
                    continue

            grouped_rows[
                (dataset_name, utt_count, prompt_name, utterance_index, clip_count)
            ].append((clip_position, similarity_to_full, neighboring_similarity))

    dataset_case_metrics: dict[str, dict[tuple[int, str], list[UtteranceMetrics]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (
        dataset_name,
        utt_count,
        prompt_name,
        _utterance_index,
        clip_count,
    ), point_rows in sorted(grouped_rows.items()):
        ordered_rows = sorted(point_rows, key=lambda item: item[0])
        clip_to_final_similarities = [similarity for _, similarity, _ in ordered_rows]
        neighboring_similarities = [
            similarity
            for _, _, similarity in ordered_rows
            if similarity is not None
        ]
        if len(clip_to_final_similarities) != clip_count:
            print(
                "[WARN] Skipping incomplete per-dataset human annotation utterance "
                f"for ST overlay: dataset={dataset_name}, utt={utt_count}, "
                f"task={prompt_name}, expected={clip_count}, "
                f"actual={len(clip_to_final_similarities)}"
            )
            continue
        if len(neighboring_similarities) != clip_count - 1:
            print(
                "[WARN] Skipping per-dataset human annotation utterance with "
                f"incomplete neighbor similarities: dataset={dataset_name}, "
                f"utt={utt_count}, task={prompt_name}, expected={clip_count - 1}, "
                f"actual={len(neighboring_similarities)}"
            )
            continue
        dataset_case_metrics[dataset_name][(utt_count, prompt_name)].append(
            UtteranceMetrics(
                clip_count=clip_count,
                clip_to_final_similarities=clip_to_final_similarities,
                neighboring_similarities=neighboring_similarities,
            )
        )

    return {
        dataset_name: dict(case_metrics)
        for dataset_name, case_metrics in dataset_case_metrics.items()
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
    grouped_values = filter_grouped_values_for_plot(
        collect_clip_to_final_bins(
            utterance_metrics=utterance_metrics,
            progress_partitions=progress_partitions,
        )
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

    if not grouped_values:
        plt.title(title)
        plt.xlabel(f"Observed clip ratio (rounded to nearest 1/{progress_partitions})")
        plt.ylabel("Cosine similarity to final clip")
        plt.xlim(0.0, 1.02)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

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
    sigma_multiplier: float | None = None,
) -> None:
    grouped_values = filter_grouped_values_for_plot(
        collect_clip_to_final_bins(
            utterance_metrics=utterance_metrics,
            progress_partitions=progress_partitions,
        )
    )
    if not grouped_values:
        return

    ratios, mean_values, std_values = mean_and_std_by_ratio(grouped_values)

    plt.figure(figsize=(8, 6))
    plot_line_with_optional_error_bars(
        ratios,
        mean_values,
        std_values,
        sigma_multiplier,
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
    model_labels = sorted(case_metrics)

    for model_label in model_labels:
        grouped_values = filter_grouped_values_for_plot(
            collect_clip_to_final_bins(
                utterance_metrics=case_metrics[model_label],
                progress_partitions=progress_partitions,
            )
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
    sigma_multiplier: float | None = None,
) -> bool:
    plt.figure(figsize=(8, 6))
    plotted_any = False
    color_map = matplotlib.colormaps["tab10"]
    model_labels = sorted(case_metrics)
    pfs_x_step = 1.0 / progress_partitions

    for model_index, model_label in enumerate(model_labels):
        grouped_values = filter_grouped_values_for_plot(
            collect_clip_to_final_bins(
                utterance_metrics=case_metrics[model_label],
                progress_partitions=progress_partitions,
            )
        )
        if not grouped_values:
            print(
                f"[WARN] No usable utterances for combined mean plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}"
            )
            continue

        ratios, mean_values, std_values = mean_and_std_by_ratio(grouped_values)
        plot_line_with_optional_error_bars(
            ratios,
            mean_values,
            std_values,
            sigma_multiplier,
            color=color_map(model_index % 10),
            x_offset=compute_series_x_offset(
                model_index,
                len(model_labels),
                pfs_x_step,
                sigma_multiplier,
                PFS_ERROR_BAR_OFFSET_FRACTION,
            ),
            marker="o",
            linewidth=1.8,
            label=model_label,
        )
        plotted_any = True

    if not plotted_any:
        plt.close()
        print(
            f"[WARN] No usable utterances for combined clip-to-final mean plot: "
            f"utt={utt_group_size}, task={task_name}"
        )
        return False

    _, mean_title = resolve_mean_variant(sigma_multiplier)
    plt.title(
        f"Combined Clip-to-Final Similarity | {utt_group_size}-utt | {task_name} | {mean_title}"
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
    human_case_turnover_metrics: Sequence[UtteranceMetrics] | None,
    stat_name: str,
    utt_group_size: int,
    task_name: str,
    progress_partitions: int,
    output_path: Path,
    percentile: int | None = None,
    sigma_multiplier: float | None = None,
    dataset_name: str | None = None,
) -> bool:
    plt.figure(figsize=(10, 7))
    plotted_any = False
    color_map = matplotlib.colormaps["tab10"]
    if stat_name != "mean" and not (stat_name == "percentile" and percentile is not None):
        raise ValueError(f"Unsupported stat_name={stat_name} percentile={percentile}")

    model_labels = sorted(case_metrics)
    series_count = len(model_labels) + (
        1 if human_case_turnover_metrics is not None or human_case_summary else 0
    )
    pfs_x_step = 1.0 / progress_partitions

    for model_index, model_label in enumerate(model_labels):
        grouped_values = filter_grouped_values_for_plot(
            collect_clip_to_final_bins(
                utterance_metrics=case_metrics[model_label],
                progress_partitions=progress_partitions,
            )
        )
        if not grouped_values:
            print(
                f"[WARN] No usable utterances for human/model aggregate plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}, stat={stat_name}"
            )
            continue

        if stat_name == "mean":
            ratios, stat_values, std_values = mean_and_std_by_ratio(grouped_values)
        else:
            ratios = sorted(grouped_values)
            stat_values = [
                float(np.percentile(grouped_values[ratio], percentile))
                for ratio in ratios
            ]
            std_values = None

        plot_line_with_optional_error_bars(
            ratios,
            stat_values,
            std_values,
            sigma_multiplier if stat_name == "mean" else None,
            color=color_map(model_index % 10),
            x_offset=compute_series_x_offset(
                model_index,
                series_count,
                pfs_x_step,
                sigma_multiplier if stat_name == "mean" else None,
                PFS_ERROR_BAR_OFFSET_FRACTION,
            ),
            marker="o",
            linewidth=1.8,
            label=model_label,
        )
        plotted_any = True

    human_stat_key = "mean" if stat_name == "mean" else f"p{percentile}"
    if human_case_turnover_metrics:
        human_grouped_values = filter_grouped_values_for_plot(
            collect_clip_to_final_bins(
                utterance_metrics=human_case_turnover_metrics,
                progress_partitions=progress_partitions,
            )
        )
        if human_grouped_values:
            if stat_name == "mean":
                ratios, values, std_values = mean_and_std_by_ratio(human_grouped_values)
            else:
                ratios = sorted(human_grouped_values)
                values = [
                    float(np.percentile(human_grouped_values[ratio], percentile))
                    for ratio in ratios
                ]
                std_values = None
            plot_line_with_optional_error_bars(
                ratios,
                values,
                std_values,
                sigma_multiplier if stat_name == "mean" else None,
                color="#111111",
                x_offset=compute_series_x_offset(
                    len(model_labels),
                    series_count,
                    pfs_x_step,
                    sigma_multiplier if stat_name == "mean" else None,
                    PFS_ERROR_BAR_OFFSET_FRACTION,
                ),
                linestyle="--",
                marker="s",
                linewidth=2.2,
                label="Human annotations",
            )
            plotted_any = True
        else:
            print(
                f"[WARN] No human annotation bins met min sample count for aggregate plot: "
                f"utt={utt_group_size}, task={task_name}, stat={human_stat_key}"
            )
    else:
        human_points = human_case_summary.get(human_stat_key, [])
        if human_points:
            ratios = [ratio for ratio, _ in human_points]
            values = [value for _, value in human_points]
            plot_line_with_optional_error_bars(
                ratios,
                values,
                None,
                None,
                color="#111111",
                linestyle="--",
                marker="s",
                linewidth=2.2,
                label="Human annotations",
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
        format_pfs_plot_title(
            task_name=task_name,
            stat_name=stat_name,
            percentile=percentile,
            dataset_name=dataset_name,
        ),
        fontsize=PFS_TITLE_FONT_SIZE,
    )
    plt.xlabel(
        f"Observed clip ratio (k/n, rounded to nearest 1/{progress_partitions})",
        fontsize=PFS_AXIS_LABEL_FONT_SIZE,
    )
    plt.ylabel("PFS", fontsize=PFS_AXIS_LABEL_FONT_SIZE)
    plt.xlim(0.0, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.xticks(fontsize=PFS_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=PFS_TICK_LABEL_FONT_SIZE)
    plt.legend(loc="lower right", fontsize=PFS_LEGEND_FONT_SIZE)
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
    plt.boxplot(
        values,
        tick_labels=[str(value) for value in clip_counts],
        showfliers=False,
    )

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
    plt.boxplot(
        values,
        tick_labels=[str(value) for value in clip_counts],
        showfliers=False,
    )

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
    human_case_metrics: Sequence[UtteranceMetrics] | None = None,
    sigma_multiplier: float | None = None,
    dataset_name: str | None = None,
) -> bool:
    plt.figure(figsize=(10, 7))
    plotted_any = False
    color_map = matplotlib.colormaps["tab10"]
    model_labels = sorted(case_metrics)
    series_count = len(model_labels) + (1 if human_case_metrics is not None else 0)
    threshold_x_step = min_positive_x_step(turnover_thresholds)

    for model_index, model_label in enumerate(model_labels):
        model_metrics = case_metrics[model_label]
        if not model_metrics:
            print(
                f"[WARN] No usable utterances for ST-threshold plot: "
                f"model={model_label}, utt={utt_group_size}, task={task_name}"
            )
            continue

        averages = []
        std_values = []
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
            std_values.append(float(np.std(values)))

        plot_line_with_optional_error_bars(
            turnover_thresholds,
            averages,
            std_values,
            sigma_multiplier,
            color=color_map(model_index % 10),
            x_offset=compute_series_x_offset(
                model_index,
                series_count,
                threshold_x_step,
                sigma_multiplier,
                STR_ERROR_BAR_OFFSET_FRACTION,
            ),
            marker="o",
            linewidth=1.8,
            label=model_label,
        )
        plotted_any = True

    if human_case_metrics is not None:
        if human_case_metrics:
            human_averages = []
            human_std_values = []
            for threshold in turnover_thresholds:
                human_values = [
                    compute_semantic_turnover_ratio(
                        clip_count=metrics.clip_count,
                        neighboring_similarities=metrics.neighboring_similarities,
                        turnover_threshold=threshold,
                    )
                    for metrics in human_case_metrics
                ]
                human_averages.append(float(np.mean(human_values)))
                human_std_values.append(float(np.std(human_values)))

            plot_line_with_optional_error_bars(
                turnover_thresholds,
                human_averages,
                human_std_values,
                sigma_multiplier,
                color="#111111",
                x_offset=compute_series_x_offset(
                    len(model_labels),
                    series_count,
                    threshold_x_step,
                    sigma_multiplier,
                    STR_ERROR_BAR_OFFSET_FRACTION,
                ),
                linestyle="--",
                marker="s",
                linewidth=2.2,
                label="Human annotations",
            )
            plotted_any = True
        else:
            print(
                f"[WARN] No human annotation ST data for aggregate plot: "
                f"utt={utt_group_size}, task={task_name}"
            )

    if not plotted_any:
        plt.close()
        print(
            f"[WARN] No usable data for combined ST-threshold plot: "
            f"utt={utt_group_size}, task={task_name}"
        )
        return False

    plt.title(
        format_str_plot_title(task_name, dataset_name=dataset_name),
        fontsize=STR_TITLE_FONT_SIZE,
    )
    plt.xlabel("Threshold t", fontsize=STR_AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Average STR", fontsize=STR_AXIS_LABEL_FONT_SIZE)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.xticks(fontsize=STR_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=STR_TICK_LABEL_FONT_SIZE)
    plt.legend(loc="upper left", fontsize=STR_LEGEND_FONT_SIZE)
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


def serialize_human_annotation_summary(
    human_annotation_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None,
) -> list[dict[str, object]]:
    if human_annotation_summary is None:
        return []

    rows: list[dict[str, object]] = []
    for (utt_group_size, task_name), stat_rows in sorted(human_annotation_summary.items()):
        for stat_name, points in sorted(stat_rows.items()):
            for progress_ratio, similarity in points:
                rows.append(
                    {
                        "utt_group_size": utt_group_size,
                        "task": task_name,
                        "stat": stat_name,
                        "progress_ratio": progress_ratio,
                        "similarity": similarity,
                    }
                )
    return rows


def serialize_human_annotation_turnover_metrics(
    human_annotation_turnover_metrics: dict[tuple[int, str], list[UtteranceMetrics]] | None,
) -> list[dict[str, object]]:
    if human_annotation_turnover_metrics is None:
        return []

    rows: list[dict[str, object]] = []
    for (utt_group_size, task_name), metrics_list in sorted(
        human_annotation_turnover_metrics.items()
    ):
        for metrics in metrics_list:
            rows.append(
                {
                    "utt_group_size": utt_group_size,
                    "task": task_name,
                    "clip_count": metrics.clip_count,
                    "clip_to_final_similarities": metrics.clip_to_final_similarities,
                    "neighboring_similarities": metrics.neighboring_similarities,
                }
            )
    return rows


def serialize_case_metrics(
    case_metrics_by_key: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for (utt_group_size, task_name), case_metrics in sorted(case_metrics_by_key.items()):
        models: list[dict[str, object]] = []
        for model_label, utterance_metrics in sorted(case_metrics.items()):
            models.append(
                {
                    "model": model_label,
                    "utterances": [
                        {
                            "clip_count": metrics.clip_count,
                            "clip_to_final_similarities": metrics.clip_to_final_similarities,
                            "neighboring_similarities": metrics.neighboring_similarities,
                        }
                        for metrics in utterance_metrics
                    ],
                }
            )
        cases.append(
            {
                "utt_group_size": utt_group_size,
                "task": task_name,
                "models": models,
            }
        )
    return cases


def load_case_metrics_from_rows(
    case_rows: Sequence[dict[str, object]],
) -> dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]:
    case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case in case_rows:
        utt_group_size = int(case["utt_group_size"])
        task_name = str(case["task"])
        for model_entry in case.get("models", []):
            model_label = str(model_entry["model"])
            for utterance in model_entry.get("utterances", []):
                case_metrics[(utt_group_size, task_name)][model_label].append(
                    UtteranceMetrics(
                        clip_count=int(utterance["clip_count"]),
                        clip_to_final_similarities=[
                            float(value)
                            for value in utterance["clip_to_final_similarities"]
                        ],
                        neighboring_similarities=[
                            float(value)
                            for value in utterance["neighboring_similarities"]
                        ],
                    )
                )
    return case_metrics


def serialize_human_annotation_summary_by_dataset(
    human_annotation_summary_by_dataset: dict[
        str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]
    ] | None,
) -> list[dict[str, object]]:
    if human_annotation_summary_by_dataset is None:
        return []
    return [
        {
            "dataset": dataset_name,
            "rows": serialize_human_annotation_summary(human_annotation_summary),
        }
        for dataset_name, human_annotation_summary in sorted(
            human_annotation_summary_by_dataset.items()
        )
    ]


def serialize_human_annotation_turnover_metrics_by_dataset(
    human_annotation_turnover_metrics_by_dataset: dict[
        str, dict[tuple[int, str], list[UtteranceMetrics]]
    ] | None,
) -> list[dict[str, object]]:
    if human_annotation_turnover_metrics_by_dataset is None:
        return []
    return [
        {
            "dataset": dataset_name,
            "rows": serialize_human_annotation_turnover_metrics(
                human_annotation_turnover_metrics
            ),
        }
        for dataset_name, human_annotation_turnover_metrics in sorted(
            human_annotation_turnover_metrics_by_dataset.items()
        )
    ]


def write_plot_data_json(
    output_path: Path,
    results_root: Path,
    additional_results_roots: Sequence[Path],
    plots_root: Path,
    model_location: str,
    turnover_thresholds: Sequence[float],
    progress_partitions: int,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    dataset_case_metrics: dict[
        str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]
    ],
    human_annotation_summary_csv: Path | None,
    human_annotation_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None,
    human_annotation_summary_by_dataset: dict[
        str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]
    ] | None,
    human_annotation_turnover_metrics: dict[tuple[int, str], list[UtteranceMetrics]] | None,
    human_annotation_turnover_metrics_by_dataset: dict[
        str, dict[tuple[int, str], list[UtteranceMetrics]]
    ] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "description": (
            "Aggregate analysis plot data. The model utterance metrics and human "
            "annotation summaries are sufficient to regenerate combined "
            "clip-to-final, human/model partial-to-full, ST-threshold, and "
            "WASTP outputs without reading result JSON files or embedding text "
            "again."
        ),
        "results_root": str(results_root.resolve()),
        "additional_results_roots": [
            str(root.resolve()) for root in additional_results_roots
        ],
        "plots_root": str(plots_root.resolve()),
        "model_location": model_location,
        "turnover_thresholds": [float(value) for value in turnover_thresholds],
        "progress_partitions": progress_partitions,
        "human_annotation_summary_csv": (
            str(human_annotation_summary_csv.expanduser().resolve())
            if human_annotation_summary_csv is not None
            else None
        ),
        "human_annotation_summary": serialize_human_annotation_summary(
            human_annotation_summary
        ),
        "human_annotation_summary_by_dataset": serialize_human_annotation_summary_by_dataset(
            human_annotation_summary_by_dataset
        ),
        "human_annotation_turnover_metrics": serialize_human_annotation_turnover_metrics(
            human_annotation_turnover_metrics
        ),
        "human_annotation_turnover_metrics_by_dataset": serialize_human_annotation_turnover_metrics_by_dataset(
            human_annotation_turnover_metrics_by_dataset
        ),
        "cases": serialize_case_metrics(combined_case_metrics),
        "dataset_cases": [
            {
                "dataset": dataset_name,
                "cases": serialize_case_metrics(case_metrics),
            }
            for dataset_name, case_metrics in sorted(dataset_case_metrics.items())
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_plot_data_points_csv(
    output_path: Path,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    progress_partitions: int,
) -> None:
    fieldnames = [
        "utt_group_size",
        "task",
        "model",
        "utterance_index",
        "clip_position",
        "clip_count",
        "progress_ratio_raw",
        "progress_ratio_binned",
        "clip_to_final_similarity",
        "neighboring_similarity_to_next",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (utt_group_size, task_name), case_metrics in sorted(combined_case_metrics.items()):
            for model_label, utterance_metrics in sorted(case_metrics.items()):
                for utterance_index, metrics in enumerate(utterance_metrics, start=1):
                    for clip_position, similarity in enumerate(
                        metrics.clip_to_final_similarities, start=1
                    ):
                        neighboring_similarity = (
                            metrics.neighboring_similarities[clip_position - 1]
                            if clip_position <= len(metrics.neighboring_similarities)
                            else ""
                        )
                        progress_ratio_raw = clip_position / metrics.clip_count
                        writer.writerow(
                            {
                                "utt_group_size": utt_group_size,
                                "task": task_name,
                                "model": model_label,
                                "utterance_index": utterance_index,
                                "clip_position": clip_position,
                                "clip_count": metrics.clip_count,
                                "progress_ratio_raw": progress_ratio_raw,
                                "progress_ratio_binned": quantize_progress_ratio(
                                    progress_ratio_raw,
                                    progress_partitions,
                                ),
                                "clip_to_final_similarity": similarity,
                                "neighboring_similarity_to_next": neighboring_similarity,
                            }
                        )


def write_plot_data_bins_csv(
    output_path: Path,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    progress_partitions: int,
) -> None:
    fieldnames = [
        "utt_group_size",
        "task",
        "model",
        "bin_index",
        "progress_ratio",
        "sample_count",
        "mean_similarity",
        "percentile_25",
        "percentile_50",
        "percentile_75",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (utt_group_size, task_name), case_metrics in sorted(combined_case_metrics.items()):
            for model_label, utterance_metrics in sorted(case_metrics.items()):
                grouped_values = collect_clip_to_final_bins(
                    utterance_metrics=utterance_metrics,
                    progress_partitions=progress_partitions,
                )
                for progress_ratio in sorted(grouped_values):
                    values = grouped_values[progress_ratio]
                    writer.writerow(
                        {
                            "utt_group_size": utt_group_size,
                            "task": task_name,
                            "model": model_label,
                            "bin_index": int(round(progress_ratio * progress_partitions)),
                            "progress_ratio": progress_ratio,
                            "sample_count": len(values),
                            "mean_similarity": float(np.mean(values)),
                            "percentile_25": float(np.percentile(values, 25)),
                            "percentile_50": float(np.percentile(values, 50)),
                            "percentile_75": float(np.percentile(values, 75)),
                        }
                    )


def write_human_plot_data_csv(
    output_path: Path,
    human_annotation_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None,
) -> None:
    fieldnames = [
        "utt_group_size",
        "task",
        "stat",
        "progress_ratio",
        "similarity",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in serialize_human_annotation_summary(human_annotation_summary):
            writer.writerow(row)


def write_analysis_plot_data(
    output_dir: Path,
    results_root: Path,
    additional_results_roots: Sequence[Path],
    plots_root: Path,
    model_location: str,
    turnover_thresholds: Sequence[float],
    progress_partitions: int,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    dataset_case_metrics: dict[
        str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]
    ],
    human_annotation_summary_csv: Path | None,
    human_annotation_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None,
    human_annotation_summary_by_dataset: dict[
        str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]
    ] | None,
    human_annotation_turnover_metrics: dict[tuple[int, str], list[UtteranceMetrics]] | None,
    human_annotation_turnover_metrics_by_dataset: dict[
        str, dict[tuple[int, str], list[UtteranceMetrics]]
    ] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_plot_data.json"
    points_csv_path = output_dir / "analysis_plot_points.csv"
    bins_csv_path = output_dir / "analysis_plot_bins.csv"
    human_csv_path = output_dir / "analysis_human_annotation_summary.csv"

    write_plot_data_json(
        output_path=json_path,
        results_root=results_root,
        additional_results_roots=additional_results_roots,
        plots_root=plots_root,
        model_location=model_location,
        turnover_thresholds=turnover_thresholds,
        progress_partitions=progress_partitions,
        combined_case_metrics=combined_case_metrics,
        dataset_case_metrics=dataset_case_metrics,
        human_annotation_summary_csv=human_annotation_summary_csv,
        human_annotation_summary=human_annotation_summary,
        human_annotation_summary_by_dataset=human_annotation_summary_by_dataset,
        human_annotation_turnover_metrics=human_annotation_turnover_metrics,
        human_annotation_turnover_metrics_by_dataset=human_annotation_turnover_metrics_by_dataset,
    )
    write_plot_data_points_csv(
        output_path=points_csv_path,
        combined_case_metrics=combined_case_metrics,
        progress_partitions=progress_partitions,
    )
    write_plot_data_bins_csv(
        output_path=bins_csv_path,
        combined_case_metrics=combined_case_metrics,
        progress_partitions=progress_partitions,
    )
    write_human_plot_data_csv(
        output_path=human_csv_path,
        human_annotation_summary=human_annotation_summary,
    )

    print(f"[INFO] Saved {json_path}")
    print(f"[INFO] Saved {points_csv_path}")
    print(f"[INFO] Saved {bins_csv_path}")
    print(f"[INFO] Saved {human_csv_path}")


def load_analysis_plot_data(
    input_path: Path,
) -> tuple[
    Path,
    dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    dict[str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]],
    list[float],
    int,
    dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None,
    dict[str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]] | None,
    dict[tuple[int, str], list[UtteranceMetrics]] | None,
    dict[str, dict[tuple[int, str], list[UtteranceMetrics]]] | None,
]:
    payload = json.loads(input_path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported plot data schema_version: {payload.get('schema_version')}"
        )

    plots_root = Path(payload["plots_root"]).expanduser().resolve()
    turnover_thresholds = [float(value) for value in payload["turnover_thresholds"]]
    progress_partitions = int(payload["progress_partitions"])
    combined_case_metrics = load_case_metrics_from_rows(payload.get("cases", []))
    dataset_case_metrics: dict[
        str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]
    ] = {}
    for dataset_entry in payload.get("dataset_cases", []):
        dataset_name = str(dataset_entry["dataset"])
        dataset_case_metrics[dataset_name] = load_case_metrics_from_rows(
            dataset_entry.get("cases", [])
        )

    human_rows = payload.get("human_annotation_summary", [])
    human_annotation_summary = None
    if human_rows:
        human_annotation_summary = defaultdict(lambda: defaultdict(list))
        for row in human_rows:
            human_annotation_summary[
                (int(row["utt_group_size"]), str(row["task"]))
            ][str(row["stat"])].append(
                (float(row["progress_ratio"]), float(row["similarity"]))
            )
        human_annotation_summary = {
            case_key: {
                stat_name: sorted(points, key=lambda item: item[0])
                for stat_name, points in stat_rows.items()
            }
            for case_key, stat_rows in human_annotation_summary.items()
        }

    human_annotation_summary_by_dataset = None
    human_summary_dataset_rows = payload.get("human_annotation_summary_by_dataset", [])
    if human_summary_dataset_rows:
        human_annotation_summary_by_dataset = {}
        for dataset_entry in human_summary_dataset_rows:
            dataset_summary = defaultdict(lambda: defaultdict(list))
            for row in dataset_entry.get("rows", []):
                dataset_summary[
                    (int(row["utt_group_size"]), str(row["task"]))
                ][str(row["stat"])].append(
                    (float(row["progress_ratio"]), float(row["similarity"]))
                )
            human_annotation_summary_by_dataset[str(dataset_entry["dataset"])] = {
                case_key: {
                    stat_name: sorted(points, key=lambda item: item[0])
                    for stat_name, points in stat_rows.items()
                }
                for case_key, stat_rows in dataset_summary.items()
            }

    human_turnover_rows = payload.get("human_annotation_turnover_metrics", [])
    human_annotation_turnover_metrics = None
    if human_turnover_rows:
        human_annotation_turnover_metrics = defaultdict(list)
        for row in human_turnover_rows:
            human_annotation_turnover_metrics[
                (int(row["utt_group_size"]), str(row["task"]))
            ].append(
                UtteranceMetrics(
                    clip_count=int(row["clip_count"]),
                    clip_to_final_similarities=[
                        float(value)
                        for value in row.get("clip_to_final_similarities", [])
                    ],
                    neighboring_similarities=[
                        float(value)
                        for value in row["neighboring_similarities"]
                    ],
                )
            )
        human_annotation_turnover_metrics = dict(human_annotation_turnover_metrics)

    human_annotation_turnover_metrics_by_dataset = None
    human_turnover_dataset_rows = payload.get(
        "human_annotation_turnover_metrics_by_dataset", []
    )
    if human_turnover_dataset_rows:
        human_annotation_turnover_metrics_by_dataset = {}
        for dataset_entry in human_turnover_dataset_rows:
            dataset_turnover_metrics = defaultdict(list)
            for row in dataset_entry.get("rows", []):
                dataset_turnover_metrics[
                    (int(row["utt_group_size"]), str(row["task"]))
                ].append(
                    UtteranceMetrics(
                        clip_count=int(row["clip_count"]),
                        clip_to_final_similarities=[
                            float(value)
                            for value in row.get("clip_to_final_similarities", [])
                        ],
                        neighboring_similarities=[
                            float(value)
                            for value in row["neighboring_similarities"]
                        ],
                    )
                )
            human_annotation_turnover_metrics_by_dataset[
                str(dataset_entry["dataset"])
            ] = dict(dataset_turnover_metrics)

    return (
        plots_root,
        combined_case_metrics,
        dataset_case_metrics,
        turnover_thresholds,
        progress_partitions,
        human_annotation_summary,
        human_annotation_summary_by_dataset,
        human_annotation_turnover_metrics,
        human_annotation_turnover_metrics_by_dataset,
    )


def generate_combined_outputs(
    plots_root: Path,
    combined_case_metrics: dict[tuple[int, str], dict[str, list[UtteranceMetrics]]],
    turnover_thresholds: Sequence[float],
    progress_partitions: int,
    human_annotation_summary: dict[tuple[int, str], dict[str, list[tuple[float, float]]]] | None = None,
    human_annotation_turnover_metrics: dict[tuple[int, str], list[UtteranceMetrics]] | None = None,
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
            human_case_turnover_metrics = (
                human_annotation_turnover_metrics.get(case_key, [])
                if human_annotation_turnover_metrics is not None
                else None
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
                        human_case_turnover_metrics=human_case_turnover_metrics,
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
                sigma_multiplier=None,
            )

            for sigma_multiplier in (1, 2):
                sigma_suffix, _ = resolve_mean_variant(sigma_multiplier)
                sigma_output_path = (
                    plots_root
                    / f"combined_clip_to_final_mean{sigma_suffix}_{utt_group_size}utt_{task_name}.png"
                )
                plot_combined_clip_to_final_mean_lines(
                    case_metrics=case_metrics,
                    utt_group_size=utt_group_size,
                    task_name=task_name,
                    progress_partitions=progress_partitions,
                    output_path=sigma_output_path,
                    sigma_multiplier=sigma_multiplier,
                )

            if human_annotation_summary is not None:
                human_model_mean_output_path = (
                    plots_root
                    / f"combined_human_model_partial_to_full_mean_{utt_group_size}utt_{task_name}.png"
                )
                if plot_combined_human_model_partial_to_full_lines(
                    case_metrics=case_metrics,
                    human_case_summary=human_case_summary,
                    human_case_turnover_metrics=human_case_turnover_metrics,
                    stat_name="mean",
                    utt_group_size=utt_group_size,
                    task_name=task_name,
                    progress_partitions=progress_partitions,
                    output_path=human_model_mean_output_path,
                    sigma_multiplier=None,
                ):
                    print(f"[INFO] Saved {human_model_mean_output_path}")

                for sigma_multiplier in (1, 2):
                    sigma_suffix, _ = resolve_mean_variant(sigma_multiplier)
                    human_model_sigma_output_path = (
                        plots_root
                        / f"combined_human_model_partial_to_full_mean{sigma_suffix}_{utt_group_size}utt_{task_name}.png"
                    )
                    if plot_combined_human_model_partial_to_full_lines(
                        case_metrics=case_metrics,
                        human_case_summary=human_case_summary,
                        human_case_turnover_metrics=human_case_turnover_metrics,
                        stat_name="mean",
                        utt_group_size=utt_group_size,
                        task_name=task_name,
                        progress_partitions=progress_partitions,
                        output_path=human_model_sigma_output_path,
                        sigma_multiplier=sigma_multiplier,
                    ):
                        print(f"[INFO] Saved {human_model_sigma_output_path}")

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
                human_case_metrics=human_case_turnover_metrics,
                sigma_multiplier=None,
            )

            for sigma_multiplier in (1, 2):
                sigma_suffix, _ = resolve_mean_variant(sigma_multiplier)
                st_sigma_output_path = (
                    plots_root
                    / f"combined_st_vs_threshold{sigma_suffix}_{utt_group_size}utt_{task_name}.png"
                )
                plot_combined_st_threshold_lines(
                    case_metrics=case_metrics,
                    turnover_thresholds=turnover_thresholds,
                    utt_group_size=utt_group_size,
                    task_name=task_name,
                    output_path=st_sigma_output_path,
                    human_case_metrics=human_case_turnover_metrics,
                    sigma_multiplier=sigma_multiplier,
                )

    write_wastp_table(
        combined_case_metrics=combined_case_metrics,
        output_csv_path=plots_root / "wastp_summary.csv",
        output_md_path=plots_root / "wastp_summary.md",
    )


def generate_per_dataset_combined_outputs(
    plots_root: Path,
    dataset_case_metrics: dict[
        str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]
    ],
    turnover_thresholds: Sequence[float],
    progress_partitions: int,
    human_annotation_summary_by_dataset: dict[
        str, dict[tuple[int, str], dict[str, list[tuple[float, float]]]]
    ] | None = None,
    human_annotation_turnover_metrics_by_dataset: dict[
        str, dict[tuple[int, str], list[UtteranceMetrics]]
    ] | None = None,
) -> None:
    for dataset_name in DATASET_NAMES:
        dataset_metrics = dataset_case_metrics.get(dataset_name, {})
        if not dataset_metrics:
            continue

        dataset_human_summary = (
            human_annotation_summary_by_dataset.get(dataset_name, {})
            if human_annotation_summary_by_dataset is not None
            else {}
        )
        dataset_human_turnover_metrics = (
            human_annotation_turnover_metrics_by_dataset.get(dataset_name, {})
            if (
                human_annotation_turnover_metrics_by_dataset is not None
                and dataset_name in human_annotation_turnover_metrics_by_dataset
            )
            else {}
        )

        for utt_group_size in UTTERANCE_GROUP_SIZES:
            output_dir = plots_root / dataset_name / f"{utt_group_size}-utt_group"
            output_dir.mkdir(parents=True, exist_ok=True)

            for task_name in TASK_NAMES:
                case_key = (utt_group_size, task_name)
                case_metrics = dataset_metrics.get(case_key, {})
                human_case_summary = dataset_human_summary.get(case_key, {})
                human_case_turnover_metrics = (
                    dataset_human_turnover_metrics.get(case_key, [])
                    if (
                        human_annotation_turnover_metrics_by_dataset is not None
                        and dataset_name in human_annotation_turnover_metrics_by_dataset
                    )
                    else None
                )
                if (
                    not case_metrics
                    and not human_case_summary
                    and not human_case_turnover_metrics
                ):
                    continue

                if case_metrics or human_case_summary or human_case_turnover_metrics:
                    for percentile in (25, 50, 75):
                        human_model_output_path = (
                            output_dir
                            / f"combined_human_model_partial_to_full_p{percentile}_{task_name}.png"
                        )
                        if plot_combined_human_model_partial_to_full_lines(
                            case_metrics=case_metrics,
                            human_case_summary=human_case_summary,
                            human_case_turnover_metrics=human_case_turnover_metrics,
                            stat_name="percentile",
                            percentile=percentile,
                            utt_group_size=utt_group_size,
                            task_name=task_name,
                            progress_partitions=progress_partitions,
                            output_path=human_model_output_path,
                            dataset_name=dataset_name,
                        ):
                            print(f"[INFO] Saved {human_model_output_path}")

                    human_model_mean_output_path = (
                        output_dir
                        / f"combined_human_model_partial_to_full_mean_{task_name}.png"
                    )
                    if plot_combined_human_model_partial_to_full_lines(
                        case_metrics=case_metrics,
                        human_case_summary=human_case_summary,
                        human_case_turnover_metrics=human_case_turnover_metrics,
                        stat_name="mean",
                        utt_group_size=utt_group_size,
                        task_name=task_name,
                        progress_partitions=progress_partitions,
                        output_path=human_model_mean_output_path,
                        sigma_multiplier=None,
                        dataset_name=dataset_name,
                    ):
                        print(f"[INFO] Saved {human_model_mean_output_path}")

                    for sigma_multiplier in (1, 2):
                        sigma_suffix, _ = resolve_mean_variant(sigma_multiplier)
                        human_model_sigma_output_path = (
                            output_dir
                            / f"combined_human_model_partial_to_full_mean{sigma_suffix}_{task_name}.png"
                        )
                        if plot_combined_human_model_partial_to_full_lines(
                            case_metrics=case_metrics,
                            human_case_summary=human_case_summary,
                            human_case_turnover_metrics=human_case_turnover_metrics,
                            stat_name="mean",
                            utt_group_size=utt_group_size,
                            task_name=task_name,
                            progress_partitions=progress_partitions,
                            output_path=human_model_sigma_output_path,
                            sigma_multiplier=sigma_multiplier,
                            dataset_name=dataset_name,
                        ):
                            print(f"[INFO] Saved {human_model_sigma_output_path}")

                st_output_path = (
                    output_dir / f"combined_st_vs_threshold_{task_name}.png"
                )
                if plot_combined_st_threshold_lines(
                    case_metrics=case_metrics,
                    turnover_thresholds=turnover_thresholds,
                    utt_group_size=utt_group_size,
                    task_name=task_name,
                    output_path=st_output_path,
                    human_case_metrics=human_case_turnover_metrics,
                    sigma_multiplier=None,
                    dataset_name=dataset_name,
                ):
                    print(f"[INFO] Saved {st_output_path}")

                for sigma_multiplier in (1, 2):
                    sigma_suffix, _ = resolve_mean_variant(sigma_multiplier)
                    st_sigma_output_path = (
                        output_dir
                        / f"combined_st_vs_threshold{sigma_suffix}_{task_name}.png"
                    )
                    if plot_combined_st_threshold_lines(
                        case_metrics=case_metrics,
                        turnover_thresholds=turnover_thresholds,
                        utt_group_size=utt_group_size,
                        task_name=task_name,
                        output_path=st_sigma_output_path,
                        human_case_metrics=human_case_turnover_metrics,
                        sigma_multiplier=sigma_multiplier,
                        dataset_name=dataset_name,
                    ):
                        print(f"[INFO] Saved {st_sigma_output_path}")


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
    dataset_case_metrics: dict[
        str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]
    ],
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
            dataset_case_metrics[dataset_root.name][(group_size, task_name)][
                model_label
            ].extend(utterance_metrics)

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

        for sigma_multiplier in (None, 1, 2):
            sigma_suffix, mean_title = resolve_mean_variant(sigma_multiplier)
            clip_to_final_mean_path = (
                output_dir / f"clip_to_final_similarity_mean_only{sigma_suffix}.png"
            )
            plot_clip_to_final_mean(
                utterance_metrics=utterance_metrics,
                title=f"{title} | {mean_title}",
                progress_partitions=progress_partitions,
                output_path=clip_to_final_mean_path,
                sigma_multiplier=sigma_multiplier,
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
    if args.from_plot_data is not None:
        (
            plots_root,
            combined_case_metrics,
            dataset_case_metrics,
            turnover_thresholds,
            progress_partitions,
            human_annotation_summary,
            human_annotation_summary_by_dataset,
            human_annotation_turnover_metrics,
            human_annotation_turnover_metrics_by_dataset,
        ) = load_analysis_plot_data(args.from_plot_data)
        if args.plots_root is not None:
            plots_root = args.plots_root.expanduser().resolve()
        plots_root.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] Regenerating combined plots from {args.from_plot_data} "
            f"into {plots_root}"
        )
        if dataset_case_metrics:
            print(
                "[INFO] Loaded per-dataset plot data for datasets: "
                f"{', '.join(sorted(dataset_case_metrics))}"
            )
        else:
            print(
                "[WARN] Saved plot data has no dataset_cases entries; "
                "per-dataset plots cannot be regenerated from this cache. "
                "Regenerate the cache without --from-plot-data and with "
                "--save-plot-data."
            )
        if human_annotation_summary is not None and not human_annotation_summary_by_dataset:
            print(
                "[WARN] Saved plot data has aggregate human annotation rows but "
                "no per-dataset human annotation rows. Per-dataset PFS plots "
                "will still be written from model data, but without per-dataset "
                "human annotation overlays."
            )
        generate_combined_outputs(
            plots_root=plots_root,
            combined_case_metrics=combined_case_metrics,
            turnover_thresholds=turnover_thresholds,
            progress_partitions=progress_partitions,
            human_annotation_summary=human_annotation_summary,
            human_annotation_turnover_metrics=human_annotation_turnover_metrics,
        )
        generate_per_dataset_combined_outputs(
            plots_root=plots_root,
            dataset_case_metrics=dataset_case_metrics,
            turnover_thresholds=turnover_thresholds,
            progress_partitions=progress_partitions,
            human_annotation_summary_by_dataset=human_annotation_summary_by_dataset,
            human_annotation_turnover_metrics_by_dataset=human_annotation_turnover_metrics_by_dataset,
        )
        return

    results_root = resolve_results_root(args.results_root)
    additional_results_roots = resolve_additional_results_roots(
        args.additional_results_root
    )
    plots_root = resolve_plots_root(results_root)
    if args.plots_root is not None:
        plots_root = args.plots_root.expanduser().resolve()
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_dir = (
        args.plot_data_dir.expanduser().resolve()
        if args.plot_data_dir is not None
        else (plots_root / "plot_data").resolve()
    )

    human_annotation_summary = None
    human_annotation_summary_by_dataset = None
    human_annotation_turnover_metrics = None
    human_annotation_turnover_metrics_by_dataset = None
    if args.human_annotation_summary_csv is not None:
        human_annotation_summary = load_human_annotation_summary_csv(
            args.human_annotation_summary_csv
        )
        human_annotation_summary_by_dataset = load_human_annotation_summary_csv_by_dataset(
            args.human_annotation_summary_csv
        )
        human_annotation_turnover_metrics = load_human_annotation_turnover_points_csv(
            args.human_annotation_summary_csv
        )
        human_annotation_turnover_metrics_by_dataset = load_human_annotation_turnover_points_csv_by_dataset(
            args.human_annotation_summary_csv
        )
        print(
            "[INFO] Loaded human annotation summary cases: "
            f"{len(human_annotation_summary)} from {args.human_annotation_summary_csv}"
        )
        print(
            "[INFO] Loaded per-dataset human annotation summary datasets: "
            f"{len(human_annotation_summary_by_dataset)} from "
            f"{args.human_annotation_summary_csv}"
        )
        print(
            "[INFO] Loaded human annotation ST cases: "
            f"{len(human_annotation_turnover_metrics)} from "
            f"{infer_human_annotation_points_csv(args.human_annotation_summary_csv)}"
        )
        print(
            "[INFO] Loaded per-dataset human annotation ST datasets: "
            f"{len(human_annotation_turnover_metrics_by_dataset)} from "
            f"{infer_human_annotation_points_csv(args.human_annotation_summary_csv)}"
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
    dataset_case_metrics: dict[
        str, dict[tuple[int, str], dict[str, list[UtteranceMetrics]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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
                dataset_case_metrics=dataset_case_metrics,
            )

    if args.save_plot_data:
        write_analysis_plot_data(
            output_dir=plot_data_dir,
            results_root=results_root,
            additional_results_roots=additional_results_roots,
            plots_root=plots_root,
            model_location=model_loc,
            turnover_thresholds=args.turnover_thresholds,
            progress_partitions=args.progress_partitions,
            combined_case_metrics=combined_case_metrics,
            dataset_case_metrics=dataset_case_metrics,
            human_annotation_summary_csv=args.human_annotation_summary_csv,
            human_annotation_summary=human_annotation_summary,
            human_annotation_summary_by_dataset=human_annotation_summary_by_dataset,
            human_annotation_turnover_metrics=human_annotation_turnover_metrics,
            human_annotation_turnover_metrics_by_dataset=human_annotation_turnover_metrics_by_dataset,
        )

    generate_combined_outputs(
        plots_root=plots_root,
        combined_case_metrics=combined_case_metrics,
        turnover_thresholds=args.turnover_thresholds,
        progress_partitions=args.progress_partitions,
        human_annotation_summary=human_annotation_summary,
        human_annotation_turnover_metrics=human_annotation_turnover_metrics,
    )
    generate_per_dataset_combined_outputs(
        plots_root=plots_root,
        dataset_case_metrics=dataset_case_metrics,
        turnover_thresholds=args.turnover_thresholds,
        progress_partitions=args.progress_partitions,
        human_annotation_summary_by_dataset=human_annotation_summary_by_dataset,
        human_annotation_turnover_metrics_by_dataset=human_annotation_turnover_metrics_by_dataset,
    )


if __name__ == "__main__":
    main()
