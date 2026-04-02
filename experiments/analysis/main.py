import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_NAMES = ("mintrec2", "seamless_interaction")
DIALOGUE_KEY_PATTERN = re.compile(r"^d\d+u\d+$")
FILE_CLIP_PATTERN = re.compile(r"^(?P<prefix>d\d+u\d+)_clip(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute clip-to-final cosine similarities for Qwen2.5 batch results "
            "and save one scatter plot per result subfolder."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Path to results/qwen2.5. Defaults to <repo>/results/qwen2.5 when omitted."
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
            "SentenceTransformer model. When set, --model is ignored and the "
            "model is loaded offline from this path (no network required)."
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
        "Could not find results/qwen2.5 automatically. "
        "Pass --results-root explicitly."
    )


def iter_result_folders(dataset_root: Path) -> List[Path]:
    result_folders: List[Path] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_dir():
            continue
        if list(path.glob("batch*.json")):
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
    if isinstance(value, dict):
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


def is_error_text(text: str) -> bool:
    return text.startswith("[ERROR]")


def compute_similarity_points(
    model: SentenceTransformer,
    ordered_clips: Sequence[Tuple[int, str]],
) -> List[Tuple[float, float]]:
    valid_clips = [
        (clip_index, text)
        for clip_index, text in ordered_clips
        if text.strip() and not is_error_text(text)
    ]
    if len(valid_clips) < 2:
        return []

    texts = [text for _, text in valid_clips]
    embeddings = model.encode(texts, convert_to_numpy=True)
    final_embedding = embeddings[-1].reshape(1, -1)
    similarities = cosine_similarity(embeddings, final_embedding).ravel()

    total_clips = len(valid_clips)
    return [
        (position / total_clips, float(similarity))
        for position, similarity in enumerate(similarities, start=1)
    ]


def build_figure_name(result_folder: Path, dataset_root: Path) -> str:
    relative_parts = result_folder.relative_to(dataset_root).parts
    stem = "__".join(relative_parts)
    return f"{stem}_clip_to_final_similarity.png"


def plot_points(points: Sequence[Tuple[float, float]], title: str, output_path: Path) -> None:
    if not points:
        return

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

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


def analyze_result_folder(model: SentenceTransformer, result_folder: Path) -> List[Tuple[float, float]]:
    all_points: List[Tuple[float, float]] = []
    json_paths = sorted(
        result_folder.glob("batch*.json"),
        key=lambda path: sort_key_for_clip_identifier(path.stem),
    )

    for json_path in json_paths:
        payload = load_json(json_path)
        for dialogue_key, dialogue_value in find_dialogue_entries(payload):
            ordered_clips = extract_ordered_clips(dialogue_key, dialogue_value)
            all_points.extend(compute_similarity_points(model, ordered_clips))

    return all_points


def analyze_dataset(model: SentenceTransformer, dataset_root: Path) -> None:
    result_folders = iter_result_folders(dataset_root)
    if not result_folders:
        print(f"[WARN] No folders with batch*.json found under {dataset_root}")
        return

    for result_folder in result_folders:
        points = analyze_result_folder(model=model, result_folder=result_folder)
        if not points:
            print(f"[WARN] No usable clip pairs found in {result_folder}")
            continue

        output_path = dataset_root / build_figure_name(result_folder, dataset_root)
        title = str(result_folder.relative_to(dataset_root)).replace("\\", " / ")
        plot_points(points=points, title=title, output_path=output_path)
        print(f"[INFO] Saved {output_path}")


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
        analyze_dataset(model=model, dataset_root=dataset_root)


if __name__ == "__main__":
    main()
