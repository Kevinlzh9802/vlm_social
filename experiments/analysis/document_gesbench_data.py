from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("/data/GesBench_Data")
DATASET_NAMES = ("mintrec2", "meld", "seamless_interaction")
FILE_TYPE_SUFFIXES = {
    "videos": {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"},
    "json": {".json"},
    "csv": {".csv"},
    "images": {".png", ".jpg", ".jpeg"},
    "text": {".txt", ".md"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a GesBench data directory, write a concise README.md into it, "
            "and print the same structure summary."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"GesBench data root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="README output path. Defaults to <root>/README.md.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum directory tree depth to print. Default: 4.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=30,
        help="Maximum entries shown per directory. Default: 30.",
    )
    return parser.parse_args()


def count_file_types(root: Path) -> dict[str, int]:
    counts = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        matched = False
        for label, suffixes in FILE_TYPE_SUFFIXES.items():
            if suffix in suffixes:
                counts[label] += 1
                matched = True
                break
        if not matched:
            counts["other_files"] += 1
    counts["directories"] = sum(1 for path in root.rglob("*") if path.is_dir())
    return dict(counts)


def sort_key(path: Path) -> tuple[int, str]:
    return (1 if path.is_file() else 0, path.name.lower())


def build_tree_lines(
    root: Path,
    max_depth: int,
    max_entries: int,
    current_depth: int = 0,
) -> list[str]:
    if current_depth == 0:
        lines = [f"{root.name}/"]
    else:
        lines = []

    if current_depth >= max_depth:
        return lines

    children = sorted(root.iterdir(), key=sort_key)
    shown_children = children[:max_entries]
    for index, child in enumerate(shown_children):
        is_last = index == len(shown_children) - 1 and len(children) <= max_entries
        connector = "`-- " if is_last else "|-- "
        prefix = "    " * current_depth
        suffix = "/" if child.is_dir() else ""
        lines.append(f"{prefix}{connector}{child.name}{suffix}")
        if child.is_dir():
            lines.extend(
                build_tree_lines(
                    child,
                    max_depth=max_depth,
                    max_entries=max_entries,
                    current_depth=current_depth + 1,
                )
            )

    hidden_count = len(children) - len(shown_children)
    if hidden_count > 0:
        prefix = "    " * current_depth
        lines.append(f"{prefix}`-- ... {hidden_count} more entries")
    return lines


def existing(path: Path) -> bool:
    return path.exists()


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def build_known_layout_lines(root: Path) -> list[str]:
    lines = []
    for dataset_name in DATASET_NAMES:
        dataset_root = root / dataset_name
        if not dataset_root.exists():
            continue
        context_root = dataset_root / "context"
        raw_root = dataset_root / "raw"
        lines.append(f"- `{dataset_name}/`: source dataset folder.")
        if context_root.exists():
            lines.append(
                f"  - `{dataset_name}/context/<n>-utt_group/batch*/`: full benchmark clip videos used by model inference and aggregate analysis."
            )
        if raw_root.exists():
            lines.append(f"  - `{dataset_name}/raw/`: original or pre-partition source data.")

    human_eval = root / "human_eval"
    task1_root = first_existing(human_eval / "task1", root / "task1")
    task2_root = first_existing(human_eval / "task2", root / "task2")
    videos_root = first_existing(human_eval / "videos", root / "videos")
    pupil_root = first_existing(human_eval / "pupil", root / "pupil")

    if human_eval.exists():
        lines.append("- `human_eval/`: human evaluation assets and derived outputs.")

    if task1_root.exists():
        lines.append(
            f"- `{task1_root.relative_to(root)}/`: task1 annotation assets; `task1_b*.json` links annotation instances to media."
        )
    if task2_root.exists():
        lines.append(
            f"- `{task2_root.relative_to(root)}/`: task2 annotation/manipulation assets; `manipulation_full/data/` stores manipulated videos when present."
        )
    if videos_root.exists():
        lines.append(
            f"- `{videos_root.relative_to(root)}/`: local media files replacing annotation media URLs."
        )
    if pupil_root.exists():
        lines.append(f"- `{pupil_root.relative_to(root)}/`: Pupil Labs eye-tracking exports.")

    if (root / "results").exists():
        lines.append(
            "- `results/`: full-benchmark model outputs consumed by `job_scripts/analysis_daic.sh`."
        )
    if (root / "plots").exists():
        lines.append(
            "- `plots/`: aggregate analysis plots and reusable plot data from `experiments/analysis/main.py`."
        )

    return lines or ["- No known GesBench subfolders were found at this root."]


def build_script_link_lines(root: Path) -> list[str]:
    task1_json = first_existing(root / "human_eval" / "task1", root / "task1")
    task2_data = first_existing(
        root / "human_eval" / "task2" / "manipulation_full" / "data",
        root / "task2" / "manipulation_full" / "data",
    )
    return [
        "- `job_scripts/analysis_daic.sh` runs `experiments/analysis/main.py` on `results/`, overlays task1 human annotation summaries, and writes aggregate plots under `plots/`.",
        "- `job_scripts/human_annotation_similarity_daic.sh` reads task1 annotations plus `task1_b*.json` media lists and creates human partial-to-full plot data.",
        "- `job_scripts/manipulation_result_similarity_daic.sh` compares task2 manipulation outputs against full benchmark reference outputs.",
        "- `job_scripts/clip_statistics_daic.sh` runs `experiments/analysis/clip_statistics.py` over full data, task2 manipulated data, and task1 task-JSON media selections.",
        f"- Task1 media JSON root expected by the current scripts: `{task1_json}`.",
        f"- Task2 manipulated-video root expected by the current scripts: `{task2_data}`.",
    ]


def build_readme(root: Path, max_depth: int, max_entries: int) -> str:
    counts = count_file_types(root)
    count_parts = [
        f"{key}: {counts[key]}"
        for key in sorted(counts)
        if counts[key] > 0
    ]
    tree_lines = build_tree_lines(
        root,
        max_depth=max_depth,
        max_entries=max_entries,
    )

    sections = [
        "# GesBench Data Layout",
        "",
        f"Root: `{root}`",
        "",
        "This README was generated from the current filesystem layout. It is a concise map of the data folders and how they connect to the analysis scripts in this repository.",
        "",
        "## Quick Counts",
        "",
        ", ".join(count_parts) if count_parts else "No files found.",
        "",
        "## Directory Tree",
        "",
        "```text",
        *tree_lines,
        "```",
        "",
        "## Main Folders",
        "",
        *build_known_layout_lines(root),
        "",
        "## Analysis Links",
        "",
        *build_script_link_lines(root),
        "",
        "## Notes",
        "",
        "- Full benchmark clip videos are grouped by dataset and utterance count under `<dataset>/context/<n>-utt_group/`.",
        "- Task1 media paths come from task JSON files and are rebased to local media files by the analysis utilities.",
        "- Task2 manipulation outputs mirror the dataset/context/utterance-group organization so statistics and comparisons can be grouped consistently.",
    ]
    return "\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GesBench data root does not exist: {root}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "README.md"
    )
    readme = build_readme(
        root=root,
        max_depth=args.max_depth,
        max_entries=args.max_entries,
    )
    output_path.write_text(readme, encoding="utf-8")
    print(readme)
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
