import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import re

import cv2

temp_cache_dir = Path(tempfile.gettempdir()) / "vlm_social_cache"
temp_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(temp_cache_dir))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FILENAME_PATTERN = re.compile(r"^dia(?P<dialogue>\d+)_utt(?P<utterance>\d+)$")


@dataclass(frozen=True)
class VideoRecord:
    dialogue_id: int
    utterance_id: int
    filename: str
    path: str
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dialogue-style video folders and export statistics.",
    )
    parser.add_argument(
        "video_folder",
        type=Path,
        help="Folder that contains files named like dia{n}_utt{m}.mp4.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where reports and figures will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan the input folder recursively instead of only its direct children.",
    )
    return parser.parse_args()


def iter_video_files(video_folder: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from sorted(
            path for path in video_folder.rglob("*") if path.is_file()
        )
        return

    yield from sorted(path for path in video_folder.iterdir() if path.is_file())


def parse_video_filename(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return None

    match = FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None

    return int(match["dialogue"]), int(match["utterance"])


def get_video_duration_seconds(video_path: Path) -> float:
    duration_from_ffprobe = get_video_duration_from_ffprobe(video_path)
    if duration_from_ffprobe is not None:
        return duration_from_ffprobe

    duration_from_cv2 = get_video_duration_from_cv2(video_path)
    if duration_from_cv2 is not None:
        return duration_from_cv2

    raise RuntimeError(f"Could not determine video duration for {video_path}")


def get_video_duration_from_ffprobe(video_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def get_video_duration_from_cv2(video_path: Path) -> float | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps is None or fps <= 0 or frame_count is None or frame_count < 0:
        return None

    return float(frame_count / fps)


def collect_video_records(video_folder: Path, recursive: bool) -> tuple[list[VideoRecord], list[str]]:
    records: list[VideoRecord] = []
    skipped_files: list[str] = []

    for path in iter_video_files(video_folder, recursive=recursive):
        parsed = parse_video_filename(path)
        if parsed is None:
            skipped_files.append(path.name)
            continue

        dialogue_id, utterance_id = parsed
        duration_seconds = get_video_duration_seconds(path)
        records.append(
            VideoRecord(
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
                filename=path.name,
                path=str(path.resolve()),
                duration_seconds=duration_seconds,
            )
        )

    records.sort(key=lambda record: (record.dialogue_id, record.utterance_id, record.filename))
    return records, skipped_files


def compute_numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "total": None,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "total": float(arr.sum()),
    }


def build_dialogue_summary(records: list[VideoRecord]) -> dict[str, object]:
    records_by_dialogue: dict[int, list[VideoRecord]] = defaultdict(list)
    for record in records:
        records_by_dialogue[record.dialogue_id].append(record)

    dialogue_ids = sorted(records_by_dialogue)
    utterance_count_values: list[int] = []
    missing_utterance_total = 0
    per_dialogue: list[dict[str, object]] = []

    for dialogue_id in dialogue_ids:
        dialogue_records = sorted(
            records_by_dialogue[dialogue_id],
            key=lambda record: record.utterance_id,
        )
        utterance_ids = sorted(record.utterance_id for record in dialogue_records)
        unique_utterance_ids = sorted(set(utterance_ids))
        duplicate_utterance_ids = sorted(
            utterance_id
            for utterance_id in set(utterance_ids)
            if utterance_ids.count(utterance_id) > 1
        )

        max_utterance_id = unique_utterance_ids[-1]
        expected_utterances = set(range(max_utterance_id + 1))
        missing_utterance_ids = sorted(expected_utterances - set(unique_utterance_ids))
        duration_values = [record.duration_seconds for record in dialogue_records]

        utterance_count_values.append(len(unique_utterance_ids))
        missing_utterance_total += len(missing_utterance_ids)

        per_dialogue.append(
            {
                "dialogue_id": dialogue_id,
                "utterance_count": len(unique_utterance_ids),
                "min_utterance_id": unique_utterance_ids[0],
                "max_utterance_id": max_utterance_id,
                "missing_utterance_ids": missing_utterance_ids,
                "duplicate_utterance_ids": duplicate_utterance_ids,
                "duration_seconds": compute_numeric_summary(duration_values),
            }
        )

    missing_dialogue_ids: list[int] = []
    if dialogue_ids:
        expected_dialogue_ids = set(range(dialogue_ids[0], dialogue_ids[-1] + 1))
        missing_dialogue_ids = sorted(expected_dialogue_ids - set(dialogue_ids))

    return {
        "dialogue_count": len(dialogue_ids),
        "dialogue_ids": dialogue_ids,
        "min_dialogue_id": dialogue_ids[0] if dialogue_ids else None,
        "max_dialogue_id": dialogue_ids[-1] if dialogue_ids else None,
        "missing_dialogue_ids": missing_dialogue_ids,
        "utterance_count_distribution": compute_numeric_summary(
            [float(value) for value in utterance_count_values]
        ),
        "dialogues_with_missing_utterances": sum(
            1 for dialogue in per_dialogue if dialogue["missing_utterance_ids"]
        ),
        "missing_utterance_total": missing_utterance_total,
        "per_dialogue": per_dialogue,
    }


def build_summary(records: list[VideoRecord], skipped_files: list[str], video_folder: Path) -> dict[str, object]:
    duration_values = [record.duration_seconds for record in records]
    dialogue_summary = build_dialogue_summary(records)

    return {
        "input_folder": str(video_folder.resolve()),
        "matched_video_count": len(records),
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files,
        "dialogues": dialogue_summary,
        "video_duration_seconds": compute_numeric_summary(duration_values),
        "videos": [asdict(record) for record in records],
    }


def format_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.3f}"
    return str(value)


def write_summary_text(summary: dict[str, object], output_path: Path) -> None:
    dialogue_summary = summary["dialogues"]
    utterance_distribution = dialogue_summary["utterance_count_distribution"]
    duration_summary = summary["video_duration_seconds"]

    lines = [
        "Dataset Video Analysis",
        "======================",
        "",
        f"Input folder: {summary['input_folder']}",
        f"Matched videos: {summary['matched_video_count']}",
        f"Skipped files: {summary['skipped_file_count']}",
        "",
        "Dialogue Summary",
        "----------------",
        f"Dialogue count: {dialogue_summary['dialogue_count']}",
        f"Dialogue id range: {dialogue_summary['min_dialogue_id']} to {dialogue_summary['max_dialogue_id']}",
        f"Missing dialogue ids inside the observed range: {dialogue_summary['missing_dialogue_ids']}",
        "",
        "Utterances Per Dialogue",
        "-----------------------",
        f"Mean utterance count: {format_float(utterance_distribution['mean'])}",
        f"Median utterance count: {format_float(utterance_distribution['median'])}",
        f"Min utterance count: {format_float(utterance_distribution['min'])}",
        f"Max utterance count: {format_float(utterance_distribution['max'])}",
        f"Dialogues with missing utterances: {dialogue_summary['dialogues_with_missing_utterances']}",
        f"Total missing utterance slots inside observed ranges: {dialogue_summary['missing_utterance_total']}",
        "",
        "Video Duration (seconds)",
        "------------------------",
        f"Mean: {format_float(duration_summary['mean'])}",
        f"Median: {format_float(duration_summary['median'])}",
        f"Std: {format_float(duration_summary['std'])}",
        f"Min: {format_float(duration_summary['min'])}",
        f"Max: {format_float(duration_summary['max'])}",
        f"25th percentile: {format_float(duration_summary['p25'])}",
        f"75th percentile: {format_float(duration_summary['p75'])}",
        f"Total duration: {format_float(duration_summary['total'])}",
        "",
        "Per-Dialogue Gaps",
        "-----------------",
    ]

    for dialogue in dialogue_summary["per_dialogue"]:
        lines.append(
            f"dia{dialogue['dialogue_id']}: "
            f"{dialogue['utterance_count']} utterances, "
            f"max utt {dialogue['max_utterance_id']}, "
            f"missing {dialogue['missing_utterance_ids']}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_video_csv(records: list[VideoRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dialogue_id",
                "utterance_id",
                "filename",
                "path",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_dialogue_csv(summary: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dialogue_id",
                "utterance_count",
                "min_utterance_id",
                "max_utterance_id",
                "missing_utterance_ids",
                "duplicate_utterance_ids",
                "duration_min_seconds",
                "duration_max_seconds",
                "duration_mean_seconds",
                "duration_median_seconds",
                "duration_total_seconds",
            ],
        )
        writer.writeheader()
        for dialogue in summary["dialogues"]["per_dialogue"]:
            duration_summary = dialogue["duration_seconds"]
            writer.writerow(
                {
                    "dialogue_id": dialogue["dialogue_id"],
                    "utterance_count": dialogue["utterance_count"],
                    "min_utterance_id": dialogue["min_utterance_id"],
                    "max_utterance_id": dialogue["max_utterance_id"],
                    "missing_utterance_ids": json.dumps(dialogue["missing_utterance_ids"]),
                    "duplicate_utterance_ids": json.dumps(dialogue["duplicate_utterance_ids"]),
                    "duration_min_seconds": duration_summary["min"],
                    "duration_max_seconds": duration_summary["max"],
                    "duration_mean_seconds": duration_summary["mean"],
                    "duration_median_seconds": duration_summary["median"],
                    "duration_total_seconds": duration_summary["total"],
                }
            )


def plot_utterance_counts(summary: dict[str, object], output_path: Path) -> None:
    per_dialogue = summary["dialogues"]["per_dialogue"]
    dialogue_ids = [dialogue["dialogue_id"] for dialogue in per_dialogue]
    utterance_counts = [dialogue["utterance_count"] for dialogue in per_dialogue]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dialogue_ids, utterance_counts, color="#4C78A8")
    ax.set_title("Utterance Count Per Dialogue")
    ax.set_xlabel("Dialogue ID")
    ax.set_ylabel("Number of Utterances")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_utterance_presence(summary: dict[str, object], output_path: Path) -> None:
    observed_x: list[int] = []
    observed_y: list[int] = []
    missing_x: list[int] = []
    missing_y: list[int] = []

    for dialogue in summary["dialogues"]["per_dialogue"]:
        dialogue_id = dialogue["dialogue_id"]
        max_utterance_id = dialogue["max_utterance_id"]
        missing_utterances = set(dialogue["missing_utterance_ids"])

        for utterance_id in range(max_utterance_id + 1):
            if utterance_id in missing_utterances:
                missing_x.append(utterance_id)
                missing_y.append(dialogue_id)
            else:
                observed_x.append(utterance_id)
                observed_y.append(dialogue_id)

    fig, ax = plt.subplots(figsize=(10, 6))
    if observed_x:
        ax.scatter(
            observed_x,
            observed_y,
            marker="s",
            s=32,
            color="#4C78A8",
            label="Observed",
        )
    if missing_x:
        ax.scatter(
            missing_x,
            missing_y,
            marker="x",
            s=40,
            color="#E45756",
            label="Missing in 0..max range",
        )

    ax.set_title("Utterance Coverage Per Dialogue")
    ax.set_xlabel("Utterance ID")
    ax.set_ylabel("Dialogue ID")
    ax.grid(True, linestyle="--", alpha=0.25)
    if observed_x or missing_x:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_duration_distribution(records: list[VideoRecord], output_path: Path) -> None:
    durations = [record.duration_seconds for record in records]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    axes[0].hist(durations, bins=min(20, max(5, len(durations))), color="#72B7B2", edgecolor="black")
    axes[0].set_title("Video Duration Distribution")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Video Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].boxplot(durations, vert=False)
    axes[1].set_xlabel("Duration (seconds)")
    axes[1].set_yticks([])
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(records: list[VideoRecord], skipped_files: list[str], video_folder: Path, output_folder: Path) -> dict[str, object]:
    output_folder.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records, skipped_files, video_folder)

    summary_json_path = output_folder / "summary.json"
    summary_txt_path = output_folder / "summary.txt"
    videos_csv_path = output_folder / "videos.csv"
    dialogues_csv_path = output_folder / "dialogues.csv"

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_text(summary, summary_txt_path)
    write_video_csv(records, videos_csv_path)
    write_dialogue_csv(summary, dialogues_csv_path)

    plot_utterance_counts(summary, output_folder / "utterance_count_per_dialogue.png")
    plot_utterance_presence(summary, output_folder / "utterance_coverage.png")
    plot_duration_distribution(records, output_folder / "video_duration_distribution.png")

    return summary


def analyze_video_folder(video_folder: Path, output_folder: Path, recursive: bool = False) -> dict[str, object]:
    if not video_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {video_folder}")
    if not video_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {video_folder}")

    records, skipped_files = collect_video_records(video_folder, recursive=recursive)
    if not records:
        raise ValueError(
            "No matching videos were found. Expected files named like dia{n}_utt{m}.mp4."
        )

    return write_outputs(records, skipped_files, video_folder, output_folder)


def main() -> None:
    args = parse_args()
    summary = analyze_video_folder(
        video_folder=args.video_folder,
        output_folder=args.output_folder,
        recursive=args.recursive,
    )

    print(f"Analyzed {summary['matched_video_count']} videos.")
    print(f"Detected {summary['dialogues']['dialogue_count']} dialogues.")
    print(f"Outputs written to {args.output_folder.resolve()}")


if __name__ == "__main__":
    main()
