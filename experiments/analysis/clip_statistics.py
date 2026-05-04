from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    from .parse_human_annotations import (
        flatten_task_media_lists,
        load_json,
        normalize_task_payload,
    )
except ImportError:
    from parse_human_annotations import (
        flatten_task_media_lists,
        load_json,
        normalize_task_payload,
    )


DATASET_NAMES = ("mintrec2", "meld", "seamless_interaction")
UTT_GROUP_RE = re.compile(r"^(?P<size>[123])-utt_group$")
TASK1_BATCH_RE = re.compile(r"^(?P<dataset>.+)_u(?P<size>[123])b\d+$")
CLIP_FILE_RE = re.compile(r"^(?P<prefix>.+)_clip_?(?P<index>\d+)\.[^.]+$")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


@dataclass(frozen=True)
class VideoRecord:
    dataset: str
    utt_count: int
    path: Path
    group_id: str
    clip_index: int


@dataclass(frozen=True)
class DurationRecord:
    dataset: str
    utt_count: int
    group_id: str
    clip_index: int
    path: str
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute clip-duration statistics for full GestaltBench data, task2 "
            "manipulation data, and task1 task-JSON-selected videos."
        )
    )
    parser.add_argument(
        "--full-root",
        type=Path,
        required=True,
        help="GestaltBench data root containing <dataset>/context/<n>-utt_group.",
    )
    parser.add_argument(
        "--task2-root",
        type=Path,
        required=True,
        help="Task2 manipulation_full/data root containing <dataset>/context/<n>-utt_group.",
    )
    parser.add_argument(
        "--task1-root",
        type=Path,
        required=True,
        help="Task1 root where output files are written.",
    )
    parser.add_argument(
        "--task1-json",
        type=Path,
        required=True,
        help="Task1 JSON file or directory containing task1.json/task1_b*.json.",
    )
    parser.add_argument(
        "--task1-local-prefix",
        type=Path,
        required=True,
        help="Local path prefix replacing task1 media URLs.",
    )
    parser.add_argument(
        "--media-url-prefix",
        action="append",
        default=[],
        help=(
            "Task1 media URL prefix to replace with --task1-local-prefix. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="clip_statistics",
        help="Base output filename. Default: clip_statistics.",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable path. Default: ffprobe.",
    )
    return parser.parse_args()


def iter_video_files(root: Path) -> Iterable[Path]:
    for extension in VIDEO_EXTENSIONS:
        yield from root.rglob(f"*{extension}")


def parse_clip_file_name(filename: str) -> tuple[str, int] | None:
    match = CLIP_FILE_RE.fullmatch(filename)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("index"))


def parse_context_video_path(path: Path) -> tuple[str, int] | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "context" or index == 0 or index + 1 >= len(parts):
            continue
        dataset = parts[index - 1]
        if dataset not in DATASET_NAMES:
            continue
        match = UTT_GROUP_RE.fullmatch(parts[index + 1])
        if match is None:
            continue
        return dataset, int(match.group("size"))
    return None


def context_group_id(path: Path) -> str | None:
    parsed_clip = parse_clip_file_name(path.name)
    if parsed_clip is None:
        return None
    clip_prefix, _clip_index = parsed_clip
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "context" or index + 2 >= len(parts):
            continue
        group_parts = parts[index + 2 : -1] + (clip_prefix,)
        return "/".join(group_parts)
    return None


def parse_task1_video_path(path_text: str) -> tuple[str, int] | None:
    parsed = urlparse(path_text)
    posix_text = parsed.path if parsed.scheme else path_text
    parts = tuple(
        part for part in PurePosixPath(posix_text).parts if part not in {"/", ""}
    )

    for index, part in enumerate(parts):
        if not re.fullmatch(r"u[123]", part):
            continue
        if index + 1 >= len(parts):
            continue
        batch_match = TASK1_BATCH_RE.fullmatch(parts[index + 1])
        if batch_match is None:
            continue
        return batch_match.group("dataset"), int(batch_match.group("size"))

    # Some task JSONs may already point to the full-data context layout.
    local_parts = Path(posix_text).parts
    for index, part in enumerate(local_parts):
        if part != "context" or index == 0 or index + 1 >= len(local_parts):
            continue
        dataset = local_parts[index - 1]
        if dataset not in DATASET_NAMES:
            continue
        match = UTT_GROUP_RE.fullmatch(local_parts[index + 1])
        if match is not None:
            return dataset, int(match.group("size"))
    return None


def task1_group_id(path_text: str) -> str | None:
    parsed = urlparse(path_text)
    posix_text = parsed.path if parsed.scheme else path_text
    parts = tuple(
        part for part in PurePosixPath(posix_text).parts if part not in {"/", ""}
    )
    if not parts:
        return None
    parsed_clip = parse_clip_file_name(parts[-1])
    if parsed_clip is None:
        return None
    clip_prefix, _clip_index = parsed_clip
    return "/".join(parts[:-1] + (clip_prefix,))


def keep_largest_clip_per_group(
    records: Iterable[VideoRecord],
) -> list[VideoRecord]:
    selected: dict[tuple[str, int, str], VideoRecord] = {}
    for record in records:
        key = (record.dataset, record.utt_count, record.group_id)
        current = selected.get(key)
        if current is None or record.clip_index > current.clip_index:
            selected[key] = record
    return sorted(
        selected.values(),
        key=lambda record: (
            record.dataset,
            record.utt_count,
            record.group_id,
            record.clip_index,
            str(record.path),
        ),
    )


def collect_context_videos(root: Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for dataset in DATASET_NAMES:
        context_root = root / dataset / "context"
        if not context_root.is_dir():
            continue
        for path in sorted(iter_video_files(context_root)):
            parsed = parse_context_video_path(path)
            if parsed is None:
                continue
            parsed_dataset, utt_count = parsed
            if parsed_dataset != dataset:
                continue
            parsed_clip = parse_clip_file_name(path.name)
            group_id = context_group_id(path)
            if parsed_clip is None or group_id is None:
                continue
            _clip_prefix, clip_index = parsed_clip
            records.append(
                VideoRecord(
                    dataset=dataset,
                    utt_count=utt_count,
                    path=path,
                    group_id=group_id,
                    clip_index=clip_index,
                )
            )
    return keep_largest_clip_per_group(records)


def resolve_task1_json_paths(task_json: Path) -> list[Path]:
    source = task_json.expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Task1 JSON path does not exist: {source}")

    paths = sorted(source.glob("task1_b*.json"))
    fallback = source / "task1.json"
    if fallback.is_file():
        paths.append(fallback)
    if not paths:
        raise FileNotFoundError(f"No task1*.json files found under {source}")
    return paths


def rebase_media_path(
    media_path: str,
    media_url_prefixes: list[str],
    local_prefix: Path,
) -> Path:
    text = media_path.strip()
    for prefix in media_url_prefixes:
        if text.startswith(prefix):
            relative = text[len(prefix) :].lstrip("/")
            return local_prefix / PurePosixPath(relative)

    parsed = urlparse(text)
    if parsed.scheme:
        path_text = parsed.path.lstrip("/")
        marker = "gestalt_bench/annotation"
        marker_index = path_text.find(marker)
        if marker_index >= 0:
            suffix = path_text[marker_index + len(marker) :]
            suffix_parts = PurePosixPath(suffix.lstrip("0123456789/")).parts
            return local_prefix.joinpath(*suffix_parts)
        return local_prefix / PurePosixPath(Path(parsed.path).name)

    return Path(text)


def collect_task1_videos(
    task_json: Path,
    media_url_prefixes: list[str],
    local_prefix: Path,
) -> tuple[list[VideoRecord], list[str]]:
    records: list[VideoRecord] = []
    warnings: list[str] = []

    for json_path in resolve_task1_json_paths(task_json):
        task_items = normalize_task_payload(load_json(json_path))
        for task_instance_id in range(len(task_items)):
            try:
                video_paths, _audio_paths = flatten_task_media_lists(
                    task_items,
                    task_instance_id,
                )
            except ValueError as exc:
                warnings.append(
                    f"{json_path.name} instance {task_instance_id}: {exc}"
                )
                continue

            for media_path in video_paths:
                parsed = parse_task1_video_path(media_path)
                if parsed is None:
                    warnings.append(
                        f"{json_path.name} instance {task_instance_id}: "
                        f"could not infer dataset/utt from {media_path}"
                    )
                    continue
                dataset, utt_count = parsed
                parsed_media = urlparse(media_path)
                parsed_clip = parse_clip_file_name(
                    PurePosixPath(
                        parsed_media.path if parsed_media.scheme else media_path
                    ).name
                )
                group_id = task1_group_id(media_path)
                if parsed_clip is None or group_id is None:
                    warnings.append(
                        f"{json_path.name} instance {task_instance_id}: "
                        f"could not infer clip group/index from {media_path}"
                    )
                    continue
                _clip_prefix, clip_index = parsed_clip
                local_path = rebase_media_path(
                    media_path,
                    media_url_prefixes=media_url_prefixes,
                    local_prefix=local_prefix,
                ).expanduser()
                records.append(
                    VideoRecord(
                        dataset=dataset,
                        utt_count=utt_count,
                        path=local_path,
                        group_id=group_id,
                        clip_index=clip_index,
                    )
                )

    return keep_largest_clip_per_group(records), warnings


def probe_duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration_text = result.stdout.strip().splitlines()[0]
    duration = float(duration_text)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"invalid duration {duration_text!r}")
    return duration


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "std_seconds": None,
            "min_seconds": None,
            "p25_seconds": None,
            "median_seconds": None,
            "p75_seconds": None,
            "max_seconds": None,
        }

    sorted_values = sorted(values)
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": mean_value,
        "std_seconds": math.sqrt(variance),
        "min_seconds": sorted_values[0],
        "p25_seconds": percentile(sorted_values, 25),
        "median_seconds": median(sorted_values),
        "p75_seconds": percentile(sorted_values, 75),
        "max_seconds": sorted_values[-1],
    }


def summarize_records(
    dataset_name: str,
    records: list[VideoRecord],
    ffprobe: str,
    source_root: Path,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    duration_records: list[DurationRecord] = []
    failures: list[dict[str, str]] = []

    for record in records:
        if not record.path.is_file():
            failures.append({"path": str(record.path), "error": "file does not exist"})
            continue
        try:
            duration = probe_duration(record.path, ffprobe=ffprobe)
        except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
            failures.append({"path": str(record.path), "error": str(exc)})
            continue
        duration_records.append(
            DurationRecord(
                dataset=record.dataset,
                utt_count=record.utt_count,
                group_id=record.group_id,
                clip_index=record.clip_index,
                path=str(record.path),
                duration_seconds=duration,
            )
        )

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    grouped_by_utt_count: dict[int, list[float]] = defaultdict(list)
    for record in duration_records:
        grouped[(record.dataset, record.utt_count)].append(record.duration_seconds)
        grouped_by_utt_count[record.utt_count].append(record.duration_seconds)

    groups = []
    for dataset, utt_count in sorted(grouped):
        values = grouped[(dataset, utt_count)]
        groups.append(
            {
                "dataset": dataset,
                "utt_count": utt_count,
                **summarize_values(values),
            }
        )

    all_datasets_by_utt_count = []
    for utt_count in sorted(grouped_by_utt_count):
        values = grouped_by_utt_count[utt_count]
        all_datasets_by_utt_count.append(
            {
                "dataset": "all_datasets",
                "utt_count": utt_count,
                **summarize_values(values),
            }
        )

    return {
        "source": dataset_name,
        "source_root": str(source_root),
        "stat_unit": "one duration per video group, using the largest clip index",
        "input_video_count": len(records),
        "valid_video_count": len(duration_records),
        "failed_video_count": len(failures),
        "groups": groups,
        "all_datasets_by_utt_count": all_datasets_by_utt_count,
        "overall": summarize_values(
            [record.duration_seconds for record in duration_records]
        ),
        "failures": failures,
        "warnings": warnings or [],
    }


def format_seconds(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def write_outputs(output_root: Path, output_name: str, payload: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{output_name}.json"
    txt_path = output_root / f"{output_name}.txt"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"Source: {payload['source']}",
        f"Source root: {payload['source_root']}",
        f"Stat unit: {payload['stat_unit']}",
        f"Input videos: {payload['input_video_count']}",
        f"Valid videos: {payload['valid_video_count']}",
        f"Failed videos: {payload['failed_video_count']}",
        "",
        "dataset\tutt_count\tcount\ttotal_s\tmean_s\tstd_s\tmin_s\tp25_s\tmedian_s\tp75_s\tmax_s",
    ]
    for group in payload["groups"]:
        lines.append(
            "\t".join(
                [
                    str(group["dataset"]),
                    str(group["utt_count"]),
                    str(group["count"]),
                    format_seconds(group["total_seconds"]),
                    format_seconds(group["mean_seconds"]),
                    format_seconds(group["std_seconds"]),
                    format_seconds(group["min_seconds"]),
                    format_seconds(group["p25_seconds"]),
                    format_seconds(group["median_seconds"]),
                    format_seconds(group["p75_seconds"]),
                    format_seconds(group["max_seconds"]),
                ]
            )
        )

    if payload["all_datasets_by_utt_count"]:
        lines.extend(
            [
                "",
                "All datasets by utterance count:",
                "dataset\tutt_count\tcount\ttotal_s\tmean_s\tstd_s\tmin_s\tp25_s\tmedian_s\tp75_s\tmax_s",
            ]
        )
        for group in payload["all_datasets_by_utt_count"]:
            lines.append(
                "\t".join(
                    [
                        str(group["dataset"]),
                        str(group["utt_count"]),
                        str(group["count"]),
                        format_seconds(group["total_seconds"]),
                        format_seconds(group["mean_seconds"]),
                        format_seconds(group["std_seconds"]),
                        format_seconds(group["min_seconds"]),
                        format_seconds(group["p25_seconds"]),
                        format_seconds(group["median_seconds"]),
                        format_seconds(group["p75_seconds"]),
                        format_seconds(group["max_seconds"]),
                    ]
                )
            )

    overall = payload["overall"]
    lines.extend(
        [
            "",
            "Overall:",
            f"count: {overall['count']}",
            f"total_seconds: {format_seconds(overall['total_seconds'])}",
            f"mean_seconds: {format_seconds(overall['mean_seconds'])}",
            f"std_seconds: {format_seconds(overall['std_seconds'])}",
            f"min_seconds: {format_seconds(overall['min_seconds'])}",
            f"p25_seconds: {format_seconds(overall['p25_seconds'])}",
            f"median_seconds: {format_seconds(overall['median_seconds'])}",
            f"p75_seconds: {format_seconds(overall['p75_seconds'])}",
            f"max_seconds: {format_seconds(overall['max_seconds'])}",
        ]
    )

    if payload["warnings"]:
        lines.extend(["", "Warnings:"])
        lines.extend(str(warning) for warning in payload["warnings"])

    if payload["failures"]:
        lines.extend(["", "Failures:"])
        lines.extend(
            f"{failure['path']}\t{failure['error']}" for failure in payload["failures"]
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Saved {json_path}")
    print(f"[INFO] Saved {txt_path}")


def main() -> None:
    args = parse_args()

    full_root = args.full_root.expanduser().resolve()
    task2_root = args.task2_root.expanduser().resolve()
    task1_root = args.task1_root.expanduser().resolve()
    task1_local_prefix = args.task1_local_prefix.expanduser().resolve()

    if not full_root.is_dir():
        raise FileNotFoundError(f"Full root does not exist: {full_root}")
    if not task2_root.is_dir():
        raise FileNotFoundError(f"Task2 root does not exist: {task2_root}")
    if not args.task1_json.exists():
        raise FileNotFoundError(f"Task1 JSON path does not exist: {args.task1_json}")

    full_payload = summarize_records(
        dataset_name="full",
        records=collect_context_videos(full_root),
        ffprobe=args.ffprobe,
        source_root=full_root,
    )
    write_outputs(full_root, args.output_name, full_payload)

    task2_payload = summarize_records(
        dataset_name="task2_manipulation_full",
        records=collect_context_videos(task2_root),
        ffprobe=args.ffprobe,
        source_root=task2_root,
    )
    write_outputs(task2_root, args.output_name, task2_payload)

    task1_records, task1_warnings = collect_task1_videos(
        task_json=args.task1_json,
        media_url_prefixes=args.media_url_prefix,
        local_prefix=task1_local_prefix,
    )
    task1_payload = summarize_records(
        dataset_name="task1",
        records=task1_records,
        ffprobe=args.ffprobe,
        source_root=args.task1_json.expanduser().resolve(),
        warnings=task1_warnings,
    )
    write_outputs(task1_root, args.output_name, task1_payload)


if __name__ == "__main__":
    main()
