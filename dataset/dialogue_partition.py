import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from video_utils import cut_video_into_clips


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FILENAME_PATTERN = re.compile(r"^dia(?P<dialogue>\d+)_utt(?P<utterance>\d+)$")
GROUP_SEQUENCE = (1, 2, 3)
GROUP_FOLDER_NAMES = {
    1: "1-utt_group",
    2: "2-utt_group",
    3: "3-utt_group",
}
MATERIALIZATION_MODES = ("nested", "context")


@dataclass(frozen=True)
class VideoRecord:
    dialogue_id: int
    utterance_id: int
    filename: str
    path: str


@dataclass(frozen=True)
class PartitionGroup:
    dialogue_id: int
    group_size: int
    utterance_ids: list[int]
    source_paths: list[str]
    group_name: str


@dataclass(frozen=True)
class SkippedWindow:
    dialogue_id: int
    group_size: int
    start_utterance_id: int
    expected_utterance_ids: list[int]
    missing_utterance_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Partition dialogue utterances into alternating 1/2/3-utterance groups "
            "and materialize grouped outputs."
        ),
    )
    parser.add_argument(
        "video_folder",
        type=Path,
        help="Folder containing videos named like dia{n}_utt{m}.mp4.",
    )
    parser.add_argument(
        "output_parent",
        type=Path,
        help="Folder where grouped outputs and summary files will be written.",
    )
    parser.add_argument(
        "--clip-length",
        type=float,
        default=0.5,
        help="Length in seconds for cumulative clips of the last utterance.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input videos recursively.",
    )
    parser.add_argument(
        "--dialogue-range",
        type=int,
        default=None,
        help="1-based hundred-range index. E.g. 1 \u2192 dialogues [0,100), 4 \u2192 [300,400).",
    )
    parser.add_argument(
        "-mode",
        "--mode",
        choices=MATERIALIZATION_MODES,
        default="nested",
        help=(
            "Output layout mode. 'nested' keeps the current structure. "
            "'context' flattens each group folder and prepends prior utterances "
            "to every generated clip."
        ),
    )
    return parser.parse_args()


def iter_video_files(video_folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in video_folder.rglob("*") if path.is_file())
    return sorted(path for path in video_folder.iterdir() if path.is_file())


def parse_video_filename(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return None

    match = FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None

    return int(match["dialogue"]), int(match["utterance"])


def utterance_label(dialogue_id: int, utterance_id: int) -> str:
    return f"d{dialogue_id}u{utterance_id}"


def group_label(dialogue_id: int, utterance_ids: list[int]) -> str:
    if len(utterance_ids) == 1:
        return utterance_label(dialogue_id, utterance_ids[0])
    return f"d{dialogue_id}u{utterance_ids[0]}-u{utterance_ids[-1]}"


def collect_video_records(video_folder: Path, recursive: bool) -> tuple[list[VideoRecord], list[str]]:
    records: list[VideoRecord] = []
    skipped_files: list[str] = []
    seen_keys: set[tuple[int, int]] = set()

    for path in iter_video_files(video_folder, recursive=recursive):
        parsed = parse_video_filename(path)
        if parsed is None:
            skipped_files.append(path.name)
            continue

        key = (parsed[0], parsed[1])
        if key in seen_keys:
            raise ValueError(
                f"Duplicate dialogue/utterance pair found for dia{parsed[0]}_utt{parsed[1]}."
            )

        seen_keys.add(key)
        records.append(
            VideoRecord(
                dialogue_id=parsed[0],
                utterance_id=parsed[1],
                filename=path.name,
                path=str(path.resolve()),
            )
        )

    records.sort(key=lambda record: (record.dialogue_id, record.utterance_id))
    return records, skipped_files


def partition_dialogue(records: list[VideoRecord]) -> tuple[list[PartitionGroup], list[SkippedWindow]]:
    record_by_utterance = {record.utterance_id: record for record in records}
    max_utterance_id = max(record_by_utterance)
    cursor = 0
    group_index = 0

    groups: list[PartitionGroup] = []
    skipped: list[SkippedWindow] = []

    while cursor <= max_utterance_id:
        group_size = GROUP_SEQUENCE[group_index]
        expected_ids = list(range(cursor, cursor + group_size))
        missing_ids = [
            utterance_id
            for utterance_id in expected_ids
            if utterance_id not in record_by_utterance
        ]

        if not missing_ids:
            utterance_ids = expected_ids
            source_paths = [record_by_utterance[utterance_id].path for utterance_id in utterance_ids]
            groups.append(
                PartitionGroup(
                    dialogue_id=records[0].dialogue_id,
                    group_size=group_size,
                    utterance_ids=utterance_ids,
                    source_paths=source_paths,
                    group_name=group_label(records[0].dialogue_id, utterance_ids),
                )
            )
            group_index = (group_index + 1) % len(GROUP_SEQUENCE)
        else:
            skipped.append(
                SkippedWindow(
                    dialogue_id=records[0].dialogue_id,
                    group_size=group_size,
                    start_utterance_id=cursor,
                    expected_utterance_ids=expected_ids,
                    missing_utterance_ids=missing_ids,
                )
            )

        cursor += group_size

    return groups, skipped


def rename_generated_clips(clip_folder: Path, prefix: str) -> dict[str, int]:
    clip_counts = {"mp4": 0, "wav": 0}

    for clip_path in sorted(clip_folder.glob("clip_*.mp4")):
        clip_index = clip_path.stem.split("_")[-1]
        new_path = clip_folder / f"{prefix}_clip{clip_index}.mp4"
        clip_path.rename(new_path)
        clip_counts["mp4"] += 1

    for clip_path in sorted(clip_folder.glob("clip_*.wav")):
        clip_index = clip_path.stem.split("_")[-1]
        new_path = clip_folder / f"{prefix}_clip{clip_index}.wav"
        clip_path.rename(new_path)
        clip_counts["wav"] += 1

    return clip_counts


def get_video_properties(video_path: str | Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for {video_path}")

    return fps, total_frames, total_frames / fps


def run_ffmpeg_with_video_codec_fallback(
    command_builder,
    context: str,
) -> None:
    last_command: list[str] | None = None
    last_result = None

    for video_codec in ("libx264", "mpeg4"):
        command = command_builder(video_codec)
        result = subprocess.run(command, capture_output=True, text=True)
        last_command = command
        last_result = result
        if result.returncode == 0:
            return
        if "Unknown encoder" not in (result.stderr or ""):
            break

    stderr_tail = "none"
    if last_result is not None and last_result.stderr:
        stderr_tail = last_result.stderr[-1000:]

    raise RuntimeError(
        f"{context} failed.\n"
        f"Command: {' '.join(last_command or [])}\n"
        f"ffmpeg stderr: {stderr_tail}"
    )


def concatenate_group_video(source_paths: list[str], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        concat_list_path = Path(handle.name)
        for source_path in source_paths:
            handle.write(f"file {Path(source_path).resolve().as_posix()}\n")

    try:
        run_ffmpeg_with_video_codec_fallback(
            lambda video_codec: [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c:v",
                video_codec,
                "-c:a",
                "aac",
                str(output_path),
            ],
            context=f"Concatenating context video for {output_path.stem}",
        )
    finally:
        concat_list_path.unlink(missing_ok=True)


def cut_contextual_cumulative_clips(
    video_path: Path,
    final_source_path: str,
    output_folder: Path,
    clip_length: float,
    prefix_duration_seconds: float,
    clip_prefix: str,
) -> dict[str, int]:
    fps, total_frames, _ = get_video_properties(final_source_path)
    clip_frames = int(clip_length * fps)

    if clip_frames <= 0:
        raise RuntimeError(
            f"Invalid clip length ({clip_length}) or FPS ({fps}) for {final_source_path}"
        )

    num_clips = total_frames // clip_frames
    clip_counts = {"mp4": 0, "wav": 0}

    for clip_index in range(1, num_clips + 1):
        duration_seconds = prefix_duration_seconds + (clip_index * clip_frames / fps)
        output_clip_path = output_folder / f"{clip_prefix}_clip{clip_index}.mp4"
        output_wav_path = output_folder / f"{clip_prefix}_clip{clip_index}.wav"

        run_ffmpeg_with_video_codec_fallback(
            lambda video_codec: [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-ss",
                "0",
                "-t",
                str(duration_seconds),
                "-map",
                "0:v",
                "-c:v",
                video_codec,
                str(output_clip_path),
                "-map",
                "0:a?",
                "-c:a",
                "pcm_s16le",
                str(output_wav_path),
            ],
            context=f"Generating contextual clip {clip_index} for {clip_prefix}",
        )

        if output_clip_path.exists():
            clip_counts["mp4"] += 1
        if output_wav_path.exists():
            clip_counts["wav"] += 1

    return clip_counts


def materialize_group_nested(
    group: PartitionGroup,
    output_parent: Path,
    clip_length: float,
) -> dict[str, object] | None:
    group_root = output_parent / GROUP_FOLDER_NAMES[group.group_size] / group.group_name

    final_utterance_id = group.utterance_ids[-1]
    final_label = utterance_label(group.dialogue_id, final_utterance_id)
    clip_folder = group_root / final_label

    if clip_folder.exists() and any(clip_folder.glob(f"{final_label}_clip*.mp4")):
        return None

    group_root.mkdir(parents=True, exist_ok=True)

    copied_whole_videos: list[str] = []
    for utterance_id, source_path in zip(group.utterance_ids[:-1], group.source_paths[:-1]):
        target_name = f"{utterance_label(group.dialogue_id, utterance_id)}.mp4"
        target_path = group_root / target_name
        shutil.copy2(source_path, target_path)
        copied_whole_videos.append(str(target_path))

    clip_folder.mkdir(parents=True, exist_ok=True)

    cut_video_into_clips(
        group.source_paths[-1],
        str(clip_folder),
        clip_length=clip_length,
        cumulative=True,
        save_separate_audio=True,
        video_include_audio=False,
    )
    clip_counts = rename_generated_clips(clip_folder, final_label)

    return {
        "dialogue_id": group.dialogue_id,
        "group_size": group.group_size,
        "group_name": group.group_name,
        "utterance_ids": group.utterance_ids,
        "mode": "nested",
        "whole_videos": copied_whole_videos,
        "clip_output_folder": str(clip_folder),
        "clip_mp4_count": clip_counts["mp4"],
        "clip_wav_count": clip_counts["wav"],
    }


def materialize_group_context(
    group: PartitionGroup,
    output_parent: Path,
    clip_length: float,
) -> dict[str, object] | None:
    group_root = output_parent / GROUP_FOLDER_NAMES[group.group_size] / group.group_name
    clip_prefix = group.group_name

    if group_root.exists() and any(group_root.glob(f"{clip_prefix}_clip*.mp4")):
        return None

    group_root.mkdir(parents=True, exist_ok=True)

    if group.group_size == 1:
        cut_video_into_clips(
            group.source_paths[-1],
            str(group_root),
            clip_length=clip_length,
            cumulative=True,
            save_separate_audio=True,
            video_include_audio=False,
        )
        clip_counts = rename_generated_clips(group_root, clip_prefix)
    else:
        prefix_duration_seconds = sum(
            get_video_properties(source_path)[2] for source_path in group.source_paths[:-1]
        )
        with tempfile.TemporaryDirectory(prefix=f"{group.group_name}_", dir=str(group_root)) as tmpdir:
            concatenated_path = Path(tmpdir) / f"{group.group_name}_context.mp4"
            concatenate_group_video(group.source_paths, concatenated_path)
            clip_counts = cut_contextual_cumulative_clips(
                video_path=concatenated_path,
                final_source_path=group.source_paths[-1],
                output_folder=group_root,
                clip_length=clip_length,
                prefix_duration_seconds=prefix_duration_seconds,
                clip_prefix=clip_prefix,
            )

    return {
        "dialogue_id": group.dialogue_id,
        "group_size": group.group_size,
        "group_name": group.group_name,
        "utterance_ids": group.utterance_ids,
        "mode": "context",
        "whole_videos": [],
        "clip_output_folder": str(group_root),
        "clip_mp4_count": clip_counts["mp4"],
        "clip_wav_count": clip_counts["wav"],
    }


def materialize_group(
    group: PartitionGroup,
    output_parent: Path,
    clip_length: float,
    mode: str,
) -> dict[str, object] | None:
    if mode == "nested":
        return materialize_group_nested(group, output_parent, clip_length)
    if mode == "context":
        return materialize_group_context(group, output_parent, clip_length)
    raise ValueError(f"Unsupported mode: {mode}")


def build_summary(
    video_folder: Path,
    output_parent: Path,
    records: list[VideoRecord],
    skipped_files: list[str],
    groups: list[PartitionGroup],
    skipped_windows: list[SkippedWindow],
    materialized_groups: list[dict[str, object]],
    clip_length: float,
    mode: str,
) -> dict[str, object]:
    groups_by_size = defaultdict(int)
    clips_by_size = defaultdict(int)
    wavs_by_size = defaultdict(int)
    dialogues_by_id: dict[int, dict[str, object]] = {}

    for record in records:
        dialogues_by_id.setdefault(
            record.dialogue_id,
            {
                "dialogue_id": record.dialogue_id,
                "available_utterance_count": 0,
                "max_utterance_id": 0,
                "group_1_count": 0,
                "group_2_count": 0,
                "group_3_count": 0,
                "skipped_window_count": 0,
                "skipped_windows": [],
            },
        )
        dialogue_info = dialogues_by_id[record.dialogue_id]
        dialogue_info["available_utterance_count"] += 1
        dialogue_info["max_utterance_id"] = max(dialogue_info["max_utterance_id"], record.utterance_id)

    for group in groups:
        groups_by_size[group.group_size] += 1
        dialogue_info = dialogues_by_id[group.dialogue_id]
        dialogue_info[f"group_{group.group_size}_count"] += 1

    for skipped in skipped_windows:
        dialogue_info = dialogues_by_id[skipped.dialogue_id]
        dialogue_info["skipped_window_count"] += 1
        dialogue_info["skipped_windows"].append(asdict(skipped))

    for materialized in materialized_groups:
        clips_by_size[materialized["group_size"]] += materialized["clip_mp4_count"]
        wavs_by_size[materialized["group_size"]] += materialized["clip_wav_count"]

    return {
        "input_folder": str(video_folder.resolve()),
        "output_parent": str(output_parent.resolve()),
        "clip_length_seconds": clip_length,
        "mode": mode,
        "matched_video_count": len(records),
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files,
        "dialogue_count": len(dialogues_by_id),
        "group_counts": {str(size): groups_by_size[size] for size in GROUP_SEQUENCE},
        "clip_mp4_counts": {str(size): clips_by_size[size] for size in GROUP_SEQUENCE},
        "clip_wav_counts": {str(size): wavs_by_size[size] for size in GROUP_SEQUENCE},
        "skipped_window_count": len(skipped_windows),
        "groups": [asdict(group) for group in groups],
        "materialized_groups": materialized_groups,
        "dialogues": [dialogues_by_id[dialogue_id] for dialogue_id in sorted(dialogues_by_id)],
        "skipped_windows": [asdict(skipped) for skipped in skipped_windows],
    }


def write_summary_files(summary: dict[str, object], output_parent: Path) -> None:
    output_parent.mkdir(parents=True, exist_ok=True)

    (output_parent / "partition_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Dialogue Partition Summary",
        "==========================",
        "",
        f"Input folder: {summary['input_folder']}",
        f"Output parent: {summary['output_parent']}",
        f"Mode: {summary['mode']}",
        f"Matched videos: {summary['matched_video_count']}",
        f"Skipped files: {summary['skipped_file_count']}",
        f"Dialogues: {summary['dialogue_count']}",
        f"Clip length (seconds): {summary['clip_length_seconds']}",
        "",
        "Successful Group Counts",
        "-----------------------",
        f"1-utt groups: {summary['group_counts']['1']}",
        f"2-utt groups: {summary['group_counts']['2']}",
        f"3-utt groups: {summary['group_counts']['3']}",
        "",
        "Generated Clip Counts",
        "---------------------",
        f"1-utt mp4 clips: {summary['clip_mp4_counts']['1']}",
        f"2-utt mp4 clips: {summary['clip_mp4_counts']['2']}",
        f"3-utt mp4 clips: {summary['clip_mp4_counts']['3']}",
        f"1-utt wav clips: {summary['clip_wav_counts']['1']}",
        f"2-utt wav clips: {summary['clip_wav_counts']['2']}",
        f"3-utt wav clips: {summary['clip_wav_counts']['3']}",
        "",
        f"Skipped windows: {summary['skipped_window_count']}",
        "",
        "Per-Dialogue",
        "------------",
    ]

    for dialogue in summary["dialogues"]:
        lines.append(
            f"dia{dialogue['dialogue_id']}: "
            f"available={dialogue['available_utterance_count']}, "
            f"max_utt={dialogue['max_utterance_id']}, "
            f"groups(1/2/3)=({dialogue['group_1_count']}/{dialogue['group_2_count']}/{dialogue['group_3_count']}), "
            f"skipped={dialogue['skipped_window_count']}"
        )

    (output_parent / "partition_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with (output_parent / "partition_groups.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dialogue_id",
                "group_size",
                "group_name",
                "utterance_ids",
                "source_paths",
            ],
        )
        writer.writeheader()
        for group in summary["groups"]:
            writer.writerow(
                {
                    "dialogue_id": group["dialogue_id"],
                    "group_size": group["group_size"],
                    "group_name": group["group_name"],
                    "utterance_ids": json.dumps(group["utterance_ids"]),
                    "source_paths": json.dumps(group["source_paths"]),
                }
            )

    with (output_parent / "partition_dialogues.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dialogue_id",
                "available_utterance_count",
                "max_utterance_id",
                "group_1_count",
                "group_2_count",
                "group_3_count",
                "skipped_window_count",
            ],
        )
        writer.writeheader()
        for dialogue in summary["dialogues"]:
            writer.writerow(
                {
                    "dialogue_id": dialogue["dialogue_id"],
                    "available_utterance_count": dialogue["available_utterance_count"],
                    "max_utterance_id": dialogue["max_utterance_id"],
                    "group_1_count": dialogue["group_1_count"],
                    "group_2_count": dialogue["group_2_count"],
                    "group_3_count": dialogue["group_3_count"],
                    "skipped_window_count": dialogue["skipped_window_count"],
                }
            )


def analyze_and_partition(
    video_folder: Path,
    output_parent: Path,
    clip_length: float,
    recursive: bool,
    mode: str,
    dialogue_range: int | None = None,
) -> dict[str, object]:
    if not video_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {video_folder}")
    if not video_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {video_folder}")

    records, skipped_files = collect_video_records(video_folder, recursive=recursive)

    if dialogue_range is not None:
        lo = (dialogue_range - 1) * 100
        hi = dialogue_range * 100
        records = [r for r in records if lo <= r.dialogue_id < hi]

    if not records:
        if dialogue_range is not None:
            print(f"No dialogues found in range [{lo}, {hi}). Nothing to do.")
            return {}
        raise ValueError(
            "No matching videos were found. Expected files named like dia{n}_utt{m}.mp4."
        )

    records_by_dialogue: dict[int, list[VideoRecord]] = defaultdict(list)
    for record in records:
        records_by_dialogue[record.dialogue_id].append(record)

    all_groups: list[PartitionGroup] = []
    all_skipped_windows: list[SkippedWindow] = []
    for dialogue_id in sorted(records_by_dialogue):
        dialogue_groups, dialogue_skipped = partition_dialogue(records_by_dialogue[dialogue_id])
        all_groups.extend(dialogue_groups)
        all_skipped_windows.extend(dialogue_skipped)

    materialized_groups = [
        result
        for group in all_groups
        if (result := materialize_group(group, output_parent, clip_length, mode)) is not None
    ]

    summary = build_summary(
        video_folder=video_folder,
        output_parent=output_parent,
        records=records,
        skipped_files=skipped_files,
        groups=all_groups,
        skipped_windows=all_skipped_windows,
        materialized_groups=materialized_groups,
        clip_length=clip_length,
        mode=mode,
    )
    write_summary_files(summary, output_parent)
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze_and_partition(
        video_folder=args.video_folder,
        output_parent=args.output_parent,
        clip_length=args.clip_length,
        recursive=args.recursive,
        mode=args.mode,
        dialogue_range=args.dialogue_range,
    )

    if not summary:
        return

    print(f"Matched videos: {summary['matched_video_count']}")
    print(f"Dialogues: {summary['dialogue_count']}")
    print(
        "Groups (1/2/3): "
        f"{summary['group_counts']['1']}/"
        f"{summary['group_counts']['2']}/"
        f"{summary['group_counts']['3']}"
    )
    print(f"Mode: {summary['mode']}")
    print(f"Outputs written to: {args.output_parent.resolve()}")


if __name__ == "__main__":
    main()
