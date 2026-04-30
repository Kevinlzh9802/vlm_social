#!/usr/bin/env python3
"""Framework for Gemma 4 ingroup inference from a flat JSON manifest.

The input JSON is expected to contain either:

- a top-level list of records, or
- a top-level object with one of: ``records``, ``entries``, ``items``, ``data``.

Each retained record is converted into one Gemma chat turn with:

1. optional audio
2. video
3. prompt text rendered from ``prompt_ingroup.json``

This file is intentionally a framework first: it handles manifest loading,
path resolution, record filtering, prompt rendering, and result writing while
reusing the Gemma multimodal prompting path from ``batch_infer_context.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    from .batch_infer_context import (
        DEFAULT_MODEL_PATH,
        DEFAULT_SYSTEM_PROMPT,
        combine_system_and_user_prompt,
        infer_turn,
        select_video_num_frames,
    )
except ImportError:
    from batch_infer_context import (
        DEFAULT_MODEL_PATH,
        DEFAULT_SYSTEM_PROMPT,
        combine_system_and_user_prompt,
        infer_turn,
        select_video_num_frames,
    )


DEFAULT_PROMPT_CONFIG_PATH = Path(__file__).resolve().with_name("prompt_ingroup.json")


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 ingroup inference from a flat JSON manifest."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Local Gemma model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="JSON file containing ~1 record per inference item.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ingroup_results.json"),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_PROMPT_CONFIG_PATH,
        help="Prompt config JSON. Defaults to gemma/prompt_ingroup.json.",
    )
    parser.add_argument(
        "--media-path-prefix",
        default=None,
        help=(
            "Source prefix to replace in each record's media path, for example "
            "a URL/media prefix stored in the JSON."
        ),
    )
    parser.add_argument(
        "--local-path-prefix",
        type=Path,
        default=None,
        help=(
            "Local filesystem prefix that replaces --media-path-prefix in each "
            "record's media path before loading the media."
        ),
    )
    parser.add_argument(
        "--video-media-path-prefix",
        default=None,
        help="Optional video-specific source prefix override.",
    )
    parser.add_argument(
        "--video-local-path-prefix",
        type=Path,
        default=None,
        help="Optional video-specific local prefix override.",
    )
    parser.add_argument(
        "--audio-media-path-prefix",
        default=None,
        help="Optional audio-specific source prefix override.",
    )
    parser.add_argument(
        "--audio-local-path-prefix",
        type=Path,
        default=None,
        help="Optional audio-specific local prefix override.",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Record key used as the stable result id. Dotted paths are supported.",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="Optional root used to resolve relative media paths.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional override for the system prompt from the prompt config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Gemma thinking mode in the chat template when supported.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default generation is deterministic.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=32,
        help="Maximum frames to sample per video.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run video-only inference by omitting separate audio inputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of retained records to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First zero-based manifest index to consider.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last zero-based manifest index to consider, inclusive.",
    )
    parser.add_argument(
        "--exclude-video-substring",
        action="append",
        default=[],
        help="Skip records whose resolved video path contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--exclude-audio-substring",
        action="append",
        default=[],
        help="Skip records whose resolved audio path contains this substring. Repeatable.",
    )
    return parser.parse_args()


def get_nested_value(record: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = record
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def load_manifest_records(input_json_path: Path) -> list[dict[str, Any]]:
    with input_json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = None
        for key in ("records", "entries", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise ValueError(
                f"{input_json_path} must be a list or contain one of: "
                "records, entries, items, data"
            )
    else:
        raise ValueError(f"Unsupported JSON payload type in {input_json_path}: {type(payload)}")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {index} is not a JSON object: {record!r}")
        normalized.append(dict(record))
    return normalized


def load_prompt_config(prompt_config_path: Path) -> dict[str, str]:
    with prompt_config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Prompt config must be a JSON object: {prompt_config_path}")

    system_prompt = payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str):
        raise ValueError(f"{prompt_config_path} field 'system_prompt' must be a string.")

    user_prompt_template = payload.get("user_prompt_template", payload.get("prompt"))
    if not isinstance(user_prompt_template, str) or not user_prompt_template.strip():
        sections: list[str] = []

        intro = payload.get("intro")
        if isinstance(intro, str) and intro.strip():
            sections.append(intro.strip())

        questions = payload.get("questions")
        if isinstance(questions, Sequence) and not isinstance(
            questions, (str, bytes, bytearray)
        ):
            question_lines = ["Questions:"]
            for index, question in enumerate(questions, start=1):
                if not isinstance(question, Mapping):
                    continue
                label = question.get("label")
                prompt = question.get("prompt")
                response_format = question.get("response_format")
                parts: list[str] = []
                if isinstance(label, str) and label.strip():
                    parts.append(f"{index}. {label.strip()}")
                if isinstance(prompt, str) and prompt.strip():
                    parts.append(prompt.strip())
                if isinstance(response_format, str) and response_format.strip():
                    parts.append(f"Response format: {response_format.strip()}")
                if parts:
                    question_lines.append("\n".join(parts))
            if len(question_lines) > 1:
                sections.append("\n\n".join(question_lines))

        examples = payload.get("examples")
        if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes, bytearray)):
            example_lines = ["Examples:"]
            for index, example in enumerate(examples, start=1):
                if isinstance(example, str) and example.strip():
                    example_lines.append(f"{index}. {example.strip()}")
                elif isinstance(example, Mapping):
                    title = example.get("title")
                    content = example.get("content")
                    parts: list[str] = []
                    if isinstance(title, str) and title.strip():
                        parts.append(f"{index}. {title.strip()}")
                    if isinstance(content, str) and content.strip():
                        parts.append(content.strip())
                    if parts:
                        example_lines.append("\n".join(parts))
            if len(example_lines) > 1:
                sections.append("\n\n".join(example_lines))

        user_prompt_template = "\n\n".join(section for section in sections if section.strip())
        if not user_prompt_template.strip():
            raise ValueError(
                f"{prompt_config_path} must define 'user_prompt_template' (or 'prompt'), "
                "or provide structured 'intro'/'questions'/'examples' fields."
            )

    return {
        "system_prompt": system_prompt.strip(),
        "user_prompt_template": user_prompt_template.strip(),
    }


def flatten_record_for_prompt(
    record: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_record_for_prompt(value, prefix=flat_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[flat_key] = value
            if prefix == "":
                flattened.setdefault(str(key), value)
    return flattened


def render_prompt(template: str, record: Mapping[str, Any]) -> str:
    format_values = SafeFormatDict(flatten_record_for_prompt(record))
    format_values["record_json"] = json.dumps(record, ensure_ascii=False)
    return template.format_map(format_values).strip()


def resolve_media_path(
    raw_path: Any,
    *,
    manifest_dir: Path,
    media_root: Path | None,
) -> Path | None:
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate

    manifest_candidate = (manifest_dir / candidate).resolve()
    if manifest_candidate.exists():
        return manifest_candidate

    if media_root is not None:
        return (media_root / candidate).resolve()
    return manifest_candidate


def resolve_prefixed_media_path(
    raw_media_path: Any,
    *,
    manifest_dir: Path,
    media_root: Path | None,
    media_path_prefix: str | None,
    local_path_prefix: Path | None,
) -> tuple[Path | None, str | None]:
    if raw_media_path is None:
        return None, None
    if not isinstance(raw_media_path, str) or not raw_media_path.strip():
        return None, None

    rewritten_path = raw_media_path
    if media_path_prefix:
        if local_path_prefix is None:
            raise ValueError(
                "--local-path-prefix is required when --media-path-prefix is provided."
            )
        if raw_media_path.startswith(media_path_prefix):
            suffix = raw_media_path[len(media_path_prefix) :].lstrip("/\\")
            rewritten_path = str(local_path_prefix.joinpath(*suffix.split("/")))

    resolved_path = resolve_media_path(
        rewritten_path,
        manifest_dir=manifest_dir,
        media_root=media_root,
    )
    return resolved_path, rewritten_path


def effective_media_prefix(
    specific_prefix: str | None,
    shared_prefix: str | None,
) -> str | None:
    return specific_prefix if specific_prefix is not None else shared_prefix


def effective_local_prefix(
    specific_prefix: Path | None,
    shared_prefix: Path | None,
) -> Path | None:
    return specific_prefix if specific_prefix is not None else shared_prefix


def select_audio_speakers(record: Mapping[str, Any]) -> tuple[int, list[int]] | None:
    participant = record.get("participant")
    if not isinstance(participant, int) or participant <= 0:
        return None

    conversation_floor = record.get("conversation_floor", [])
    if conversation_floor is None:
        conversation_floor = []
    if not isinstance(conversation_floor, Sequence) or isinstance(
        conversation_floor, (str, bytes, bytearray)
    ):
        return None

    floor_ids: list[int] = []
    for speaker_id in conversation_floor:
        if not isinstance(speaker_id, int) or speaker_id <= 0:
            return None
        if speaker_id != participant and speaker_id not in floor_ids:
            floor_ids.append(speaker_id)
    return participant, floor_ids


def path_matches_any(path: Path | None, patterns: Sequence[str]) -> bool:
    if path is None:
        return False
    path_text = str(path)
    return any(pattern and pattern in path_text for pattern in patterns)


def safe_filename_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "record"


def load_audio_for_mix(audio_path: Path, target_sampling_rate: int | None = None) -> tuple[Any, int]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required to aggregate conversation-floor audio") from exc

    audio, sampling_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
        audio = librosa.resample(
            y=audio,
            orig_sr=sampling_rate,
            target_sr=target_sampling_rate,
        )
        sampling_rate = target_sampling_rate
    return audio, int(sampling_rate)


def write_audio_for_mix(audio_path: Path, audio: Any, sampling_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to write aggregated audio") from exc

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio, sampling_rate, subtype="FLOAT")


def aggregate_conversation_floor_audio(
    *,
    participant_audio_path: Path,
    conversation_floor_audio_paths: Sequence[Path],
    output_audio_path: Path,
    record_id: str,
    length_tolerance_seconds: float = 1.0,
    length_tolerance_ratio: float = 0.05,
) -> list[str]:
    import numpy as np

    warnings: list[str] = []
    participant_audio, sampling_rate = load_audio_for_mix(participant_audio_path)
    target_length = len(participant_audio)
    tolerance_samples = max(
        int(sampling_rate * length_tolerance_seconds),
        int(target_length * length_tolerance_ratio),
    )
    mixed_tracks: list[Any] = []

    for audio_path in conversation_floor_audio_paths:
        audio, _ = load_audio_for_mix(audio_path, target_sampling_rate=sampling_rate)
        length_delta = len(audio) - target_length
        if abs(length_delta) > tolerance_samples:
            warning = (
                f"record {record_id}: discarded conversation-floor audio "
                f"{audio_path} because length differs by {length_delta} sample(s)"
            )
            print(f"[WARN] {warning}", flush=True)
            warnings.append(warning)
            continue

        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            audio = audio[:target_length]
        mixed_tracks.append(audio)

    if not mixed_tracks:
        warning = (
            f"record {record_id}: no valid conversation-floor audio tracks to aggregate; "
            "using silence for audio2"
        )
        print(f"[WARN] {warning}", flush=True)
        warnings.append(warning)
        mixed_audio = np.zeros(target_length, dtype=np.float32)
    else:
        mixed_audio = np.sum(np.stack(mixed_tracks, axis=0), axis=0).astype(np.float32)
        peak = float(np.max(np.abs(mixed_audio))) if mixed_audio.size else 0.0
        if peak > 1.0:
            mixed_audio = mixed_audio / peak

    write_audio_for_mix(output_audio_path, mixed_audio, sampling_rate)
    return warnings


def prepare_record(
    *,
    record: dict[str, Any],
    record_index: int,
    id_key: str,
    manifest_dir: Path,
    media_root: Path | None,
    media_path_prefix: str | None,
    local_path_prefix: Path | None,
    video_media_path_prefix: str | None,
    video_local_path_prefix: Path | None,
    audio_media_path_prefix: str | None,
    audio_local_path_prefix: Path | None,
    no_audio: bool,
    aggregated_audio_dir: Path,
    exclude_video_substrings: Sequence[str],
    exclude_audio_substrings: Sequence[str],
    user_prompt_template: str,
) -> tuple[dict[str, Any] | None, str | None]:
    record_id = get_nested_value(record, id_key)
    if record_id is None:
        record_id = record.get("id", record_index)

    resolved_video_media_path_prefix = effective_media_prefix(
        video_media_path_prefix, media_path_prefix
    )
    resolved_video_local_path_prefix = effective_local_prefix(
        video_local_path_prefix, local_path_prefix
    )
    source_video_path = record.get("video")
    video_path, rewritten_video_path = resolve_prefixed_media_path(
        source_video_path,
        manifest_dir=manifest_dir,
        media_root=media_root,
        media_path_prefix=resolved_video_media_path_prefix,
        local_path_prefix=resolved_video_local_path_prefix,
    )
    if video_path is None:
        return None, "missing_video_path"
    if path_matches_any(video_path, exclude_video_substrings):
        return None, "excluded_video"
    if not video_path.exists():
        return None, "video_not_found"

    source_audio_paths: list[str] = []
    rewritten_audio_paths: list[str] = []
    audio_paths: list[str] = []
    participant_speaker_id: int | None = None
    conversation_floor_speaker_ids: list[int] = []
    participant_audio_path: str | None = None
    conversation_floor_audio_paths: list[str] = []
    aggregated_conversation_floor_audio_path: str | None = None
    audio_warnings: list[str] = []
    if not no_audio:
        resolved_audio_media_path_prefix = effective_media_prefix(
            audio_media_path_prefix, media_path_prefix
        )
        resolved_audio_local_path_prefix = effective_local_prefix(
            audio_local_path_prefix, local_path_prefix
        )
        audio_entries = record.get("audio")
        if not isinstance(audio_entries, Sequence) or isinstance(
            audio_entries, (str, bytes, bytearray)
        ):
            return None, "invalid_audio_list"

        speaker_selection = select_audio_speakers(record)
        if speaker_selection is None:
            return None, "invalid_speaker_selection"

        participant_speaker_id, conversation_floor_speaker_ids = speaker_selection

        def resolve_audio_for_speaker(
            speaker_id: int,
        ) -> tuple[str, str, Path] | tuple[None, None, None]:
            audio_index = speaker_id - 1
            if audio_index < 0 or audio_index >= len(audio_entries):
                return None, None, None

            source_audio_path = audio_entries[audio_index]
            resolved_audio_path, rewritten_audio_path = resolve_prefixed_media_path(
                source_audio_path,
                manifest_dir=manifest_dir,
                media_root=media_root,
                media_path_prefix=resolved_audio_media_path_prefix,
                local_path_prefix=resolved_audio_local_path_prefix,
            )
            if resolved_audio_path is None:
                return None, None, None
            if path_matches_any(resolved_audio_path, exclude_audio_substrings):
                return None, None, None
            if not resolved_audio_path.exists():
                return None, None, None

            return (
                str(source_audio_path),
                "" if rewritten_audio_path is None else str(rewritten_audio_path),
                resolved_audio_path,
            )

        resolved_participant_audio = resolve_audio_for_speaker(participant_speaker_id)
        if resolved_participant_audio == (None, None, None):
            return None, "participant_audio_not_found"

        source_participant_audio, rewritten_participant_audio, participant_audio = (
            resolved_participant_audio
        )
        source_audio_paths.append(str(source_participant_audio))
        rewritten_audio_paths.append(str(rewritten_participant_audio))
        participant_audio_path = str(participant_audio)

        resolved_conversation_floor_paths: list[Path] = []
        for speaker_id in conversation_floor_speaker_ids:
            resolved_floor_audio = resolve_audio_for_speaker(speaker_id)
            if resolved_floor_audio == (None, None, None):
                return None, "conversation_floor_audio_not_found"
            source_audio_path, rewritten_audio_path, resolved_audio_path = resolved_floor_audio
            source_audio_paths.append(str(source_audio_path))
            rewritten_audio_paths.append(str(rewritten_audio_path))
            conversation_floor_audio_paths.append(str(resolved_audio_path))
            resolved_conversation_floor_paths.append(resolved_audio_path)

        aggregate_audio_name = (
            f"{record_index:06d}_{safe_filename_part(record_id)}_conversation_floor.wav"
        )
        aggregate_audio_path = aggregated_audio_dir / aggregate_audio_name
        try:
            audio_warnings = aggregate_conversation_floor_audio(
                participant_audio_path=participant_audio,
                conversation_floor_audio_paths=resolved_conversation_floor_paths,
                output_audio_path=aggregate_audio_path,
                record_id=str(record_id),
            )
        except Exception as exc:
            print(
                f"[WARN] record {record_id}: failed to aggregate conversation-floor audio: {exc}",
                flush=True,
            )
            return None, "audio_aggregation_failed"

        aggregated_conversation_floor_audio_path = str(aggregate_audio_path)
        audio_paths = [str(participant_audio), str(aggregate_audio_path)]

    return (
        {
            "record_index": record_index,
            "record_id": str(record_id),
            "source_video_path": None if source_video_path is None else str(source_video_path),
            "rewritten_video_path": rewritten_video_path,
            "video_path": str(video_path),
            "speaker_ids": [] if no_audio else [
                participant_speaker_id,
                *conversation_floor_speaker_ids,
            ],
            "participant_speaker_id": participant_speaker_id,
            "conversation_floor_speaker_ids": conversation_floor_speaker_ids,
            "source_audio_paths": source_audio_paths,
            "rewritten_audio_paths": rewritten_audio_paths,
            "audio_paths": audio_paths,
            "participant_audio_path": participant_audio_path,
            "conversation_floor_audio_paths": conversation_floor_audio_paths,
            "aggregated_conversation_floor_audio_path": (
                aggregated_conversation_floor_audio_path
            ),
            "audio_warnings": audio_warnings,
            "user_prompt": render_prompt(user_prompt_template, record),
            "source_record": record,
        },
        None,
    )


def main() -> None:
    args = parse_args()

    if args.start_index < 0:
        raise ValueError(f"--start-index must be non-negative: {args.start_index}")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError(
            f"--end-index must be greater than or equal to --start-index: "
            f"{args.end_index} < {args.start_index}"
        )

    input_json_path = args.input_json.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    prompt_config_path = args.prompt_config.expanduser().resolve()
    media_root = None if args.media_root is None else args.media_root.expanduser().resolve()
    local_path_prefix = (
        None if args.local_path_prefix is None else args.local_path_prefix.expanduser().resolve()
    )
    video_local_path_prefix = (
        None
        if args.video_local_path_prefix is None
        else args.video_local_path_prefix.expanduser().resolve()
    )
    audio_local_path_prefix = (
        None
        if args.audio_local_path_prefix is None
        else args.audio_local_path_prefix.expanduser().resolve()
    )

    if not input_json_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")
    if not prompt_config_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompt_config_path}")

    prompt_config = load_prompt_config(prompt_config_path)
    system_prompt = (
        args.system_prompt.strip()
        if isinstance(args.system_prompt, str) and args.system_prompt.strip()
        else prompt_config["system_prompt"]
    )

    all_records = load_manifest_records(input_json_path)
    end_index = len(all_records) - 1 if args.end_index is None else args.end_index
    selected_records = [
        (record_index, record)
        for record_index, record in enumerate(all_records)
        if args.start_index <= record_index <= end_index
    ]
    print(f"[INFO] Loaded {len(all_records)} record(s) from {input_json_path}")
    print(
        f"[INFO] Selected manifest index range: "
        f"{args.start_index}-{end_index} ({len(selected_records)} record(s))"
    )
    print(f"[INFO] Prompt config: {prompt_config_path}")
    print(f"[INFO] Media path prefix: {args.media_path_prefix}")
    print(f"[INFO] Local path prefix: {local_path_prefix}")
    print(f"[INFO] Video media path prefix: {args.video_media_path_prefix}")
    print(f"[INFO] Video local path prefix: {video_local_path_prefix}")
    print(f"[INFO] Audio media path prefix: {args.audio_media_path_prefix}")
    print(f"[INFO] Audio local path prefix: {audio_local_path_prefix}")
    print(f"[INFO] No audio: {args.no_audio}")
    print(f"[INFO] Max video frames: {args.max_video_frames}")

    kept_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    manifest_dir = input_json_path.parent
    aggregated_audio_dir = output_path.parent / "_audio_mixes" / output_path.stem

    for record_index, record in selected_records:
        prepared, skip_reason = prepare_record(
            record=record,
            record_index=record_index,
            id_key=args.id_key,
            manifest_dir=manifest_dir,
            media_root=media_root,
            media_path_prefix=args.media_path_prefix,
            local_path_prefix=local_path_prefix,
            video_media_path_prefix=args.video_media_path_prefix,
            video_local_path_prefix=video_local_path_prefix,
            audio_media_path_prefix=args.audio_media_path_prefix,
            audio_local_path_prefix=audio_local_path_prefix,
            no_audio=args.no_audio,
            aggregated_audio_dir=aggregated_audio_dir,
            exclude_video_substrings=args.exclude_video_substring,
            exclude_audio_substrings=args.exclude_audio_substring,
            user_prompt_template=prompt_config["user_prompt_template"],
        )
        if prepared is None:
            skip_reason = skip_reason or "filtered_out"
            skip_counter[skip_reason] += 1
            skipped_records.append(
                {
                    "record_index": record_index,
                    "record_id": str(get_nested_value(record, args.id_key) or record_index),
                    "skip_reason": skip_reason,
                }
            )
            continue
        kept_records.append(prepared)
        if args.limit is not None and len(kept_records) >= args.limit:
            break

    print(
        f"[INFO] Retained {len(kept_records)} record(s); "
        f"skipped {sum(skip_counter.values())} before inference."
    )
    if skip_counter:
        for reason, count in sorted(skip_counter.items()):
            print(f"[INFO]   skip {reason}: {count}")
    if not kept_records:
        print("[WARN] Nothing to process. Exiting.")
        return

    print(f"[INFO] Loading Gemma model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    results: list[dict[str, Any]] = []
    total_records = len(kept_records)
    for processed_index, item in enumerate(kept_records, start=1):
        print(
            f"[{processed_index}/{total_records}] "
            f"{item['record_id']} -> {Path(item['video_path']).name}",
            flush=True,
        )
        video_num_frames = select_video_num_frames(
            video_path=item["video_path"],
            max_video_frames=args.max_video_frames,
        )
        text_prompt = combine_system_and_user_prompt(
            system_prompt=system_prompt,
            user_prompt=item["user_prompt"],
        )

        user_content: list[dict[str, Any]] = []
        if not args.no_audio:
            for audio_path in item["audio_paths"]:
                user_content.append({"type": "audio", "audio": audio_path})
        user_content.extend(
            [
                {
                    "type": "video",
                    "video": item["video_path"],
                    "num_frames": video_num_frames,
                },
                {"type": "text", "text": text_prompt},
            ]
        )

        try:
            response = infer_turn(
                model=model,
                processor=processor,
                messages=[{"role": "user", "content": user_content}],
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        except Exception as exc:
            response = f"[ERROR] {exc}"
            print(f"  [WARN] Error: {exc}", flush=True)

        results.append(
            {
                "record_index": item["record_index"],
                "record_id": item["record_id"],
                "source_video_path": item["source_video_path"],
                "rewritten_video_path": item["rewritten_video_path"],
                "video_path": item["video_path"],
                "speaker_ids": item["speaker_ids"],
                "participant_speaker_id": item["participant_speaker_id"],
                "conversation_floor_speaker_ids": item["conversation_floor_speaker_ids"],
                "source_audio_paths": item["source_audio_paths"],
                "rewritten_audio_paths": item["rewritten_audio_paths"],
                "audio_paths": item["audio_paths"],
                "participant_audio_path": item["participant_audio_path"],
                "conversation_floor_audio_paths": item["conversation_floor_audio_paths"],
                "aggregated_conversation_floor_audio_path": (
                    item["aggregated_conversation_floor_audio_path"]
                ),
                "audio_warnings": item["audio_warnings"],
                "system": system_prompt,
                "user": item["user_prompt"],
                "assistant": response,
            }
        )

    error_count = sum(1 for item in results if str(item["assistant"]).startswith("[ERROR]"))
    summary = {
        "input_json": str(input_json_path),
        "prompt_config": str(prompt_config_path),
        "record_count": len(all_records),
        "selected_record_count": len(selected_records),
        "start_index": args.start_index,
        "end_index": end_index,
        "retained_count": len(kept_records),
        "skipped_count": len(skipped_records),
        "processed_count": len(results),
        "error_count": error_count,
        "no_audio": args.no_audio,
        "max_video_frames": args.max_video_frames,
        "aggregated_audio_dir": None if args.no_audio else str(aggregated_audio_dir),
        "skip_reasons": dict(sorted(skip_counter.items())),
    }

    output_payload = {
        "__summary__": summary,
        "__skipped__": skipped_records,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
