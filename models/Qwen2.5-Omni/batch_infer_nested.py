#!/usr/bin/env python3
"""Batch inference with Qwen2.5-Omni over nested dialogue folders.

Expected layout
===============
<data_root>/
    sample_A/
        d1u1.mp4
        d1u2.mp4
        d1u3/
            d1u3_clip1.mp4
            d1u3_clip1.wav
            d1u3_clip2.mp4
            d1u3_clip2.wav
    sample_B/
        ...

This script is dedicated to the "nested" dataset layout and only supports a
multi-round conversation setting:

- ``--group 3``: round 1 uses the concatenation of the two whole-context
  videos (for example ``d1u1.mp4 + d1u2.mp4``), then the conversation
  continues with every clip from the final utterance folder (for example
  ``d1u3/*.mp4``).
- ``--group 2``: round 1 uses only the most recent whole-context video
  (for example ``d1u2.mp4``), then continues with the final utterance clips.
- ``--group 1``: no context round; the conversation starts directly from the
  final utterance clips.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from prompt_utils import PROMPT_CONFIG_PATH, build_prompt_variant_key, load_prompt_templates
from qwen_omni_utils import process_mm_info
from torch_compat import ensure_cuda_runtime_compatibility
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

UTTERANCE_PATTERN = re.compile(r"^d(?P<dialogue>\d+)u(?P<utterance>\d+)$")
CLIP_INDEX_PATTERN = re.compile(r"^(?P<prefix>.+?)_clip(?P<index>\d+)$")


def parse_utterance_label(label: str) -> tuple[int, int] | None:
    match = UTTERANCE_PATTERN.fullmatch(label)
    if match is None:
        return None
    return int(match["dialogue"]), int(match["utterance"])


def clip_sort_key(path: Path) -> tuple[str, int, str]:
    match = CLIP_INDEX_PATTERN.fullmatch(path.stem)
    if match is None:
        return path.stem, -1, path.name
    return match["prefix"], int(match["index"]), path.name


def build_context_stem(source_videos: Sequence[Path]) -> str:
    joined = "+".join(path.stem for path in source_videos)
    return f"{joined}_context"


# ---------------------------------------------------------------------------
# Folder traversal
# ---------------------------------------------------------------------------

def collect_nested_rounds(data_root: str, group: int) -> Dict[str, List[dict]]:
    """Scan *data_root* for nested dialogue groups.

    Each immediate subfolder is expected to contain zero or more whole-video
    context files named like ``d1u1.mp4`` and exactly one final utterance clip
    folder named like ``d1u3/``. The returned round list is always ordered as:
    optional context round first, followed by final-utterance clips.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        whole_videos: Dict[int, Path] = {}
        clip_folders: Dict[int, Path] = {}
        dialogue_ids: set[int] = set()

        for child in sorted(subfolder.iterdir()):
            if child.is_file() and child.suffix.lower() == ".mp4":
                parsed = parse_utterance_label(child.stem)
                if parsed is None:
                    continue
                dialogue_id, utterance_id = parsed
                dialogue_ids.add(dialogue_id)
                whole_videos[utterance_id] = child.resolve()
            elif child.is_dir():
                parsed = parse_utterance_label(child.name)
                if parsed is None:
                    continue
                dialogue_id, utterance_id = parsed
                dialogue_ids.add(dialogue_id)
                clip_folders[utterance_id] = child.resolve()

        if not clip_folders:
            print(f"[WARN] No nested clip folder found in {subfolder}, skipping.")
            continue
        if len(dialogue_ids) > 1:
            print(f"[WARN] Mixed dialogue ids found in {subfolder}, skipping.")
            continue

        final_utterance_id = max(clip_folders)
        clip_folder = clip_folders[final_utterance_id]
        clip_videos = sorted(clip_folder.glob("*.mp4"), key=clip_sort_key)
        if not clip_videos:
            print(f"[WARN] No clip videos found in {clip_folder}, skipping.")
            continue

        context_utterance_ids = list(
            range(final_utterance_id - (group - 1), final_utterance_id)
        )
        if any(utterance_id <= 0 for utterance_id in context_utterance_ids):
            print(
                f"[WARN] Invalid context utterance ids for {subfolder.name} "
                f"with group={group}, skipping."
            )
            continue

        missing_context = [
            utterance_id
            for utterance_id in context_utterance_ids
            if utterance_id not in whole_videos
        ]
        if missing_context:
            print(
                f"[WARN] Missing context video(s) {missing_context} in {subfolder}, skipping."
            )
            continue

        rounds: List[dict] = []
        if context_utterance_ids:
            context_videos = [whole_videos[utterance_id] for utterance_id in context_utterance_ids]
            rounds.append(
                {
                    "kind": "context",
                    "stem": build_context_stem(context_videos),
                    "source_videos": [str(path) for path in context_videos],
                }
            )

        for clip_video in clip_videos:
            clip_audio = clip_folder / f"{clip_video.stem}.wav"
            rounds.append(
                {
                    "kind": "clip",
                    "stem": clip_video.stem,
                    "video": str(clip_video.resolve()),
                    "audio": str(clip_audio.resolve()) if clip_audio.exists() else None,
                }
            )

        result[subfolder.name] = rounds

    return result


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def run_ffmpeg_with_video_codec_fallback(command_builder, context: str) -> None:
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


def concatenate_videos(source_paths: Sequence[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        concat_list_path = Path(handle.name)
        for source_path in source_paths:
            handle.write(f"file {source_path.resolve().as_posix()}\n")

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
                "-fflags",
                "+genpts",
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


def extract_wav_from_video(video_path: Path, output_path: Path) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-1000:] or "none"
        raise RuntimeError(
            f"Extracting wav from {video_path} failed.\n"
            f"Command: {' '.join(command)}\n"
            f"ffmpeg stderr: {stderr_tail}"
        )
    return output_path


def ensure_audio_path(video_path: Path, audio_path: str | None, temp_dir: Path) -> Path:
    if audio_path is not None:
        candidate = Path(audio_path)
        if candidate.is_file():
            return candidate

    extracted_path = temp_dir / f"{video_path.stem}.wav"
    if extracted_path.exists():
        return extracted_path

    return extract_wav_from_video(video_path, extracted_path)


def resolve_round_media(
    round_spec: dict,
    temp_dir: Path,
    include_audio: bool = True,
) -> tuple[Path, Path | None]:
    if round_spec["kind"] == "context":
        source_videos = [Path(path) for path in round_spec["source_videos"]]
        if len(source_videos) == 1:
            context_video = source_videos[0]
        else:
            context_video = temp_dir / f"{round_spec['stem']}.mp4"
            concatenate_videos(source_videos, context_video)
        if not include_audio:
            return context_video, None
        context_audio = ensure_audio_path(context_video, None, temp_dir)
        return context_video, context_audio

    video_path = Path(round_spec["video"])
    if not include_audio:
        return video_path, None
    audio_path = ensure_audio_path(video_path, round_spec["audio"], temp_dir)
    return video_path, audio_path


# ---------------------------------------------------------------------------
# Chat inference
# ---------------------------------------------------------------------------

def infer_turn(
    model,
    processor,
    messages: List[dict],
    use_audio_in_video: bool,
    max_new_tokens: int,
) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(
        messages, use_audio_in_video=use_audio_in_video
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=False,
    )

    prompt_length = inputs.input_ids.shape[1]
    generated_ids = output[:, prompt_length:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return response[0] if response else ""


def is_error_response(response: str) -> bool:
    return response.startswith("[ERROR]")


def build_error_summary(results: Dict[str, List[dict]]) -> Dict[str, dict]:
    subfolder_summary: Dict[str, dict] = {}
    total_files = 0
    total_errors = 0

    for subfolder_name, entries in results.items():
        file_count = len(entries)
        error_count = sum(1 for entry in entries if is_error_response(entry["assistant"]))
        total_files += file_count
        total_errors += error_count
        subfolder_summary[subfolder_name] = {
            "files": file_count,
            "errors": error_count,
            "error_ratio": (error_count / file_count) if file_count else 0.0,
        }

    return {
        "overall": {
            "files": total_files,
            "errors": total_errors,
            "error_ratio": (total_errors / total_files) if total_files else 0.0,
        },
        "subfolders": subfolder_summary,
    }


def print_error_summary(summary: Dict[str, dict]) -> None:
    overall = summary["overall"]
    print("\n[INFO] Error summary")
    print(
        "[INFO] Overall: "
        f"{overall['errors']}/{overall['files']} "
        f"({overall['error_ratio']:.4f})"
    )

    for subfolder_name, subfolder_stats in summary["subfolders"].items():
        print(
            f"[INFO] {subfolder_name}: "
            f"{subfolder_stats['errors']}/{subfolder_stats['files']} "
            f"({subfolder_stats['error_ratio']:.4f})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Qwen2.5-Omni inference over nested dialogue folders. "
            "This script only supports the multi-round setting."
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "/scratch/zli33/models/Qwen2.5-Omni-7B"),
        help="Model id or local model path.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent folder containing nested sample subfolders.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to the output JSON file (default: results.json).",
    )
    parser.add_argument(
        "--prompt-choice",
        required=True,
        help="Prompt family under prompts/prompts.json, e.g. 'intention' or 'affordance'.",
    )
    parser.add_argument(
        "--mode",
        choices=("nested",),
        default="nested",
        help="Inference layout mode. This script only supports 'nested'.",
    )
    parser.add_argument(
        "--utt-count",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help="Utterance count used to select the single_utt or multi_utt prompt variant.",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=("single-turn", "multi-turn"),
        default="multi-turn",
        help="Conversation mode. Nested inference requires 'multi-turn'.",
    )
    parser.add_argument(
        "--group",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help=(
            "Nested context setting. "
            "1: no context, 2: use the most recent whole video, "
            "3: use the concatenation of the two whole-context videos."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated thinker tokens.",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Also use the audio track embedded in video input.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run video-only inference by omitting separate audio inputs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_cuda_runtime_compatibility()

    if args.conversation_mode != "multi-turn":
        raise ValueError("Nested inference only supports conversation mode 'multi-turn'.")
    if args.group != args.utt_count:
        raise ValueError(
            f"Nested inference expects --group and --utt-count to match, got "
            f"group={args.group}, utt_count={args.utt_count}."
        )
    if args.no_audio and args.use_audio_in_video:
        raise ValueError("--no-audio cannot be combined with --use-audio-in-video.")

    prompt_variant_key = build_prompt_variant_key(args.prompt_choice, args.utt_count)
    prompt_templates = load_prompt_templates(
        prompt_choice=args.prompt_choice,
        conversation_mode=args.conversation_mode,
        utt_count=args.utt_count,
    )
    print(
        f"[INFO] Prompt config: {PROMPT_CONFIG_PATH} "
        f"(mode: {args.mode}, conversation_mode: {args.conversation_mode}, "
        f"choice: {args.prompt_choice}, variant: {prompt_variant_key})"
    )
    print(f"[INFO] First-turn prompt:\n{prompt_templates['first']}\n")
    print(f"[INFO] Follow-up prompt:\n{prompt_templates['after']}\n")
    print(f"[INFO] Utterance count: {args.utt_count}")
    print(f"[INFO] Nested group setting: {args.group}\n")
    print(f"[INFO] No audio: {args.no_audio}\n")

    nested_rounds = collect_nested_rounds(args.data_root, args.group)
    total_rounds = sum(len(rounds) for rounds in nested_rounds.values())
    print(
        f"[INFO] Found {total_rounds} round(s) across {len(nested_rounds)} subfolder(s)."
    )
    if total_rounds == 0:
        print("[WARN] Nothing to process. Exiting.")
        return

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] Loading model: {args.model}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)

    results: Dict[str, list] = {}
    processed = 0

    for subfolder_name, rounds in nested_rounds.items():
        subfolder_results: list = []
        conversation_messages = [
            {"role": "system", "content": [{"type": "text", "text": args.system_prompt}]}
        ]

        with tempfile.TemporaryDirectory(prefix=f"{subfolder_name}_") as tmpdir_name:
            temp_dir = Path(tmpdir_name)

            for round_index, round_spec in enumerate(rounds):
                processed += 1
                prompt_template = (
                    prompt_templates["first"] if round_index == 0 else prompt_templates["after"]
                )
                print(
                    f"[{processed}/{total_rounds}] {subfolder_name}/{round_spec['stem']}  ...",
                    flush=True,
                )

                appended_user = False
                try:
                    video_path, audio_path = resolve_round_media(
                        round_spec,
                        temp_dir,
                        include_audio=not args.no_audio,
                    )
                    user_content = [
                        {"type": "video", "video": str(video_path)},
                        {"type": "text", "text": prompt_template},
                    ]
                    if not args.no_audio:
                        user_content.insert(0, {"type": "audio", "audio": str(audio_path)})
                    user_message = {"role": "user", "content": user_content}
                    conversation_messages.append(user_message)
                    appended_user = True

                    response = infer_turn(
                        model=model,
                        processor=processor,
                        messages=conversation_messages,
                        use_audio_in_video=args.use_audio_in_video,
                        max_new_tokens=args.max_new_tokens,
                    )
                    conversation_messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response}],
                        }
                    )
                except Exception as exc:
                    if appended_user:
                        conversation_messages.pop()
                    response = f"[ERROR] {exc}"
                    print(f"  [WARN] Error: {exc}")

                subfolder_results.append(
                    {
                        "file": round_spec["stem"],
                        "round_type": round_spec["kind"],
                        "system": args.system_prompt,
                        "user": prompt_template,
                        "assistant": response,
                    }
                )

        results[subfolder_name] = subfolder_results

    summary = build_error_summary(results)
    print_error_summary(summary)

    output_payload = {"__summary__": summary, **results}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[INFO] Results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
