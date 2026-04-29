#!/usr/bin/env python3
"""Batch inference with Qwen2.5-Omni over a folder of video/audio pairs.

Expected data layout
====================
<data_root>/
    subfolder_A/
        d1u1-u2_clip_01.mp4
        d1u1-u2_clip_01.wav
        d3u2-u3_clip_02.mp4
        d3u2-u3_clip_02.wav
    subfolder_B/
        ...

The script discovers every {.mp4, .wav} pair that shares the same stem
inside each immediate subfolder of *data_root*, feeds each pair to the
model together with a user-defined prompt template, and writes the
collected responses to a single JSON file. Pairs in the same subfolder are
processed either as one continuous multi-turn chat or as independent
single-turn chats ordered by filename.

Output JSON structure
=====================
{
    "__summary__": {
        "overall": {"files": 10, "errors": 2, "error_ratio": 0.2},
        "subfolders": {
            "subfolder_A": {"files": 2, "errors": 1, "error_ratio": 0.5},
            "subfolder_B": {"files": 8, "errors": 1, "error_ratio": 0.125}
        }
    },
    "subfolder_A": [
        {"file": "d1u1-u2_clip_01", "system": "...", "user": "...", "assistant": "..."},
        {"file": "d3u2-u3_clip_02", "system": "...", "user": "...", "assistant": "..."}
    ],
    "subfolder_B": [...]
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from prompt_utils import PROMPT_CONFIG_PATH, build_prompt_variant_key, load_prompt_templates
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


# ---------------------------------------------------------------------------
# Folder traversal
# ---------------------------------------------------------------------------

def collect_av_pairs(data_root: str) -> Dict[str, List[dict]]:
    """Scan *data_root* for video/audio pairs.

    For every immediate subfolder in *data_root*, find all `.mp4` files and
    check whether a matching `.wav` file with the same stem exists.  Only
    complete pairs are returned.  The stem is kept verbatim, so filenames
    such as ``d1u1-u2_clip_01.mp4`` and ``d1u1-u2_clip_01.wav`` remain
    identified as ``d1u1-u2_clip_01``.

    Returns
    -------
    dict
        Mapping from subfolder name to a sorted list of dicts, each with
        keys ``"stem"``, ``"video"``, ``"audio"`` (absolute paths).
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        # Keep the full filename stem, including any dialogue/utterance prefix.
        video_files = {f.stem: f for f in sorted(subfolder.glob("*.mp4"))}
        pairs: List[dict] = []

        for stem, video_path in sorted(video_files.items()):
            audio_path = subfolder / f"{stem}.wav"
            if audio_path.exists():
                pairs.append(
                    {
                        "stem": stem,
                        "video": str(video_path),
                        "audio": str(audio_path),
                    }
                )
            else:
                print(f"[WARN] No matching .wav for {video_path}, skipping.")

        if pairs:
            result[subfolder.name] = pairs

    return result


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
    """Run one assistant turn for the current chat history."""
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
    """Return whether a model response represents a failed inference."""
    return response.startswith("[ERROR]")


def build_error_summary(results: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Build per-subfolder and overall error ratios from collected results."""
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
    """Print per-subfolder and overall error ratios."""
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
        description="Batch Qwen2.5-Omni inference over a folder of video/audio pairs."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "/scratch/zli33/models/Qwen2.5-Omni-7B"),
        help="Model id or local model path.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent folder containing subfolders of mp4/wav pairs.",
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
        choices=("context",),
        default="context",
        help="Inference layout mode. This script only supports 'context'.",
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
        default="single-turn",
        help=(
            "How to process pairs inside each subfolder: "
            "'multi-turn' keeps one running chat per subfolder, "
            "'single-turn' starts a fresh chat for each pair."
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load prompt templates ---
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
    if args.conversation_mode == "single-turn":
        print(f"[INFO] Prompt text:\n{prompt_templates['text']}\n")
    else:
        print(f"[INFO] First-turn prompt:\n{prompt_templates['first']}\n")
        print(f"[INFO] Follow-up prompt:\n{prompt_templates['after']}\n")
    print(f"[INFO] Utterance count: {args.utt_count}")
    print(f"[INFO] Conversation mode: {args.conversation_mode}\n")

    # --- Collect all video/audio pairs ---
    av_pairs = collect_av_pairs(args.data_root)
    total_pairs = sum(len(v) for v in av_pairs.values())
    print(f"[INFO] Found {total_pairs} video/audio pair(s) across {len(av_pairs)} subfolder(s).")
    if total_pairs == 0:
        print("[WARN] Nothing to process. Exiting.")
        return

    # --- Load model ---
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] Loading model: {args.model}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)

    # --- Run inference ---
    results: Dict[str, list] = {}
    processed = 0

    for subfolder_name, pairs in av_pairs.items():
        sorted_pairs = sorted(
            pairs,
            key=lambda pair: (Path(pair["video"]).name, Path(pair["audio"]).name),
        )
        subfolder_results: list = []
        conversation_messages = [
            {"role": "system", "content": [{"type": "text", "text": args.system_prompt}]}
        ]
        for pair_index, pair in enumerate(sorted_pairs):
            processed += 1
            if args.conversation_mode == "single-turn":
                prompt_template = prompt_templates["text"]
            else:
                prompt_template = (
                    prompt_templates["first"] if pair_index == 0 else prompt_templates["after"]
                )
            print(
                f"[{processed}/{total_pairs}] {subfolder_name}/{pair['stem']}  ...",
                flush=True,
            )
            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": pair["audio"]},
                    {"type": "video", "video": pair["video"]},
                    {"type": "text", "text": prompt_template},
                ],
            }

            if args.conversation_mode == "multi-turn":
                messages = conversation_messages
                messages.append(user_message)
            else:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": args.system_prompt}],
                    },
                    user_message,
                ]

            try:
                response = infer_turn(
                    model=model,
                    processor=processor,
                    messages=messages,
                    use_audio_in_video=args.use_audio_in_video,
                    max_new_tokens=args.max_new_tokens,
                )
                if args.conversation_mode == "multi-turn":
                    conversation_messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response}],
                        }
                    )
            except Exception as exc:
                if args.conversation_mode == "multi-turn":
                    conversation_messages.pop()
                response = f"[ERROR] {exc}"
                print(f"  [WARN] Error: {exc}")

            subfolder_results.append(
                {
                    "file": pair["stem"],
                    "system": args.system_prompt,
                    "user": prompt_template,
                    "assistant": response,
                }
            )

        results[subfolder_name] = subfolder_results

    summary = build_error_summary(results)
    print_error_summary(summary)

    output_payload = {"__summary__": summary, **results}

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[INFO] Results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
