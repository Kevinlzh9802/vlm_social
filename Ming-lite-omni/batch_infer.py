#!/usr/bin/env python3
"""Batch inference over subfolders containing .mp4 files and optional .wav pairs."""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import torch
from prompt_utils import (
    PROMPT_CONFIG_PATH,
    build_prompt_variant_key,
    load_prompt_templates,
)
from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available

from modeling_bailingmm import BailingMMNativeForConditionalGeneration


def natural_sort_key(value: str):
    """Sort strings by text chunks and embedded integers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def collect_av_pairs(data_root: Path, include_audio: bool = True) -> Dict[str, List[dict]]:
    """Collect .mp4 files, optionally requiring matched .wav files."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}
    for subfolder in sorted(
        data_root.iterdir(), key=lambda path: natural_sort_key(path.name)
    ):
        if not subfolder.is_dir():
            continue

        pairs: List[dict] = []
        video_files = {
            f.stem: f
            for f in sorted(
                subfolder.glob("*.mp4"), key=lambda path: natural_sort_key(path.name)
            )
        }
        for stem, video_path in sorted(
            video_files.items(), key=lambda item: natural_sort_key(item[0])
        ):
            pair = {
                "stem": stem,
                "video": str(video_path),
            }
            if include_audio:
                audio_path = subfolder / f"{stem}.wav"
                if not audio_path.exists():
                    print(f"[WARN] Missing audio for {video_path}, skip.")
                    continue
                pair["audio"] = str(audio_path)
            pairs.append(pair)

        if pairs:
            result[subfolder.name] = pairs

    return result


def resolve_attn_implementation(requested: str, device: str) -> str:
    if requested not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported attention implementation: {requested}")

    if device != "cuda":
        resolved = "eager" if requested == "auto" else requested
        print(f"[INFO] Attention implementation: requested={requested}, resolved={resolved}")
        return resolved

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA attention selection requested, but CUDA is unavailable.")

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    has_fa2 = is_flash_attn_2_available()

    if requested == "flash_attention_2":
        if not has_fa2:
            print("[WARN] flash-attn is not installed; falling back to sdpa.")
            resolved = "sdpa"
        elif major >= 10:
            print(
                "[WARN] flash_attention_2 is not reliable on Blackwell-class GPUs "
                f"(detected {device_name}, compute capability {major}.{minor}); "
                "falling back to sdpa."
            )
            resolved = "sdpa"
        else:
            resolved = "flash_attention_2"
    elif requested == "auto":
        resolved = "flash_attention_2" if has_fa2 and 8 <= major < 10 else "sdpa"
    else:
        resolved = requested

    print(
        "[INFO] Attention implementation: "
        f"requested={requested}, resolved={resolved}, "
        f"gpu='{device_name}', capability={major}.{minor}, flash_attn_2_available={has_fa2}"
    )
    return resolved


def cuda_diagnostics() -> str:
    lines = [
        f"torch={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')}",
    ]
    try:
        lines.append(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    except Exception as exc:
        lines.append(f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}")
    try:
        lines.append(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    except Exception as exc:
        lines.append(f"torch.cuda.device_count() raised {type(exc).__name__}: {exc}")
    try:
        if torch.cuda.device_count() > 0:
            lines.append(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        lines.append(f"torch.cuda.get_device_name(0) raised {type(exc).__name__}: {exc}")
    try:
        torch.empty(1, device="cuda")
        lines.append("torch.empty(..., device='cuda')=ok")
    except Exception as exc:
        lines.append(f"torch.empty(..., device='cuda') raised {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def wait_for_cuda(timeout_seconds: int = 60, interval_seconds: int = 10) -> None:
    deadline = time.monotonic() + timeout_seconds
    attempt = 1
    last_diagnostics = ""

    while True:
        if torch.cuda.is_available():
            try:
                torch.empty(1, device="cuda")
                return
            except Exception:
                pass

        last_diagnostics = cuda_diagnostics()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "CUDA is requested but not usable after "
                f"{timeout_seconds} seconds.\n{last_diagnostics}"
            )

        print(
            "[WARN] CUDA is not usable yet; retrying "
            f"in {interval_seconds}s (attempt {attempt}).\n{last_diagnostics}",
            flush=True,
        )
        time.sleep(interval_seconds)
        attempt += 1


def load_model_and_processor(model_path: str, device: str, attn_implementation: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    resolved_attn_implementation = resolve_attn_implementation(attn_implementation, device)

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": resolved_attn_implementation,
        "low_cpu_mem_usage": True,
        # Batch AV inference does not use Ming's image-generation path, so skip
        # loading the extra diffusion/connector checkpoints.
        "load_image_gen_modules": False,
    }
    if device == "cuda":
        kwargs["device_map"] = {"": 0}

    model = BailingMMNativeForConditionalGeneration.from_pretrained(model_path, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model = model.eval()
    return model, processor


@torch.inference_mode()
def infer_messages(
    model,
    processor,
    messages: List[dict],
    max_new_tokens: int,
) -> str:
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    for key in inputs.keys():
        if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
            inputs[key] = inputs[key].to(dtype=torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def run_batch(
    model,
    processor,
    pairs_by_subfolder: Dict[str, List[dict]],
    prompt_templates: Dict[str, str],
    conversation_mode: str,
    max_new_tokens: int,
    max_frames: int,
    sample: str,
    no_audio: bool,
) -> Dict[str, List[dict]]:
    output: Dict[str, List[dict]] = {}

    for subfolder, pairs in pairs_by_subfolder.items():
        print(f"[INFO] Processing {subfolder}: {len(pairs)} pairs")
        items: List[dict] = []
        conversation_messages: List[dict] = []

        for pair_idx, pair in enumerate(pairs):
            stem = pair["stem"]
            if conversation_mode == "single-turn":
                prompt_template = prompt_templates["text"]
            else:
                prompt_template = (
                    prompt_templates["first"]
                    if pair_idx == 0
                    else prompt_templates["after"]
                )

            content = [
                {
                    "type": "video",
                    "video": pair["video"],
                    "max_frames": max_frames,
                    "sample": sample,
                },
            ]
            if not no_audio:
                content.append({"type": "audio", "audio": pair["audio"]})
            content.append({"type": "text", "text": prompt_template})

            user_message = {
                "role": "HUMAN",
                "content": content,
            }

            if conversation_mode == "multi-turn":
                messages = conversation_messages
                messages.append(user_message)
            else:
                messages = [user_message]

            try:
                response = infer_messages(
                    model=model,
                    processor=processor,
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                )
                if conversation_mode == "multi-turn":
                    conversation_messages.append(
                        {
                            "role": "ASSISTANT",
                            "content": [{"type": "text", "text": response}],
                        }
                    )
                print(f"[OK] {subfolder}/{stem}")
            except Exception as exc:
                if conversation_mode == "multi-turn":
                    conversation_messages.pop()
                response = f"[ERROR] {type(exc).__name__}: {exc}"
                print(f"[ERR] {subfolder}/{stem}: {response}")

            items.append(
                {
                    "file": stem,
                    "response": response,
                }
            )

        output[subfolder] = items

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch video/audio or video-only inference for Ming-Lite-Omni."
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        type=str,
        default=os.environ.get("MODEL_PATH", "."),
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
        choices=("context", "nested"),
        default="context",
        help="Inference layout mode.",
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
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--sample", type=str, default="uniform", choices=["uniform", "fps"])
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run video-only inference: do not require .wav files or pass audio inputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        wait_for_cuda()

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
    print(f"[INFO] No audio: {args.no_audio}\n")

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    pairs = collect_av_pairs(data_root, include_audio=not args.no_audio)
    total_pairs = sum(len(value) for value in pairs.values())
    input_label = "video" if args.no_audio else "video/audio"
    print(
        f"[INFO] Found {total_pairs} {input_label} item(s) across {len(pairs)} subfolder(s)."
    )
    if total_pairs == 0:
        print("[WARN] Nothing to process. Exiting.")
        return

    print(f"[INFO] Loading model: {args.model_path}")
    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )

    results = run_batch(
        model=model,
        processor=processor,
        pairs_by_subfolder=pairs,
        prompt_templates=prompt_templates,
        conversation_mode=args.conversation_mode,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        sample=args.sample,
        no_audio=args.no_audio,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Saved results to {output_path}")


if __name__ == "__main__":
    main()
