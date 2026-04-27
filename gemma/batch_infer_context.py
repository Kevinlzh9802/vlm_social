#!/usr/bin/env python3
"""Batch inference with Gemma 4 E4B over video/audio context folders.

Expected data layout and output JSON schema match the repository's
``batch_infer_context.py`` Qwen runner:

<data_root>/
    subfolder_A/
        d1u1-u2_clip_01.mp4
        d1u1-u2_clip_01.wav
        ...

The script discovers matching ``.mp4``/``.wav`` stems in each immediate
subfolder, runs either independent single-turn chats or one multi-turn chat
per subfolder, and writes all assistant responses to one JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/scratch/zli33/models/GemmaE4B")
DEFAULT_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "api_models" / "configs" / "prompts.json"
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful multimodal assistant."
DEFAULT_GEMMA4_CHAT_TEMPLATE = """{{ bos_token }}
{%- set loop_messages = messages -%}
{%- if messages and messages[0]['role'] in ['system', 'developer'] -%}
{{ '<|turn>system\n' }}
{%- if enable_thinking is defined and enable_thinking -%}
{{ '<|think|>\n' }}
{%- endif -%}
{%- for item in messages[0]['content'] -%}
{%- if item['type'] == 'text' -%}
{{ item['text'] | trim }}
{%- endif -%}
{%- endfor -%}
{{ '<turn|>\n' }}
{%- set loop_messages = messages[1:] -%}
{%- elif enable_thinking is defined and enable_thinking -%}
{{ '<|turn>system\n<|think|>\n<turn|>\n' }}
{%- endif -%}
{%- for message in loop_messages -%}
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
{{ '<|turn>' + role + '\n' }}
{%- for item in message['content'] -%}
{%- if item['type'] == 'text' -%}
{{ item['text'] | trim }}
{%- elif item['type'] == 'image' -%}
{{ '<|image|>' }}
{%- elif item['type'] == 'audio' -%}
{{ '<|audio|>' }}
{%- elif item['type'] == 'video' -%}
{{ '<|video|>' }}
{%- endif -%}
{%- endfor -%}
{{ '<turn|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|turn>model\n' }}
{%- endif -%}"""


def collect_av_pairs(data_root: str | Path, require_audio: bool = True) -> dict[str, list[dict]]:
    """Scan *data_root* for video/audio pairs or video-only inputs."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: dict[str, list[dict]] = {}
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        pairs: list[dict] = []
        for video_path in sorted(subfolder.glob("*.mp4")):
            audio_path = subfolder / f"{video_path.stem}.wav"
            if audio_path.exists():
                audio_value = str(audio_path)
            elif require_audio:
                print(f"[WARN] No matching .wav for {video_path}, skipping.")
                continue
            else:
                audio_value = None

            pairs.append(
                {
                    "stem": video_path.stem,
                    "video": str(video_path),
                    "audio": audio_value,
                }
            )

        if pairs:
            result[subfolder.name] = pairs

    return result


def get_video_frame_count(video_path: str | Path) -> int | None:
    """Return the video frame count when it can be inferred cheaply."""
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(video_path)) as container:
            video_stream = next(
                (stream for stream in container.streams if stream.type == "video"),
                None,
            )
            if video_stream is None:
                return None
            if video_stream.frames and video_stream.frames > 0:
                return int(video_stream.frames)
            if (
                video_stream.duration is not None
                and video_stream.time_base is not None
                and video_stream.average_rate is not None
            ):
                duration_seconds = float(video_stream.duration * video_stream.time_base)
                estimated_frames = int(duration_seconds * float(video_stream.average_rate))
                if estimated_frames > 0:
                    return estimated_frames

            decoded_frames = 0
            for _ in container.decode(video=0):
                decoded_frames += 1
            return decoded_frames if decoded_frames > 0 else None
    except Exception:
        return None


def select_video_num_frames(video_path: str | Path, max_video_frames: int) -> int:
    if max_video_frames <= 0:
        raise ValueError(f"max_video_frames must be positive, got {max_video_frames}")
    frame_count = get_video_frame_count(video_path)
    if frame_count is None:
        return max_video_frames
    return max(1, min(max_video_frames, frame_count))


def build_prompt_variant_key(prompt_choice: str, utt_count: int) -> str:
    utt_suffix = "single_utt" if utt_count == 1 else "multi_utt"
    return f"{prompt_choice}_{utt_suffix}"


def load_prompt_templates(
    prompt_config_path: Path,
    prompt_choice: str,
    conversation_mode: str,
    utt_count: int,
) -> dict[str, str]:
    with prompt_config_path.open(encoding="utf-8") as handle:
        prompt_config = json.load(handle)

    section_name = (
        "single_turn_prompts" if conversation_mode == "single-turn" else "multi_turn_prompts"
    )
    prompt_section = prompt_config.get(section_name, {})
    if not isinstance(prompt_section, Mapping):
        raise ValueError(f"Invalid prompt section in {prompt_config_path}: {section_name}")

    prompt_variant_key = build_prompt_variant_key(prompt_choice, utt_count)
    templates = prompt_section.get(prompt_variant_key)
    if not isinstance(templates, Mapping):
        available = ", ".join(sorted(prompt_section)) or "none"
        raise KeyError(
            f"Prompt variant '{prompt_variant_key}' not found in {prompt_config_path}. "
            f"Available variants in {section_name}: {available}"
        )

    required_fields = ("text",) if conversation_mode == "single-turn" else ("first", "after")
    missing = [
        field
        for field in required_fields
        if not isinstance(templates.get(field), str) or not templates[field].strip()
    ]
    if missing:
        raise ValueError(
            f"Prompt variant '{prompt_variant_key}' is missing required fields: "
            f"{', '.join(missing)}"
        )

    return {field: str(templates[field]).strip() for field in required_fields}


def normalize_message_content(messages: list[dict]) -> list[dict]:
    """Represent every message content as a list of typed content parts.

    Some Gemma processor versions assume content parts are dicts and will fail
    with "string indices must be integers" when system or assistant turns are
    plain strings.
    """
    normalized_messages: list[dict] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_content = [{"type": "text", "text": content}]
        elif isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
            normalized_content = []
            for item in content:
                if isinstance(item, str):
                    normalized_content.append({"type": "text", "text": item})
                else:
                    normalized_content.append(item)
        else:
            normalized_content = [{"type": "text", "text": str(content)}]

        normalized_message = dict(message)
        normalized_message["content"] = normalized_content
        normalized_messages.append(normalized_message)
    return normalized_messages


def min_message_video_num_frames(messages: list[dict]) -> int | None:
    video_num_frames: list[int] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
            continue
        for item in content:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "video"
                and isinstance(item.get("num_frames"), int)
            ):
                video_num_frames.append(int(item["num_frames"]))
    return min(video_num_frames) if video_num_frames else None


def combine_system_and_user_prompt(system_prompt: str, user_prompt: str) -> str:
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if not system_prompt:
        return user_prompt
    return f"{system_prompt}\n\n{user_prompt}"


def apply_gemma_chat_template(processor: Any, messages: list[dict], enable_thinking: bool) -> Any:
    """Apply the Gemma chat template, tolerating older processor signatures."""
    messages = normalize_message_content(messages)
    video_num_frames = min_message_video_num_frames(messages)
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if video_num_frames is not None:
        kwargs["num_frames"] = video_num_frames
    try:
        return processor.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(messages, **kwargs)
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise

    if not getattr(processor, "_vlm_social_warned_missing_chat_template", False):
        print(
            "[WARN] Processor has no chat template; using built-in Gemma 4 fallback template.",
            flush=True,
        )
        setattr(processor, "_vlm_social_warned_missing_chat_template", True)
    try:
        return processor.apply_chat_template(
            messages,
            chat_template=DEFAULT_GEMMA4_CHAT_TEMPLATE,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            chat_template=DEFAULT_GEMMA4_CHAT_TEMPLATE,
            **kwargs,
        )


def get_model_input_device(model: Any) -> torch.device | str:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def coerce_parsed_response(parsed: Any) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, Mapping):
        for key in ("answer", "response", "content", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, Sequence) and not isinstance(parsed, (bytes, bytearray)):
        text_items = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        if text_items:
            return text_items[-1]
    for attr in ("answer", "response", "content", "text"):
        value = getattr(parsed, attr, None)
        if isinstance(value, str):
            return value.strip()
    return str(parsed).strip()


def strip_common_special_tokens(text: str) -> str:
    replacements = (
        "<bos>",
        "<eos>",
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<turn|>",
        "<|turn|>",
    )
    cleaned = text
    for token in replacements:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def parse_response(processor: Any, raw_response: str) -> str:
    if hasattr(processor, "parse_response"):
        parsed = coerce_parsed_response(processor.parse_response(raw_response))
        if parsed:
            return parsed
    return strip_common_special_tokens(raw_response)


def infer_turn(
    model: Any,
    processor: Any,
    messages: list[dict],
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    inputs = apply_gemma_chat_template(
        processor=processor,
        messages=messages,
        enable_thinking=enable_thinking,
    ).to(get_model_input_device(model))
    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return parse_response(processor, raw_response)


def is_error_response(response: str) -> bool:
    return response.startswith("[ERROR]")


def build_error_summary(results: dict[str, list[dict]]) -> dict[str, dict]:
    subfolder_summary: dict[str, dict] = {}
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


def print_error_summary(summary: dict[str, dict]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Gemma 4 E4B inference over context video/audio folders."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Local Gemma model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent folder containing subfolders of mp4/wav pairs.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_PROMPT_CONFIG_PATH,
        help="Prompt JSON config with single_turn_prompts and multi_turn_prompts.",
    )
    parser.add_argument(
        "--prompt-choice",
        required=True,
        help="Prompt family in the prompt config, e.g. intention or affordance.",
    )
    parser.add_argument(
        "--mode",
        choices=("context",),
        default="context",
        help="Inference layout mode. This script supports context mode.",
    )
    parser.add_argument(
        "--utt-count",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help="Utterance count used to select the prompt variant.",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=("single-turn", "multi-turn"),
        default="single-turn",
        help="Use one fresh chat per pair or one running chat per subfolder.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt. Use --enable-thinking to request Gemma reasoning mode.",
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
        help=(
            "Maximum frames to sample per video. Shorter clips use their available "
            "frame count to avoid sampling more frames than exist."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run video-only inference by omitting separate .wav inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_config_path = args.prompt_config.expanduser().resolve()
    if not prompt_config_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompt_config_path}")

    prompt_templates = load_prompt_templates(
        prompt_config_path=prompt_config_path,
        prompt_choice=args.prompt_choice,
        conversation_mode=args.conversation_mode,
        utt_count=args.utt_count,
    )
    prompt_variant_key = build_prompt_variant_key(args.prompt_choice, args.utt_count)
    print(
        f"[INFO] Prompt config: {prompt_config_path} "
        f"(mode: {args.mode}, conversation_mode: {args.conversation_mode}, "
        f"choice: {args.prompt_choice}, variant: {prompt_variant_key})"
    )
    if args.conversation_mode == "single-turn":
        print(f"[INFO] Prompt text:\n{prompt_templates['text']}\n")
    else:
        print(f"[INFO] First-turn prompt:\n{prompt_templates['first']}\n")
        print(f"[INFO] Follow-up prompt:\n{prompt_templates['after']}\n")
    print(f"[INFO] Utterance count: {args.utt_count}")
    print(f"[INFO] Conversation mode: {args.conversation_mode}")
    print(f"[INFO] No audio: {args.no_audio}")
    print(f"[INFO] Thinking enabled: {args.enable_thinking}\n")
    print(f"[INFO] Max video frames: {args.max_video_frames}\n")

    av_pairs = collect_av_pairs(args.data_root, require_audio=not args.no_audio)
    total_pairs = sum(len(pairs) for pairs in av_pairs.values())
    input_label = "video item(s)" if args.no_audio else "video/audio pair(s)"
    print(f"[INFO] Found {total_pairs} {input_label} across {len(av_pairs)} subfolder(s).")
    if total_pairs == 0:
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

    results: dict[str, list] = {}
    processed = 0
    for subfolder_name, pairs in av_pairs.items():
        sorted_pairs = sorted(
            pairs,
            key=lambda pair: (
                Path(pair["video"]).name,
                Path(pair["audio"]).name if pair["audio"] is not None else "",
            ),
        )
        subfolder_results: list = []
        conversation_messages: list[dict] = []

        for pair_index, pair in enumerate(sorted_pairs):
            processed += 1
            if args.conversation_mode == "single-turn":
                prompt_template = prompt_templates["text"]
            else:
                prompt_template = (
                    prompt_templates["first"] if pair_index == 0 else prompt_templates["after"]
                )

            print(f"[{processed}/{total_pairs}] {subfolder_name}/{pair['stem']} ...", flush=True)

            video_num_frames = select_video_num_frames(
                video_path=pair["video"],
                max_video_frames=args.max_video_frames,
            )
            if video_num_frames < args.max_video_frames:
                print(
                    f"  [INFO] Short video: sampling {video_num_frames} frame(s) "
                    f"instead of {args.max_video_frames}.",
                    flush=True,
                )

            text_prompt = prompt_template
            if args.conversation_mode == "single-turn" or pair_index == 0:
                text_prompt = combine_system_and_user_prompt(
                    system_prompt=args.system_prompt,
                    user_prompt=prompt_template,
                )

            user_content: list[dict] = []
            if not args.no_audio:
                user_content.append({"type": "audio", "audio": pair["audio"]})
            user_content.extend(
                [
                    {
                        "type": "video",
                        "video": pair["video"],
                        "num_frames": video_num_frames,
                    },
                    {"type": "text", "text": text_prompt},
                ]
            )
            user_message = {"role": "user", "content": user_content}

            if args.conversation_mode == "multi-turn":
                messages = conversation_messages
                messages.append(user_message)
            else:
                messages = [user_message]

            try:
                response = infer_turn(
                    model=model,
                    processor=processor,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.enable_thinking,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[INFO] Results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
