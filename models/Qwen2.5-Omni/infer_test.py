#!/usr/bin/env python3
import argparse
import os

import torch
from qwen_omni_utils import process_mm_info
from torch_compat import ensure_cuda_runtime_compatibility
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-turn Qwen2.5-Omni inference with text + audio + video input."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "/path/to/GesBench/models/Qwen2.5-Omni"),
        help="Model id or local model path.",
    )
    parser.add_argument(
        "--audio",
        default="path/to/samples/clip_11.wav",
        help="Audio URL or local file path.",
    )
    parser.add_argument(
        "--video",
        default="/path/to/samples/clip_11.mp4",
        help="Video URL or local file path.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Use both the audio and video to describe what you hear and see. "
            "Be concise and mention key evidence from each modality."
        ),
        help="User text prompt.",
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


def main() -> None:
    args = parse_args()
    ensure_cuda_runtime_compatibility()

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"[INFO] Loading model: {args.model}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": args.system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": args.audio},
                {"type": "video", "video": args.video},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(
        messages, use_audio_in_video=args.use_audio_in_video
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print("[INFO] Running generation...")
    output = model.generate(
        **inputs,
        use_audio_in_video=args.use_audio_in_video,
        return_audio=False,
        thinker_max_new_tokens=args.max_new_tokens,
        thinker_do_sample=False,
    )
    response = processor.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
