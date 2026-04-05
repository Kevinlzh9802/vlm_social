import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types as genai_types

from prompt_utils import (
    PROMPT_CONFIG_PATH as SHARED_PROMPT_CONFIG_PATH,
    build_prompt_variant_key as shared_build_prompt_variant_key,
    load_prompt_templates as shared_load_prompt_templates,
)


MODE_TO_MODEL = {
    "2.5-flash": "gemini-2.5-flash",
}
DEFAULT_MODE = "2.5-flash"
DEFAULT_MODEL = MODE_TO_MODEL[DEFAULT_MODE]
DEFAULT_PROMPT = "Describe what is happening in this video."
DEFAULT_SYSTEM_PROMPT = "You are Gemini, a helpful multimodal assistant."
DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent / "api_keys" / "gemini.txt"
PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prompts.json"


def load_api_key(api_key_path: Path = DEFAULT_API_KEY_PATH) -> str:
    with open(api_key_path, "r", encoding="utf-8") as file:
        api_key = file.read().strip()

    if not api_key:
        raise ValueError(f"Gemini API key file is empty: {api_key_path}")
    return api_key


def get_client(api_key: Optional[str] = None, api_key_path: Path = DEFAULT_API_KEY_PATH):
    key = api_key or load_api_key(api_key_path=api_key_path)
    return genai.Client(api_key=key)


def build_prompt_variant_key(prompt_choice: str, utt_count: int) -> str:
    if shared_build_prompt_variant_key is not None:
        return shared_build_prompt_variant_key(prompt_choice=prompt_choice, utt_count=utt_count)
    del prompt_choice
    return "single_utt" if utt_count == 1 else "multi_utt"


def _load_prompt_config(prompt_config_path: Path) -> Dict[str, Any]:
    with open(prompt_config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_prompt_text(value: Any, required_key: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get(required_key)
        if isinstance(candidate, str):
            return candidate
    raise KeyError(f"Could not resolve prompt field '{required_key}'.")


def _resolve_single_turn_prompt(variant_config: Any) -> Dict[str, str]:
    if isinstance(variant_config, str):
        return {"text": variant_config}
    if not isinstance(variant_config, dict):
        raise KeyError("Single-turn prompt config must be a string or dict.")

    if "text" in variant_config and isinstance(variant_config["text"], str):
        return {"text": variant_config["text"]}
    if "single-turn" in variant_config:
        return {"text": _resolve_prompt_text(variant_config["single-turn"], "text")}
    raise KeyError("Missing single-turn prompt text.")


def _resolve_multi_turn_prompt(variant_config: Any) -> Dict[str, str]:
    if not isinstance(variant_config, dict):
        raise KeyError("Multi-turn prompt config must be a dict.")

    if "multi-turn" in variant_config and isinstance(variant_config["multi-turn"], dict):
        variant_config = variant_config["multi-turn"]

    first = _resolve_prompt_text(variant_config.get("first"), "text")
    after = _resolve_prompt_text(variant_config.get("after"), "text")
    return {"first": first, "after": after}


def load_prompt_templates(
    prompt_choice: str,
    conversation_mode: str,
    utt_count: int,
    prompt_config_path: Path = PROMPT_CONFIG_PATH,
) -> Dict[str, str]:
    if (
        shared_load_prompt_templates is not None
        and SHARED_PROMPT_CONFIG_PATH is not None
        and Path(prompt_config_path) == Path(SHARED_PROMPT_CONFIG_PATH)
    ):
        return shared_load_prompt_templates(
            prompt_choice=prompt_choice,
            conversation_mode=conversation_mode,
            utt_count=utt_count,
        )

    config = _load_prompt_config(prompt_config_path)
    prompt_family = config.get(prompt_choice)
    if not isinstance(prompt_family, dict):
        raise KeyError(
            f"Prompt choice '{prompt_choice}' was not found in {prompt_config_path}."
        )

    variant_key = build_prompt_variant_key(prompt_choice=prompt_choice, utt_count=utt_count)
    variant_config = prompt_family.get(variant_key)
    if variant_config is None:
        raise KeyError(
            f"Prompt variant '{variant_key}' was not found under '{prompt_choice}'."
        )

    if conversation_mode == "single-turn":
        return _resolve_single_turn_prompt(variant_config)
    if conversation_mode == "multi-turn":
        return _resolve_multi_turn_prompt(variant_config)
    raise ValueError(f"Unsupported conversation mode: {conversation_mode}")


def resolve_prompt(
    prompt: Optional[str],
    prompt_choice: Optional[str],
    conversation_mode: str,
    utt_count: Optional[int],
    turn_index: int,
    prompt_config_path: Path = PROMPT_CONFIG_PATH,
) -> str:
    if prompt_choice:
        if utt_count is None:
            raise ValueError("--utt-count is required when --prompt-choice is used.")
        prompt_templates = load_prompt_templates(
            prompt_choice=prompt_choice,
            conversation_mode=conversation_mode,
            utt_count=utt_count,
            prompt_config_path=prompt_config_path,
        )
        if conversation_mode == "single-turn":
            return prompt_templates["text"]
        return prompt_templates["first"] if turn_index == 0 else prompt_templates["after"]

    if prompt is not None:
        return prompt
    return DEFAULT_PROMPT


def build_generation_config(system_prompt: Optional[str]):
    if not system_prompt or genai_types is None:
        return None
    return genai_types.GenerateContentConfig(system_instruction=system_prompt)


def generate_video_response(
    video_path: str,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = DEFAULT_MODEL,
    client=None,
    system_prompt: Optional[str] = None,
) -> str:
    """Upload a video and return Gemini's text response."""
    if client is None:
        client = get_client()

    uploaded_video = client.files.upload(file=video_path)
    config = build_generation_config(system_prompt=system_prompt)
    contents = [prompt, uploaded_video]

    if config is None and system_prompt:
        contents = [f"System instruction: {system_prompt}\n\nUser request: {prompt}", uploaded_video]

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    text = getattr(response, "text", None)
    if not text:
        raise ValueError(f"Gemini returned an empty response for video: {video_path}")
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gemini inference on a single video.")
    parser.add_argument(
        "--video-path",
        type=str,
        default="data/clip_11.mp4",
        help="Path to a video file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Direct prompt text. Ignored when --prompt-choice is provided.",
    )
    parser.add_argument(
        "--prompt-choice",
        type=str,
        default=None,
        help="Prompt family key in api_models/configs/prompts.json.",
    )
    parser.add_argument(
        "--prompts-config",
        type=Path,
        default=PROMPT_CONFIG_PATH,
        help="Path to prompts.json.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_TO_MODEL.keys()),
        default=DEFAULT_MODE,
        help="Gemini mode alias. Currently only 2.5-flash is enabled.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional explicit Gemini model override. Defaults to the model behind --mode.",
    )
    parser.add_argument(
        "--utt-count",
        "--utt",
        dest="utt_count",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Utterance count used to select single_utt or multi_utt prompt variants.",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=("single-turn", "multi-turn"),
        default="single-turn",
        help="Prompt layout mode used when selecting templates from prompts.json.",
    )
    parser.add_argument(
        "--turn-index",
        type=int,
        default=0,
        help="Turn index for multi-turn prompt selection. 0 uses the first-turn prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional Gemini system instruction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model or MODE_TO_MODEL[args.mode]
    prompt_text = resolve_prompt(
        prompt=args.prompt,
        prompt_choice=args.prompt_choice,
        conversation_mode=args.conversation_mode,
        utt_count=args.utt_count,
        turn_index=args.turn_index,
        prompt_config_path=args.prompts_config,
    )

    if args.prompt_choice:
        prompt_variant_key = build_prompt_variant_key(
            prompt_choice=args.prompt_choice,
            utt_count=args.utt_count,
        )
        print(
            f"[INFO] Prompt config: {args.prompts_config} "
            f"(mode: {args.mode}, conversation_mode: {args.conversation_mode}, "
            f"choice: {args.prompt_choice}, variant: {prompt_variant_key}, "
            f"turn_index: {args.turn_index})"
        )
    else:
        print(f"[INFO] Using direct prompt with mode: {args.mode}")

    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Prompt text:\n{prompt_text}\n")

    response_text = generate_video_response(
        video_path=args.video_path,
        prompt=prompt_text,
        model_name=model_name,
        system_prompt=args.system_prompt,
    )
    print(response_text)


if __name__ == "__main__":
    main()
