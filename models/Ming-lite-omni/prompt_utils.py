#!/usr/bin/env python3
"""Helpers for loading prompt variants from prompts/prompts.json."""

import json
from pathlib import Path
from typing import Dict


PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "prompts" / "prompts.json"


def build_prompt_variant_key(prompt_choice: str, utt_count: int) -> str:
    """Map a prompt family and utterance count to the JSON variant key."""
    variant_suffix = "single_utt" if utt_count == 1 else "multi_utt"
    return f"{prompt_choice}_{variant_suffix}"


def _load_prompt_payload() -> dict:
    if not PROMPT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Prompt config not found: {PROMPT_CONFIG_PATH}")

    payload = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Prompt config must contain a top-level object: {PROMPT_CONFIG_PATH}"
        )
    return payload


def load_prompt_templates(
    prompt_choice: str,
    conversation_mode: str,
    utt_count: int,
) -> Dict[str, str]:
    """Load the prompt variant required for the selected conversation mode."""
    payload = _load_prompt_payload()
    variant_key = build_prompt_variant_key(prompt_choice, utt_count)

    section_name = (
        "single_turn_prompts"
        if conversation_mode == "single-turn"
        else "multi_turn_prompts"
    )
    section = payload.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(
            f"Missing '{section_name}' in prompt config: {PROMPT_CONFIG_PATH}"
        )

    prompt_entry = section.get(variant_key)
    if not isinstance(prompt_entry, dict):
        raise KeyError(
            f"Prompt variant '{variant_key}' not found in '{section_name}' of "
            f"{PROMPT_CONFIG_PATH}"
        )

    if conversation_mode == "single-turn":
        text = prompt_entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Prompt variant '{variant_key}' must contain a non-empty 'text' field."
            )
        return {"text": text.strip()}

    first = prompt_entry.get("first")
    after = prompt_entry.get("after")
    if not isinstance(first, str) or not first.strip():
        raise ValueError(
            f"Prompt variant '{variant_key}' must contain a non-empty 'first' field."
        )
    if not isinstance(after, str) or not after.strip():
        raise ValueError(
            f"Prompt variant '{variant_key}' must contain a non-empty 'after' field."
        )
    return {"first": first.strip(), "after": after.strip()}
