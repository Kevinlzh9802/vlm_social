#!/usr/bin/env python3
"""Helpers for loading prompt templates from a shared JSON config."""

import json
from pathlib import Path
from typing import Dict


PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "prompts" / "prompts.json"
SINGLE_TURN_PROMPT_FIELDS = ("text",)
MULTI_TURN_PROMPT_FIELDS = ("first", "after")
CONVERSATION_MODE_TO_SECTION = {
    "single-turn": "single_turn_prompts",
    "multi-turn": "multi_turn_prompts",
}


def list_prompt_choices() -> list[str]:
    """Return the available batch prompt choices defined in the JSON config."""
    prompt_config = _load_prompt_config()
    prompt_choices: set[str] = set()

    for conversation_mode in CONVERSATION_MODE_TO_SECTION:
        section = _get_prompt_section(prompt_config, conversation_mode)
        for prompt_key in section:
            prompt_choice, _, suffix = prompt_key.rpartition("_")
            if suffix == "utt" and prompt_choice.endswith(("_single", "_multi")):
                base_choice, _, _ = prompt_choice.rpartition("_")
                if base_choice:
                    prompt_choices.add(base_choice)

    return sorted(prompt_choices)


def build_prompt_variant_key(prompt_choice: str, utt_count: int) -> str:
    """Return the prompt variant key for the requested utterance count."""
    utt_suffix = "single_utt" if utt_count == 1 else "multi_utt"
    return f"{prompt_choice}_{utt_suffix}"


def load_prompt_templates(
    prompt_choice: str,
    conversation_mode: str,
    utt_count: int,
) -> Dict[str, str]:
    """Load prompt templates for a prompt choice and conversation setting."""
    prompt_config = _load_prompt_config()
    prompt_section = _get_prompt_section(prompt_config, conversation_mode)
    prompt_variant_key = build_prompt_variant_key(prompt_choice, utt_count)
    templates = prompt_section.get(prompt_variant_key)
    if not isinstance(templates, dict):
        available = ", ".join(sorted(prompt_section)) or "none"
        raise KeyError(
            f"Prompt variant '{prompt_variant_key}' not found in {PROMPT_CONFIG_PATH}. "
            f"Available variants in '{CONVERSATION_MODE_TO_SECTION[conversation_mode]}': "
            f"{available}"
        )

    required_fields = (
        SINGLE_TURN_PROMPT_FIELDS
        if conversation_mode == "single-turn"
        else MULTI_TURN_PROMPT_FIELDS
    )
    missing_fields = [
        field
        for field in required_fields
        if not isinstance(templates.get(field), str) or not templates[field].strip()
    ]
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise ValueError(
            f"Prompt variant '{prompt_variant_key}' in {PROMPT_CONFIG_PATH} is missing "
            f"required field(s): {missing_text}"
        )

    return {
        field: templates[field].strip()
        for field in required_fields
    }


def _load_prompt_config() -> dict:
    if not PROMPT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Prompt config not found: {PROMPT_CONFIG_PATH}")

    with PROMPT_CONFIG_PATH.open(encoding="utf-8") as handle:
        prompt_config = json.load(handle)

    if not isinstance(prompt_config, dict):
        raise ValueError(
            f"Invalid prompt config in {PROMPT_CONFIG_PATH}: root must be a JSON object."
        )

    return prompt_config


def _get_prompt_section(prompt_config: dict, conversation_mode: str) -> dict:
    if conversation_mode not in CONVERSATION_MODE_TO_SECTION:
        available_modes = ", ".join(sorted(CONVERSATION_MODE_TO_SECTION))
        raise ValueError(
            f"Unsupported conversation mode '{conversation_mode}'. "
            f"Expected one of: {available_modes}"
        )

    section_name = CONVERSATION_MODE_TO_SECTION[conversation_mode]
    prompt_section = prompt_config.get(section_name, {})
    if not isinstance(prompt_section, dict):
        raise ValueError(
            f"Invalid prompt config in {PROMPT_CONFIG_PATH}: "
            f"'{section_name}' must be a JSON object."
        )

    return prompt_section
