#!/usr/bin/env python3
"""Submit a Gemini Batch API job for a folder of video/audio inputs.

This script is **submit-only**: it uploads media files, creates the batch job,
and registers the job in a shared registry JSON file — then exits.
Use ``gemini_retrieve.py`` to scan the registry and download results for
all completed jobs.

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

Registry JSON (``--registry``)
==============================
A single JSON file (list of entries) shared across all experiments::

    [
        {
            "run_id": "intention_1utt_batch01_20260405T143012_a3f2",
            "job_name": "batches/123456789",
            "status": "pending",
            "submitted_at": "2026-04-05T14:30:12",
            "dataset": "mintrec2",
            "prompt_choice": "intention",
            "utt_count": 1,
            "gemini_mode": "2.5-flash",
            "model_name": "gemini-2.5-flash",
            "output_json": "/path/to/batch01.json",
            "video_map": { ... }
        },
        ...
    ]
"""

import argparse
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def check_network(host: str = "generativelanguage.googleapis.com", port: int = 443, timeout: int = 10) -> None:
    """Verify outbound connectivity before doing any real work."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        print(f"[INFO] Network check passed ({host}:{port} reachable).")
    except OSError as exc:
        print(
            f"[ERROR] Network check failed — cannot reach {host}:{port}: {exc}\n"
            f"  The Gemini Batch API requires internet access. "
            f"Make sure the job runs on a node with outbound connectivity.",
            file=sys.stderr,
        )
        sys.exit(1)


check_network()

from google import genai
from google.genai import types as genai_types

from gemini import (
    DEFAULT_API_KEY_PATH,
    DEFAULT_SYSTEM_PROMPT,
    MODE_TO_MODEL,
    DEFAULT_MODE,
    PROMPT_CONFIG_PATH,
    get_client,
    resolve_prompt,
)


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "gemini_registry.json"


# ---------------------------------------------------------------------------
# Folder traversal
# ---------------------------------------------------------------------------

def collect_media_inputs(data_root: str, require_audio: bool = True) -> Dict[str, List[dict]]:
    """Scan *data_root* for .mp4 files and optional same-stem .wav files.

    Returns a mapping from subfolder name to a sorted list of dicts
    with keys ``"stem"``, ``"video"``, and ``"audio"``. When
    ``require_audio`` is true, videos without matching ``.wav`` files are
    skipped.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        entries: List[dict] = []
        for video_path in sorted(subfolder.glob("*.mp4")):
            audio_path = subfolder / f"{video_path.stem}.wav"
            if audio_path.exists():
                audio_value: Optional[str] = str(audio_path)
            elif require_audio:
                print(f"[WARN] No matching .wav for {video_path}, skipping.")
                continue
            else:
                audio_value = None

            entries.append({
                "stem": video_path.stem,
                "video": str(video_path),
                "audio": audio_value,
            })

        if entries:
            result[subfolder.name] = entries
    return result


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_media(
    client: Any,
    video_map: Dict[str, List[dict]],
) -> Dict[str, dict]:
    """Upload all referenced media via the File API.

    Returns a mapping from local media path to file metadata needed for
    ``file_data`` request parts.
    """
    path_to_file: Dict[str, dict] = {}
    total = sum(
        1 + int(bool(entry.get("audio")))
        for entries in video_map.values()
        for entry in entries
    )
    idx = 0

    for subfolder, entries in video_map.items():
        for entry in entries:
            idx += 1
            video_path = entry["video"]
            print(f"[{idx}/{total}] Uploading {subfolder}/{entry['stem']}.mp4 ...")
            uploaded = client.files.upload(file=video_path)
            path_to_file[video_path] = {
                "file_uri": getattr(uploaded, "uri", None)
                or f"https://generativelanguage.googleapis.com/v1beta/{uploaded.name}",
                "mime_type": getattr(uploaded, "mime_type", "video/mp4") or "video/mp4",
            }
            print(f"  -> {uploaded.name}")

            audio_path = entry.get("audio")
            if audio_path:
                idx += 1
                print(f"[{idx}/{total}] Uploading {subfolder}/{entry['stem']}.wav ...")
                uploaded_audio = client.files.upload(file=audio_path)
                path_to_file[audio_path] = {
                    "file_uri": getattr(uploaded_audio, "uri", None)
                    or f"https://generativelanguage.googleapis.com/v1beta/{uploaded_audio.name}",
                    "mime_type": (
                        getattr(uploaded_audio, "mime_type", "audio/wav") or "audio/wav"
                    ),
                }
                print(f"  -> {uploaded_audio.name}")

    return path_to_file


# ---------------------------------------------------------------------------
# JSONL construction
# ---------------------------------------------------------------------------

def build_jsonl_requests(
    video_map: Dict[str, List[dict]],
    path_to_file: Dict[str, dict],
    prompt_text: str,
    system_prompt: Optional[str],
) -> List[dict]:
    """Build one JSONL line per media item.

    Each line has ``"key"`` (subfolder/stem) and ``"request"`` (a
    GenerateContentRequest dict referencing uploaded video/audio file URIs).
    """
    lines: List[dict] = []

    for subfolder, entries in video_map.items():
        for entry in entries:
            video_file = path_to_file[entry["video"]]
            parts: List[dict] = [
                {"text": prompt_text},
                {"file_data": video_file},
            ]

            audio_path = entry.get("audio")
            if audio_path:
                parts.append({"file_data": path_to_file[audio_path]})

            request_body: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts,
                    }
                ],
            }

            if system_prompt:
                request_body["system_instruction"] = {
                    "parts": [{"text": system_prompt}]
                }

            lines.append({
                "key": f"{subfolder}/{entry['stem']}",
                "request": request_body,
            })

    return lines


def write_jsonl(lines: List[dict], output_path: Path) -> None:
    """Write a list of dicts as a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Batch job lifecycle
# ---------------------------------------------------------------------------

def submit_batch_job(
    client: Any,
    jsonl_path: Path,
    model_name: str,
    display_name: str = "gemini-batch-video-job",
) -> Any:
    """Upload the JSONL file and create a batch job."""
    print(f"[INFO] Uploading JSONL request file: {jsonl_path}")
    uploaded_jsonl = client.files.upload(
        file=str(jsonl_path),
        config=genai_types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )
    print(f"[INFO] Uploaded JSONL as: {uploaded_jsonl.name}")

    batch_job = client.batches.create(
        model=model_name,
        src=uploaded_jsonl.name,
        config={"display_name": display_name},
    )
    print(f"[INFO] Created batch job: {batch_job.name}")
    return batch_job


# ---------------------------------------------------------------------------
# Run-ID generation
# ---------------------------------------------------------------------------

def generate_run_id(prompt_choice: str, utt_count: int, batch_id: str) -> str:
    """Build a unique, human-readable run identifier.

    Format: ``<prompt>_<utt>utt_<batch>_<timestamp>_<random4>``
    e.g.  ``intention_1utt_batch01_20260405T143012_a3f2``
    """
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    rand = os.urandom(2).hex()  # 4 hex chars
    return f"{prompt_choice}_{utt_count}utt_{batch_id}_{ts}_{rand}"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry(registry_path: Path) -> List[dict]:
    """Load the shared registry, returning an empty list if absent."""
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text(encoding="utf-8"))


def save_registry(registry_path: Path, entries: List[dict]) -> None:
    """Write the registry atomically-ish (write-then-rename)."""
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(registry_path)


def append_registry_entry(
    registry_path: Path,
    run_id: str,
    job_name: str,
    dataset: str,
    prompt_choice: str,
    utt_count: int,
    gemini_mode: str,
    model_name: str,
    output_json: str,
    data_root: str,
    video_map: Dict[str, List[dict]],
    no_audio: bool,
) -> None:
    """Append a new entry to the shared registry."""
    entries = load_registry(registry_path)
    entries.append({
        "run_id": run_id,
        "job_name": job_name,
        "status": "pending",
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "prompt_choice": prompt_choice,
        "utt_count": utt_count,
        "gemini_mode": gemini_mode,
        "model_name": model_name,
        "output_json": output_json,
        "data_root": data_root,
        "video_map": video_map,
        "no_audio": no_audio,
    })
    save_registry(registry_path, entries)
    print(f"[INFO] Registered run '{run_id}' in {registry_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload videos and submit a Gemini Batch API job (submit-only).",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent folder containing subfolders of .mp4 files.",
    )
    parser.add_argument(
        "--output",
        default="results_gemini_batch.json",
        help="Path where the final results JSON should eventually be written.",
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
        help="Gemini mode alias.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional explicit Gemini model override.",
    )
    parser.add_argument(
        "--utt-count",
        "--utt",
        dest="utt_count",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Utterance count for prompt variant selection.",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=("single-turn",),
        default="single-turn",
        help="Batch API only supports single-turn for now.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System instruction for all requests.",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help="Path to Gemini API key file.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help="Path to write the intermediate JSONL file. Defaults to <output>.requests.jsonl.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to the shared registry JSON file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name (stored in registry for bookkeeping).",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default="",
        help="Batch identifier string (stored in registry and used in run_id).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run video-only requests by omitting separate same-stem .wav inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_name = args.model or MODE_TO_MODEL[args.mode]
    prompt_text = resolve_prompt(
        prompt=args.prompt,
        prompt_choice=args.prompt_choice,
        conversation_mode=args.conversation_mode,
        utt_count=args.utt_count,
        turn_index=0,
        prompt_config_path=args.prompts_config,
    )

    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Prompt text:\n{prompt_text}\n")

    # --- Collect media inputs ---
    video_map = collect_media_inputs(args.data_root, require_audio=not args.no_audio)
    total = sum(len(v) for v in video_map.values())
    input_label = "video item(s)" if args.no_audio else "video/audio pair(s)"
    print(f"[INFO] Found {total} {input_label} across {len(video_map)} subfolder(s).")
    if total == 0:
        print("[WARN] Nothing to process.")
        return

    # --- Upload media ---
    client = get_client(api_key_path=args.api_key_path)
    path_to_file = upload_media(client, video_map)

    # --- Build JSONL ---
    jsonl_lines = build_jsonl_requests(
        video_map=video_map,
        path_to_file=path_to_file,
        prompt_text=prompt_text,
        system_prompt=args.system_prompt,
    )

    jsonl_path = args.jsonl_path or Path(args.output).with_suffix(".requests.jsonl")
    write_jsonl(jsonl_lines, jsonl_path)
    print(f"[INFO] Wrote {len(jsonl_lines)} request(s) to {jsonl_path}")

    # --- Submit batch job ---
    batch_id_label = args.batch_id or Path(args.output).stem
    run_id = generate_run_id(
        prompt_choice=args.prompt_choice or "custom",
        utt_count=args.utt_count or 0,
        batch_id=batch_id_label,
    )

    batch_job = submit_batch_job(
        client=client,
        jsonl_path=jsonl_path,
        model_name=model_name,
        display_name=run_id,
    )

    # --- Register in shared registry ---
    append_registry_entry(
        registry_path=args.registry,
        run_id=run_id,
        job_name=batch_job.name,
        dataset=args.dataset,
        prompt_choice=args.prompt_choice or "",
        utt_count=args.utt_count or 0,
        gemini_mode=args.mode,
        model_name=model_name,
        output_json=str(Path(args.output).resolve()),
        data_root=args.data_root,
        video_map=video_map,
        no_audio=args.no_audio,
    )

    print(f"\n[INFO] Batch job submitted: {batch_job.name}")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Use gemini_retrieve.py --registry {args.registry} to collect results later.")


if __name__ == "__main__":
    main()
