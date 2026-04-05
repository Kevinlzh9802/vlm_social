#!/usr/bin/env python3
"""Batch Gemini inference over a folder of videos using the Gemini Batch API.

Expected data layout
====================
<data_root>/
    subfolder_A/
        d1u1-u2_clip_01.mp4
        d3u2-u3_clip_02.mp4
    subfolder_B/
        ...

The script discovers every .mp4 file inside each immediate subfolder of
*data_root*, uploads them via the File API, builds a JSONL batch request
referencing the uploaded files, submits a single asynchronous batch job,
polls until completion, and writes results to a JSON file.

The Batch API runs at 50% of standard pricing with a target 24-hour
turnaround (often much faster).

Output JSON structure
=====================
{
    "__summary__": { ... },
    "subfolder_A": [
        {"file": "d1u1-u2_clip_01", "prompt": "...", "response": "..."},
        ...
    ],
    ...
}
"""

import argparse
import json
import socket
import sys
import time
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
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    MODE_TO_MODEL,
    DEFAULT_MODE,
    PROMPT_CONFIG_PATH,
    build_prompt_variant_key,
    get_client,
    load_prompt_templates,
    resolve_prompt,
)


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ---------------------------------------------------------------------------
# Folder traversal
# ---------------------------------------------------------------------------

def collect_videos(data_root: str) -> Dict[str, List[dict]]:
    """Scan *data_root* for .mp4 files in each immediate subfolder.

    Returns a mapping from subfolder name to a sorted list of dicts
    with keys ``"stem"`` and ``"video"`` (absolute path string).
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        videos = sorted(subfolder.glob("*.mp4"))
        if videos:
            result[subfolder.name] = [
                {"stem": v.stem, "video": str(v)} for v in videos
            ]
    return result


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_videos(
    client: Any,
    video_map: Dict[str, List[dict]],
) -> Dict[str, str]:
    """Upload all videos via the File API.

    Returns a mapping from local video path to the uploaded file name
    (e.g. ``"files/abc123"``).
    """
    path_to_file_name: Dict[str, str] = {}
    total = sum(len(v) for v in video_map.values())
    idx = 0

    for subfolder, entries in video_map.items():
        for entry in entries:
            idx += 1
            video_path = entry["video"]
            print(f"[{idx}/{total}] Uploading {subfolder}/{entry['stem']}.mp4 ...")
            uploaded = client.files.upload(file=video_path)
            path_to_file_name[video_path] = uploaded.name
            print(f"  -> {uploaded.name}")

    return path_to_file_name


# ---------------------------------------------------------------------------
# JSONL construction
# ---------------------------------------------------------------------------

def build_jsonl_requests(
    video_map: Dict[str, List[dict]],
    path_to_file_name: Dict[str, str],
    prompt_text: str,
    system_prompt: Optional[str],
) -> List[dict]:
    """Build one JSONL line per video.

    Each line has ``"key"`` (subfolder/stem) and ``"request"`` (a
    GenerateContentRequest dict referencing the uploaded file URI).
    """
    lines: List[dict] = []

    for subfolder, entries in video_map.items():
        for entry in entries:
            file_name = path_to_file_name[entry["video"]]
            # The file URI format used in batch JSONL
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"

            request_body: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt_text},
                            {"file_data": {"file_uri": file_uri}},
                        ],
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


def poll_batch_job(
    client: Any,
    job_name: str,
    poll_interval: int = 30,
) -> Any:
    """Poll until the batch job reaches a terminal state."""
    print(f"[INFO] Polling job: {job_name}")
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        if state in COMPLETED_STATES:
            print(f"[INFO] Job finished: {state}")
            return job
        print(f"[INFO] State: {state} — waiting {poll_interval}s ...")
        time.sleep(poll_interval)


def download_results(client: Any, job: Any) -> List[dict]:
    """Download and parse the result JSONL from a succeeded batch job."""
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch job did not succeed. Final state: {job.state.name}. "
            f"Error: {getattr(job, 'error', 'N/A')}"
        )

    results: List[dict] = []

    # File-based output
    if job.dest and job.dest.file_name:
        print(f"[INFO] Downloading result file: {job.dest.file_name}")
        content_bytes = client.files.download(file=job.dest.file_name)
        for line in content_bytes.decode("utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))

    # Inline output fallback
    elif job.dest and job.dest.inlined_responses:
        for resp in job.dest.inlined_responses:
            entry: Dict[str, Any] = {}
            if resp.response:
                try:
                    entry["text"] = resp.response.text
                except AttributeError:
                    entry["text"] = str(resp.response)
            if resp.error:
                entry["error"] = str(resp.error)
            results.append(entry)
    else:
        print("[WARN] No results found in batch job output.")

    return results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_results(
    raw_results: List[dict],
    video_map: Dict[str, List[dict]],
) -> Dict[str, Any]:
    """Reshape raw JSONL results into the subfolder-grouped output format."""
    # Build a lookup from key -> response text
    key_to_response: Dict[str, str] = {}
    key_to_error: Dict[str, str] = {}

    for entry in raw_results:
        key = entry.get("key", "")
        resp = entry.get("response")
        if resp:
            # JSONL result: response is a GenerateContentResponse dict
            try:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                text = str(resp)
            key_to_response[key] = text
        if entry.get("error"):
            key_to_error[key] = str(entry["error"])
        # Inline results already have "text"
        if "text" in entry and not key:
            key_to_response[key] = entry["text"]

    grouped: Dict[str, list] = {}
    total_files = 0
    total_errors = 0

    for subfolder, entries in video_map.items():
        subfolder_results = []
        for e in entries:
            key = f"{subfolder}/{e['stem']}"
            response = key_to_response.get(key, "")
            error = key_to_error.get(key)
            total_files += 1
            if error or not response:
                total_errors += 1
            subfolder_results.append({
                "file": e["stem"],
                "response": response,
                **({"error": error} if error else {}),
            })
        grouped[subfolder] = subfolder_results

    summary = {
        "overall": {
            "files": total_files,
            "errors": total_errors,
            "error_ratio": (total_errors / total_files) if total_files else 0.0,
        },
    }

    return {"__summary__": summary, **grouped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Gemini inference over a folder of videos (Batch API)."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent folder containing subfolders of .mp4 files.",
    )
    parser.add_argument(
        "--output",
        default="results_gemini_batch.json",
        help="Path to the output JSON file.",
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
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status polls (default: 30).",
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

    # --- Collect videos ---
    video_map = collect_videos(args.data_root)
    total = sum(len(v) for v in video_map.values())
    print(f"[INFO] Found {total} video(s) across {len(video_map)} subfolder(s).")
    if total == 0:
        print("[WARN] Nothing to process.")
        return

    # --- Upload videos ---
    client = get_client(api_key_path=args.api_key_path)
    path_to_file_name = upload_videos(client, video_map)

    # --- Build JSONL ---
    jsonl_lines = build_jsonl_requests(
        video_map=video_map,
        path_to_file_name=path_to_file_name,
        prompt_text=prompt_text,
        system_prompt=args.system_prompt,
    )

    jsonl_path = args.jsonl_path or Path(args.output).with_suffix(".requests.jsonl")
    write_jsonl(jsonl_lines, jsonl_path)
    print(f"[INFO] Wrote {len(jsonl_lines)} request(s) to {jsonl_path}")

    # --- Submit batch job ---
    batch_job = submit_batch_job(
        client=client,
        jsonl_path=jsonl_path,
        model_name=model_name,
    )

    # --- Poll ---
    completed_job = poll_batch_job(
        client=client,
        job_name=batch_job.name,
        poll_interval=args.poll_interval,
    )

    # --- Retrieve results ---
    raw_results = download_results(client, completed_job)
    output = format_results(raw_results, video_map)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[INFO] Results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
