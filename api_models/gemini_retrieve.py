#!/usr/bin/env python3
"""Retrieve results for all completed Gemini batch jobs in the registry.

This script is the counterpart to ``gemini_batch.py`` (submit-only).
It reads the shared registry JSON, queries each ``"pending"`` job,
downloads results for those that have finished, writes the output JSON,
and marks the entry as ``"retrieved"`` (or ``"failed"``/``"expired"``/
``"cancelled"``).

Running it multiple times is safe — already-retrieved entries are skipped.

Usage
=====
    python gemini_retrieve.py --registry gemini_registry.json --api-key-path /path/to/key.txt

It needs no dataset/prompt/utt parameters — everything is in the registry.
"""

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Network pre-check (same as gemini_batch.py)
# ---------------------------------------------------------------------------

def check_network(host: str = "generativelanguage.googleapis.com", port: int = 443, timeout: int = 10) -> None:
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        print(f"[INFO] Network check passed ({host}:{port} reachable).")
    except OSError as exc:
        print(
            f"[ERROR] Network check failed — cannot reach {host}:{port}: {exc}\n"
            f"  Make sure the job runs on a node with outbound connectivity.",
            file=sys.stderr,
        )
        sys.exit(1)


check_network()

from google import genai
from google.genai import types as genai_types

from gemini import (
    DEFAULT_API_KEY_PATH,
    get_client,
)
from gemini_batch import (
    DEFAULT_REGISTRY_PATH,
    load_registry,
    save_registry,
)


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

TERMINAL_STATUS_MAP = {
    "JOB_STATE_SUCCEEDED": "retrieved",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


# ---------------------------------------------------------------------------
# Download & format  (moved here from the old gemini_batch.py)
# ---------------------------------------------------------------------------

def download_results(client: Any, job: Any) -> List[dict]:
    """Download and parse the result JSONL from a succeeded batch job."""
    results: List[dict] = []

    if job.dest and job.dest.file_name:
        print(f"[INFO] Downloading result file: {job.dest.file_name}")
        content_bytes = client.files.download(file=job.dest.file_name)
        for line in content_bytes.decode("utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
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


def format_results(
    raw_results: List[dict],
    video_map: Dict[str, List[dict]],
) -> Dict[str, Any]:
    """Reshape raw JSONL results into the subfolder-grouped output format."""
    key_to_response: Dict[str, str] = {}
    key_to_error: Dict[str, str] = {}

    for entry in raw_results:
        key = entry.get("key", "")
        resp = entry.get("response")
        if resp:
            try:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                text = str(resp)
            key_to_response[key] = text
        if entry.get("error"):
            key_to_error[key] = str(entry["error"])
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
# Core retrieval logic
# ---------------------------------------------------------------------------

def process_registry(client: Any, registry_path: Path) -> None:
    """Check every pending entry, retrieve completed ones, update the registry."""
    entries = load_registry(registry_path)
    if not entries:
        print("[INFO] Registry is empty — nothing to retrieve.")
        return

    pending = [e for e in entries if e.get("status") == "pending"]
    if not pending:
        print("[INFO] No pending jobs in registry.")
        return

    print(f"[INFO] Found {len(pending)} pending job(s) in {registry_path}\n")
    modified = False

    for entry in pending:
        run_id = entry["run_id"]
        job_name = entry["job_name"]

        print(f"--- [{run_id}] checking {job_name} ...")
        try:
            job = client.batches.get(name=job_name)
        except Exception as exc:
            print(f"  [WARN] Could not query job: {exc}")
            continue

        state = job.state.name
        print(f"  State: {state}")

        if state not in COMPLETED_STATES:
            print(f"  Still running — skipping.")
            continue

        new_status = TERMINAL_STATUS_MAP.get(state, state)
        entry["status"] = new_status
        entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        modified = True

        if state != "JOB_STATE_SUCCEEDED":
            error_msg = str(getattr(job, "error", "N/A"))
            entry["error"] = error_msg
            print(f"  Job {new_status}: {error_msg}")
            continue

        # --- Download & write results ---
        video_map = entry.get("video_map", {})
        output_json = entry.get("output_json", "")
        if not output_json:
            print(f"  [WARN] No output_json path in registry entry — skipping download.")
            continue

        try:
            raw_results = download_results(client, job)
            output = format_results(raw_results, video_map)

            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(f"  Results saved to {output_path}")
        except Exception as exc:
            entry["status"] = "retrieve_error"
            entry["error"] = str(exc)
            print(f"  [ERROR] Failed to download/format results: {exc}")

    if modified:
        save_registry(registry_path, entries)
        print(f"\n[INFO] Registry updated: {registry_path}")
    else:
        print(f"\n[INFO] No jobs completed yet — registry unchanged.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve results for all completed Gemini batch jobs in the registry.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to the shared registry JSON file.",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help="Path to Gemini API key file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = get_client(api_key_path=args.api_key_path)
    process_registry(client, args.registry)


if __name__ == "__main__":
    main()
