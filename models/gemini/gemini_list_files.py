#!/usr/bin/env python3
"""List all files uploaded to the Gemini Files API.

Usage
-----
    python gemini_list_files.py
    python gemini_list_files.py --api-key-path path/to/key.txt
    python gemini_list_files.py --json           # output as JSON
    python gemini_list_files.py --delete-all     # delete every listed file
"""

import argparse
import json
import sys
from pathlib import Path

from google import genai

DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent / "configs" / "gemini_api.txt"


def load_api_key(api_key_path: Path) -> str:
    if not api_key_path.exists():
        print(
            f"[ERROR] API key file not found: {api_key_path}\n"
            "  Create the file and paste your Gemini API key into it.",
            file=sys.stderr,
        )
        sys.exit(1)
    key = api_key_path.read_text(encoding="utf-8").strip()
    if not key:
        print(f"[ERROR] API key file is empty: {api_key_path}", file=sys.stderr)
        sys.exit(1)
    return key


def list_files(client: genai.Client) -> list:
    return list(client.files.list())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List all files uploaded to the Gemini Files API.",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help="Path to a text file containing the Gemini API key.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print output as a JSON array instead of a human-readable table.",
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete every file returned by the list (use with caution).",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print the file count, skip the full listing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.api_key_path)
    client = genai.Client(api_key=api_key)

    files = list_files(client)

    if not files:
        print("No files found.")
        return

    if args.count_only:
        print(f"Total: {len(files)} file(s)")
    elif args.as_json:
        records = []
        for f in files:
            records.append({
                "name": f.name,
                "display_name": getattr(f, "display_name", None),
                "mime_type": getattr(f, "mime_type", None),
                "size_bytes": getattr(f, "size_bytes", None),
                "state": str(getattr(f, "state", "")),
                "create_time": str(getattr(f, "create_time", "")),
                "expiration_time": str(getattr(f, "expiration_time", "")),
                "uri": getattr(f, "uri", None),
            })
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        print(f"{'NAME':<30} {'DISPLAY NAME':<35} {'MIME TYPE':<25} {'SIZE':>12}  STATE")
        print("-" * 120)
        for f in files:
            name = f.name or ""
            display = getattr(f, "display_name", "") or ""
            mime = getattr(f, "mime_type", "") or ""
            size = getattr(f, "size_bytes", None)
            size_str = f"{size:,}" if size is not None else "N/A"
            state = str(getattr(f, "state", ""))
            print(f"{name:<30} {display:<35} {mime:<25} {size_str:>12}  {state}")
        print(f"\nTotal: {len(files)} file(s)")

    if args.delete_all:
        confirm = input(f"\nDelete all {len(files)} file(s)? [y/N] ").strip().lower()
        if confirm == "y":
            for f in files:
                client.files.delete(name=f.name)
                print(f"  Deleted {f.name}")
            print("Done.")
        else:
            print("Aborted.")


if __name__ == "__main__":
    main()
