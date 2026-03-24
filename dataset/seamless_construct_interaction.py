#!/usr/bin/env python3
"""
Group Seamless Interaction participant-centric files into dyadic interaction folders.

Example:
    python group_seamless_interactions.py \
        --data-root /path/to/extracted_batches \
        --filelist /path/to/assets/filelist.csv \
        --interactions /path/to/assets/interactions.csv \
        --label naturalistic \
        --batches 0 1 2 \
        --out-root /path/to/grouped_interactions \
        --link-mode symlink

Output:
    grouped_interactions/
      V00_S0926_I00000480/
        manifest.json
        p1/
          V00_S0926_I00000480_P0092A.mp4
          V00_S0926_I00000480_P0092A.wav
          V00_S0926_I00000480_P0092A.json
          transcript.json
          vad.json
        p2/
          ...

Notes:
- Uses local files actually present under --data-root.
- Uses filelist.csv only to filter to chosen label/split/batches if desired.
- Keeps only interactions with exactly 2 participants and both participants
  having at least video/audio/json by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FILE_ID_RE = re.compile(
    r"^(V\d+)_S(\d+)_I(\d+)_P([A-Za-z0-9]+)$"
)


@dataclass(frozen=True)
class FileIdParts:
    vendor: str          # e.g. V00
    session: str         # e.g. 0926
    interaction: str     # e.g. 00000480
    participant: str     # e.g. 0092A or 1281

    @property
    def file_id(self) -> str:
        return f"{self.vendor}_S{self.session}_I{self.interaction}_P{self.participant}"

    @property
    def interaction_key(self) -> Tuple[str, str, str]:
        return (self.vendor, self.session, self.interaction)

    @property
    def interaction_id(self) -> str:
        return f"{self.vendor}_S{self.session}_I{self.interaction}"


def parse_file_id(file_id: str) -> Optional[FileIdParts]:
    m = FILE_ID_RE.match(file_id)
    if not m:
        return None
    vendor, session, interaction, participant = m.groups()
    return FileIdParts(vendor=vendor, session=session, interaction=interaction, participant=participant)


def infer_file_id_from_path(path: Path) -> Optional[str]:
    """
    Accepts file names like:
        V00_S0926_I00000480_P0092A.mp4
        V00_S0926_I00000480_P0092A.json
    """
    stem = path.stem
    parts = parse_file_id(stem)
    return parts.file_id if parts else None


def load_filelist_rows(
    filelist_csv: Path,
    label: Optional[str],
    split: Optional[str],
    batches: Optional[set[int]],
) -> Dict[str, dict]:
    """
    Maps file_id -> row dict after filtering.
    """
    rows: Dict[str, dict] = {}
    with filelist_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["file_id"].strip()
            if label is not None and row.get("label", "").strip() != label:
                continue
            if split is not None and row.get("split", "").strip() != split:
                continue
            if batches is not None:
                try:
                    batch_idx = int(row.get("batch_idx", ""))
                except ValueError:
                    continue
                if batch_idx not in batches:
                    continue
            rows[file_id] = row
    return rows


def load_interactions_csv(interactions_csv: Optional[Path]) -> Dict[str, dict]:
    """
    Maps prompt_hash (e.g. '00000480') -> row dict
    because the repo documents that prompt_hash corresponds to I<interaction>.
    """
    if interactions_csv is None:
        return {}

    rows: Dict[str, dict] = {}
    with interactions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_hash = row.get("prompt_hash", "").strip()
            if prompt_hash:
                rows[prompt_hash] = row
    return rows


def scan_local_files(data_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Returns:
        file_id -> {".mp4": path, ".wav": path, ".json": path, ".npz": path, ...}
    """
    found: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".wav", ".json", ".npz"}:
            continue
        file_id = infer_file_id_from_path(path)
        if file_id is None:
            continue
        found[file_id][path.suffix.lower()] = path
    return found


def safe_read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_json_read_error": str(e)}


def extract_transcript_and_vad(sample_json: dict) -> Tuple[object, object]:
    """
    The repo examples show keys:
        'metadata:transcript'
        'metadata:vad'
    but we allow a few fallback variants to be robust.
    """
    transcript = None
    vad = None

    for key in ("metadata:transcript", "transcript", "metadata_transcript"):
        if key in sample_json:
            transcript = sample_json[key]
            break

    for key in ("metadata:vad", "vad", "metadata_vad"):
        if key in sample_json:
            vad = sample_json[key]
            break

    return transcript, vad


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def participant_sort_key(participant_id: str) -> Tuple[int, str]:
    """
    Put A before B when present, else lexical fallback.
    Examples:
      0092A < 0092B
      1281A < 1281B
      0731   lexical
    """
    suffix_order = 9
    if participant_id.endswith("A"):
        suffix_order = 0
    elif participant_id.endswith("B"):
        suffix_order = 1
    return (suffix_order, participant_id)


def build_grouped_interactions(
    local_files: Dict[str, Dict[str, Path]],
    allowed_file_ids: Optional[set[str]],
) -> Dict[Tuple[str, str, str], Dict[str, Dict[str, Path]]]:
    """
    Returns:
      (vendor, session, interaction) -> participant_id -> modality map
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Path]]] = defaultdict(dict)

    for file_id, modalities in local_files.items():
        if allowed_file_ids is not None and file_id not in allowed_file_ids:
            continue
        parts = parse_file_id(file_id)
        if parts is None:
            continue
        grouped[parts.interaction_key][parts.participant] = modalities

    return grouped


def interaction_has_required_modalities(
    participant_modalities: Dict[str, Path],
    required_modalities: Tuple[str, ...] = (".mp4", ".wav", ".json"),
) -> bool:
    return all(m in participant_modalities for m in required_modalities)


def write_json(path: Path, obj: object) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Root of extracted HF tar files")
    parser.add_argument("--filelist", type=Path, required=True, help="Path to assets/filelist.csv")
    parser.add_argument("--interactions", type=Path, default=None, help="Path to assets/interactions.csv")
    parser.add_argument("--label", type=str, default="naturalistic", help="Dataset label to keep")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter: train/dev/test")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of batch_idx values to keep, e.g. --batches 0 1 2",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--link-mode",
        type=str,
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place files into grouped folders",
    )
    parser.add_argument(
        "--strict-two-person",
        action="store_true",
        help="Keep only interactions with exactly 2 participants",
    )
    parser.add_argument(
        "--require-modalities",
        nargs="*",
        default=[".mp4", ".wav", ".json"],
        help="Required modalities per participant",
    )
    parser.add_argument(
        "--write-transcript-json",
        action="store_true",
        help="Write transcript.json and vad.json extracted from each participant json",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    batches_set = set(args.batches) if args.batches is not None else None

    print("[1/6] Loading filelist.csv filters...")
    filelist_rows = load_filelist_rows(
        filelist_csv=args.filelist,
        label=args.label,
        split=args.split,
        batches=batches_set,
    )
    allowed_file_ids = set(filelist_rows.keys())
    print(f"  Allowed file_ids from filelist filters: {len(allowed_file_ids)}")

    print("[2/6] Loading interactions.csv...")
    interactions_meta = load_interactions_csv(args.interactions)
    print(f"  Prompt rows loaded: {len(interactions_meta)}")

    print("[3/6] Scanning local extracted files...")
    local_files = scan_local_files(args.data_root)
    print(f"  Local file_ids discovered: {len(local_files)}")

    print("[4/6] Grouping into interactions...")
    grouped = build_grouped_interactions(local_files, allowed_file_ids=allowed_file_ids)
    print(f"  Candidate interaction groups: {len(grouped)}")

    total_written = 0
    skipped_not_two = 0
    skipped_missing_modalities = 0

    summary = {
        "data_root": str(args.data_root),
        "filelist": str(args.filelist),
        "interactions_csv": str(args.interactions) if args.interactions else None,
        "label": args.label,
        "split": args.split,
        "batches": sorted(list(batches_set)) if batches_set is not None else None,
        "link_mode": args.link_mode,
        "require_modalities": args.require_modalities,
        "interactions_written": 0,
        "skipped_not_two": 0,
        "skipped_missing_modalities": 0,
        "interactions": [],
    }

    print("[5/6] Writing grouped interaction folders...")
    for interaction_key in sorted(grouped.keys()):
        vendor, session, interaction = interaction_key
        participant_map = grouped[interaction_key]
        participants_sorted = sorted(participant_map.keys(), key=participant_sort_key)

        if args.strict_two_person and len(participants_sorted) != 2:
            skipped_not_two += 1
            continue

        if len(participants_sorted) < 2:
            skipped_not_two += 1
            continue

        # For safety, keep only the first two sorted participants if there are more than two.
        # This should usually not happen for dyadic data.
        participants_sorted = participants_sorted[:2]

        if not all(
            interaction_has_required_modalities(participant_map[p], tuple(args.require_modalities))
            for p in participants_sorted
        ):
            skipped_missing_modalities += 1
            continue

        interaction_id = f"{vendor}_S{session}_I{interaction}"
        interaction_dir = args.out_root / interaction_id
        interaction_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "interaction_id": interaction_id,
            "vendor": vendor,
            "session_id": session,
            "interaction_hash": interaction,  # matches prompt_hash in interactions.csv per repo docs
            "participants": [],
            "prompt_metadata": interactions_meta.get(interaction, None),
        }

        for idx, participant_id in enumerate(participants_sorted, start=1):
            role_name = f"p{idx}"
            person_dir = interaction_dir / role_name
            person_dir.mkdir(parents=True, exist_ok=True)

            modalities = participant_map[participant_id]
            person_record = {
                "role": role_name,
                "participant_id": participant_id,
                "file_id": f"{vendor}_S{session}_I{interaction}_P{participant_id}",
                "files": {},
                "transcript_present": False,
                "vad_present": False,
            }

            sample_json = None
            for suffix, src_path in sorted(modalities.items()):
                dst_path = person_dir / src_path.name
                link_or_copy(src_path, dst_path, mode=args.link_mode)
                person_record["files"][suffix] = str(dst_path)

                if suffix == ".json":
                    sample_json = safe_read_json(src_path)

            if sample_json is not None:
                transcript, vad = extract_transcript_and_vad(sample_json)

                if args.write_transcript_json:
                    if transcript is not None:
                        write_json(person_dir / "transcript.json", transcript)
                    if vad is not None:
                        write_json(person_dir / "vad.json", vad)

                person_record["transcript_present"] = transcript is not None
                person_record["vad_present"] = vad is not None

                # Keep a tiny preview in manifest, not the whole transcript.
                if isinstance(transcript, list):
                    person_record["transcript_preview_n"] = min(3, len(transcript))
                    person_record["transcript_preview"] = transcript[:3]
                else:
                    person_record["transcript_preview"] = transcript if transcript is not None else None

                if isinstance(vad, list):
                    person_record["vad_preview_n"] = min(3, len(vad))
                    person_record["vad_preview"] = vad[:3]
                else:
                    person_record["vad_preview"] = vad if vad is not None else None

            manifest["participants"].append(person_record)

        write_json(interaction_dir / "manifest.json", manifest)
        summary["interactions"].append(
            {
                "interaction_id": interaction_id,
                "participants": [p["participant_id"] for p in manifest["participants"]],
                "prompt_id_unique": (
                    manifest["prompt_metadata"].get("prompt_id_unique")
                    if manifest["prompt_metadata"] is not None
                    else None
                ),
                "interaction_type": (
                    manifest["prompt_metadata"].get("interaction_type")
                    if manifest["prompt_metadata"] is not None
                    else None
                ),
            }
        )
        total_written += 1

    summary["interactions_written"] = total_written
    summary["skipped_not_two"] = skipped_not_two
    summary["skipped_missing_modalities"] = skipped_missing_modalities

    print("[6/6] Writing summary...")
    write_json(args.out_root / "_grouping_summary.json", summary)

    print("\nDone.")
    print(f"Interactions written:        {total_written}")
    print(f"Skipped (not 2 participants): {skipped_not_two}")
    print(f"Skipped (missing modalities): {skipped_missing_modalities}")
    print(f"Output root: {args.out_root}")


if __name__ == "__main__":
    main()