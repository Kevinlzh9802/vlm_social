#!/usr/bin/env python3
"""
Parse grouped Seamless Interaction interactions into utterance-level clips.

Input structure (from the previous grouping step):
<grouped_root>/
  V00_S0032_I00000486/
    manifest.json
    p1/
      V00_S0032_I00000486_P0049.mp4
      V00_S0032_I00000486_P0049.wav
      V00_S0032_I00000486_P0049.json
    p2/
      V00_S0032_I00000486_PXXXX.mp4
      V00_S0032_I00000486_PXXXX.wav
      V00_S0032_I00000486_PXXXX.json

Output structure:
<out_root>/
  dialogue_info/
    dia1.json
    dia2.json
    ...
  dia1_utt1.mp4
  dia1_utt2.mp4
  ...
  dia2_utt1.mp4
  ...
  _utterance_split_summary.json

Default behavior:
- transcript-based segmentation
- each transcript segment becomes one utterance
- utterances across both participants are ordered globally by start time
- numbering dia{n}_utt{m} uses the 1-based conversation index for n and the
  conversation timeline order for m
- each output MP4 contains both video and audio (muxed)
- all clips are placed flat under out_root; per-dialogue metadata JSONs go
  into out_root/dialogue_info/

Requires:
- ffmpeg installed and available on PATH
- ffprobe installed and available on PATH

Example:
python split_interactions_to_utterances.py \
    --grouped-root /data/seamless/grouped_naturalistic \
    --out-root /data/seamless/utterance_level \
    --mode transcript \
    --padding 0.0 \
    --min-duration 0.05
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ---------- Data structures ----------

@dataclass
class Utterance:
    interaction_id: str
    speaker_role: str          # "p1" or "p2"
    speaker_file_id: str       # full sample id from source json["id"]
    local_index: int           # utterance index within speaker stream
    global_index: int          # utterance index in merged dialogue order
    start: float
    end: float
    duration: float
    transcript: str
    words: List[dict]
    source_video: str
    source_audio: str
    source_json: str


# ---------- Utilities ----------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_single_file(folder: Path, suffixes: Tuple[str, ...]) -> Optional[Path]:
    matches = []
    for s in suffixes:
        matches.extend(folder.glob(f"*{s}"))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    if len(matches) > 1:
        # Prefer non-derived files
        matches = sorted(matches)
    return matches[0]


def _detect_h264_encoder() -> Tuple[str, List[str]]:
    """Return (encoder_name, extra_flags) for the best available H.264 encoder."""
    candidates = [
        ("libx264", ["-preset", "fast", "-crf", "18"]),
        ("libopenh264", []),
    ]
    for name, flags in candidates:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
        if name in result.stdout:
            return name, flags
    raise RuntimeError(
        "No supported H.264 encoder found in ffmpeg "
        "(need libx264 or libopenh264)"
    )


_H264_ENCODER: Optional[Tuple[str, List[str]]] = None


def get_h264_encoder() -> Tuple[str, List[str]]:
    global _H264_ENCODER
    if _H264_ENCODER is None:
        _H264_ENCODER = _detect_h264_encoder()
    return _H264_ENCODER


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def run_ffmpeg_trim_av(
    video_src: Path,
    audio_src: Path,
    dst: Path,
    start: float,
    end: float,
    reencode: bool = True,
) -> None:
    """Trim video and audio from separate source files and mux into one MP4.

    When *reencode* is True (the default), ``-ss`` is placed *after* both
    ``-i`` inputs (output-side seek).  ffmpeg decodes every frame from the
    start of each file so every packet has a valid PTS — no mosaic artifacts
    and no timestamp errors.  Audio is encoded to AAC for MP4 compatibility.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)

    if reencode:
        enc_name, enc_flags = get_h264_encoder()
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_src),
            "-i", str(audio_src),
            "-ss", f"{start:.6f}",
            "-t", f"{duration:.6f}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", enc_name,
            *enc_flags,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(dst),
        ]
    else:
        # Stream-copy video; audio must still be encoded for MP4 container
        # (source is PCM WAV which MP4 cannot hold as-is).
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-i", str(video_src),
            "-ss", f"{start:.6f}",
            "-i", str(audio_src),
            "-t", f"{duration:.6f}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            str(dst),
        ]

    subprocess.run(cmd, check=True)


def clamp_interval(start: float, end: float, max_duration: float) -> Tuple[float, float]:
    start = max(0.0, start)
    end = min(max_duration, end)
    if end < start:
        end = start
    return start, end


def build_utterance_tag(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}"


def build_dialogue_json_name(dialogue_id: int) -> str:
    return f"dia{dialogue_id}.json"


# ---------- Segmentation logic ----------

def extract_segments_from_transcript_json(
    sample_json: dict,
    interaction_id: str,
    speaker_role: str,
    source_video: Path,
    source_audio: Path,
    source_json_path: Path,
    min_duration: float,
) -> List[Utterance]:
    """
    Expected schema:
    {
      "id": "...",
      "metadata:transcript": [
        {
          "words": [...],
          "start": ...,
          "end": ...,
          "transcript": "..."
        },
        ...
      ]
    }
    """
    sample_id = sample_json.get("id", source_json_path.stem)
    transcript_segments = sample_json.get("metadata:transcript", [])

    utterances: List[Utterance] = []
    for i, seg in enumerate(transcript_segments, start=1):
        if not isinstance(seg, dict):
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        transcript = str(seg.get("transcript", "")).strip()
        words = seg.get("words", [])
        duration = end - start

        if duration < min_duration:
            continue

        utterances.append(
            Utterance(
                interaction_id=interaction_id,
                speaker_role=speaker_role,
                speaker_file_id=sample_id,
                local_index=i,
                global_index=-1,  # assigned later after merge
                start=start,
                end=end,
                duration=duration,
                transcript=transcript,
                words=words if isinstance(words, list) else [],
                source_video=str(source_video),
                source_audio=str(source_audio),
                source_json=str(source_json_path),
            )
        )

    return utterances


def extract_segments_from_vad_json(
    sample_json: dict,
    interaction_id: str,
    speaker_role: str,
    source_video: Path,
    source_audio: Path,
    source_json_path: Path,
    min_duration: float,
) -> List[Utterance]:
    """
    Fallback mode if needed.
    Supports common simple VAD layouts:
      - list of {"start":..., "end":...}
      - list of [start, end]
    Transcript text is left empty.
    """
    sample_id = sample_json.get("id", source_json_path.stem)
    vad_segments = sample_json.get("metadata:vad", [])

    utterances: List[Utterance] = []
    for i, seg in enumerate(vad_segments, start=1):
        start = None
        end = None

        if isinstance(seg, dict):
            if "start" in seg and "end" in seg:
                start = float(seg["start"])
                end = float(seg["end"])
        elif isinstance(seg, list) and len(seg) >= 2:
            start = float(seg[0])
            end = float(seg[1])

        if start is None or end is None:
            continue

        duration = end - start
        if duration < min_duration:
            continue

        utterances.append(
            Utterance(
                interaction_id=interaction_id,
                speaker_role=speaker_role,
                speaker_file_id=sample_id,
                local_index=i,
                global_index=-1,
                start=start,
                end=end,
                duration=duration,
                transcript="",
                words=[],
                source_video=str(source_video),
                source_audio=str(source_audio),
                source_json=str(source_json_path),
            )
        )

    return utterances


def assign_global_order(utterances: List[Utterance]) -> List[Utterance]:
    utterances_sorted = sorted(
        utterances,
        key=lambda u: (u.start, u.end, u.speaker_role, u.local_index)
    )
    for idx, utt in enumerate(utterances_sorted, start=1):
        utt.global_index = idx
    return utterances_sorted


def merge_contiguous_utterances(utterances: List[Utterance]) -> List[Utterance]:
    """Merge consecutive utterances from the same speaker into one.

    The input list must already be sorted in global timeline order (i.e. the
    output of ``assign_global_order``).  Two utterances are considered
    *contiguous* when they are adjacent in the sorted list **and** belong to
    the same ``speaker_role`` (which also implies the same source files).

    Merged utterances take the earliest ``start``, latest ``end``,
    concatenated ``transcript`` (space-separated), and combined ``words``
    lists.  ``local_index`` is set to the first constituent's value and
    ``global_index`` is reassigned afterwards.
    """
    if not utterances:
        return []

    merged: List[Utterance] = []
    current = utterances[0]

    for nxt in utterances[1:]:
        if nxt.speaker_role == current.speaker_role:
            # Merge nxt into current.
            new_end = max(current.end, nxt.end)
            new_start = min(current.start, nxt.start)
            combined_transcript = " ".join(
                t for t in (current.transcript, nxt.transcript) if t
            )
            combined_words = current.words + nxt.words
            current = Utterance(
                interaction_id=current.interaction_id,
                speaker_role=current.speaker_role,
                speaker_file_id=current.speaker_file_id,
                local_index=current.local_index,
                global_index=-1,
                start=new_start,
                end=new_end,
                duration=new_end - new_start,
                transcript=combined_transcript,
                words=combined_words,
                source_video=current.source_video,
                source_audio=current.source_audio,
                source_json=current.source_json,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)

    # Reassign global indices.
    for idx, utt in enumerate(merged, start=1):
        utt.global_index = idx

    return merged


# ---------- Per-interaction processing ----------

def process_interaction(
    interaction_dir: Path,
    out_root: Path,
    dialogue_id: int,
    mode: str,
    padding: float,
    min_duration: float,
    reencode: bool,
    overwrite: bool,
    merge_contiguous: bool = True,
) -> dict:
    interaction_id = interaction_dir.name
    dialogue_info_dir = out_root / "dialogue_info"
    dialogue_info_dir.mkdir(parents=True, exist_ok=True)

    participants = []
    for role in ("p1", "p2"):
        person_dir = interaction_dir / role
        if not person_dir.is_dir():
            continue

        video_path = find_single_file(person_dir, (".mp4",))
        audio_path = find_single_file(person_dir, (".wav",))
        json_path = None

        candidate_jsons = [
            p for p in person_dir.glob("*.json")
            if p.name not in {"transcript.json", "vad.json"}
        ]
        if candidate_jsons:
            json_path = sorted(candidate_jsons)[0]

        if video_path is None or audio_path is None or json_path is None:
            continue

        participants.append(
            {
                "role": role,
                "video": video_path,
                "audio": audio_path,
                "json": json_path,
            }
        )

    if len(participants) != 2:
        return {
            "interaction_id": interaction_id,
            "dialogue_id": dialogue_id,
            "status": "skipped",
            "reason": f"expected 2 participants, found {len(participants)}",
            "utterances_written": 0,
        }

    all_utts: List[Utterance] = []
    participant_summaries = []

    for p in participants:
        sample_json = read_json(p["json"])
        if mode == "transcript":
            utts = extract_segments_from_transcript_json(
                sample_json=sample_json,
                interaction_id=interaction_id,
                speaker_role=p["role"],
                source_video=p["video"],
                source_audio=p["audio"],
                source_json_path=p["json"],
                min_duration=min_duration,
            )
        elif mode == "vad":
            utts = extract_segments_from_vad_json(
                sample_json=sample_json,
                interaction_id=interaction_id,
                speaker_role=p["role"],
                source_video=p["video"],
                source_audio=p["audio"],
                source_json_path=p["json"],
                min_duration=min_duration,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        participant_summaries.append(
            {
                "role": p["role"],
                "video": str(p["video"]),
                "audio": str(p["audio"]),
                "json": str(p["json"]),
                "segments_found": len(utts),
            }
        )
        all_utts.extend(utts)

    all_utts = assign_global_order(all_utts)

    if merge_contiguous:
        pre_merge_count = len(all_utts)
        all_utts = merge_contiguous_utterances(all_utts)
    else:
        pre_merge_count = len(all_utts)

    # Cache media durations
    durations = {}
    for p in participants:
        durations[str(p["video"])] = ffprobe_duration(p["video"])
        durations[str(p["audio"])] = ffprobe_duration(p["audio"])

    utterance_index = []
    dialogue_json_out = dialogue_info_dir / build_dialogue_json_name(dialogue_id)

    for utt in all_utts:
        tag = build_utterance_tag(dialogue_id, utt.global_index)
        clip_out = out_root / f"{tag}.mp4"

        if clip_out.exists() and not overwrite:
            utterance_index.append(
                {
                    "tag": tag,
                    "dialogue_id": dialogue_id,
                    **asdict(utt),
                    "clip_out": str(clip_out),
                    "status": "exists_skipped",
                }
            )
            continue

        padded_start = utt.start - padding
        padded_end = utt.end + padding

        video_max = durations[utt.source_video]
        audio_max = durations[utt.source_audio]
        clip_max = min(video_max, audio_max)

        clip_start, clip_end = clamp_interval(padded_start, padded_end, clip_max)

        run_ffmpeg_trim_av(
            video_src=Path(utt.source_video),
            audio_src=Path(utt.source_audio),
            dst=clip_out,
            start=clip_start,
            end=clip_end,
            reencode=reencode,
        )

        utterance_index.append(
            {
                "tag": tag,
                "dialogue_id": dialogue_id,
                **asdict(utt),
                "clip_start": clip_start,
                "clip_end": clip_end,
                "clip_out": str(clip_out),
                "status": "written",
            }
        )

    write_json(
        dialogue_json_out,
        {
            "interaction_id": interaction_id,
            "dialogue_id": dialogue_id,
            "mode": mode,
            "padding": padding,
            "min_duration": min_duration,
            "merge_contiguous": merge_contiguous,
            "num_utterances_pre_merge": pre_merge_count,
            "num_utterances_total": len(all_utts),
            "participants": participant_summaries,
            "utterances": utterance_index,
        },
    )

    return {
        "interaction_id": interaction_id,
        "dialogue_id": dialogue_id,
        "dialogue_json": str(dialogue_json_out),
        "status": "ok",
        "merge_contiguous": merge_contiguous,
        "utterances_pre_merge": pre_merge_count,
        "utterances_written": sum(1 for x in utterance_index if x["status"] == "written"),
        "utterances_total": len(all_utts),
    }


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped-root", type=Path, required=True, help="Per-interaction grouped root")
    parser.add_argument("--out-root", type=Path, required=True, help="Output utterance-level root")
    parser.add_argument(
        "--mode",
        type=str,
        default="transcript",
        choices=["transcript", "vad"],
        help="Segmentation mode; transcript is default",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Seconds added before start and after end of each utterance",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="Skip utterances shorter than this duration in seconds",
    )
    parser.add_argument(
        "--no-reencode",
        action="store_true",
        default=False,
        help=(
            "Use stream-copy instead of re-encoding. Much faster but cuts "
            "can only land on keyframes, which typically causes mosaic "
            "artifacts at the start of each clip."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing utterance clips",
    )
    parser.add_argument(
        "--interaction-glob",
        type=str,
        default="V*_S*_I*",
        help="Glob pattern for interaction folders under grouped-root",
    )
    parser.add_argument(
        "--no-merge-contiguous",
        action="store_true",
        default=False,
        help=(
            "Disable merging of contiguous utterances from the same speaker. "
            "By default consecutive same-speaker segments are merged into one "
            "utterance."
        ),
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH")

    args.out_root.mkdir(parents=True, exist_ok=True)

    interaction_dirs = sorted(
        [p for p in args.grouped_root.glob(args.interaction_glob) if p.is_dir()]
    )

    summary = {
        "grouped_root": str(args.grouped_root),
        "out_root": str(args.out_root),
        "mode": args.mode,
        "padding": args.padding,
        "min_duration": args.min_duration,
        "reencode": not args.no_reencode,
        "merge_contiguous": not args.no_merge_contiguous,
        "num_interactions_seen": len(interaction_dirs),
        "results": [],
    }

    for i, interaction_dir in enumerate(interaction_dirs, start=1):
        print(f"[{i}/{len(interaction_dirs)}] {interaction_dir.name}")
        try:
            result = process_interaction(
                interaction_dir=interaction_dir,
                out_root=args.out_root,
                dialogue_id=i,
                mode=args.mode,
                padding=args.padding,
                min_duration=args.min_duration,
                reencode=not args.no_reencode,
                overwrite=args.overwrite,
                merge_contiguous=not args.no_merge_contiguous,
            )
        except Exception as e:
            result = {
                "interaction_id": interaction_dir.name,
                "dialogue_id": i,
                "status": "error",
                "error": str(e),
            }
        summary["results"].append(result)

    write_json(args.out_root / "_utterance_split_summary.json", summary)
    print("Done.")


if __name__ == "__main__":
    main()