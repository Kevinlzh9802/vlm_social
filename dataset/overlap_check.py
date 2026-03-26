#!/usr/bin/env python3
"""
Word-level duration analysis and conversation filtering for Seamless data.

For each interaction:
1. Extract word-level start/end from participant JSONs.
2. Plot word durations on a timeline (words >4s flagged red), 20 per page.
3. Filter conversations:
   a. Remove all words with duration > threshold (default 4s).
   b. Remove conversations where the speech ratio between participants is
      >5 or <1/5 (after removing long words).
   c. Remove conversations where combined speech is <10% of video length.
4. Output a filtered interaction list for seamless_construct_utterance.py.

Requires: matplotlib

Example:
    python overlap_check.py \
        --grouped-root /data/seamless/grouped_interaction \
        --out-dir /data/seamless/overlap_analysis
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@dataclass
class Word:
    speaker: str
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_participant_json(person_dir: Path) -> Optional[Path]:
    candidates = [
        p for p in person_dir.glob("*.json")
        if p.name not in {"transcript.json", "vad.json"}
    ]
    return sorted(candidates)[0] if candidates else None


# ---------------------------------------------------------------------------
# Extract word-level intervals from metadata:transcript
# ---------------------------------------------------------------------------

def extract_words(sample_json: dict, speaker: str) -> List[Word]:
    words: List[Word] = []
    for seg in sample_json.get("metadata:transcript", []):
        if not isinstance(seg, dict):
            continue
        for w in seg.get("words", []):
            if not isinstance(w, dict):
                continue
            raw_start = w.get("start")
            raw_end = w.get("end")
            if raw_start is None or raw_end is None:
                continue
            start = float(raw_start)
            end = float(raw_end)
            text = str(w.get("word", ""))
            words.append(Word(speaker, start, end, text))
    return words


# ---------------------------------------------------------------------------
# Plot one interaction onto an Axes
# ---------------------------------------------------------------------------

COLORS = {"p1": "#4C72B0", "p2": "#DD8452"}
FLAG_COLOR = "#D62728"
Y_POS = {"p1": 1.0, "p2": 0.5}
BAR_H = 0.35


def plot_on_ax(
    ax: plt.Axes,
    interaction_id: str,
    p1_words: List[Word],
    p2_words: List[Word],
    conv_length: float,
    flag_threshold: float,
    p1_flagged: int,
    p2_flagged: int,
) -> None:
    for w in p1_words:
        dur = w.end - w.start
        color = FLAG_COLOR if dur > flag_threshold else COLORS["p1"]
        ax.barh(Y_POS["p1"], dur, left=w.start,
                height=BAR_H, color=color, edgecolor="none")
    for w in p2_words:
        dur = w.end - w.start
        color = FLAG_COLOR if dur > flag_threshold else COLORS["p2"]
        ax.barh(Y_POS["p2"], dur, left=w.start,
                height=BAR_H, color=color, edgecolor="none")

    ax.set_yticks([Y_POS["p2"], Y_POS["p1"]])
    ax.set_yticklabels(["P2", "P1"], fontsize=7)
    ax.set_xlim(0, conv_length * 1.02)
    ax.set_ylim(0.15, 1.35)
    ax.tick_params(axis="x", labelsize=6)

    ax.set_title(
        f"{interaction_id}  |  {conv_length:.0f}s  "
        f"words: P1={len(p1_words)} P2={len(p2_words)}  "
        f"flagged(>{flag_threshold:.0f}s): P1={p1_flagged} P2={p2_flagged}",
        fontsize=7, pad=2,
    )


# ---------------------------------------------------------------------------
# Flush a page of subplots
# ---------------------------------------------------------------------------

ROWS_PER_PAGE = 20


def flush_page(
    axes_data: list,
    page_num: int,
    out_dir: Path,
    flag_threshold: float,
) -> None:
    n = len(axes_data)
    fig, axes = plt.subplots(n, 1, figsize=(16, 1.6 * n), squeeze=False)

    legend_handles = [
        mpatches.Patch(color=COLORS["p1"], label="P1"),
        mpatches.Patch(color=COLORS["p2"], label="P2"),
        mpatches.Patch(color=FLAG_COLOR, label=f">{flag_threshold:.0f}s"),
    ]

    for i, data in enumerate(axes_data):
        ax = axes[i, 0]
        plot_on_ax(ax, **data)
        if i == 0:
            ax.legend(handles=legend_handles, loc="upper right",
                      fontsize=6, ncol=3)

    fig.tight_layout(h_pad=0.6)
    out_path = out_dir / f"page_{page_num:03d}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name} ({n} plots)")


# ---------------------------------------------------------------------------
# Per-interaction analysis
# ---------------------------------------------------------------------------

def analyse_interaction(
    interaction_dir: Path,
    flag_threshold: float,
) -> Optional[dict]:
    interaction_id = interaction_dir.name

    jsons = {}
    for role in ("p1", "p2"):
        person_dir = interaction_dir / role
        if not person_dir.is_dir():
            return None
        jp = find_participant_json(person_dir)
        if jp is None:
            return None
        jsons[role] = read_json(jp)

    p1_words = extract_words(jsons["p1"], "p1")
    p2_words = extract_words(jsons["p2"], "p2")

    all_endpoints = (
        [w.start for w in p1_words] + [w.end for w in p1_words]
        + [w.start for w in p2_words] + [w.end for w in p2_words]
    )
    if not all_endpoints:
        return None

    # Use max endpoint as approximate video length (video starts at t=0).
    video_length = max(all_endpoints)
    conv_length = video_length

    p1_flagged = sum(1 for w in p1_words if w.end - w.start > flag_threshold)
    p2_flagged = sum(1 for w in p2_words if w.end - w.start > flag_threshold)

    # Clean speech: only words within threshold
    p1_clean = sum(w.end - w.start for w in p1_words if w.end - w.start <= flag_threshold)
    p2_clean = sum(w.end - w.start for w in p2_words if w.end - w.start <= flag_threshold)
    combined_clean = p1_clean + p2_clean

    # Speech ratio (larger / smaller)
    if p1_clean > 0 and p2_clean > 0:
        speech_ratio = max(p1_clean, p2_clean) / min(p1_clean, p2_clean)
    else:
        speech_ratio = float("inf")

    # Talk fraction = combined clean speech / video length
    talk_fraction = combined_clean / video_length if video_length > 0 else 0.0

    return {
        "interaction_id": interaction_id,
        "video_length_s": round(video_length, 3),
        "p1_words_total": len(p1_words),
        "p2_words_total": len(p2_words),
        "p1_flagged": p1_flagged,
        "p2_flagged": p2_flagged,
        "p1_clean_speech_s": round(p1_clean, 3),
        "p2_clean_speech_s": round(p2_clean, 3),
        "combined_clean_speech_s": round(combined_clean, 3),
        "speech_ratio": round(speech_ratio, 3),
        "talk_fraction": round(talk_fraction, 5),
        "_plot_data": {
            "interaction_id": interaction_id,
            "p1_words": p1_words,
            "p2_words": p2_words,
            "conv_length": conv_length,
            "flag_threshold": flag_threshold,
            "p1_flagged": p1_flagged,
            "p2_flagged": p2_flagged,
        },
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_interactions(
    results: List[dict],
    max_speech_ratio: float,
    min_talk_fraction: float,
) -> tuple[List[dict], dict]:
    """Apply the three filters and return (accepted, rejection_stats)."""
    rejected_ratio = []
    rejected_quiet = []
    accepted = []

    for r in results:
        if r["speech_ratio"] > max_speech_ratio:
            rejected_ratio.append(r["interaction_id"])
            continue
        if r["talk_fraction"] < min_talk_fraction:
            rejected_quiet.append(r["interaction_id"])
            continue
        accepted.append(r)

    stats = {
        "total_analysed": len(results),
        "rejected_speech_ratio": len(rejected_ratio),
        "rejected_too_quiet": len(rejected_quiet),
        "accepted": len(accepted),
        "rejected_ratio_ids": rejected_ratio,
        "rejected_quiet_ids": rejected_quiet,
    }
    return accepted, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Word-level duration analysis and filtering for Seamless data.",
    )
    parser.add_argument("--grouped-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--flag-threshold", type=float, default=4.0,
        help="Words longer than this (seconds) are flagged / removed (default 4.0)",
    )
    parser.add_argument(
        "--max-speech-ratio", type=float, default=5.0,
        help="Max allowed ratio between participant speech durations (default 5.0)",
    )
    parser.add_argument(
        "--min-talk-fraction", type=float, default=0.10,
        help="Min combined speech / video length to keep (default 0.10)",
    )
    parser.add_argument("--interaction-glob", type=str, default="V*_S*_I*")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    interaction_dirs = sorted(
        p for p in args.grouped_root.glob(args.interaction_glob) if p.is_dir()
    )
    print(f"Found {len(interaction_dirs)} interactions.")

    results = []
    page_buffer: list = []
    page_num = 1

    for idx, idir in enumerate(interaction_dirs, 1):
        print(f"[{idx}/{len(interaction_dirs)}] {idir.name}")
        info = analyse_interaction(idir, args.flag_threshold)
        if info is None:
            continue

        page_buffer.append(info["_plot_data"])
        results.append(
            {k: v for k, v in info.items() if k != "_plot_data"}
        )

        if len(page_buffer) == ROWS_PER_PAGE:
            flush_page(page_buffer, page_num, args.out_dir, args.flag_threshold)
            page_buffer = []
            page_num += 1

    if page_buffer:
        flush_page(page_buffer, page_num, args.out_dir, args.flag_threshold)

    if not results:
        print("No interactions processed.")
        return

    # ---- Filtering ----
    accepted, filter_stats = filter_interactions(
        results, args.max_speech_ratio, args.min_talk_fraction,
    )

    accepted_ids = [r["interaction_id"] for r in accepted]

    # Output for seamless_construct_utterance.py (--filter-json)
    filter_output = {
        "description": (
            "Filtered interaction list for seamless_construct_utterance.py. "
            "Pass this file via --filter-json to only process these interactions."
        ),
        "filter_params": {
            "flag_threshold_s": args.flag_threshold,
            "max_speech_ratio": args.max_speech_ratio,
            "min_talk_fraction": args.min_talk_fraction,
        },
        "filter_stats": filter_stats,
        "interaction_ids": accepted_ids,
        "per_interaction": accepted,
    }
    filter_json_path = args.out_dir / "filtered_interactions.json"
    with filter_json_path.open("w", encoding="utf-8") as f:
        json.dump(filter_output, f, ensure_ascii=False, indent=2)

    # Full summary (all interactions before filtering)
    total_words = sum(r["p1_words_total"] + r["p2_words_total"] for r in results)
    total_flagged = sum(r["p1_flagged"] + r["p2_flagged"] for r in results)

    full_summary = {
        "num_interactions": len(results),
        "flag_threshold_s": args.flag_threshold,
        "total_words": total_words,
        "total_flagged": total_flagged,
        "flagged_ratio": round(total_flagged / total_words, 5) if total_words else 0.0,
        "per_interaction": results,
    }
    summary_path = args.out_dir / "word_duration_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Interactions analysed : {len(results)}")
    print(f"Total words           : {total_words}")
    if total_words:
        print(f"Flagged (>{args.flag_threshold:.0f}s)      : {total_flagged} ({total_flagged/total_words:.2%})")
    print(f"-------- Filtering --------")
    print(f"Rejected (ratio>{args.max_speech_ratio:.0f})  : {filter_stats['rejected_speech_ratio']}")
    print(f"Rejected (talk<{args.min_talk_fraction:.0%})  : {filter_stats['rejected_too_quiet']}")
    print(f"Accepted              : {filter_stats['accepted']}")
    print(f"-------- Output -----------")
    print(f"Full summary          : {summary_path}")
    print(f"Filtered list         : {filter_json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
