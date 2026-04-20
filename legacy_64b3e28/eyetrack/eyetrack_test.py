"""
Extract eye gaze positions from a Pupil Core recording for given time intervals.

Pupil Core Recording Format (https://docs.pupil-labs.com/core/software/recording-format/):
  - gaze.pldata        : msgpack-encoded gaze data (norm_pos, confidence, timestamp, …)
  - gaze_timestamps.npy: numpy array of timestamps (Pupil Time, float64 seconds)
  - world_timestamps.npy: numpy array of world-camera frame timestamps

Each gaze datum contains:
  - norm_pos  : [x, y] in normalized coordinates (origin = bottom-left, range [0,1])
  - confidence: detection quality (0.0–1.0)
  - timestamp : Pupil Time in seconds

Usage:
    python eyetrack_test.py <recording_dir> --intervals 10.0 15.0 20.0 25.0
    (extracts gaze for intervals [10–15] and [20–25] seconds relative to recording start)
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import msgpack
import numpy as np


def load_pldata(pldata_path: str) -> list:
    """Read a .pldata file and return a list of deserialized payloads."""
    data = []
    with open(pldata_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        while True:
            try:
                msg = unpacker.unpack()  # each message is [topic, payload_bytes]
                topic = msg[0]
                payload_bytes = msg[1]
                datum = msgpack.unpackb(payload_bytes, raw=False)
                datum["topic"] = topic
                data.append(datum)
            except msgpack.OutOfData:
                break
    return data


def load_gaze(recording_dir: str) -> Tuple[np.ndarray, list]:
    """
    Load gaze timestamps and gaze data from a Pupil Core recording directory.
    Returns (timestamps, gaze_data_list).
    """
    ts_path = os.path.join(recording_dir, "gaze_timestamps.npy")
    pldata_path = os.path.join(recording_dir, "gaze.pldata")

    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Gaze timestamps not found: {ts_path}")
    if not os.path.exists(pldata_path):
        raise FileNotFoundError(f"Gaze pldata not found: {pldata_path}")

    timestamps = np.load(ts_path)
    gaze_data = load_pldata(pldata_path)

    if len(timestamps) != len(gaze_data):
        print(
            f"Warning: timestamp count ({len(timestamps)}) != "
            f"gaze datum count ({len(gaze_data)}). Using minimum of both."
        )
    return timestamps, gaze_data


def extract_gaze_in_interval(
    timestamps: np.ndarray,
    gaze_data: list,
    start_time: float,
    end_time: float,
    confidence_threshold: float = 0.6,
) -> List[dict]:
    """
    Extract gaze samples whose timestamps fall within [start_time, end_time].
    Times are in the same Pupil Time coordinate as the recording.
    Only samples with confidence >= confidence_threshold are returned.
    """
    mask = (timestamps >= start_time) & (timestamps <= end_time)
    indices = np.where(mask)[0]

    results = []
    n_data = len(gaze_data)
    for idx in indices:
        if idx >= n_data:
            continue
        datum = gaze_data[idx]
        conf = datum.get("confidence", 0.0)
        if conf < confidence_threshold:
            continue
        norm_pos = datum.get("norm_pos", [None, None])
        results.append(
            {
                "timestamp": float(timestamps[idx]),
                "norm_pos_x": norm_pos[0],
                "norm_pos_y": norm_pos[1],
                "confidence": conf,
            }
        )
    return results


def parse_intervals(values: List[float]) -> List[Tuple[float, float]]:
    """Parse a flat list of floats into (start, end) pairs."""
    if len(values) % 2 != 0:
        raise ValueError("Intervals must be specified as pairs: start1 end1 start2 end2 ...")
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def main():
    parser = argparse.ArgumentParser(
        description="Extract gaze positions from a Pupil Core recording."
    )
    parser.add_argument("recording_dir", help="Path to the Pupil Core recording folder.")
    parser.add_argument(
        "--intervals",
        nargs="+",
        type=float,
        required=True,
        help=(
            "Time intervals as pairs of start/end offsets (seconds from recording start). "
            "Example: --intervals 10 15 20 25  →  [10s–15s] and [20s–25s]"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum gaze confidence threshold (default: 0.6).",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Treat interval times as absolute Pupil Time instead of offsets from recording start.",
    )
    args = parser.parse_args()

    intervals = parse_intervals(args.intervals)
    timestamps, gaze_data = load_gaze(args.recording_dir)

    if len(timestamps) == 0:
        print("No gaze data found in recording.")
        return

    recording_start = timestamps[0]
    print(f"Recording start (Pupil Time): {recording_start:.6f} s")
    print(f"Recording end   (Pupil Time): {timestamps[-1]:.6f} s")
    print(f"Total gaze samples: {len(timestamps)}")
    print(f"Confidence threshold: {args.confidence}")
    print()

    for start_offset, end_offset in intervals:
        if args.absolute:
            t_start, t_end = start_offset, end_offset
        else:
            t_start = recording_start + start_offset
            t_end = recording_start + end_offset

        samples = extract_gaze_in_interval(
            timestamps, gaze_data, t_start, t_end, args.confidence
        )

        label_start = start_offset if not args.absolute else t_start
        label_end = end_offset if not args.absolute else t_end
        print(f"--- Interval [{label_start:.2f}s – {label_end:.2f}s] ---")
        print(f"  Gaze samples found: {len(samples)}")

        if samples:
            xs = [s["norm_pos_x"] for s in samples]
            ys = [s["norm_pos_y"] for s in samples]
            print(f"  Mean gaze position (norm): x={np.mean(xs):.4f}, y={np.mean(ys):.4f}")
            print(f"  Std  gaze position (norm): x={np.std(xs):.4f},  y={np.std(ys):.4f}")
            print(f"  First sample: t={samples[0]['timestamp']:.6f}  "
                  f"pos=({samples[0]['norm_pos_x']:.4f}, {samples[0]['norm_pos_y']:.4f})  "
                  f"conf={samples[0]['confidence']:.2f}")
            print(f"  Last  sample: t={samples[-1]['timestamp']:.6f}  "
                  f"pos=({samples[-1]['norm_pos_x']:.4f}, {samples[-1]['norm_pos_y']:.4f})  "
                  f"conf={samples[-1]['confidence']:.2f}")
        print()


if __name__ == "__main__":
    main()