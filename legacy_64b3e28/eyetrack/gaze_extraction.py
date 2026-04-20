from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from urllib.parse import unquote

try:
    from .annotation_intervals import AnnotatorTimings, VideoTiming, parse_timestamp
except ImportError:
    from annotation_intervals import AnnotatorTimings, VideoTiming, parse_timestamp


MEDIA_URL_PREFIX = "http://localhost:5000/api/media/gestalt_bench/annotation1/"


@dataclass
class VideoEntry:
    video_id: int
    video_url: str
    video_path: Path


@dataclass
class VideoGazeData:
    annotator_number: int
    timing: VideoTiming
    video_entry: VideoEntry
    samples: List[dict]


def load_pldata(pldata_path: Path) -> list:
    """Read a .pldata file and return a list of deserialized payloads."""
    import msgpack

    data = []
    with pldata_path.open("rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        while True:
            try:
                topic, payload_bytes = unpacker.unpack()
                datum = msgpack.unpackb(payload_bytes, raw=False)
                datum["topic"] = topic
                data.append(datum)
            except msgpack.OutOfData:
                break
    return data


def load_gaze(recording_dir: Path) -> Tuple[Any, list]:
    """Load gaze timestamps and gaze data from a Pupil Core recording directory."""
    import numpy as np

    ts_path = recording_dir / "gaze_timestamps.npy"
    pldata_path = recording_dir / "gaze.pldata"

    if not ts_path.exists():
        raise FileNotFoundError(f"Gaze timestamps not found: {ts_path}")
    if not pldata_path.exists():
        raise FileNotFoundError(f"Gaze pldata not found: {pldata_path}")

    timestamps = np.load(ts_path)
    gaze_data = load_pldata(pldata_path)
    if len(timestamps) != len(gaze_data):
        logging.warning(
            "timestamp count (%s) != gaze datum count (%s). Using minimum of both.",
            len(timestamps),
            len(gaze_data),
        )
    return timestamps, gaze_data


def extract_gaze_in_interval(
    timestamps: Any,
    gaze_data: list,
    start_time: float,
    end_time: float,
    confidence_threshold: float = 0.6,
) -> List[dict]:
    """Extract gaze samples whose timestamps fall within [start_time, end_time]."""
    import numpy as np

    indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
    results = []
    for idx in indices:
        if idx >= len(gaze_data):
            continue
        datum = gaze_data[idx]
        confidence = datum.get("confidence", 0.0)
        if confidence < confidence_threshold:
            continue
        norm_pos = datum.get("norm_pos") or [None, None]
        results.append(
            {
                "timestamp": float(timestamps[idx]),
                "norm_pos_x": norm_pos[0],
                "norm_pos_y": norm_pos[1],
                "confidence": confidence,
            }
        )
    return results


def get_first_numeric(mapping: dict, keys: Tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in mapping:
            continue
        try:
            return float(mapping[key])
        except (TypeError, ValueError):
            logging.error("metadata key %s is not numeric: %r", key, mapping[key])
            return None
    return None


def load_recording_time_offset(recording_dir: Path) -> float:
    """
    Return the offset that converts UNIX/system time to Pupil Time.

    Pupil Time = system_time + offset
    """
    info_path = recording_dir / "info.player.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found. Cannot align annotation datetimes with Pupil Time."
        )

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    system_start = get_first_numeric(info, ("start_time_system_s", "start_time_system"))
    pupil_start = get_first_numeric(info, ("start_time_synced_s", "start_time_synced"))
    if system_start is None or pupil_start is None:
        raise KeyError(
            "info.player.json must contain start_time_system_s and start_time_synced_s "
            "to align annotation datetimes with gaze timestamps."
        )
    return pupil_start - system_start


def annotation_time_to_pupil_time(value: str, system_to_pupil_offset: float) -> float:
    return parse_timestamp(value).timestamp() + system_to_pupil_offset


def resolve_video_path(video_url: str, local_path_prefix: Path, media_url_prefix: str) -> Path:
    if video_url.startswith(media_url_prefix):
        relative_path = video_url[len(media_url_prefix) :]
    else:
        logging.warning(
            "video URL does not start with %s; treating it as a relative path: %s",
            media_url_prefix,
            video_url,
        )
        relative_path = video_url
    return local_path_prefix.joinpath(*unquote(relative_path).lstrip("/\\").split("/"))


def iter_video_entries(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        if "id" in node and "video" in node:
            yield node
        for value in node.values():
            yield from iter_video_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_video_entries(value)


def load_video_entries(
    video_json_path: Path,
    local_path_prefix: Path,
    media_url_prefix: str,
) -> Dict[int, VideoEntry]:
    with video_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    videos = {}
    for list_index, item in enumerate(iter_video_entries(payload)):
        video_id = int(item.get("id", list_index))
        video_url = str(item["video"])
        videos[list_index] = VideoEntry(
            video_id=video_id,
            video_url=video_url,
            video_path=resolve_video_path(video_url, local_path_prefix, media_url_prefix),
        )
    return videos


def get_video_entry_for_timing(
    timing: VideoTiming,
    video_entries: Dict[int, VideoEntry] | None,
    local_path_prefix: Path,
    media_url_prefix: str,
) -> VideoEntry | None:
    if timing.video_path:
        annotation_video_path = resolve_video_path(
            timing.video_path,
            local_path_prefix,
            media_url_prefix,
        )
        if annotation_video_path.exists():
            return VideoEntry(
                video_id=timing.video_number,
                video_url=timing.video_path,
                video_path=annotation_video_path,
            )

        logging.warning(
            "Annotation video path does not exist; trying video JSON fallback for video %s: %s",
            timing.video_number,
            annotation_video_path,
        )

    if video_entries is None:
        logging.error(
            "No annotation video path and no video JSON fallback for video number %s.",
            timing.video_number,
        )
        return None

    video_entry = video_entries.get(timing.video_number)
    if video_entry is None:
        logging.error("No video list entry found for video number %s.", timing.video_number)
        return None
    if not video_entry.video_path.exists():
        logging.error("Video file does not exist: %s", video_entry.video_path)
        return None
    return video_entry


def extract_video_gaze_data(
    all_timings: List[AnnotatorTimings],
    video_entries: Dict[int, VideoEntry] | None,
    timestamps: Any,
    gaze_data: list,
    system_to_pupil_offset: float,
    confidence_threshold: float,
    local_path_prefix: Path,
    media_url_prefix: str,
) -> List[VideoGazeData]:
    extracted = []
    for annotator_timings in all_timings:
        for timing in annotator_timings.timings:
            if timing.video_start_time is None or timing.video_end_time is None:
                logging.warning(
                    "Skipping annotator %s video %s because start/end time is missing.",
                    annotator_timings.annotator_number,
                    timing.video_number,
                )
                continue

            video_entry = get_video_entry_for_timing(
                timing,
                video_entries,
                local_path_prefix,
                media_url_prefix,
            )
            if video_entry is None:
                continue

            start_time = annotation_time_to_pupil_time(
                timing.video_start_time,
                system_to_pupil_offset,
            )
            end_time = annotation_time_to_pupil_time(
                timing.video_end_time,
                system_to_pupil_offset,
            )
            samples = extract_gaze_in_interval(
                timestamps,
                gaze_data,
                start_time,
                end_time,
                confidence_threshold,
            )
            extracted.append(
                VideoGazeData(
                    annotator_number=annotator_timings.annotator_number,
                    timing=timing,
                    video_entry=video_entry,
                    samples=samples,
                )
            )
    return extracted
