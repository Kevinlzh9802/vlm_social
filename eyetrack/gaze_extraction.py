from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from urllib.parse import unquote

try:
    from .annotation_intervals import AnnotatorTimings, VideoTiming, parse_timestamp
except ImportError:
    from annotation_intervals import AnnotatorTimings, VideoTiming, parse_timestamp


MEDIA_URL_PREFIX = "http://localhost:5000/api/media/gestalt_bench/annotation1/"
SURFACE_EXPORT_DIR_RE = re.compile(r"^\d{3}$")


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


VideoEntryIndex = Dict[tuple[int | None, int], VideoEntry]


def find_latest_surface_gaze_csv(recording_dir: Path) -> Path:
    exports_dir = recording_dir / "exports"
    if not exports_dir.is_dir():
        raise FileNotFoundError(f"Surface export folder not found: {exports_dir}")

    export_dirs = [
        path
        for path in exports_dir.iterdir()
        if path.is_dir() and SURFACE_EXPORT_DIR_RE.fullmatch(path.name)
    ]
    if not export_dirs:
        raise FileNotFoundError(
            f"No 3-digit surface export folders found under {exports_dir}"
        )

    latest_export_dir = max(export_dirs, key=lambda path: int(path.name))
    surface_dir = latest_export_dir / "surfaces"
    candidates = sorted(surface_dir.glob("gaze_positions_on_surface_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No gaze_positions_on_surface_*.csv found under {surface_dir}"
        )
    if len(candidates) > 1:
        logging.warning(
            "Multiple surface gaze CSV files found under %s; using %s",
            surface_dir,
            candidates[-1].name,
        )
    return candidates[-1]


def gaze_source_metadata(recording_dir: Path) -> dict[str, str]:
    surface_csv_path = find_latest_surface_gaze_csv(recording_dir)
    return {
        "gaze_source_type": "surface_csv",
        "gaze_source_path": str(surface_csv_path),
        "gaze_timestamps_path": str(recording_dir / "gaze_timestamps.npy"),
        "gaze_pldata_path": str(recording_dir / "gaze.pldata"),
    }


def load_gaze(recording_dir: Path) -> Tuple[Any, list]:
    """Load gaze timestamps and screen-referenced gaze samples from a Pupil export."""
    import numpy as np

    surface_csv_path = find_latest_surface_gaze_csv(recording_dir)
    timestamps: list[float] = []
    gaze_data: list[dict] = []
    with surface_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"world_timestamp", "x_norm", "y_norm"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise KeyError(
                f"{surface_csv_path} is missing required columns: {sorted(missing_columns)}"
            )

        for row_index, row in enumerate(reader, start=2):
            on_surface_raw = row.get("on_surf")
            if on_surface_raw is not None and str(on_surface_raw).strip().lower() in {
                "0",
                "false",
                "no",
            }:
                continue
            try:
                timestamp = float(row["world_timestamp"])
                x_norm = float(row["x_norm"])
                y_norm = float(row["y_norm"])
            except (TypeError, ValueError) as exc:
                logging.warning(
                    "Skipping invalid surface gaze row %s in %s: %s",
                    row_index,
                    surface_csv_path,
                    exc,
                )
                continue

            confidence_raw = row.get("confidence")
            try:
                confidence = 1.0 if confidence_raw in (None, "") else float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 1.0

            timestamps.append(timestamp)
            gaze_data.append(
                {
                    "timestamp": timestamp,
                    "norm_pos": [x_norm, y_norm],
                    "norm_pos_x": x_norm,
                    # Surface CSV coordinates already use Pupil-style normalization:
                    # bottom-left=(0,0), top-right=(1,1).
                    "norm_pos_y": y_norm,
                    "confidence": confidence,
                }
            )

    if not timestamps:
        logging.warning("No usable surface gaze rows found in %s", surface_csv_path)
        return np.array([], dtype=float), []

    return np.array(timestamps, dtype=float), gaze_data


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
        norm_pos = datum.get("norm_pos")
        if norm_pos is not None:
            norm_pos_x, norm_pos_y = norm_pos[0], norm_pos[1]
        else:
            norm_pos_x = datum.get("norm_pos_x")
            norm_pos_y = datum.get("norm_pos_y")
        results.append(
            {
                "timestamp": float(timestamps[idx]),
                "norm_pos_x": norm_pos_x,
                "norm_pos_y": norm_pos_y,
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


def resolve_optional_media_path(
    media_url: str | None,
    local_path_prefix: Path,
    media_url_prefix: str,
) -> Path | None:
    if not media_url:
        return None
    media_path = resolve_video_path(media_url, local_path_prefix, media_url_prefix)
    if not media_path.exists():
        logging.warning("Resolved media path does not exist: %s", media_path)
        return None
    return media_path


def iter_video_entries(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        if "id" in node and "video" in node:
            yield node
        for value in node.values():
            yield from iter_video_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_video_entries(value)


def iter_indexed_video_entries(node: Any) -> Iterator[tuple[int | None, int, dict]]:
    """Yield video entries keyed by optional task instance and local video index."""
    if isinstance(node, list):
        if all(isinstance(item, list) for item in node):
            for task_instance_id, task_items in enumerate(node):
                for video_index, item in enumerate(task_items):
                    if isinstance(item, dict) and "video" in item:
                        yield task_instance_id, video_index, item
            return

        for video_index, item in enumerate(node):
            if isinstance(item, dict) and "video" in item:
                yield None, video_index, item
            else:
                yield from iter_indexed_video_entries(item)
        return

    for list_index, item in enumerate(iter_video_entries(node)):
        yield None, list_index, item


def load_video_entries(
    video_json_path: Path,
    local_path_prefix: Path,
    media_url_prefix: str,
) -> VideoEntryIndex:
    with video_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    videos = {}
    for task_instance_id, video_index, item in iter_indexed_video_entries(payload):
        video_id = int(item.get("id", video_index))
        video_url = str(item["video"])
        videos[(task_instance_id, video_index)] = VideoEntry(
            video_id=video_id,
            video_url=video_url,
            video_path=resolve_video_path(video_url, local_path_prefix, media_url_prefix),
        )
    return videos


def get_video_entry_for_timing(
    timing: VideoTiming,
    video_entries: VideoEntryIndex | None,
    local_path_prefix: Path,
    media_url_prefix: str,
    task_instance_id: int | None = None,
    use_annotation_video_path: bool = False,
) -> VideoEntry | None:
    # Default to the shared task JSON by task_instance_id first, then local
    # video_number.  The frontend may write incorrect annotation video_path
    # values, so only use those paths when explicitly requested.
    if video_entries is not None:
        lookup_keys = []
        if task_instance_id is not None:
            lookup_keys.append((task_instance_id, timing.video_number))
        lookup_keys.append((None, timing.video_number))

        for lookup_key in lookup_keys:
            video_entry = video_entries.get(lookup_key)
            if video_entry is not None and video_entry.video_path.exists():
                return video_entry
        logging.warning(
            "Video JSON lookup failed for task instance %s video number %s.",
            task_instance_id,
            timing.video_number,
        )

    if use_annotation_video_path and timing.video_path:
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
            "Annotation video path does not exist for video %s: %s.",
            timing.video_number,
            annotation_video_path,
        )

    logging.error(
        "No valid video source found for video number %s.",
        timing.video_number,
    )
    return None


def extract_video_gaze_data(
    all_timings: List[AnnotatorTimings],
    video_entries: VideoEntryIndex | None,
    timestamps: Any,
    gaze_data: list,
    system_to_pupil_offset: float,
    confidence_threshold: float,
    local_path_prefix: Path,
    media_url_prefix: str,
    task_instance_id: int | None = None,
    use_annotation_video_path: bool = False,
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
                task_instance_id,
                use_annotation_video_path,
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
