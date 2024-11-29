from pathlib import Path
from datetime import datetime, timedelta
import subprocess

from constants import RAW_VIDEOS_FRAMERATE


def subprocess_run_with_guardfile(cmd: list[str], guardfile_path: Path):
    """
    Runs a subprocess command, and writes the command to a guardfile, which is deleted after subprocess runs
    successfully as a way to monitor whether there was a crash mid-processing, or a forced stop.

    """
    running_message: str = "Running: " + " ".join(cmd)
    with open(guardfile_path, "w") as f:
        f.write(running_message)
    subprocess.run(cmd, check=True)
    guardfile_path.unlink()


def get_video_duration_in_seconds(file_path: Path) -> float:
    # fmt: off
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    # fmt: on
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(output.stdout)


def extract_timecode_from_conflab_raw_go_pro_video_metadata(
    video_path: Path,
) -> datetime:
    # fmt: off
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-show_streams",
        "-i", str(video_path),
    ]
    # fmt: on
    res = subprocess.run(cmd, capture_output=True)
    video_timecode = res.stdout.decode("utf-8").rstrip()
    # Find TAG:timecode=<VAL> in the string
    video_timecode = video_timecode.split("TAG:timecode=")[1].split("\n")[0]
    # The metadata in the files do not have date information, only HH:MM:SS, so we manually add it
    video_date = [2019, 10, 24]
    video_time = video_timecode.split(":")
    video_time[-1] = round(1000000 * (int(video_time[-1]) / RAW_VIDEOS_FRAMERATE))
    video_time = list(map(int, video_time))
    # Finally, we add 2 hours to account for a timezone difference
    return datetime(*video_date, *video_time) + timedelta(hours=2)