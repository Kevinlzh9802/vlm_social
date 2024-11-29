"""
This script traverses over the raw videos for each camera, and extracts video segments of 2 minutes each.

It replicates the ffmpeg commands which are listed in the videoSplitCamX.sh scripts, found in staff-bulk

 staff-bulk/ewi/insy/SPCDataSets/conflab-mm/processed/annotation/videoSegments/cam2/videoSplitCam2.sh

"""

from pathlib import Path
import sys
import math

grandparent_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent_dir))

from constants import (  # noqa: E402
    camera_id_to_dict_of_video_index_to_raw_video_file_basename,
    camera_was_rotated_map,
    # CAMERAS_OF_INTEREST,
    # RAW_VIDEOS_FOLDER_IN_STAFF_BULK,
    # VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK,
    # VIDEO_SEGMENTS_FOLDER_IN_LOCAL,
    # check_if_staff_bulk_is_mounted,
)
from ffmpeg_utils import get_video_duration_in_seconds, subprocess_run_with_guardfile  # noqa: E402

CAMERAS_OF_INTEREST = [4]
RAW_VIDEOS_FOLDER_IN_LOCAL = Path.home() / "tudelft" / "projects" / "datasets" / "conflab" / "data_raw" / "cameras" / "video"
VIDEO_SEGMENTS_FOLDER_IN_LOCAL = Path.home() / "tudelft" / "projects" / "datasets" / "modification" / "conflab_seg_custom"

def main():
    # Iterate over all the cameras for which we are interested in extracting segments
    for camera_index in CAMERAS_OF_INTEREST:
        camera_was_rotated = camera_was_rotated_map[f"cam{camera_index}"]

        # Iterate over all raw videos for the given camera
        for (
            video_index,
            raw_video_file_basename,
        ) in camera_id_to_dict_of_video_index_to_raw_video_file_basename[
            f"cam{camera_index}"
        ].items():
            if video_index < 3:
                # Skip videos with an index less than 3
                continue

            raw_video_file_path = (
                RAW_VIDEOS_FOLDER_IN_LOCAL
                / f"cam{camera_index:02}"
                / raw_video_file_basename
            )
            if camera_was_rotated:
                # Handle rotated cameras
                raw_rotated_file_basename = raw_video_file_basename.replace(
                    ".MP4", "_rot.MP4"
                )
                # rot videos already exists
                raw_rotated_video_file_path_in_local = (
                    RAW_VIDEOS_FOLDER_IN_LOCAL
                    / f"cam{camera_index:02}"
                    / raw_rotated_file_basename
                )
                # raw_rotated_video_file_path_in_local = (
                #     VIDEO_SEGMENTS_FOLDER_IN_LOCAL
                #     / "raw_overhead"
                #     / f"cam{camera_index:02}"
                #     / raw_rotated_file_basename
                # )
                #
                # if not raw_rotated_video_file_path_in_local.exists():
                #     raw_rotated_video_file_path_in_local.parent.mkdir(
                #         parents=True, exist_ok=True
                #     )
                #     # Rotate the video using ffmpeg
                #     cmd = [
                #         "ffmpeg",
                #         "-i",
                #         str(raw_video_file_path),
                #         "-c",
                #         "copy",
                #         "-metadata:s:v:0",
                #         "rotate=0",
                #         str(raw_rotated_video_file_path_in_local),
                #     ]
                #     print("================= ROTATING =======================")
                #     subprocess_run_with_guardfile(
                #         cmd,
                #         raw_rotated_video_file_path_in_local.with_suffix(
                #             ".isincomplete.txt"
                #         ),
                #     )

                raw_video_file_path = raw_rotated_video_file_path_in_local

            # Extract video duration and calculate the number of 2-minute segments
            video_duration_in_seconds = get_video_duration_in_seconds(raw_video_file_path)
            number_of_segments = math.ceil(video_duration_in_seconds / 120)

            # Process each video segment
            for segment_index in range(1, number_of_segments + 1):
                video_segments_folder_path_in_local_path_for_camera = (
                    VIDEO_SEGMENTS_FOLDER_IN_LOCAL / f"cam{camera_index}"
                )

                video_segment_file_basename = f"vid{video_index}-seg{segment_index}.mp4"
                video_segment_scaled_file_basename = (
                    f"vid{video_index}-seg{segment_index}-scaled.mp4"
                )
                video_segment_scaled_and_denoised_file_basename = (
                    f"vid{video_index}-seg{segment_index}-scaled-denoised.mp4"
                )

                # Check if the scaled and denoised segment already exists locally
                video_segment_scaled_and_denoised_file_path_in_local = (
                    video_segments_folder_path_in_local_path_for_camera
                    / video_segment_scaled_and_denoised_file_basename
                )
                if video_segment_scaled_and_denoised_file_path_in_local.exists():
                    print(
                        f"[LOCAL] Video segment {video_segment_scaled_and_denoised_file_path_in_local} already exists"
                    )
                    continue
                else:
                    print(
                        f"Video segment {video_segment_scaled_and_denoised_file_basename} needs to be generated"
                    )

                video_segments_folder_path_in_local_path_for_camera.mkdir(
                    parents=True, exist_ok=True
                )

                # Trim the video segment if it doesn't already exist
                video_segment_file_path = (
                    video_segments_folder_path_in_local_path_for_camera
                    / video_segment_file_basename
                )
                if not video_segment_file_path.exists():
                    fast_seek_position = (
                        f"00:{(2*(segment_index-1)-1):02}:40"
                        if segment_index > 1
                        else "00:00:00"
                    )
                    slow_seek_position = "00:00:20" if segment_index > 1 else "00:00:00"
                    cmd = [
                        "ffmpeg",
                        "-ss",
                        fast_seek_position,
                        "-i",
                        str(raw_video_file_path),
                        "-vcodec",
                        "copy",
                        "-acodec",
                        "copy",
                        "-copyinkf",
                        "-ss",
                        slow_seek_position,
                        "-t",
                        "00:02:00",
                        str(video_segment_file_path),
                    ]
                    print("================= Trimming =======================")
                    subprocess_run_with_guardfile(
                        cmd, video_segment_file_path.with_suffix(".isincomplete.txt")
                    )

                # Scale the video segment if it doesn't already exist
                video_segment_scaled_file_path = (
                    video_segments_folder_path_in_local_path_for_camera
                    / video_segment_scaled_file_basename
                )
                if not video_segment_scaled_file_path.exists():
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(video_segment_file_path),
                        "-s",
                        "960x540",
                        "-c:a",
                        "copy",
                        "-copyinkf",
                        str(video_segment_scaled_file_path),
                    ]
                    print("================= Scaling =======================")
                    subprocess_run_with_guardfile(
                        cmd,
                        video_segment_scaled_file_path.with_suffix(".isincomplete.txt"),
                    )

                # Denoise the video segment
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_segment_scaled_file_path),
                    "-vf",
                    "hqdn3d=luma_tmp=30",
                    "-vcodec",
                    "libx264",
                    "-tune",
                    "film",
                    str(video_segment_scaled_and_denoised_file_path_in_local),
                ]
                print("================= Denoising =======================")
                subprocess_run_with_guardfile(
                    cmd,
                    video_segment_scaled_and_denoised_file_path_in_local.with_suffix(
                        ".isincomplete.txt"
                    ),
                )

if __name__ == "__main__":
    main()
