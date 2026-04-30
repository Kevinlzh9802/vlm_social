import cv2
import os
import subprocess


def _build_ffmpeg_clip_command(
    video_path,
    start_sec,
    duration_sec,
    output_clip_path,
    video_include_audio,
    video_codec,
):
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
    ]

    if video_include_audio:
        ffmpeg_cmd += [
            "-map", "0:v",
            "-map", "0:a?",
            "-c:v", video_codec,
            "-c:a", "aac",
            "-shortest",
            output_clip_path,
        ]
    else:
        ffmpeg_cmd += [
            "-map", "0:v",
            "-c:v", video_codec,
            output_clip_path,
        ]

    return ffmpeg_cmd


def _build_ffmpeg_wav_command(
    video_path,
    start_sec,
    duration_sec,
    output_wav_path,
):
    return [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", video_path,
        "-map", "0:a?",
        "-c:a", "pcm_s16le",
        output_wav_path,
    ]

def save_last_frame(video_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all video files in the folder
    for video_file in os.listdir(video_folder):
        # Check if the file is a video by its extension
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]  # Get the name without extension

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error opening video file: {video_file}")
                continue

            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Set the video position to the last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

            # Read the last frame
            ret, frame = cap.read()
            if ret:
                # Save the last frame as an image
                output_image_path = os.path.join(output_folder, f"{video_name}_last.jpg")
                cv2.imwrite(output_image_path, frame)
                print(f"Last frame of {video_file} saved as {output_image_path}")
            else:
                print(f"Failed to read last frame of {video_file}")

            # Release the video capture object
            cap.release()

def cut_video_into_clips(
    video_path,
    output_folder,
    clip_length=0.5,
    cumulative=True,
    save_separate_audio=False,
    video_include_audio=True,
):
    """
    Cut video into clips. Optionally save separate WAV per clip and choose whether
    the MP4 has audio (only when save_separate_audio is True).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file to get properties
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(clip_length * fps)
    cap.release()

    if fps <= 0 or clip_frames <= 0:
        print(f"Invalid FPS ({fps}) or clip length ({clip_length}) for {video_path}")
        return

    # Calculate the number of clips
    num_clips = total_frames // clip_frames

    for i in range(num_clips):
        if cumulative:
            start_frame = 0
            num_frames = (i + 1) * clip_frames
        else:
            start_frame = i * clip_frames
            num_frames = clip_frames

        start_sec = start_frame / fps
        duration_sec = num_frames / fps

        output_clip_path = os.path.join(output_folder, f"clip_{i + 1}.mp4")
        base_name = f"clip_{i + 1}"
        output_wav_path = os.path.join(output_folder, f"{base_name}.wav")

        ret = None
        for video_codec in ("libx264", "mpeg4"):
            ffmpeg_cmd = _build_ffmpeg_clip_command(
                video_path=video_path,
                start_sec=start_sec,
                duration_sec=duration_sec,
                output_clip_path=output_clip_path,
                video_include_audio=video_include_audio,
                video_codec=video_codec,
            )
            ret = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if ret.returncode == 0:
                break
            if "Unknown encoder" not in (ret.stderr or ""):
                break

        if ret is None or ret.returncode != 0:
            print(f"  ffmpeg stderr: {ret.stderr[-500:] if ret and ret.stderr else 'none'}")
            print(f"Clip {i + 1} failed for {video_path}")
        else:
            print(f"Clip {i + 1} saved as {output_clip_path}")
            if save_separate_audio:
                audio_ret = subprocess.run(
                    _build_ffmpeg_wav_command(
                        video_path=video_path,
                        start_sec=start_sec,
                        duration_sec=duration_sec,
                        output_wav_path=output_wav_path,
                    ),
                    capture_output=True,
                    text=True,
                )
                if audio_ret.returncode != 0:
                    print(
                        "  audio ffmpeg stderr: "
                        f"{audio_ret.stderr[-500:] if audio_ret.stderr else 'none'}"
                    )
                    print(f"  Audio extraction failed for clip {i + 1} of {video_path}")
                elif os.path.exists(output_wav_path):
                    print(f"  Audio saved as {output_wav_path}")

def cut_videos_in_folder(
    video_folder,
    output_folder,
    clip_length=0.5,
    cumulative=True,
    save_separate_audio=False,
    video_include_audio=True,
):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all video files in the folder
    for video_file in os.listdir(video_folder):
        # Check if the file is a video by its extension
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, video_file)
            output_subfolder = os.path.join(output_folder, os.path.splitext(video_file)[0])
            cut_video_into_clips(
                video_path,
                output_subfolder,
                clip_length,
                cumulative,
                save_separate_audio,
                video_include_audio,
            )

def main():
    dataset_path = '<project-root>/projects/datasets/MIntRec2.0'
    # Example usage
    video_folder = os.path.join(dataset_path, 'in-scope-20260223T102217Z-1-002', 'in-scope', 'raw_data')
    output_folder = os.path.join(dataset_path, 'modification', 'in-scope-20260223T102217Z-1-002', 'in-scope', 'raw_data_segmented')

    # save_last_frame(video_folder, output_folder)
    cut_videos_in_folder(video_folder, output_folder, clip_length=0.5, cumulative=True, save_separate_audio=True, video_include_audio=False)

if __name__ == '__main__':
    main()
