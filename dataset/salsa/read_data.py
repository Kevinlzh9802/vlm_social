import cv2
import random
import os
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip


dataset_parent_path = '/home/zonghuan/tudelft/projects/datasets/'
dataset_name = 'salsa'
dataset_path = os.path.join(dataset_parent_path, dataset_name)
IMG_PER_VIDEO = 200
SEGMENT_PER_VIDEO = 20
SEGMENT_DURATION = 5

def select_random_frames_dataset(d_path, v_paths, n):
    all_frames = []
    for v_path in v_paths:
        v_path_complete = os.path.join(d_path, v_path)
        all_frames.append(select_random_frames(v_path_complete, n))
    return all_frames

def select_random_frames(v_path, n):
    # Open the video file
    cap = cv2.VideoCapture(v_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n > total_frames:
        print(f"Error: The video only has {total_frames} frames, but {n} were requested.")
        return []

    # Randomly select 'n' frame indices
    # frame_indices = sorted(random.sample(range(total_frames), n))
    frame_indices = np.linspace(1000, total_frames-1000, n).round().astype(int).tolist()
    selected_frames = []

    for idx in frame_indices:
        # Set the video capture to the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Read the frame
        ret, frame = cap.read()
        if ret:
            selected_frames.append(frame)
        else:
            print(f"Error: Could not read frame at index {idx}")

    # Release the video capture
    cap.release()
    return selected_frames, frame_indices


def dataset_spec_video_path(name):
    if name == 'salsa':
        return salsa_video_paths()
    else:
        raise NotImplemented

def salsa_video_paths():
    cpp_names = [f'salsa_cpp_cam{x}.avi' for x in range(1, 5)]
    ps = [f'salsa_ps_cam{x}.avi' for x in range(1, 5)]
    return cpp_names + ps


def video_to_img_selection(n_frames):
    video_paths = dataset_spec_video_path(dataset_name)
    dataset_modify_path = os.path.join(dataset_parent_path, 'modification', dataset_name + f'_even_{n_frames}')
    if not os.path.exists(dataset_modify_path):
        os.makedirs(dataset_modify_path)
    selected_frames = select_random_frames_dataset(dataset_path, video_paths, n_frames)

    # Optionally, show the selected frames (useful for testing)
    for video_idx, frames in enumerate(selected_frames):
        video_name = video_paths[video_idx].split('.')[0]
        for k in range(len(frames[0])):
            frame = frames[0][k]
            frame_num = str(frames[1][k]).zfill(8)
            frame_idx = str(k).zfill(3)
            cv2.imwrite(os.path.join(dataset_modify_path, f'{video_name}_{frame_num}_{frame_idx}.jpg'), frame)
        # cv2.imshow(f'Frame {i + 1}', fr)
        # cv2.waitKey(0)  # Wait for a key press to display each frame
    # cv2.destroyAllWindows()

def single_video_clip(video_path, segment_duration, write_path):
    video = VideoFileClip(video_path)
    # Loop through the video and create segments
    video_name = video_path.split('/')[-1].split('.')[0]
    frame_indices = np.linspace(5, video.duration - 10, 20).round().astype(int).tolist()
    for clip_idx, start_time in enumerate(frame_indices):
        # Define the end time for each segment
        end_time = min(start_time + segment_duration, video.duration)
        # Extract the segment
        video_segment = video.subclip(start_time, end_time)
        # Define the output file name
        str_idx = str(clip_idx).zfill(3)
        str_start = str(start_time).zfill(6)
        str_end = str(end_time).zfill(6)
        output_filename = os.path.join(write_path, f"{video_name}_segment_{str_idx}_{str_start}_{str_end}.mp4")
        # Write the segment to a file
        video_segment.write_videofile(output_filename, codec="libx264")

    # Close the video file to free resources
    video.close()

def video_to_clips():
    # Load the video file
    video_paths = salsa_video_paths()
    dataset_modify_path = os.path.join(dataset_parent_path, 'modification',
                                       dataset_name + f'_video_clips_{SEGMENT_DURATION}')
    if not os.path.exists(dataset_modify_path):
        os.makedirs(dataset_modify_path)
    for video_path in video_paths:
        v_path = os.path.join(dataset_path, video_path)
        single_video_clip(v_path, segment_duration=SEGMENT_DURATION, write_path=dataset_modify_path)

    print("Video segments have been saved.")

def main():
    video_to_img_selection(IMG_PER_VIDEO)
    # video_to_clips()

if __name__ == '__main__':
    main()
