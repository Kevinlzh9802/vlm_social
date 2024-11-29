import pandas as pd
import cv2
import os
import re
import numpy as np
import json


dataset_path = '/home/zonghuan/tudelft/projects/datasets/conflab/'
csv_path = os.path.join(dataset_path, 'annotations/f_formations/seg3.csv')
cameras_path = os.path.join(dataset_path, 'data_raw/cameras/video/')
fformation_seg = 3

def time_to_seconds(timestamp):
    """Convert timestamp (MM:SS) to seconds."""
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds


def parse_annotations(annotation_str) -> dict:
    """
    Parse the annotation string to extract bounding box IDs and camera info.

    Args:
        annotation_str (str): The string containing annotations in the format
                              '(<1,2,3>, cam4), (<4,5,6>, cam6)'.

    Returns:
        list: A list of tuples [(person_ids, camera), ...].
    """
    matches = re.findall(r'\(<(.+?)>,\s*(cam\d+)\)', annotation_str)
    camera_groups = {}
    for ids, camera in matches:
        person_ids = [int(id_) for id_ in ids.split(",") if id_ != '']  # Extract IDs as a list of integers
        if camera not in camera_groups:
            camera_groups[camera] = []
        camera_groups[camera].append(person_ids)
    return camera_groups


def find_files(folder_path, seg=3):
    """
    Find all files in the folder matching the given pattern.

    Args:
        folder_path (str): Path to the folder to search.
        seg: which segment to use. Default is 3.

    Returns:
        list: A list of file paths matching the pattern.

    """
    pattern = re.compile(r"[a-zA-Z]+(\d{2})\d*.*\.mp4$", re.IGNORECASE)
    matching_files = []

    for file in os.listdir(folder_path):
        match = pattern.match(file)
        if match:
            # Extract the first two digits immediately following the letter sequence
            digits = match.group(1)
            if len(digits) >= 2 and digits[1] == str(seg):  # Check if the second digit is '3'
                matching_files.append(os.path.join(folder_path, file))

    return matching_files


def augment_string(input_string):
    """
    Augment strings to pad single-digit numbers with a leading zero.

    Args:
        input_string (str): The input string to augment.

    Returns:
        str: The augmented string.
    """

    # Regular expression to find numbers in the string
    def pad_match(match):
        number = match.group()
        return number.zfill(2) if len(number) == 1 else number

    return re.sub(r'\d+', pad_match, input_string)

def generate_per_timestamp(timestamp: int, annotations: dict, video_dir: str):
    # for camera, person_ids in groups.items():
    #     print(f"Timestamp: {timestamp}  Camera: {camera} -> IDs: {person_ids}")
    for camera, groups in annotations.items():
        camera = augment_string(camera)
        video_folder = os.path.join(video_dir, camera)
        video_path = find_files(video_folder, seg=fformation_seg)[0]
        if not os.path.exists(video_path):
            print(f"Video {video_path} not found.")
            continue

        capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        # capture.set(cv2.CAP_PROP_VIDEO_STREAM, 0)
        fps = 60
        frame_number = int(timestamp * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = capture.read()

        if not ret:
            print(f"Could not extract frame from {video_path} at {timestamp}.")
            continue

        print(camera, groups)


def read_csv_and_process(csv_file, video_dir):
    """
    Read the CSV file and extract frames and bounding boxes.

    Args:
        csv_file (str): Path to the CSV file.
        video_dir (str): Directory containing videos.

    Returns:
        None
    """
    # Read CSV
    df = pd.read_csv(csv_file, header=None)
    # timestamp_selected = np.linspace()
    for _, row in df.iterrows():
        timestamp = row[0]
        annotation_str = row[1]  # Single annotation string column

        # Convert timestamp to seconds
        time_in_seconds = time_to_seconds(timestamp)

        # Parse annotations
        annotations = parse_annotations(annotation_str)

        generate_per_timestamp(time_in_seconds, annotations, cameras_path)



def main():
    # Example usage
    # read_csv_and_process(csv_path, cameras_path)
    coco_json = json.load(open('/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco/cam2_vid3_seg1_coco.json'))
    c = 9

if __name__ == '__main__':
    main()
