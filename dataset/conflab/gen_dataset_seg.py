import csv
import os
import math
import re
from pathlib import Path

# Define constants
SEGMENT_DURATION = 120  # Duration of each segment in seconds
OUTPUT_FOLDER = "/home/zonghuan/tudelft/projects/datasets/modification/fformation_3_segments/"  # Directory for saving group information files

# Function to parse group information string
def parse_group_info(group_info_str):
    """
    Parse the group information string to extract groupings by camera.

    Args:
        group_info_str (str): Group information in the format '(<2,3,4>, cam2)(<5,6,7>, cam4)'.

    Returns:
        dict: A dictionary where keys are cameras and values are lists of group IDs.
    """
    matches = re.findall(r'\(<(.+?)>,\s*(cam\d+)\)', group_info_str)
    camera_groups = {}
    for ids, camera in matches:
        id_group = [int(id_) for id_ in ids.split(",") if id_ != '']
        group_ids = [tuple(map(int, id_group))]  # Convert group IDs to a tuple
        if camera not in camera_groups:
            camera_groups[camera] = []
        camera_groups[camera].extend(group_ids)
    return camera_groups

# Function to process the input CSV and generate group files
def process_csv(input_csv_path, segment_duration, output_folder):
    """
    Process the input CSV to generate group information files for each camera and segment.

    Args:
        input_csv_path (str): Path to the input CSV file.
        segment_duration (int): Duration of each segment in seconds.
        output_folder (str): Directory where the output files will be saved.

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    with open(input_csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Group data by segments and cameras
    segment_data = {}
    for row in rows:
        timestamp = row[0]
        group_info_str = row[1]

        # Determine which segment the timestamp belongs to
        time_in_seconds = int(timestamp.split(":")[0]) * 60 + int(timestamp.split(":")[1])
        segment_index = math.ceil(time_in_seconds / segment_duration)

        # Parse the group information string
        camera_groups = parse_group_info(group_info_str)

        # Add group data to the respective segment and camera
        for camera, groups in camera_groups.items():
            segment_key = f"{camera}/seg{segment_index}"
            if segment_key not in segment_data:
                segment_data[segment_key] = []
            segment_data[segment_key].append((timestamp, groups))

    # Write segment-specific CSV files
    for segment_key, data in segment_data.items():
        camera, segment = segment_key.split("/")
        segment_folder = os.path.join(output_folder, camera)
        Path(segment_folder).mkdir(parents=True, exist_ok=True)

        output_file_path = os.path.join(segment_folder, f"{segment}-groups.csv")
        with open(output_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Groups"])
            for timestamp, groups in data:
                writer.writerow([timestamp, str(groups)])

    print(f"Group information files generated in {output_folder}")


def main():
    # Example usage
    input_csv_path = "/home/zonghuan/tudelft/projects/datasets/conflab/annotations/f_formations/seg3.csv"  # Path to your input CSV file
    process_csv(input_csv_path, SEGMENT_DURATION, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
