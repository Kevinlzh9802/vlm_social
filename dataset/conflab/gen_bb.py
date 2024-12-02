import os
import json
import cv2
import pandas as pd
from pathlib import Path
from conflab_utils import filter_kp_xy

def process_csv_files(csv_folder, json_folder, video_folder, output_folder):
    """
    Process CSV files to generate bounding boxes and draw them on video frames.

    Args:
        csv_folder (str): Path to the folder containing camera-specific CSV files.
        json_folder (str): Path to the folder containing JSON annotations.
        video_folder (str): Path to the folder containing videos.
        output_folder (str): Path to the folder to save frames with bounding boxes.
        filter_kp_xy (function): Function to convert keypoints to bounding boxes.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    metadata = []  # To store metadata for the JSON file
    image_id_counter = 0  # Counter for generating unique image IDs

    for root, _, files in os.walk(csv_folder):
        cam = root.split('/')[-1]  # get camera number
        for file in files:
            if not file.endswith('.csv'):
                continue

            csv_path = os.path.join(root, file)
            print(f"Processing {csv_path}...")

            # Extract camera and segment information
            cam_segment = Path(file).stem.split('-')
            seg = cam_segment[0]

            # Load corresponding JSON file
            json_path = os.path.join(json_folder, f"{cam}_vid3_{seg}_coco.json")
            if not os.path.exists(json_path):
                print(f"JSON file {json_path} not found. Skipping...")
                continue

            with open(json_path, 'r') as json_file:
                skeleton_data = json.load(json_file)

            # Load CSV data
            df = pd.read_csv(csv_path)

            # Load corresponding video file
            video_path = os.path.join(video_folder, f'{cam}', f"vid3-{seg}.mp4")
            if not os.path.exists(video_path):
                print(f"Video file {video_path} not found. Skipping...")
                continue

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = os.path.join(output_folder, f"{cam}_vid3_{seg}_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            for index, row in df.iterrows():
                timestamp = row['Timestamp']
                person_groups = eval(row['Groups'])  # Parse groups string into a list of tuples

                # Convert timestamp to skeleton annotation index
                timestamp_seconds = (int(timestamp.split(':')[0]) * 60 + int(timestamp.split(':')[1])) % (60 * 2)
                skeleton_timestamp_index = timestamp_seconds * 60  # Convert to 60Hz

                try:
                    skeletons = skeleton_data['annotations']['skeletons'][skeleton_timestamp_index]
                except:
                    print(f"No skeleton data for timestamp {timestamp}. Skipping...")
                    continue

                # Read the corresponding frame
                frame_number = int(timestamp_seconds * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    print(f"Could not read frame at timestamp {timestamp}. Skipping...")
                    continue

                # Prepare metadata for this frame
                image_id = f"{image_id_counter:06}"
                image_filename = f"{image_id}_{cam}_{seg}.jpg"
                image_filepath = os.path.join(output_folder, image_filename)
                image_metadata = {
                    "id": image_id,
                    "camera": cam,
                    "segment": seg,
                    "timestamp": timestamp,
                    "groups": [],
                    "bounding_boxes": []
                }

                # Draw bounding boxes for each person in the groups
                for group in person_groups:
                    for person_id in group:
                        if str(person_id) not in skeletons:
                            continue

                        skeleton = skeletons[str(person_id)]
                        keypoints = skeleton['keypoints']
                        _, bbox = filter_kp_xy(keypoints, frame_width, frame_height)

                        if bbox:
                            # Draw bounding box on the frame
                            x, y, w, h = bbox
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # cv2.putText(
                            #     frame,
                            #     f"ID: {person_id}",
                            #     (x, y - 10),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.5,
                            #     (0, 255, 0),
                            #     1
                            # )
                            # Append bounding box to metadata
                            image_metadata["bounding_boxes"].append({
                                "person_id": person_id,
                                "bbox": [x, y, w, h]
                            })

                    # Append the group to metadata
                    image_metadata["groups"].append(group)

                # Save the frame as an image
                cv2.imwrite(image_filepath, frame)
                print(f"Saved frame: {image_filepath}")

                # Add metadata to the list
                metadata.append(image_metadata)

                # Increment the image ID counter
                image_id_counter += 1

            cap.release()
            out.release()
            print(f"Output saved to {output_video_path}")

    metadata_json_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metadata saved to {metadata_json_path}")


def process_bounding_boxes(image_folder, metadata_json_path, output_folder):
    """
    Process images and bounding boxes to create cropped person images.

    Args:
        image_folder (str): Path to the folder containing the generated images.
        metadata_json_path (str): Path to the JSON file containing bounding box metadata.
        output_folder (str): Path to the folder to store cropped images.
    """
    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Load metadata from the JSON file
    with open(metadata_json_path, 'r') as json_file:
        metadata = json.load(json_file)

    for entry in metadata:
        image_id = entry['id']
        camera = entry['camera']
        segment = entry['segment']
        bounding_boxes = entry['bounding_boxes']

        # Construct the image filename
        image_filename = f"{image_id}_{camera}_{segment}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Skipping...")
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}. Skipping...")
            continue

        # Create a folder for the cropped bounding boxes for this image
        image_output_folder = os.path.join(output_folder, Path(image_filename).stem)
        Path(image_output_folder).mkdir(parents=True, exist_ok=True)

        # Crop and save each bounding box
        for bbox_data in bounding_boxes:
            person_id = bbox_data['person_id']
            x, y, w, h = bbox_data['bbox']

            # Crop the bounding box
            cropped_image = image[y:y + h, x:x + w]

            # Save the cropped image with the person ID as the filename
            cropped_image_filename = f"{person_id}.jpg"
            cropped_image_path = os.path.join(image_output_folder, cropped_image_filename)

            # Write the cropped image to disk
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Saved cropped image: {cropped_image_path}")


# Example usage



def main():
    # Example usage
    # csv_folder = "/home/zonghuan/tudelft/projects/datasets/modification/fformation_3_segments/"  # Folder containing CSV files
    # json_folder = "/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco/"  # Folder containing JSON annotations
    # video_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab_seg_custom"  # Folder containing videos
    # output_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab_bbox"  # Folder to save output videos
    #
    # process_csv_files(csv_folder, json_folder, video_folder, output_folder)

    image_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab_bbox"  # Folder containing generated images
    metadata_json_path = "/home/zonghuan/tudelft/projects/datasets/modification/conflab_bbox/metadata.json"  # Path to the metadata JSON file
    output_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab_gallery"  # Folder to store cropped bounding box images

    process_bounding_boxes(image_folder, metadata_json_path, output_folder)

if __name__ == '__main__':
    main()
