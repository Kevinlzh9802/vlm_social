import os
import json
from PIL import Image, ImageDraw
import math

# Define paths
DATASET_DIR = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/image/gallery_bbox"
GALLERY_DIR = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/image/gallery_bbox/gallery"
METADATA_FILE = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/image/gallery_bbox/metadata.json"
OUTPUT_DIR = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/image/position_concat"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def concat_images_with_padding(original_img, gallery_img, padding=20):
    """Concatenate the gallery image to the right of the original image with padding."""
    orig_width, orig_height = original_img.size
    gallery_width, gallery_height = gallery_img.size

    # Calculate new height and width with padding
    new_width = orig_width + gallery_width + 3 * padding
    new_height = max(orig_height, gallery_height) + 2 * padding

    # Create a new image with a white background
    new_image = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

    # Paste the original image with padding
    orig_y_offset = (new_height - orig_height) // 2
    new_image.paste(original_img, (padding, orig_y_offset))

    # Paste the gallery image to the right of the original image with padding
    gallery_y_offset = (new_height - gallery_height) // 2
    new_image.paste(gallery_img, (orig_width + 2 * padding, gallery_y_offset))

    return new_image

def categorize_position(bbox, image_width, image_height, threshold=0.1):
    """Categorize the bbox center into 'top-left', 'top-right', 'bottom-left', 'bottom-right'."""
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2

    half_width = image_width / 2
    half_height = image_height / 2

    # Calculate the deviation percentage from the center
    deviation_x = abs(center_x - half_width) / image_width
    deviation_y = abs(center_y - half_height) / image_height

    # Only accept bboxes that are sufficiently deviated from the center
    if deviation_x < threshold or deviation_y < threshold:
        return None  # The bbox is too close to the center, skip this one

    if center_x <= half_width and center_y <= half_height:
        return "top-left"
    elif center_x > half_width and center_y <= half_height:
        return "top-right"
    elif center_x <= half_width and center_y > half_height:
        return "bottom-left"
    else:
        return "bottom-right"


def find_bbox_by_person_id(data_list, person_id):
    # Filter list entries where 'person_id' matches
    matches = [item for item in data_list if item.get('person_id') == person_id]

    # Check for no or multiple matches
    if len(matches) == 0:
        print(f"Warning: No entry found with person_id = {person_id}")
        return None
    elif len(matches) > 1:
        print(f"Warning: Multiple entries found with person_id = {person_id}")
        return None

    # Return the 'bbox' field from the matched entry
    return matches[0].get('bbox')

def main():
    # Dictionary to store the positional info
    positional_info = {}
    # Load the metadata.json file
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Iterate through all images in the dataset
    for original_filename in os.listdir(DATASET_DIR):
        if original_filename.endswith('.jpg'):
            original_image_path = os.path.join(DATASET_DIR, original_filename)
            gallery_subfolder = os.path.join(GALLERY_DIR, os.path.splitext(original_filename)[0])

            # Skip if the corresponding gallery subfolder doesn't exist
            if not os.path.exists(gallery_subfolder):
                print(f"No gallery subfolder found for {original_filename}")
                continue

            # Load the original image
            original_image = Image.open(original_image_path)

            # Iterate through each gallery image in the corresponding subfolder
            for gallery_filename in os.listdir(gallery_subfolder):
                if gallery_filename.endswith('.jpg'):
                    gallery_image_path = os.path.join(gallery_subfolder, gallery_filename)
                    gallery_image = Image.open(gallery_image_path)

                    # Get bbox from metadata
                    original_image_name = os.path.splitext(original_filename)[0]
                    gallery_image_name = os.path.splitext(gallery_filename)[0]

                    try:
                        image_id = int(original_image_name.split('_')[0])
                        person_id = int(gallery_image_name)
                        bbox = find_bbox_by_person_id(metadata[image_id]['bounding_boxes'], person_id)
                        # bbox = metadata[image_id]['bounding_boxes'][gallery_image_name]
                        position_category = categorize_position(bbox, original_image.width, original_image.height)

                        # Skip if the bbox is too close to the center
                        if position_category is None:
                            print(f"Skipped {original_image_name} -> {gallery_image_name} due to ambiguity")
                            continue

                        # Concatenate the images with padding
                        concatenated_image = concat_images_with_padding(original_image, gallery_image)

                        # Save the concatenated image
                        output_image_name = f"{original_image_name}_{gallery_image_name}.jpg"
                        output_image_path = os.path.join(OUTPUT_DIR, output_image_name)
                        concatenated_image.save(output_image_path)

                        # Record the positional information
                        positional_info[output_image_name] = {
                            "original_image": original_image_path,
                            "gallery_image": gallery_image_path,
                            "bbox": bbox,
                            "position": position_category
                        }

                        print(f"Processed {output_image_name}: {position_category}")

                    except KeyError:
                        print(f"No bbox found for {original_image_name} -> {gallery_image_name}")

    # Save the positional information to a JSON file
    output_json_path = os.path.join(OUTPUT_DIR, "positional_info.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(positional_info, json_file, indent=4)

    print(f"Processing complete. Output saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
