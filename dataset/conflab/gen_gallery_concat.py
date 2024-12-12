import os
from PIL import Image, ImageDraw, ImageFont


def find_jpg_images(folder_path):
    """
    Find all .jpg images in the specified folder.
    """
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]


def load_gallery_images(gallery_folder):
    """
    Load all images in the gallery folder and return them as a list of (image, id) tuples.
    """
    gallery_images = []
    for filename in sorted(os.listdir(gallery_folder)):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(gallery_folder, filename)
            img = Image.open(img_path)
            img_id = os.path.splitext(filename)[0]
            gallery_images.append((img, img_id))
    return gallery_images


def create_composite_image(original_image_path, gallery_images, output_path, spacing=35):
    """
    Create a composite image by concatenating the original image with the gallery images.
    """
    # Load original image
    original_image = Image.open(original_image_path)

    # Skip if the gallery is empty
    if not gallery_images:
        print(f"Skipping {original_image_path} because the gallery is empty.")
        return

    # Resize gallery images to a fixed width while maintaining aspect ratio
    gallery_width = 150  # Width for gallery images
    resized_gallery_images = []
    for img, img_id in gallery_images:
        img_resized = img.copy()
        img_resized.thumbnail((gallery_width, gallery_width))
        resized_gallery_images.append((img_resized, img_id))

    # Calculate layout for gallery images (arrange in rows and columns)
    num_columns = 4  # Number of columns in the gallery
    num_rows = (len(resized_gallery_images) + num_columns - 1) // num_columns

    # Determine the height of each row based on the max height of images in that row
    row_heights = []
    for i in range(num_rows):
        current_row_images = resized_gallery_images[i * num_columns: (i + 1) * num_columns]
        row_heights.append(max(img.size[1] for img, _ in current_row_images))

    # Calculate total gallery height
    total_gallery_height = sum(row_heights) + num_rows * spacing
    total_gallery_width = num_columns * gallery_width + (num_columns - 1) * spacing

    # Create a blank canvas for the gallery
    gallery_canvas = Image.new("RGB", (total_gallery_width, total_gallery_height), "white")

    # Load font for IDs (default font for simplicity)
    # try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 36)  # Increased font size
    # except IOError:
    #     font = ImageFont.load_default()

    # Paste gallery images onto the canvas, vertically centered
    x_offset, y_offset = 0, 0
    for i, (img, img_id) in enumerate(resized_gallery_images):
        row_index = i // num_columns
        y_centered = (row_heights[row_index] - img.height) // 2
        gallery_canvas.paste(img, (x_offset, y_offset + y_centered))

        # Draw the ID below the image in red color
        draw = ImageDraw.Draw(gallery_canvas)
        text_position = (x_offset + img.width // 2, y_offset + y_centered + img.height + 10)
        draw.text(text_position, img_id, font=font, fill="red", anchor="mm")

        # Update position for next image
        x_offset += gallery_width + spacing
        if (i + 1) % num_columns == 0:
            x_offset = 0
            y_offset += row_heights[row_index] + spacing

    # Create a new blank image to concatenate original and gallery images
    combined_width = original_image.width + spacing + gallery_canvas.width
    combined_height = max(original_image.height, gallery_canvas.height)
    combined_image = Image.new("RGB", (combined_width, combined_height), "white")

    # Paste the original image and gallery canvas onto the combined image
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(gallery_canvas, (original_image.width + spacing, 0))

    # Save the combined image
    combined_image.save(output_path)
    print(f"Saved composite image to {output_path}")


def process_images(original_folder, gallery_base_folder, output_folder):
    """
    Process all images in the original folder and create composite images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    original_images = find_jpg_images(original_folder)

    for image_name in original_images:
        original_image_path = os.path.join(original_folder, image_name)
        gallery_folder = os.path.join(gallery_base_folder, os.path.splitext(image_name)[0])

        if os.path.exists(gallery_folder):
            gallery_images = load_gallery_images(gallery_folder)
            if gallery_images:
                output_path = os.path.join(output_folder, image_name)
                create_composite_image(original_image_path, gallery_images, output_path)
            else:
                print(f"Skipping {image_name} because the gallery folder is empty.")
        else:
            print(f"Gallery folder not found for {image_name}")

def main():
    original_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/gallery_bbox"
    gallery_base_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/gallery_bbox/gallery"
    output_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/gallery_concat"
    process_images(original_folder, gallery_base_folder, output_folder)

# Example usage
if __name__ == "__main__":
    main()
    # font_dir = "/usr/share/fonts/truetype"  # Update based on your OS
    # for root, _, files in os.walk(font_dir):
    #     for file in files:
    #         if file.endswith(".ttf"):
    #             print(os.path.join(root, file))
