import cv2 as cv
import os
from pathlib import Path
import numpy as np
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip

# dataset_path = "/home/zonghuan/tudelft/projects/datasets/conflab/data_processed/cameras/video_segments/cam2"
read_path = "/home/zonghuan/tudelft/projects/datasets/IdiapPoster/"
# video_name = "vid2-seg8-scaled-denoised.mp4"
write_path = "/home/zonghuan/tudelft/projects/datasets/modification/IdiapPoster/"
IMG_PER_SUBFOLDER = 3
SEGMENT_DURATION = 5

def split_video(video_path, img_path):
    os.chdir(img_path)
    cap = cv.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_name = 'f' + str(frame_num).zfill(6) + '.png'
        cv.imwrite(frame_name, frame)
        frame_num += 1

    cap.release()
    cv.destroyAllWindows()


def images_to_video_single(image_folder, output_video, fps=30):
    # Get list of image file names sorted by name (assuming names are sequential or alphabetical)
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Sort the images by name or order

    if len(images) == 0:
        print("No images found")
        return

    # Read the first image to determine frame size
    first_image = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 files
    video = cv.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through each image and write it to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = cv.imread(image_path)

        if img is not None:
            # Add image frame to video
            video.write(img)
        else:
            print(f"Warning: '{image_name}' could not be read and is skipped.")

    # Release the video writer object
    video.release()
    print(f"Video '{output_video}' has been created successfully.")

def images_to_video():
    dataset_path = Path(read_path)
    fps = 25  # Set frames per second
    for cam_part in sorted(dataset_path.iterdir()):
        if cam_part.is_dir():
            # CamX_partY
            for cam_subpart in sorted(cam_part.iterdir()):
                if cam_subpart.is_dir():
                    output_video = os.path.join(write_path, f'{cam_subpart.name}.mp4')
                    images_to_video_single(cam_subpart, output_video, fps)

def select_images(d_folder):
    dataset_path = Path(d_folder)
    part_count = 0
    for cam_part in sorted(dataset_path.iterdir()):
        if cam_part.is_dir():

            # CamX_partY
            for cam_subpart in sorted(cam_part.iterdir()):
                if cam_subpart.is_dir():
                    files = sorted([f for f in Path(cam_subpart).glob('*')])
                    if len(files) > 0:
                        selection = np.linspace(0, len(files), IMG_PER_SUBFOLDER + 2).round().astype(int).tolist()
                        selection = selection[1:-1]
                        for sel in selection:
                            file = files[sel]
                            write_name = f'{file.parts[-2]}_{file.stem}_{str(part_count).zfill(3)}.jpg'
                            shutil.copy(file, os.path.join(write_path, 'IdiapPoster_imgs', write_name))
                            part_count += 1

def clip_short(l_path, s_path):
    long_clips = sorted([f for f in Path(l_path).glob('*')])
    for long_clip in long_clips:
        video = VideoFileClip(str(long_clip))
        start_time = int(round(0.5 * video.duration - 0.5 * SEGMENT_DURATION))
        end_time = start_time + SEGMENT_DURATION
        video_segment = video.subclip(start_time, end_time)

        str_start = str(start_time).zfill(6)
        str_end = str(end_time).zfill(6)
        output_filename = os.path.join(s_path, f"{long_clip.stem}_segment_{str_start}_{str_end}.mp4")

        video_segment.write_videofile(output_filename, codec="libx264")

def main():
    # images_to_video()
    # select_images(read_path)
    path_long = os.path.join(read_path, 'IdiapPoster_clips')
    path_short = os.path.join(write_path, 'IdiapPoster_clips_short')
    clip_short(path_long, path_short)

if __name__ == '__main__':
    main()
