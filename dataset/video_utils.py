import cv2
import os

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

def main():
    # Example usage
    video_folder = ('/home/zonghuan/tudelft/projects/datasets/modification/CocktailParty/CocktailParty_clips'
                    '/CocktailParty_clips_short/')
    output_folder = ('/home/zonghuan/tudelft/projects/datasets/modification/CocktailParty/CocktailParty_clips'
                     '/CocktailParty_clips_short_last/')
    save_last_frame(video_folder, output_folder)

if __name__ == '__main__':
    main()
