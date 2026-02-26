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

def cut_video_into_clips(video_path, output_folder, clip_length=0.5, cumulative=True):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(clip_length * fps)

    # Calculate the number of clips
    num_clips = total_frames // clip_frames

    for i in range(num_clips):
        # Set the video position to the start of the current clip
        if cumulative:
            start_frame = 0
            num_frames = (i + 1) * clip_frames
        else:
            start_frame = i * clip_frames
            num_frames = clip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create a VideoWriter object to save the clip
        output_clip_path = os.path.join(output_folder, f"clip_{i + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_clip_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"Clip {i + 1} saved as {output_clip_path}")

    # Release the video capture object
    cap.release()

def cut_videos_in_folder(video_folder, output_folder, clip_length=0.5, cumulative=True):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all video files in the folder
    for video_file in os.listdir(video_folder):
        # Check if the file is a video by its extension
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, video_file)
            output_subfolder = os.path.join(output_folder, os.path.splitext(video_file)[0])
            cut_video_into_clips(video_path, output_subfolder, clip_length, cumulative)

def main():
    # Example usage
    video_folder = ('/home/zonghuan/tudelft/projects/datasets/MIntRec2.0/in-scope-20260223T102217Z-1-004'
                    '/in-scope/raw_data')
    output_folder = ('/home/zonghuan/tudelft/projects/datasets/MIntRec2.0/modification/'
                     'in-scope-20260223T102217Z-1-004/in-scope/raw_data')
    # save_last_frame(video_folder, output_folder)
    cut_videos_in_folder(video_folder, output_folder, clip_length=0.5, cumulative=True)

if __name__ == '__main__':
    main()
