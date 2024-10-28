import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pandas as pd
from threading import Thread
import time


class AnnotationFramework:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Framework")

        self.video_frame = tk.Frame(root)
        self.video_frame.grid(row=0, column=0)

        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=0, column=1, padx=10)

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()

        self.playback_controls = tk.Frame(self.control_frame)
        self.playback_controls.pack()

        # Buttons 1 to 5
        self.buttons = []
        for i in range(1, 6):
            btn = tk.Button(self.playback_controls, text=str(i), width=5, height=2,
                            command=lambda k=i: self.on_button_press(k))
            btn.pack(side=tk.TOP, padx=5, pady=5)
            self.buttons.append(btn)

        # Finish button
        self.finish_button = tk.Button(self.playback_controls, text="Finish", width=10, height=2, command=self.finish)
        self.finish_button.pack(side=tk.BOTTOM, pady=10)

        self.cap = None
        self.video_path = None
        self.frame_count = 0
        self.fps = 30
        self.annotations = []
        self.current_frame = 0
        self.playing = False
        self.key_presses = [0] * 5  # Array for keys 1-5

        self.load_video()
        self.play_thread = Thread(target=self.play_video)
        self.play_thread.start()

    def load_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.annotations = np.zeros((self.frame_count, 5), dtype=int)

    def on_button_press(self, button_index):
        if self.playing:
            self.key_presses[button_index - 1] = 1

    def play_video(self):
        while True:
            if self.cap is None or not self.playing:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = cv2.resize(frame_rgb, (640, 480))
                photo = tk.PhotoImage(image=tk.Image.fromarray(frame_image))
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.root.update_idletasks()
                self.root.update()
                # Store the key presses for the current frame
                self.annotations[self.current_frame - 1] = self.key_presses
                # Reset key presses
                self.key_presses = [0] * 5
                time.sleep(1 / self.fps)
            else:
                self.playing = False
                break

    def finish(self):
        self.playing = False
        if self.cap is not None:
            self.cap.release()
        # Generate CSV
        df = pd.DataFrame(self.annotations, columns=[f"Button_{i + 1}" for i in range(5)])
        df['Frame'] = range(self.frame_count)
        df.to_csv("annotations.csv", index=False)
        print("Annotations saved to annotations.csv")
        self.root.quit()

    def rewind(self):
        self.current_frame = max(0, self.current_frame - self.fps * 5)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def fast_forward(self):
        self.current_frame = min(self.frame_count - 1, self.current_frame + self.fps * 5)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def pause(self):
        self.playing = False

    def play(self):
        if not self.playing:
            self.playing = True
            self.play_video()

    def setup_controls(self):
        # Playback controls
        rewind_button = tk.Button(self.playback_controls, text="Rewind", width=10, height=2, command=self.rewind)
        rewind_button.pack(side=tk.LEFT, padx=5, pady=5)

        pause_button = tk.Button(self.playback_controls, text="Pause", width=10, height=2, command=self.pause)
        pause_button.pack(side=tk.LEFT, padx=5, pady=5)

        play_button = tk.Button(self.playback_controls, text="Play", width=10, height=2, command=self.play)
        play_button.pack(side=tk.LEFT, padx=5, pady=5)

        fast_forward_button = tk.Button(self.playback_controls, text="Fast Forward", width=10, height=2,
                                        command=self.fast_forward)
        fast_forward_button.pack(side=tk.LEFT, padx=5, pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationFramework(root)
    app.setup_controls()
    root.mainloop()
