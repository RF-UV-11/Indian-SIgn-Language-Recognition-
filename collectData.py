import cv2
import os
import tkinter as tk
from tkinter import messagebox
from threading import Thread

class SignLanguageDataCollector:
    def __init__(self, root):
        """
        Initialize the sign language data collection GUI.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("Sign Language Data Collection")
        self.output_dir = "sign_language_videos"
        self.frame_width = 640
        self.frame_height = 480

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create buttons for each letter A-Z
        self.create_buttons()

    def create_buttons(self):
        """
        Create buttons for each letter A-Z, initiating video capture for each.
        """
        for i in range(26):
            char = chr(ord('A') + i)
            btn = tk.Button(self.root, text=char, width=5, height=2,
                            command=lambda char=char: Thread(target=self.start_capture, args=(char,)).start())
            btn.grid(row=i//6, column=i%6, padx=10, pady=10)

    def start_capture(self, sign):
        """
        Start video capture for the selected sign language letter.

        Args:
            sign (str): The sign language letter.
        """
        sign_dir = os.path.join(self.output_dir, sign)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

        video_count = len([name for name in os.listdir(sign_dir) if os.path.isfile(os.path.join(sign_dir, name))])
        if video_count >= 100:
            messagebox.showinfo("Limit Reached", f"Already have 100 videos for sign '{sign}'")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the camera.")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(sign_dir, f"{sign}_{video_count + 1}.avi")
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (self.frame_width, self.frame_height))

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Video Capture', frame)
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageDataCollector(root)
    root.mainloop()
