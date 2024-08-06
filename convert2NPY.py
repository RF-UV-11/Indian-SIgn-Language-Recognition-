import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

class KeypointsExtractor:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, target_frame_count=30):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=max_num_hands, 
                                         min_detection_confidence=min_detection_confidence)
        self.target_frame_count = target_frame_count
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_hand_keypoints(self, results):
        """
        Extracts hand keypoints (x, y, z coordinates) from MediaPipe Hand landmarks.

        Args:
            results (mediapipe.Hands): Results containing hand landmarks.

        Returns:
            np.array: Array of hand keypoints for detected hands.
                      Each row represents keypoints for a single hand (21 landmarks x 3 coordinates).
        """
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                keypoints.append(hand_keypoints)
        
        # Ensuring the array has a fixed shape even if no hands are detected
        max_hands = 2
        num_hand_keypoints = 21 * 3  # 21 landmarks * 3 coordinates (x, y, z)
        hand_keypoints_array = np.zeros((max_hands, num_hand_keypoints))
        for i in range(min(len(keypoints), max_hands)):
            hand_keypoints_array[i, :len(keypoints[i])] = keypoints[i]

        return hand_keypoints_array.flatten()

    def process_video(self, video_path, save_dir='keypoints_data'):
        """
        Processes a video, extracts hand keypoints using MediaPipe Hands, and saves keypoints as .npy files.

        Args:
            video_path (str): Path to the input video file.
            save_dir (str): Directory to save extracted keypoints.

        Returns:
            None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip_count = max(frame_count // self.target_frame_count, 1)

        # Create directory to save keypoints
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sign = video_name[0]
        save_path = os.path.join(save_dir, sign, video_name)
        os.makedirs(save_path, exist_ok=True)

        frame_number = 0
        while cap.isOpened() and frame_number < self.target_frame_count:
            # Set frame position to skip frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number * frame_skip_count)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB and process with MediaPipe Hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Extract hand keypoints and save as .npy file
            hand_keypoints = self.extract_hand_keypoints(results)
            npy_filepath = os.path.join(save_path, f'{frame_number}.npy')
            np.save(npy_filepath, hand_keypoints)

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {video_path}")

    def Extract(self,csv_file_path="data.csv", save_dir='keypoints_data'):
        """
        Extract function to process videos listed in a DataFrame and extract hand keypoints.

        Args:
            csv_file_path (str): Path to the CSV file containing video paths.
            save_dir (str): Directory to save extracted keypoints.

        Returns:
            None
        """
        extractor = self.KeypointsExtractor()

        # Load DataFrame containing video paths
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file_path}")
            return

        # Process each video in the DataFrame
        for _, row in df.iterrows():
            video_path = row['Video_Path']
            extractor.process_video(video_path, save_dir)

        print("Hand keypoints extraction completed.")


def main():
    KeypointsExtractor = KeypointsExtractor()
    KeypointsExtractor.Extract()


if __name__ == "__main__":
    main()