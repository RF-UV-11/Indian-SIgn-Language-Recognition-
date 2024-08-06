import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


class HandSignRecognizer:
    def __init__(self, model_path, sequence_length=30, confidence_threshold=0.70):
        """
        Initialize the HandSignRecognizer with the model, sequence length, and confidence threshold.

        Args:
            model_path (str): Path to the trained model file.
            sequence_length (int): Number of frames used for prediction.
            confidence_threshold (float): Minimum probability to consider a prediction valid.
        """
        self.model = load_model(model_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.sequence = []

    def extract_hand_keypoints(self, results):
        """
        Extracts hand keypoints (x, y, z coordinates) from MediaPipe Hand landmarks.

        Args:
            results (mediapipe.Hands): Results containing hand landmarks.

        Returns:
            np.array: Flattened array of hand keypoints for detected hands.
        """
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
                keypoints.append(hand_keypoints)

        max_hands = 2
        num_hand_keypoints = 21 * 3  # 21 landmarks * 3 coordinates (x, y, z)
        hand_keypoints_array = np.zeros((max_hands, num_hand_keypoints))
        for i in range(min(len(keypoints), max_hands)):
            hand_keypoints_array[i, :len(keypoints[i])] = keypoints[i]

        return hand_keypoints_array.flatten()

    def predict_sign(self):
        """
        Predict the sign based on the current sequence of hand keypoints.

        Returns:
            str: Predicted class name if the confidence is above the threshold, otherwise an empty string.
        """
        if len(self.sequence) == self.sequence_length:
            input_data = np.expand_dims(self.sequence, axis=0)
            predictions = self.model.predict(input_data)
            predicted_class = np.argmax(predictions, axis=-1)[0]
            max_probability = np.max(predictions)

            if max_probability > self.confidence_threshold:
                return self.class_names[predicted_class]
        return ""

    def process_frame(self, frame):
        """
        Process a single frame for hand sign recognition.

        Args:
            frame (np.array): The input video frame.

        Returns:
            np.array: The frame with the hand landmarks and prediction displayed.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_keypoints = self.extract_hand_keypoints(results)
        self.sequence.append(hand_keypoints)

        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)

        predicted_class_name = self.predict_sign()
        if predicted_class_name:
            cv2.putText(frame, f'Class: {predicted_class_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def run(self):
        """
        Run the real-time hand sign recognition using the webcam.
        """
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Real-time Hand sign Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = HandSignRecognizer(model_path='model/model.h5')
    recognizer.run()
