import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

class HandSignModel:
    def __init__(self, data_dir='keypoints_data', sequence_length=30):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.num_classes = len(os.listdir(data_dir))  # Number of classes
        self.model = None
        self.history = None

    def load_data(self):
        """
        Load the dataset from the specified directory.
        Each sign class is stored in a separate subdirectory.
        """
        X = []
        y = []

        for sign in os.listdir(self.data_dir):
            sign_path = os.path.join(self.data_dir, sign)
            if os.path.isdir(sign_path):
                class_label = ord(sign) - ord('A')  # Convert 'A'-'Z' to 0-25
                for video_dir in os.listdir(sign_path):
                    video_path = os.path.join(sign_path, video_dir)
                    if os.path.isdir(video_path):
                        frames = []
                        for i in range(self.sequence_length):
                            frame_path = os.path.join(video_path, f"{i}.npy")
                            if os.path.exists(frame_path):
                                frame_data = np.load(frame_path)
                                frames.append(frame_data)
                        if len(frames) == self.sequence_length:
                            X.append(frames)
                            y.append(class_label)

        X = np.array(X)
        y = np.array(y)

        # Normalize the data
        X = X / np.amax(X)

        # One-hot encode labels
        y = to_categorical(y, num_classes=self.num_classes)

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.5):
        """
        Split the data into training, validation, and test sets.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        """
        Build the LSTM model for sign classification.
        """
        self.model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, 126), return_sequences=True),
            Dropout(0.5),
            LSTM(64, return_sequences=False),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=32):
        """
        Train the model with the provided training and validation data.
        """
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def save_model(self, model_path):
        """
        Save the trained model to a file.
        """
        self.model.save(model_path)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def run(self):
        """
        Main function to run the data loading, training, and evaluation process.
        """
        X, y = self.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        self.build_model()
        self.train(X_train, y_train, X_val, y_val)
        self.evaluate(X_test, y_test)
        self.save_model('model/model.h5')



def main():
    hand_sign_model = HandSignModel()
    hand_sign_model.run()


if __name__ == "__main__":
    main()