# Indian Sign Language Recognition Project

This project involves recognizing Indian Sign Language (ISL) gestures using a deep learning model. The project includes various scripts for collecting data, processing it, training a model, and testing the model.

## Project Setup

### Virtual Environment

1. **Create and Activate Virtual Environment:**

   ```bash
   python -m venv .\isl
   .\isl\Scripts\activate
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Collecting Data

**Script:** `collectData.py`

**Description:** Collect video data of Indian Sign Language gestures using a webcam. Videos will be saved in the `sign_language_videos` directory.

**Usage:**

   ```bash
   python collectData.py
   ```

### 2. Creating DataFrame

**Script:** `makeDataFrame.py`

**Description:** Create a DataFrame from the collected video files. The DataFrame will be saved as `data.csv`.

**Usage:**

   ```bash
   python makeDataFrame.py
   ```

### 3. Converting Videos to NPY and Extracting Keypoints

**Script:** `convert2NPY.py`

**Description:** Convert videos into `.npy` files and extract keypoints. Keypoints will be saved in the `keypoints_data` directory.

**Usage:**

   ```bash
   python convert2NPY.py
   ```

### 4. Training the Model

**Script:** `handSignModel.py`

**Description:** Train the model using the keypoints data. The trained model will be saved in the `model` directory.

**Usage:**

   ```bash
   python handSignModel.py
   ```

### 5. Testing the Model

**Script:** `handSignRecognizer.py`

**Description:** Test the trained model in real-time using a webcam. This script will display predictions for gestures in real-time.

**Usage:**

   ```bash
   python handSignRecognizer.py
   ```

## Directory Structure

```
.
├── collectData.py
├── convert2NPY.py
├── handSignModel.py
├── handSignRecognizer.py
├── makeDataFrame.py
├── model/
│   └── model.h5
├── requirements.txt
├── sign_language_videos/
│   └── (collected video files)
└── keypoints_data/
    └── (extracted keypoints data)
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```