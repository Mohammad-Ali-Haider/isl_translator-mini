# Real-Time Indian Sign Language (ISL) Gesture Recognition

This project is a real-time gesture recognition system for Indian Sign Language (ISL) that utilizes computer vision and deep learning. It leverages the MediaPipe library for accurate hand landmark detection and a PyTorch-based neural network for classification.

The system is designed to recognize both **single-hand** and **dual-hand** gestures, making it robust and versatile for a wider range of signs.

## Features

-   **Real-Time Detection**: Classifies ISL gestures instantly using a webcam feed.
-   **Dual-Hand Support**: Capable of recognizing gestures that require one or two hands.
-   **Robust Normalization**: Landmark normalization is performed on a per-hand basis, making the detection invariant to the hand's size and distance from the camera.
-   **Modular Workflow**: The project is structured with separate scripts for dataset creation, data preprocessing, model training, and real-time inference.
-   **Lightweight Model**: Uses a simple and efficient feed-forward neural network for fast predictions.

## File Structure

The project is organized into several key Python scripts:

-   `create_dataset_dual_hand.py`: Processes a directory of gesture images (e.g., from the Indian Sign Language dataset) and uses MediaPipe to extract and save raw hand landmark data to `hand_landmarks.csv`.
-   `convert_csv_dual_hand_support.py`: Reads the raw landmark data, applies per-hand normalization to the coordinates, and separates the data into single-hand, dual-hand, and a combined, padded dataset (`hand_landmarks_xy_normalized_all.csv`).
-   `main_dual_hand.py`: The main training script. It loads the normalized data, trains the gesture classification model, and saves the trained model (`hand_gesture_model_dual.pth`) and class mappings (`label_mapping_dual.npy`).
-   `detect_dual_hand.py`: The script for real-time inference. It captures video from the webcam, processes hand landmarks, and uses the trained model to predict the gesture being signed.
-   `Model.py`: Defines the architecture of the PyTorch neural network (`LinearModel`).
-   `helper.py`: Contains various utility functions used across the project, primarily for data loading and augmentation.

## Quick Start: Running the Detector

Follow these steps to get the real-time gesture detector running with the pre-trained model.

### 1. Prerequisites

-   Python 3.11+
-   A webcam

### 2. Installation

First, install `uv`, a fast Python package installer and resolver.

```bash
pip install uv
```

Next, clone the repository and navigate into the project directory.
```bash
git clone https://github.com/your-username/MAI-Project.git
cd MAI-Project
```

Finally, create a virtual environment and install the required dependencies using uv. Make sure you have a requirements.txt file with the necessary packages.
```bash
uv sync
```

Then to run the model/detect sign language
```bash
uv run detect_dual_hand.py
```