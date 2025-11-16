import os
import cv2
import mediapipe as mp
import pandas as pd
import torch
import numpy as np

from Model import LinearModel


# -------------------------------------------------------------------
# Directory printing
# -------------------------------------------------------------------
def print_directories(path):
    """Print subdirectories in the dataset path."""
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            print(f"Directory: {entry}")


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def data_loading(dataset_path, num_images_per_directory=None):
    """
    Load images from dataset path.
    Expects subfolders named by pose labels (e.g., "1", "2", ..., "9").
    """
    image_data = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png']

    for entry in os.listdir(dataset_path):
        entry_path = os.path.join(dataset_path, entry)
        if os.path.isdir(entry_path):
            counter = 0
            pose = entry  # folder name becomes the pose label
            current_directory_images = []

            for file_name in os.listdir(entry_path):
                file_path = os.path.join(entry_path, file_name)

                # Stop if we reached the max per directory
                if num_images_per_directory is not None and counter >= num_images_per_directory:
                    break

                if os.path.isfile(file_path):
                    _, file_extension = os.path.splitext(file_name)
                    if file_extension.lower() in valid_image_extensions:
                        image = cv2.imread(file_path)
                        if image is not None:
                            current_directory_images.append((image, pose))
                            counter += 1

            image_data.extend(current_directory_images)

    print(f"Loaded {len(image_data)} images from dataset.")
    return image_data


# -------------------------------------------------------------------
# Augment helpers
# -------------------------------------------------------------------
def rotate_image(image, angle):
    """Rotate image by specified angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def adjust_brightness(image, factor):
    """Adjust brightness of image by a given factor."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# -------------------------------------------------------------------
# Feature extraction using MediaPipe Hands
# -------------------------------------------------------------------
def feature_extraction(image_data, augment=False):
    """
    Extract hand landmarks using MediaPipe.
    Returns a list of dicts:
      {
        'pose': <label>,
        'hands': [ [ {x,y,z}, ...21 landmarks ], ... ]
      }
    """
    image_hand_landmarks_data = []

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,  # Support up to 2 hands for dual-hand gestures
        min_detection_confidence=0.5
    )

    for image, pose in image_data:
        images_to_process = [image]

        if augment:
            # Rotations
            images_to_process.append(rotate_image(image, 10))
            images_to_process.append(rotate_image(image, -10))
            images_to_process.append(rotate_image(image, 20))
            images_to_process.append(rotate_image(image, -20))

            # Brightness changes
            images_to_process.append(adjust_brightness(image, 1.3))
            images_to_process.append(adjust_brightness(image, 0.7))

            # Horizontal flip
            images_to_process.append(cv2.flip(image, 1))

        for img in images_to_process:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                image_hands = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    image_hands.append(landmarks)

                image_hand_landmarks_data.append({
                    'pose': pose,
                    'hands': image_hands
                })

    hands.close()
    print(f"Extracted landmarks from {len(image_hand_landmarks_data)} images.")
    return image_hand_landmarks_data


# -------------------------------------------------------------------
# CSV saver for landmarks (ONLY x,y and digits 1-9)
# -------------------------------------------------------------------
def save_landmarks_to_csv_xy_only(landmarks_data, csv_path):
    """
    Save landmarks to CSV with per-sample normalization:
      - Only poses '1'..'9'
      - Only x,y coordinates
      - Each sample normalized relative to its own min/max (distance-invariant)
      Each row: pose, x_0, y_0, x_1, y_1, ...
    """
    valid_poses = {str(i) for i in range(1, 10)}  # "1".."9"
    df = pd.DataFrame()

    for item in landmarks_data:
        pose = item['pose']

        # Skip if pose not in 1..9
        if pose not in valid_poses:
            continue

        for hand in item['hands']:
            # Extract all x and y coordinates
            x_coords = [landmark['x'] for landmark in hand]
            y_coords = [landmark['y'] for landmark in hand]
            
            # Find min and max for THIS hand sample
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Compute ranges (avoid division by zero)
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0
            
            # Normalize each landmark relative to this sample's bounds
            row = {'pose': pose}
            for idx, landmark in enumerate(hand):
                normalized_x = (landmark['x'] - x_min) / x_range
                normalized_y = (landmark['y'] - y_min) / y_range
                row[f'x_{idx}'] = normalized_x
                row[f'y_{idx}'] = normalized_y
            
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples to {csv_path} (per-sample normalized)")
    return df


# -------------------------------------------------------------------
# Model save/load
# -------------------------------------------------------------------
def save_model(model, path="hand_gesture_model.pth"):
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")


def load_model(path="hand_gesture_model.pth", input_size=42, num_classes=9):
    """
    Load a trained model.
    Args:
        path: Path to the model file
        input_size: Number of input features (42 for x,y only, 63 for x,y,z)
        num_classes: Number of output classes (9 for digits 1-9)
    """
    print(f"Loading model from {path}...")
    loaded_model = LinearModel(input_size=input_size, num_classes=num_classes)
    loaded_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    loaded_model.eval()
    print("Model loaded successfully.")
    return loaded_model
