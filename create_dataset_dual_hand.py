"""
Process the Indian dataset to extract hand landmarks with dual-hand support.
Saves landmarks sequentially: first 21 for hand1, next 21 for hand2 (if present).
Output: hand_landmarks.csv with format [pose, x0, y0, z0, ..., x20, y20, z20, x21, y21, z21, ..., x41, y41, z41]
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


def process_dataset(dataset_path, output_csv, num_images_per_class=None):
    """
    Process images from dataset and extract hand landmarks.
    
    Args:
        dataset_path: Path to Indian dataset folder
        output_csv: Output CSV file path
        num_images_per_class: Maximum images to process per class (None = all)
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,  # Support up to 2 hands
        min_detection_confidence=0.5
    )
    
    all_samples = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Get all class folders (1, 2, 3, ..., 9, A, B, C, ...)
    class_folders = sorted([f for f in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, f))])
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    print(f"\nProcessing dataset from: {dataset_path}")
    print(f"Max images per class: {num_images_per_class if num_images_per_class else 'ALL'}")
    print("="*60)
    
    total_single_hand = 0
    total_dual_hand = 0
    total_skipped = 0
    
    for class_label in class_folders:
        class_path = os.path.join(dataset_path, class_label)
        image_files = [f for f in os.listdir(class_path) 
                      if os.path.splitext(f)[1].lower() in valid_image_extensions]
        
        if num_images_per_class:
            image_files = image_files[:num_images_per_class]
        
        single_hand_count = 0
        dual_hand_count = 0
        skipped_count = 0
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                skipped_count += 1
                continue
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                
                # Create row: [pose, x0, y0, z0, ..., x20, y20, z20] or [..., x41, y41, z41]
                row = [class_label]
                
                # Process each detected hand (up to 2)
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])
                
                # Pad with zeros if only 1 hand detected (to keep consistent structure)
                if num_hands == 1:
                    # Add 63 zeros for missing second hand (21 landmarks × 3)
                    row.extend([0.0] * 63)
                    single_hand_count += 1
                elif num_hands == 2:
                    dual_hand_count += 1
                
                all_samples.append(row)
            else:
                # No hand detected - skip this image
                skipped_count += 1
        
        total_single_hand += single_hand_count
        total_dual_hand += dual_hand_count
        total_skipped += skipped_count
        
        print(f"Class '{class_label}': {single_hand_count} single-hand, {dual_hand_count} dual-hand, {skipped_count} skipped")
    
    hands.close()
    
    print("\n" + "="*60)
    print(f"Total samples collected: {len(all_samples)}")
    print(f"  - Single-hand: {total_single_hand}")
    print(f"  - Dual-hand: {total_dual_hand}")
    print(f"  - Skipped (no hands): {total_skipped}")
    
    # Create DataFrame
    # 42 landmarks × 3 coordinates = 126 features + 1 pose column = 127 columns
    column_names = ['pose']
    for i in range(42):  # 42 landmarks total (21 per hand × 2 hands max)
        column_names.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    df = pd.DataFrame(all_samples, columns=column_names)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to: {output_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.shape[1]} (1 pose + 126 features)")
    
    return df


def main():
    DATASET_PATH = "../Indian"
    OUTPUT_CSV = "./hand_landmarks.csv"
    NUM_IMAGES_PER_CLASS = None  # Set to None to process all images, or e.g., 200 for testing
    
    print("Indian Sign Language Dataset Processor")
    print("Dual-Hand Support Enabled\n")
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path not found: {DATASET_PATH}")
        return
    
    df = process_dataset(DATASET_PATH, OUTPUT_CSV, NUM_IMAGES_PER_CLASS)
    
    print("\nDataset processing complete!")
    print("\nNext steps:")
    print("1. Run: python convert_csv_dual_hand_support.py")
    print("2. Run: python main_dual_hand.py")
    print("3. Run: python detect_dual_hand.py")


if __name__ == "__main__":
    main()
