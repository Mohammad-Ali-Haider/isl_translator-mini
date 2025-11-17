"""
Script to convert hand_landmarks.csv to per-sample normalized x,y coordinates.
Supports both single-hand and dual-hand gestures.
- Single hand: 42 features (21 landmarks × 2)
- Dual hand: 84 features (42 landmarks × 2)
"""

import pandas as pd
import numpy as np


def normalize_landmarks_per_hand(hand_coords):
    """
    Normalize a single hand's landmarks relative to its bounding box.
    
    Args:
        hand_coords: numpy array of shape (21, 2) with x,y coordinates
    
    Returns:
        normalized_coords: numpy array of shape (42,) with normalized x,y alternating
    """
    x_coords = hand_coords[:, 0]
    y_coords = hand_coords[:, 1]
    
    # Find min/max for this hand
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Avoid division by zero
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    # Normalize
    x_norm = (x_coords - x_min) / x_range
    y_norm = (y_coords - y_min) / y_range
    
    # Interleave x and y: [x0, y0, x1, y1, ...]
    normalized = np.empty(42, dtype=np.float32)
    normalized[0::2] = x_norm
    normalized[1::2] = y_norm
    
    return normalized


def process_sample(X_row):
    """
    Process a single sample - could be single or dual hand.
    Detects hand count by checking if landmarks are filled with non-zero values.
    
    Args:
        X_row: numpy array with features (x,y,z pattern repeating)
    
    Returns:
        normalized features (42 for single hand, 84 for dual hand)
        num_hands (1 or 2)
    """
    # Extract x,y coordinates only (ignore z)
    x_coords = X_row[0::3]
    y_coords = X_row[1::3]
    
    n_landmarks = len(x_coords)
    
    # Check if first 21 landmarks (hand 1) are filled
    hand1_filled = False
    if n_landmarks >= 21:
        # Check if first 21 points have non-zero values
        hand1_x = x_coords[:21]
        hand1_y = y_coords[:21]
        # Consider hand filled if most points are non-zero
        non_zero_count = np.sum((hand1_x != 0) | (hand1_y != 0))
        hand1_filled = non_zero_count >= 15  # At least 15 out of 21 landmarks
    
    # Check if next 21 landmarks (hand 2) are filled
    hand2_filled = False
    if n_landmarks >= 42:
        # Check if landmarks 21-41 have non-zero values
        hand2_x = x_coords[21:42]
        hand2_y = y_coords[21:42]
        non_zero_count = np.sum((hand2_x != 0) | (hand2_y != 0))
        hand2_filled = non_zero_count >= 15  # At least 15 out of 21 landmarks
    
    # Determine number of hands
    if hand1_filled and hand2_filled:
        # Dual hand (21 landmarks per hand)
        # First hand: landmarks 0-20
        hand1_coords = np.column_stack([x_coords[:21], y_coords[:21]])
        hand1_norm = normalize_landmarks_per_hand(hand1_coords)
        
        # Second hand: landmarks 21-41
        hand2_coords = np.column_stack([x_coords[21:42], y_coords[21:42]])
        hand2_norm = normalize_landmarks_per_hand(hand2_coords)
        
        # Concatenate both hands
        normalized = np.concatenate([hand1_norm, hand2_norm])
        return normalized, 2
        
    elif hand1_filled:
        # Single hand (first 21 landmarks only)
        hand1_coords = np.column_stack([x_coords[:21], y_coords[:21]])
        normalized = normalize_landmarks_per_hand(hand1_coords)
        return normalized, 1
    
    else:
        raise ValueError(f"No valid hand landmarks found in sample")


def main():
    INPUT_CSV = "./hand_landmarks.csv"
    OUTPUT_CSV_SINGLE = "./hand_landmarks_xy_normalized_single.csv"
    OUTPUT_CSV_DUAL = "./hand_landmarks_xy_normalized_dual.csv"
    OUTPUT_CSV_COMBINED = "./hand_landmarks_xy_normalized_all.csv"
    
    print(f"Loading data from {INPUT_CSV}...")
    data = pd.read_csv(INPUT_CSV, low_memory=False)
    
    print(f"Original data shape: {data.shape}")
    
    # Get poses and features
    if 'pose' in data.columns:
        poses = data['pose'].values
        X = data.drop(columns=['pose']).values
    else:
        poses = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
    
    print(f"\nNumber of samples: {len(poses)}")
    print(f"Original features per sample: {X.shape[1]}")
    
    # Process all samples
    print("\nProcessing samples...")
    single_hand_data = []
    dual_hand_data = []
    
    for i, (pose, x_row) in enumerate(zip(poses, X)):
        try:
            normalized, num_hands = process_sample(x_row)
            
            if num_hands == 1:
                single_hand_data.append([pose] + normalized.tolist())
            elif num_hands == 2:
                dual_hand_data.append([pose] + normalized.tolist())
                
        except Exception as e:
            print(f"  Warning: Skipping sample {i} (pose={pose}): {e}")
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(poses)} samples...")
    
    print(f"\nFound {len(single_hand_data)} single-hand samples")
    print(f"Found {len(dual_hand_data)} dual-hand samples")
    
    # Create column names
    def make_column_names(num_hands):
        cols = ['pose']
        for h in range(num_hands):
            hand_prefix = f'h{h}_' if num_hands > 1 else ''
            for i in range(21):
                cols.append(f'{hand_prefix}x_{i}')
                cols.append(f'{hand_prefix}y_{i}')
        return cols
    
    # Save single-hand dataset
    if single_hand_data:
        df_single = pd.DataFrame(single_hand_data, columns=make_column_names(1))
        df_single['pose'] = df_single['pose'].astype(str)
        df_single.to_csv(OUTPUT_CSV_SINGLE, index=False)
        print(f"\n✓ Saved {len(df_single)} single-hand samples to {OUTPUT_CSV_SINGLE}")
        print(f"  Shape: {df_single.shape}")
        print(f"  Poses: {sorted(df_single['pose'].unique())[:20]}...")
    
    # Save dual-hand dataset
    if dual_hand_data:
        df_dual = pd.DataFrame(dual_hand_data, columns=make_column_names(2))
        df_dual['pose'] = df_dual['pose'].astype(str)
        df_dual.to_csv(OUTPUT_CSV_DUAL, index=False)
        print(f"\n✓ Saved {len(df_dual)} dual-hand samples to {OUTPUT_CSV_DUAL}")
        print(f"  Shape: {df_dual.shape}")
        print(f"  Poses: {sorted(df_dual['pose'].unique())[:20]}...")
    
    # Save combined dataset with indicator
    if single_hand_data or dual_hand_data:
        # Add hand count column and pad features to max size (84)
        all_data = []
        
        for row in single_hand_data:
            # Pad single hand (42 features) to 84 with zeros
            padded = row + [0.0] * 42
            all_data.append([row[0], 1] + padded[1:])  # pose, num_hands, features
        
        for row in dual_hand_data:
            all_data.append([row[0], 2] + row[1:])  # pose, num_hands, features
        
        cols_combined = ['pose', 'num_hands'] + make_column_names(2)[1:]  # Use dual-hand column names
        df_combined = pd.DataFrame(all_data, columns=cols_combined)
        df_combined['pose'] = df_combined['pose'].astype(str)
        df_combined['num_hands'] = df_combined['num_hands'].astype(int)
        df_combined.to_csv(OUTPUT_CSV_COMBINED, index=False)
        print(f"\n✓ Saved {len(df_combined)} combined samples to {OUTPUT_CSV_COMBINED}")
        print(f"  Shape: {df_combined.shape}")
        print(f"  Single-hand gestures: {(df_combined['num_hands'] == 1).sum()}")
        print(f"  Dual-hand gestures: {(df_combined['num_hands'] == 2).sum()}")


if __name__ == "__main__":
    main()
