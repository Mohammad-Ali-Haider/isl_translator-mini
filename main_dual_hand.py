"""
Training script for ISL with dual-hand support.
Handles both single-hand and dual-hand gestures with per-sample normalization.
"""

import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np

from helper import save_model, load_model
from Model import LinearModel


CSV_PATH = "./hand_landmarks_xy_normalized_all.csv"


def main():
    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    print(f"Loading data from {CSV_PATH} ...")
    data = pd.read_csv(CSV_PATH)

    print(f"Total samples: {len(data)}")
    print(f"\nHand distribution:")
    print(data["num_hands"].value_counts())
    print("\nPose distribution (first 20 classes):")
    print(data["pose"].value_counts().head(20))

    # ------------------------------------------------------------------
    # Prepare X, y with label mapping
    # ------------------------------------------------------------------
    X = data.drop(columns=['pose', 'num_hands']).values  # All features (padded to 84)
    y_labels = data['pose'].values.astype(str)
    num_hands_info = data['num_hands'].values  # Track which samples use 1 or 2 hands

    # Create label mapping
    unique_labels = sorted(set(y_labels), key=lambda x: (x.isalpha(), x))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert string labels to numeric indices
    y = np.array([label_to_idx[label] for label in y_labels])
    
    num_features = X.shape[1]  # Should be 84 (42 per hand, padded)
    num_classes = len(unique_labels)
    
    print(f"\nNumber of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping (first 10): {list(label_to_idx.items())[:10]}...")
    
    # Identify which classes use dual hands
    dual_hand_classes = set()
    for label, num_hands in zip(y_labels, num_hands_info):
        if num_hands == 2:
            dual_hand_classes.add(label)
    
    print(f"\nDual-hand gestures: {sorted(dual_hand_classes)}")
    
    # Save label mapping and hand info for inference
    np.save('label_mapping_dual.npy', {
        'label_to_idx': label_to_idx, 
        'idx_to_label': idx_to_label,
        'dual_hand_classes': list(dual_hand_classes)
    })
    print(f"Label mapping saved to label_mapping_dual.npy")

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=1234, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=1234, stratify=y_temp
    )

    print(f"\nTraining samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print("Note: Features are per-sample normalized (distance-invariant)")
    print("      Supports both single-hand (42 features) and dual-hand (84 features)")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = LinearModel(input_size=num_features, num_classes=num_classes, dropout_rate=0.4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # ------------------------------------------------------------------
    # Training with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0
    max_epochs = 300
    model_path = "hand_gesture_model_dual.pth"

    print("\n" + "="*60)
    print("Starting training (with dual-hand support)...")
    print("="*60)

    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val_tensor).float().mean()

        # Logging
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{max_epochs} - "
                f"Train Loss: {train_loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"Val Acc: {val_accuracy.item():.4f}"
            )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Loading Best Model for Evaluation")
    print("="*60)
    inference_model = load_model(model_path, input_size=num_features, num_classes=num_classes)

    inference_model.eval()
    with torch.no_grad():
        test_outputs = inference_model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(test_outputs, 1)
        predicted_np = predicted.numpy()

        test_accuracy = (predicted_np == y_test).mean()
        print(f"\nOverall Test Accuracy: {test_accuracy:.4f}")

        print("\nPer-class accuracy:")
        unique_classes = sorted(set(y_test))
        for class_idx in unique_classes:
            class_mask = (y_test == class_idx)
            if class_mask.sum() > 0:
                class_acc = (predicted_np[class_mask] == y_test[class_mask]).mean()
                original_label = idx_to_label[class_idx]
                hand_type = "2H" if original_label in dual_hand_classes else "1H"
                print(f"  {original_label} ({hand_type}): {class_acc:.4f} ({class_mask.sum()} samples)")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Label mapping saved to: label_mapping_dual.npy")


if __name__ == "__main__":
    main()
