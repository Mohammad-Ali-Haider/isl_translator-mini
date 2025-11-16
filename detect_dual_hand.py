"""
Real-time detection for ISL with dual-hand support.
Handles both single-hand and dual-hand gestures.
"""

import cv2
import torch
import mediapipe as mp
import numpy as np

from helper import load_model


def normalize_hand(x_coords, y_coords):
    """Normalize a single hand's coordinates to [0, 1] relative to its bounding box."""
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    normalized = []
    for x, y in zip(x_coords, y_coords):
        normalized.extend([(x - x_min) / x_range, (y - y_min) / y_range])
    
    return normalized


def main():
    # Load label mapping
    try:
        label_mapping = np.load('label_mapping_dual.npy', allow_pickle=True).item()
        idx_to_label = label_mapping['idx_to_label']
        dual_hand_classes = set(label_mapping.get('dual_hand_classes', []))
        num_classes = len(idx_to_label)
        print(f"Loaded label mapping with {num_classes} classes")
        print(f"Single-hand gestures: {num_classes - len(dual_hand_classes)}")
        print(f"Dual-hand gestures: {len(dual_hand_classes)} - {sorted(dual_hand_classes)}")
    except FileNotFoundError:
        print("ERROR: label_mapping_dual.npy not found. Please run main_dual_hand.py first.")
        return

    # Model expects 84 features (42 per hand, padded for single hand)
    model = load_model("hand_gesture_model_dual.pth", input_size=84, num_classes=num_classes)
    
    print("\nUsing per-sample normalization (distance-invariant)")
    print("Supports both single-hand and dual-hand gestures")
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    # Initialize MediaPipe Hands with support for 2 hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Support up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n" + "="*60)
    print("Webcam started. Press 'q' to quit.")
    print("="*60)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)
            
            # Extract and normalize landmarks for all detected hands
            all_features = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                normalized = normalize_hand(x_coords, y_coords)
                all_features.extend(normalized)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Pad features to 84 if single hand detected
            if num_hands_detected == 1:
                all_features.extend([0.0] * 42)  # Pad with zeros for missing second hand
            
            features_np = np.array(all_features, dtype=np.float32)

            if features_np.shape[0] == 84:  # Verify correct feature size
                input_tensor = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    predicted_class = idx_to_label[predicted.item()]
                    confidence_score = confidence.item()

                # Determine if prediction matches hand count
                is_dual_hand_gesture = predicted_class in dual_hand_classes
                expected_hands = 2 if is_dual_hand_gesture else 1
                hand_count_match = (num_hands_detected == expected_hands)

                # Display prediction with confidence
                hand_info = f"{num_hands_detected}H detected"
                display_text = f'{predicted_class} ({confidence_score:.2f}) [{hand_info}]'
                
                # Color based on confidence and hand count match
                if confidence_score > 0.7 and hand_count_match:
                    color = (0, 255, 0)  # Green - high confidence, correct hand count
                elif confidence_score > 0.5:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                
                if not hand_count_match:
                    color = (0, 0, 255)  # Red - wrong hand count
                
                cv2.putText(frame, display_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Show expected hand count for current prediction
                expected_text = f'Expected: {expected_hands} hand(s)'
                cv2.putText(frame, expected_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Show top 3 predictions
                top_k = min(3, num_classes)
                top_probs, top_indices = torch.topk(probabilities[0], top_k)
                
                y_offset = 100
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    label = idx_to_label[idx.item()]
                    prob_val = prob.item()
                    hands_needed = "2H" if label in dual_hand_classes else "1H"
                    cv2.putText(frame, f'{i+1}. {label} ({hands_needed}): {prob_val:.2f}', 
                               (10, y_offset + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('ISL Detection (Dual-Hand Support)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed.")


if __name__ == "__main__":
    main()
