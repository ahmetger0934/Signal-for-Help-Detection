import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.ndimage import rotate

# Load the trained model
model = tf.keras.models.load_model('best_hand_gesture_model.keras')

# Assuming your model has a class label for the "Signal for Help"
signal_for_help_label_index = 0  # Adjust this based on your model's label for "Signal for Help"

# Initialize Mediapipe Hands with enhanced settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Mediapipe Drawing for visualization
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Set higher resolution for better detection at a distance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Function to apply CLAHE to the ROI to enhance contrast
def apply_clahe(roi):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(roi)


# Function to augment hand ROI for better recognition
def augment_hand_roi(hand_roi):
    augmented_images = []

    # Flip the hand region horizontally
    flipped_hand = cv2.flip(hand_roi, 1)
    augmented_images.append(flipped_hand)

    # Rotate the hand region
    for angle in [90, 180, 270]:
        rotated_hand = rotate(hand_roi, angle, reshape=False)
        augmented_images.append(rotated_hand)

    # Scaling to simulate distance variation
    for scale in [0.8, 1.2]:
        scaled_hand = cv2.resize(hand_roi, (0, 0), fx=scale, fy=scale)
        if scaled_hand.shape[0] > 0 and scaled_hand.shape[1] > 0:
            scaled_hand = cv2.resize(scaled_hand, (28, 28))
            augmented_images.append(scaled_hand)

    return augmented_images


# Function to preprocess the hand ROI
def preprocess_hand_roi(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = apply_clahe(roi)  # Enhance contrast using CLAHE
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    return roi


# Main loop for processing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the bounding box around the hand
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Convert the bounding box coordinates to pixel values
            h, w, _ = frame.shape
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            x_max = int(x_max * w)
            y_max = int(y_max * h)

            # Ensure the bounding box is within the frame dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Extract the region of interest (ROI) within the bounding box
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # If the ROI is detected
            if hand_roi.size > 0:
                hand_roi_preprocessed = preprocess_hand_roi(hand_roi)
                augmented_rois = augment_hand_roi(hand_roi_preprocessed)
                augmented_rois.append(hand_roi_preprocessed)  # Include the original ROI

                for aug_roi in augmented_rois:
                    if aug_roi.shape == (28, 28, 1):  # Ensure the shape is correct
                        aug_roi = np.expand_dims(aug_roi, axis=0)

                        # Predict the gesture on augmented ROIs
                        prediction = model.predict(aug_roi)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)

                        # Check if the detected gesture is the "Signal for Help"
                        if predicted_class == signal_for_help_label_index and confidence > 0.75:
                            print("WARNING: Signal for Help Detected! The person may be in trouble!")

                            # Draw a red bounding box around the detected hand
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                            # Display the warning on the screen
                            label_text = f'SIGNAL FOR HELP DETECTED ({confidence:.2f})'
                            cv2.putText(frame, label_text, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 0, 255), 2, cv2.LINE_AA)

                            break  # Exit the loop once the signal is detected

    # Display the frame
    cv2.imshow('Signal for Help Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
hands.close()
cv2.destroyAllWindows()