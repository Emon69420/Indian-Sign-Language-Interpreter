import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the ROI size (region where the hand will be captured)
roi_top, roi_bottom, roi_right, roi_left = 100, 400, 100, 400

# Frame count for file naming
frame_count = 0

# Function to create directory if it doesn't exist
def create_folder(label):
    folder_path = f'own-data/{label}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Function to save image in the respective folder
def save_image(frame, label):
    global frame_count
    folder_path = create_folder(label)
    img_path = os.path.join(folder_path, f'{label}_{frame_count}.jpg')
    cv2.imwrite(img_path, frame)
    frame_count += 1
    print(f"Image saved: {img_path}")

# Function to preprocess input images
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (64, 64))  # Resize to match model input
    return image / 255.0  # Normalize pixel values

# Start capturing video
print("Press the corresponding key to save the image in the respective folder. Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame (optional, for mirror effect)
    frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]  # Extract region of interest (ROI)

    # Convert the ROI to RGB for MediaPipe processing
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Process the ROI to detect hands
    results = hands.process(rgb_roi)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on ROI
            mp_drawing.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with ROI
    cv2.imshow('Data Collection - Sign Language', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If a key is pressed, save the ROI image to the corresponding folder
    if key in range(ord('A'), ord('Z') + 1):  # For letters A-Z
        label = chr(key)  # Convert key press to corresponding label (e.g., 'A')
        save_image(roi, label)
    elif key in range(ord('0'), ord('9') + 1):  # For digits 0-9
        label = chr(key)  # Convert key press to corresponding label (e.g., '1')
        save_image(roi, label)

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
