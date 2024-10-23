import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model(r'C:\Users\emong\PycharmProjects\LangAisign\sign_language_landmark_model.h5')

# Define the actions (labels) based on your dataset
actions = os.listdir(r'C:\Users\emong\PycharmProjects\LangAisign\own-data')  # Assuming the labels are the folder names

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks


# Function to preprocess input images
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (64, 64))  # Resize to match model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image / 255.0  # Normalize pixel values


# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe to detect hands
    results = hands.process(rgb_frame)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Crop the region of interest (ROI) containing the hand
        # For simplicity, we'll use the whole frame, but you can adjust it to a smaller region
        processed_frame = preprocess_image(frame)

        # Make prediction
        prediction = model.predict(processed_frame)
        predicted_label = actions[np.argmax(prediction)]

        # Display the prediction on the frame
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        # If no hand is detected, show a message
        cv2.putText(frame, 'No hand detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the video frame
    cv2.imshow('Sign Language Interpreter', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
