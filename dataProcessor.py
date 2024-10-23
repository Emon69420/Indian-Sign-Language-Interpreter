import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Load data with added print statements to know progress
def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    print(f"Found {len(class_names)} classes to process.")

    for label in class_names:
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            print(f"Processing class: {label}")
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:  # Check if the image is read correctly
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64
                    images.append(img)
                    labels.append(label)

                    # Apply landmark extraction
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    if results.multi_hand_landmarks:
                        print(f"Hand landmarks found in image: {img_file}")
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        print(f"No hand landmarks found in image: {img_file}")
                else:
                    print(f"Could not read image: {img_path}")

    hands.close()  # Close the MediaPipe Hands solution
    return np.array(images), np.array(labels)


# Save the processed data to file for future use
def save_data(X, y, X_filename='X_data.npy', y_filename='y_labels.npy'):
    np.save(X_filename, X)
    np.save(y_filename, y)
    print(f"Data saved to {X_filename} and {y_filename}")


# Load your dataset
data_dir = r'C:\Users\emong\PycharmProjects\LangAisign\own-data'
X, y = load_data(data_dir)

# Convert labels to integers
label_to_index = {label: index for index, label in enumerate(np.unique(y))}
y = np.array([label_to_index[label] for label in y])

# Save the data
save_data(X, y)
