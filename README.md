# Sign Language Detection Using CNN and MediaPipe

This project implements a real-time indian sign language recognition system using convolutional neural networks (CNN) and hand landmarks detected via MediaPipe. The goal is to classify different hand gestures into corresponding sign language characters. (sign language -> English)

## Table of Contents
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Dataset Collection](#dataset-collection)
- [Model Training](#model-training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Usage](#usage)

## Project Structure
```
├── own-data/label              # Directory for storing labeled hand gesture images
├── modelTraining.py            # Script for building and training the CNN model
├── imageVasooli.py             # Script for collecting gesture images using a webcam, vasooli is a hindi word for collection
├── endGame.py                  # Script for real-time sign language detection
├── X_data.npy                  # Preprocessed image dataset (hand gesture landmarks)
├── y_labels.npy                # Labels corresponding to the images in the dataset
├── sign_language_landmark_model.h5   # Saved trained model
└── README.md                   # This file
```

## How it Works

Data Collection: A webcam is used to capture images of hand gestures. MediaPipe is utilized to detect hand landmarks, which are then processed and stored as images in the dataset.

Model Training: A Convolutional Neural Network (CNN) is built using TensorFlow and Keras. The model is trained on hand gesture images that have been preprocessed to 64x64 pixels. It learns to recognize and classify gestures into corresponding sign language characters.

Real-time Prediction: Once the model is trained, it can predict sign language gestures in real-time by processing frames captured via the webcam.

## Image Collection
Run the data_collection.py script to capture hand gesture images.

Press a letter key (A-Z) or number key (0-9) to label the gesture, and the corresponding ROI image will be saved. Please turn on caps lock while pressing these 

This would create a new folder ./own-data and store images under a folder with the name same as that of the letter you pressed during the image collection process, structure would be 

./own-data/{labelName}/allTheImages

## Processing the data
run the file dataProcessor.py to add landmarks on the images captured, then it makes it a numpy array that then stores the images in one array and labels in another {X_data has images with landmark and y_labels has the corresponding labels}

## Model Training

Use the modelTraining.py script to train the CNN Model

The dataset (X_data.npy and y_labels.npy) is loaded, and the model is trained to recognize gestures.

The trained model is saved as sign_language_landmark_model.h5

## Testing and Evaluation
After training, the model is evaluated using the test dataset. The performance metrics, such as accuracy, will be printed at the end of the training.

## Usage
To run real-time sign language recognition 
The system will start the webcam and detect hand gestures in real-time, displaying the predicted gesture on the screen.

For accurate results use a large data collection with different hand sizes, skin colors, lighting conditions and different backgrounds

