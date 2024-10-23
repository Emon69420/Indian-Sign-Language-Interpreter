from flask import Flask, request, render_template
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model(r'C:\Users\emong\PycharmProjects\LangAisign\sign_language_model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if 'image' is part of the request
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    # Ensure a file has been selected
    if file.filename == '':
        return "No selected file"

    # Read the image from the uploaded file
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Preprocess the image as required by your model
    img = cv2.resize(img, (64, 64))  # Adjust size based on your model input
    img = img.astype('float32') / 255.0  # Normalize if necessary
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index

    # Return the predicted class (you might want to map this to the actual label)
    return f'Predicted class: {predicted_class}'


if __name__ == '__main__':
    app.run(debug=True)
