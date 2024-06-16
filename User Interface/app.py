from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Flask Setup
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Model load
model = model = tf.keras.models.load_model("chest_xray_trained.h5")

# API endpoint
@app.route('/')
@cross_origin(supports_credentials=True)
def homepage():
    return "Chest X-ray Pneumonia Detector"

@app.route('/', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_image_processed():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    # st.image(image, use_column_width=True)

    index, conf_score = predict_image(image, model)

    data = []
    data.append({
            "result": ["Healthy", "Pneumonia"][index],
            # "confScore": conf_score
        })
        
    return jsonify(data)


def predict_image(image, model,threshold=0.5):
    # Resize and fit the image to (256, 256) using Lanczos resampling
    image = ImageOps.fit(image, (256, 256), Image.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalize pixel values to [-1, 1] 
    normalized_image_array = (image_array.astype(np.float32) / 255.0) * 2.0 - 1.0

    # Create a numpy array with shape (1, 256, 256, 3) to match model input
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Perform prediction using the model
    prediction = model.predict(data)

    print("-------------" )
    print(prediction[0][0])
    index = 0 if prediction[0][0] > threshold else 1
    confidence_score = prediction[0][0]  # Confidence score of the predicted class

    # Print prediction for debugging
    print("=======================================")
    print(prediction)
    print(index)
    print(confidence_score)
    print("=======================================")
    return index, confidence_score

if __name__ == '__main__':
    app.run(debug=False)
