from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('breast_cancer_model.h5')

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/service.html')
def understand_bc():
    return render_template('service.html')

@app.route('/feature.html')
def detection_tool():
    return render_template('feature.html')

@app.route('/team.html')
def risk_assessment_tool():
    return render_template('team.html')

@app.route('/appointment.html')
def book_appointment():
    return render_template('appointment.html')

@app.route('/testimonial.html')
def testimonials():
    return render_template('testimonial.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))  # Resize to match your model input
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        return jsonify({
            'result': predicted_label
        })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'result': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'result': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process the image with your model
        result = predict_from_file(filepath)

        return jsonify({'result': result})
    
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # Adjust size to your model's input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict_from_file(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])  # Example for classification
    return class_labels[predicted_class]

if __name__ == '__main__':
    app.run(debug=True)
