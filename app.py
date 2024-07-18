from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('breast_cancer_model.h5')

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

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
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
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
            'prediction': predicted_label,
            'confidence': float(np.max(prediction))
        })

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(debug=True)
