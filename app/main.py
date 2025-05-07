import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def add_image_analysis_feature():
    st.subheader("Scancer Ultrasound Image Detection Tool")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and preprocess the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert and preprocess the image
        image = image.convert("RGB")
        image_array = np.array(image)
        
        # Resize the image to the expected input size of the model
        image_array = cv2.resize(image_array, (128, 128))  # Resize to match model input size
        image_array = image_array.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        
        # Ensure the array has the correct shape for the model
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Load the TensorFlow model
        model = load_model("model/breast_cancer_model.keras")  # Updated file extension
        
        # Predict the class
        prediction = model.predict(image_array)
        class_labels = ["Benign", "Malignant", "Normal"]
        class_index = np.argmax(prediction[0])
        class_label = class_labels[class_index]
        
        st.write(f"The image is predicted to be: **{class_label}**")


def main():
    st.set_page_config(
        page_title="Scancer",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Scancer")
        st.write("This app helps diagnose breast cancer tissue sample. This app predicts using a machine learning model particularly deep learning to diagnose whether a breast mass is benign or malignant based on the measurements it receives from a cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        add_predictions(input_data)
    
    # Add the image analysis feature
    st.markdown("---")  # Optional: Add a separator line for clarity
    add_image_analysis_feature()  # Ensure this function is called to display the image upload button and results

if __name__ == '__main__':
    main()
