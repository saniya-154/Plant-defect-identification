import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import requests
import gdown

# Load remedies JSON
working_dir = os.path.dirname(os.path.abspath(__file__))
remedies_path = os.path.join(working_dir,"data" , "disease_remedies.json")

with open(remedies_path, "r") as f:
    disease_remedies = json.load(f) 

# Set page config for a better UI experience
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")


# Define working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_dir = os.path.join(working_dir, "trained_model")
model_path = os.path.join(trained_model_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Make sure the trained_model directory exists
os.makedirs(trained_model_dir, exist_ok=True)

# Check if model exists, if not download
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id=1_qbU34eVwpH4hthBmSzpZUZGIFges8ZI", "model.h5", quiet=False, fuzzy=True)


# Load the pre-trained model and class indices
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

def load_and_preprocess_image(image, target_size=(224, 224)):
    """Preprocesses an uploaded image for model prediction."""
    img = Image.open(image).resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image_class(model, image, class_indices):
    """Runs model prediction on the uploaded image."""
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence_score = np.max(predictions) * 100  # Convert to percentage
    return predicted_class_name, confidence_score


# Streamlit UI
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🌿 Plant Disease Classifier</h1>
    <p style='text-align: center; color: #7f8c8d;'>Upload a plant leaf image to detect possible diseases.</p>
""", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Drag and drop an image or click to browse", type=["jpg", "jpeg", "png"], help="Upload an image of a plant leaf")

if uploaded_image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_image).resize((200, 200))
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown("**Click the button below to classify the disease.**")
        if st.button("🔍 Classify", help="Click to analyze the uploaded image"):
            with st.spinner("Analyzing the image..."):
                predicted_class_name, confidence_score = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"**Prediction:** {predicted_class_name}")
                st.info(f"**Confidence:** {confidence_score:.2f}%")

                # Show disease information
                remedy_info = disease_remedies.get(predicted_class_name)

                if remedy_info:
                    st.subheader("🩺 Disease Information")
                    st.write(f"**Description:** {remedy_info.get('description', 'No description available.')}")
                    st.write(f"**Remedy:** {remedy_info.get('remedy', 'No remedy available.')}")
                    st.write(f"**Organic Remedy:** {remedy_info.get('organic_remedy', 'No organic remedy available.')}")
                else:
                    st.warning("No remedy information available for this disease.")            


# Add a footer with styling
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            width: 50%;
            text-align: center;
            color: gray;
        }
    </style>
    <div class="footer">
        Developed By CodeCraft
    </div>
""", unsafe_allow_html=True)