import streamlit as st
import numpy as np
import os
from PIL import Image
import requests
from utils import preprocess, model_arc, gen_labels  # Import model_arc from utils

# Function to download a file from GitHub
def download_file_from_github(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        st.success(f"Downloaded: {output_path}")
    else:
        st.error(f"Failed to download: {url}")

# URLs to your files in GitHub
keras_file_url = 'https://raw.githubusercontent.com/Tejas3104/Models/main/keras.h5'
labels_file_url = 'https://raw.githubusercontent.com/Tejas3104/Models/main/labels.txt'

# Download the files
download_file_from_github(keras_file_url, './keras.h5')
download_file_from_github(labels_file_url, './labels.txt')

# Load the model weights
model_weights_path = './keras.h5'
labels_path = './labels.txt'

# Load the labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model = model_arc()  # Get the architecture from utils.py
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)  # Load saved weights
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

# Load the model when the app starts
model = load_model()

# Define background and UI
background_image_url = "https://png.pngtree.com/thumb_back/fh260/background/20220217/pngtree-green-simple-atmospheric-waste-classification-illustration-background-image_953325.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Waste Classification Model")
st.write("Upload an image of waste for classification.")

# File uploader widget for image input
image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader_1")

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the uploaded image
    image_array = preprocess(image)

    # Predict using the loaded model
    prediction = model.predict(image_array)

    # Get the predicted class index and label
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = labels[predicted_class[0]]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_label}")
