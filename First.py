import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from utils import preprocess, model_arc, gen_labels

# Function to download a file from GitHub
def download_file_from_github(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        st.write(f"Downloaded: {output_path}")
    else:
        st.error(f"Failed to download file from {url}")

# Paths to the files on GitHub
model_url = 'https://github.com/Tejas3104/Models/raw/main/keras.h5'
labels_url = 'https://github.com/Tejas3104/Models/raw/main/labels.txt'

# Local paths where the files will be saved
model_path = './models/keras.h5'
labels_path = './models/labels.txt'

# Ensure the directory exists
os.makedirs('./models', exist_ok=True)

# Download the model and labels if not present
if not os.path.exists(model_path):
    st.write("Downloading model from GitHub...")
    download_file_from_github(model_url, model_path)

if not os.path.exists(labels_path):
    st.write("Downloading labels from GitHub...")
    download_file_from_github(labels_url, labels_path)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model = model_arc()  # Get the architecture from utils.py
    if os.path.exists(model_path):
        model.load_weights(model_path)  # Load saved weights
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

    # Read and load class labels from the labels.txt file
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()

    # Get the predicted class index and label
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = labels[predicted_class[0]]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_label}")
