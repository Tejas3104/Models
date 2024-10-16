import streamlit as st
import numpy as np
import os
from PIL import Image
import requests

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

# Your existing code...
@st.cache_resource
def load_model():
    model = model_arc()  # Get the architecture from utils.py
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)  # Load saved weights
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

model = load_model()

# Rest of your Streamlit app code...
