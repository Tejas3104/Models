import streamlit as st
import numpy as np
import os
from PIL import Image
from utils import preprocess, model_arc, gen_labels
import gdown  # Importing gdown to download from Google Drive

# Function to download the model from Google Drive
def download_model_from_drive():
    file_id = '1yNbICx_rQWw-fwGsIdkAXVQrg3ct8PDn'  # Your new Google Drive file ID
    url = f'https://drive.google.com/uc?id={file_id}'
    output = './weights/modelnew.weights.h5'  # Path to save the downloaded file

    # Check if the model weights are already downloaded, if not, download them
    if not os.path.exists(output):
        st.write("Downloading model weights from Google Drive...")
        os.makedirs('./weights', exist_ok=True)  # Ensure the weights folder exists
        gdown.download(url, output, quiet=False)
    else:
        st.write("Model weights already downloaded.")

# Download the model if not present
download_model_from_drive()

# Path to the downloaded model weights
model_weights_path = './weights/modelnew.weights.h5'

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
    
    # Get class labels (you need to define these in utils.py)
    labels = gen_labels()
    predicted_label = labels[predicted_class[0]]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_label}")
