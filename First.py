import streamlit as st
import os
from keras.models import load_model
from keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D class to handle loading without 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove unsupported 'groups' argument if present
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Function to load the model
def load_model_func():
    model_path = 'keras_model.h5'  # or provide the absolute path
    print(f"Trying to load model from: {model_path}")
    
    # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model with custom_objects
    model = load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print("Model loaded successfully.")
    return model

# Load the labels from the labels file
def load_labels():
    labels_path = 'labels.txt'  # or provide the absolute path
    print(f"Trying to load labels from: {labels_path}")

    # Check if the labels file exists
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
    print("Labels loaded successfully.")
    return labels

# Check the current working directory
print("Current Working Directory:", os.getcwd())

# Load the model and labels when the app starts
try:
    model = load_model_func()
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    labels = load_labels()
except Exception as e:
    st.error(f"Error loading labels: {e}")

# Streamlit app layout
st.title("Waste Classification App")
st.write("Upload an image of waste to classify it.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Here you can process the uploaded image and make predictions using the model
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.success("Image uploaded successfully!")
    
    # Add your model prediction code here
    # Example:
    # image = preprocess_image(uploaded_file)
    # predictions = model.predict(image)
    # predicted_label = labels[np.argmax(predictions)]
    # st.write(f"Predicted label: {predicted_label}")
