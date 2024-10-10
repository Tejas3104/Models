import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess(image):
    """
    Preprocess the input image for model prediction.
    - Convert RGBA to RGB if needed.
    - Resize to the expected input size of the model.
    - Convert the image to a numpy array.
    - Normalize pixel values (if required).
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    target_size = (224, 224)  # Example for VGG16
    image = image.resize(target_size)
    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    return image

def model_arc():
    """
    Build and return the model architecture based on VGG16.
    """
    model = Sequential()
    
    # Load pre-trained VGG16 model (without top layers)
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base layers
    for layer in vgg_base.layers:
        layer.trainable = False
    
    model.add(vgg_base)  # Add the pre-trained VGG16 base
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 classes for classification
    
    return model

def gen_labels():
    """
    Generate labels for the classes.
    Modify this list according to your specific classes.
    """
    return ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash", "Compost"]
