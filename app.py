import tensorflow.keras
from PIL import Image,ImageOps
import numpy as np

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

with ope('labels.txt','r') as f:
  class_names = f.read().split('\n')

data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)

image = Image.open('test.jpg')

size = (224,224)
image = ImageOps.fit(image,size,Image.ANTIALIAS)

image_array = np.asaray(image)

image.show()

normalized_image_array=(image_array.astype(np.float32)/127.0)-1

data[0]=normalized_image_array

prediction = model.predict(data)

print(prediction)

index=np.argmax(prediction)

class_name=class_names[index]

confidence_score=prediction[index]

print("Class : ", class_name)
print("Confidence Score : ", confidence_score)


