from tensorflow import keras  # TensorFlow is required for Keras to work
import tensorflow as tf
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Define and register the custom layer before loading the model
from tensorflow.keras.layers import DepthwiseConv2D


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' from kwargs if it exists
        kwargs.pop('groups', None)
        super().__init__(**kwargs)


# Register the custom layer
tf.keras.utils.get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

"""
# 얼굴상 판단기
"""

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

uploaded_file = st.file_uploader("얼굴 사진을 업로드 해주세요.", type=['jpeg', 'png', 'jpg', 'webp'])

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

col1, col2, col3 = st.columns([1,2,1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col2:
        st.image(image, use_column_width=True)

    # Ensure image is in RGB mode
    image = image.convert('RGB')

    # Resize the image to a 224x224 with the same strategy as in TM2:
    size = (224, 224)

    # Resize and crop the image to maintain the aspect ratio
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    face_types = {"0 dog": "강아지상", "1 cat": "고양이상", "2 bear": "곰상", "3 dinosaur": "공룡상", "4 rabbit": "토끼상"}

    # Display the results
    st.markdown("<h1 style='text-align: center; color: white;'>"f"얼굴상: {face_types[class_name]}</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>"f"수치: {confidence_score*100:.0f}%""</h1>", unsafe_allow_html=True)

    # Optionally display the class name and confidence score on the image
    # st.image(image, caption=f"{class_name} ({confidence_score:.2f})")

    # Print prediction and confidence score to the console
    print("Class:", class_name)
    print("Confidence Score:", confidence_score)
