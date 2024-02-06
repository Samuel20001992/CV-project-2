import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import lime
from lime import lime_image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('model.h5')  

# Define image dimensions expected by the model
img_width, img_height = 150, 150

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    return img_array

# Function to explain the prediction using Lime
def explain_prediction(image, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    return explanation

# Function to overlay boundaries on the original image
def overlay_boundaries(original_image, explanation):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imshow(temp)
    plt.imshow(mask, alpha=0.5, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('overlayed_image.png')
    overlayed_image = Image.open('overlayed_image.png')
    return overlayed_image

# Streamlit App
st.title("Insect Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess the image
    img_resized = image.resize((img_width, img_height))
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img_array)
    prediction_label = "Ant" if prediction < 0.5 else "Bee"
    st.write("Prediction: ", prediction_label)

    # Explain the prediction using Lime
    explanation = explain_prediction(img_resized, model)
    st.write("Explanation for Prediction:")
    
    # Overlay boundaries on the original image
    overlayed_image = overlay_boundaries(image, explanation)
    st.image(overlayed_image, caption='Explanation with Boundaries', use_column_width=True)
