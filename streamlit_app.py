import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load saved model and preprocessing pipeline
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
selector = joblib.load("selector.pkl")

# Constants
IMG_SIZE = (48, 48)
EXPECTED_FEATURES = 2816  # 2304 raw + 512 VGG

# VGG16 feature extractor
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block5_pool").output)

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# UI layout
st.set_page_config(page_title="Gender Prediction", layout="centered")
st.title("Gender Prediction from Face Image")
st.write("Upload a face image or take a photo with your webcam to predict gender.")

# --- Preprocessing function ---
def preprocess_image(pil_image):
    img = ImageOps.exif_transpose(pil_image.convert("RGB"))
    open_cv_image = np.array(img)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(40, 40))
    if len(faces) == 0:
        return None, img

    (x, y, w, h) = faces[0]
    face = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face, IMG_SIZE)

    # Raw features
    raw_flattened = face_resized.flatten()

    # VGG features
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    face_rgb_resized = cv2.resize(face_rgb, IMG_SIZE)
    arr = img_to_array(face_rgb_resized)
    arr = np.expand_dims(arr, axis=0)
    vgg_features = vgg_model.predict(arr, verbose=0).flatten()

    combined = np.concatenate([raw_flattened, vgg_features])
    combined = scaler.transform([combined])
    if pca is not None:
        combined = pca.transform(combined)
    if selector is not None:
        combined = selector.transform(combined)

    return combined, Image.fromarray(face_resized)

# --- UI control ---
input_method = st.radio("Select Input Method", ("Upload Image", "Use Webcam"))

# --- Upload Section ---
if input_method == "Upload Image":
    st.subheader("Upload a Face Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            features, preview = preprocess_image(image)

            if features is None:
                st.warning("No face detected. Try another image.")
            else:
                st.image(preview, caption="Detected Face", width=150)
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                label = "Man" if pred == 0 else "Woman"
                st.success(f"Prediction: {label}")
                st.write(f"Confidence → Man: %{proba[0]*100:.2f}, Woman: %{proba[1]*100:.2f}")
        except Exception as e:
            st.error(f"Error occurred: {e}")

# --- Webcam Section ---
elif input_method == "Use Webcam":
    st.subheader("Take a Photo")
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        try:
            image = Image.open(camera_image)
            features, preview = preprocess_image(image)

            if features is None:
                st.warning("No face detected. Try again.")
            else:
                st.image(preview, caption="Detected Face", width=150)
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                label = "Man" if pred == 0 else "Woman"
                st.success(f"Prediction: {label}")
                st.write(f"Confidence → Man: %{proba[0]*100:.2f}, Woman: %{proba[1]*100:.2f}")
        except Exception as e:
            st.error(f"Error occurred: {e}")
