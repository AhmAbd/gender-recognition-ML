import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Gender Prediction", layout="centered")
st.title("Gender Recognition from Image")
st.write("Upload a face image or take a webcam snapshot to predict gender.")

# === Load Saved Artifacts ===
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"
SELECTOR_PATH = "selector.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    selector = joblib.load(SELECTOR_PATH)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# === Load VGG16 model for feature extraction ===
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(48, 48, 3))
vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block5_pool").output)

# === Load face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    st.error("Could not load face detector.")
    st.stop()

def extract_features(image_array):
    rgb_img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb_img, (48, 48))
    img_array = img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg_model.predict(img_array, verbose=0)
    return features.reshape(1, -1)

def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(40, 40))
    
    if len(faces) == 0:
        st.warning("No face detected in the image.")
        return None, None

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face_pil = Image.fromarray(face).resize((48, 48))
    face_array = np.array(face_pil).flatten()

    vgg_features = extract_features(np.array(face_pil))
    full_features = np.concatenate([face_array, vgg_features.flatten()])

    if full_features.shape[0] != 2816:
        st.error(f"Expected 2816 features, got {full_features.shape[0]}")
        return None, None

    # Scale
    scaled = scaler.transform([full_features])

    # PCA
    if pca is not None:
        scaled = pca.transform(scaled)

    # Selector
    if selector is not None:
        scaled = selector.transform(scaled)

    return scaled, face_pil

# === Image Upload ===
st.subheader("📁 Upload Image")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        pil_image = Image.open(uploaded_file)
        features, preview = preprocess_image(pil_image)
        if features is not None:
            st.image(preview, caption="Detected Face", width=200)

            prediction = model.predict(features)[0]
            probas = model.predict_proba(features)[0]

            label = "Man" if prediction == 0 else "Woman"
            st.success(f"Prediction: {label}")
            st.write(f"Confidence → Man: {probas[0]*100:.2f}%, Woman: {probas[1]*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {e}")
