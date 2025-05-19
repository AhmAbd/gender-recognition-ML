import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import joblib
import os

# Load model and preprocessing pipeline
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
selector = joblib.load("selector.pkl")

# Constants
IMG_SIZE = (48, 48)
VGG_FEATURES = 512

# Dummy VGG features since they're only used during training
def dummy_vgg_features():
    return np.zeros(VGG_FEATURES)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set up app UI
st.set_page_config(page_title="Gender Detection", layout="centered")
st.title("Gender Prediction from Face")
st.write("Upload a face image or use your webcam to take one.")

# Image preprocessing
def preprocess_image(image_pil):
    img = ImageOps.exif_transpose(image_pil.convert("L"))
    img_resized = img.resize(IMG_SIZE)
    face_np = np.array(img_resized)

    img_rgb = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(40, 40))

    if len(faces) == 0:
        return None, img_resized

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)

    raw_features = face.flatten()
    vgg_features = dummy_vgg_features()
    all_features = np.concatenate([raw_features, vgg_features])

    all_features = scaler.transform([all_features])
    if pca:
        all_features = pca.transform(all_features)
    if selector:
        all_features = selector.transform(all_features)

    return all_features, Image.fromarray(face)

# Input method selector
input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# === Webcam ===
if input_method == "Use Webcam":
    st.subheader("Take a Photo")
    captured_image = st.camera_input("Capture a face image")

    if captured_image is not None:
        try:
            image = Image.open(captured_image)
            features, face_preview = preprocess_image(image)

            if features is None:
                st.warning("No face detected. Try again.")
            else:
                st.image(face_preview, caption="Detected Face", width=150)
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                label = "Man" if pred == 0 else "Woman"
                st.success(f"Prediction: {label}")
                st.write(f"Confidence → Man: {proba[0]*100:.2f}%, Woman: {proba[1]*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

# === Upload ===
elif input_method == "Upload Image":
    st.subheader("Upload a Face Image")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            features, face_preview = preprocess_image(image)

            if features is None:
                st.warning("No face detected. Try another image.")
            else:
                st.image(face_preview, caption="Detected Face", width=150)
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                label = "Man" if pred == 0 else "Woman"
                st.success(f"Prediction: {label}")
                st.write(f"Confidence → Man: {proba[0]*100:.2f}%, Woman: {proba[1]*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")
