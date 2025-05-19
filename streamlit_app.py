import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import joblib
import os

# Load saved model and preprocessing pipeline
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
selector = joblib.load("selector.pkl")

# Constants
IMG_SIZE = (48, 48)
EXPECTED_FEATURES = 2816  # 2304 raw + 512 dummy VGG features

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set page
st.set_page_config(page_title="Gender Detection", layout="centered")
st.title("Cinsiyet Tahmini Uygulaması")
st.write("Bir yüz görseli yükleyin veya kameranızı kullanarak cinsiyeti tahmin edin.")

# Dummy VGG feature generator
def dummy_vgg_features():
    return np.zeros(512)

# Image preprocessing
def preprocess_image(image_pil):
    img = ImageOps.exif_transpose(image_pil.convert("L"))
    img_resized = img.resize(IMG_SIZE)
    face_np = np.array(img_resized)

    face_cv = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(face_cv, cv2.COLOR_RGB2GRAY)
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

    return all_features, img_resized

# --- Input Method Selector ---
input_method = st.radio("Giriş yöntemi seçin:", ("Görsel Yükle", "Kamera Kullan"))

# --- Camera Section ---
if input_method == "Kamera Kullan":
    st.subheader("Kamera ile Fotoğraf Çek")
    camera_input_image = st.camera_input("Fotoğraf çek")

    if camera_input_image is not None:
        try:
            image = Image.open(camera_input_image)
            features, preview = preprocess_image(image)

            if features is None:
                st.warning("Yüz algılanamadı. Lütfen tekrar deneyin.")
            else:
                st.image(preview, caption="Algılanan Yüz", width=150)
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]

                label = "Erkek" if prediction == 0 else "Kadın"
                st.success(f"Tahmin: {label}")
                st.write(f"Güven → Erkek: %{proba[0]*100:.2f}, Kadın: %{proba[1]*100:.2f}")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# --- Upload Section ---
elif input_method == "Görsel Yükle":
    st.subheader("Fotoğraf Yükleyin")
    uploaded_file = st.file_uploader("Bir yüz görseli yükleyin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            features, preview = preprocess_image(image)

            if features is None:
                st.warning("Yüz algılanamadı. Lütfen farklı bir görsel deneyin.")
            else:
                st.image(preview, caption="Yüklenen Görsel", width=150)
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]

                label = "Erkek" if prediction == 0 else "Kadın"
                st.success(f"Tahmin: {label}")
                st.write(f"Güven → Erkek: %{proba[0]*100:.2f}, Kadın: %{proba[1]*100:.2f}")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")
